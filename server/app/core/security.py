from datetime import datetime, timezone, timedelta
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.core.database import get_db
from app.models.user import User

bearer_scheme = HTTPBearer()

# Local email+password auth uses bcrypt directly. (passlib 1.7.x cannot drive
# the installed bcrypt 5.x backend — its version probe + $2$ workaround crash —
# so we call the bcrypt library, which is the same algorithm passlib wraps.)
# bcrypt rejects secrets longer than 72 bytes, so we truncate explicitly.
def _to_72_bytes(plain: str) -> bytes:
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    """Return a bcrypt hash (utf-8 string) of the given plaintext password."""
    return bcrypt.hashpw(_to_72_bytes(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash. Never raises."""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(_to_72_bytes(plain), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict) -> str:
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_access_token(credentials.credentials)
    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def require_developer(user: User = Depends(get_current_user)) -> User:
    """Passes for admin or developer roles. Blocks plain members."""
    if user.role not in ("admin", "developer") and not user.is_admin:
        raise HTTPException(status_code=403, detail="Developer or admin access required")
    return user


# ---------------------------------------------------------------------------
# Org-RBAC v2 guards (Lane B)
#
# Additive: these new dependencies are only mounted on the new org-scoped
# endpoints. Existing routes keep using require_admin / require_developer, so
# behavior is unchanged unless a route explicitly opts in.
# ---------------------------------------------------------------------------

# Roles considered org-level administrators (full org visibility).
_ORG_ADMIN_ROLES = ("org_owner", "org_admin")


async def require_platform_admin(user: User = Depends(get_current_user)) -> User:
    """403 unless the user is a platform-level superuser."""
    if not getattr(user, "is_platform_admin", False):
        raise HTTPException(status_code=403, detail="Platform admin access required")
    return user


async def require_google_sso(user: User = Depends(get_current_user)) -> User:
    """403 unless the user authenticated via Google SSO."""
    if getattr(user, "auth_provider", None) != "google":
        raise HTTPException(status_code=403, detail="Google SSO required")
    return user


async def require_org_owner(user: User = Depends(get_current_user)) -> User:
    """403 unless role == 'org_owner'. Composes require_google_sso —
    org owners MUST be Google-authenticated."""
    await require_google_sso(user)
    if user.role != "org_owner":
        raise HTTPException(status_code=403, detail="Organization owner access required")
    return user


def require_org_role(*roles: str):
    """Factory → dependency that passes only if user.role is in `roles`."""

    async def _guard(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of roles: {sorted(roles)}",
            )
        return user

    return _guard


async def require_org_context(user: User = Depends(get_current_user)) -> User:
    """403 if the user has no organization — "no org => nothing"."""
    if getattr(user, "organization_id", None) is None:
        raise HTTPException(status_code=403, detail="No organization context")
    return user


async def domain_scoped_guard(domain: str, user: User, db: AsyncSession) -> bool:
    """Authorize a user for a specific domain within their organization.

    Org owners / org admins are allowed for every domain in their org.
    Everyone else must have a matching ManagerDomainAssignment row.

    Raises 403 when not authorized; returns True when allowed.
    """
    if user.role in _ORG_ADMIN_ROLES:
        return True

    from app.models.manager_domain_assignment import ManagerDomainAssignment

    org_id = getattr(user, "organization_id", None)
    if org_id is None:
        raise HTTPException(status_code=403, detail="No organization context")

    row = await db.execute(
        select(ManagerDomainAssignment.id).where(
            ManagerDomainAssignment.user_id == user.id,
            ManagerDomainAssignment.organization_id == org_id,
            ManagerDomainAssignment.domain_tag == domain,
        )
    )
    if row.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail=f"No access to domain '{domain}'")
    return True

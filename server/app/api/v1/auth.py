import time

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.config import Config

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logger import auth_logger, db_logger
from app.core.security import create_access_token, verify_password
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.auth import TokenOut, UserOut


class LoginBody(BaseModel):
    email: EmailStr
    password: str

router = APIRouter(prefix="/auth", tags=["auth"])

# ── OAuth setup ──
settings = get_settings()
oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


@router.get("/google/login")
async def google_login(request: Request):
    """Redirect user to Google's consent screen."""
    auth_logger.info("google_login_initiated")
    # Google must redirect back to the SERVER callback, not the frontend
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, str(redirect_uri))


@router.get("/google/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    start = time.perf_counter()
    auth_logger.info("google_callback_started")

    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    if not user_info or not user_info.get("email"):
        auth_logger.warning("google_callback_failed", reason="no_email")
        return RedirectResponse(f"{settings.FRONTEND_URL}/login?error=no_email")

    email = user_info["email"]
    name = user_info.get("name")
    picture = user_info.get("picture")

    # Upsert user
    db_start = time.perf_counter()
    db_logger.info("query_started", query="upsert_user", email=email)
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user:
        user.name = name
        user.picture = picture
        # Org-RBAC overhaul: a Google-authenticated login marks the provider.
        user.auth_provider = "google"
    else:
        # First ever user becomes admin automatically
        any_user = await db.execute(select(User.id).limit(1))
        is_first_user = any_user.scalar_one_or_none() is None
        user = User(
            email=email,
            name=name,
            picture=picture,
            is_admin=is_first_user,
            auth_provider="google",
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)
    db_logger.info("query_complete", query="upsert_user", duration_ms=round((time.perf_counter() - db_start) * 1000, 2))

    # Create JWT — role/org_id claims are additive for clients; the backend
    # still re-fetches the User row in get_current_user (claims are not trusted
    # as the source of truth for permission checks).
    access_token = create_access_token(
        {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "org_id": user.organization_id,
        }
    )

    auth_logger.info("google_callback_complete", email=email, is_admin=user.is_admin, duration_ms=round((time.perf_counter() - start) * 1000, 2))

    # Redirect to frontend with token in URL fragment (not query param for security)
    return RedirectResponse(f"{settings.FRONTEND_URL}/auth/callback?token={access_token}")


@router.post("/login", response_model=TokenOut)
async def local_login(body: LoginBody, db: AsyncSession = Depends(get_db)):
    """Local email + password login for org-created users (admin/manager/user).

    Org owners are Google-SSO-only and are rejected here. On success this mints
    the SAME JWT as the Google flow (identical claims) and returns it as TokenOut.
    """
    email = str(body.email).strip().lower()
    auth_logger.info("local_login_attempt", email=email)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    # Generic 401 for unknown users (do not leak which emails exist).
    if user is None:
        auth_logger.warning("local_login_failed", email=email, reason="not_found")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Google-SSO accounts (incl. org owners) must not use local password login.
    if (
        getattr(user, "auth_provider", None) == "google"
        or user.role == "org_owner"
        or user.hashed_password is None
    ):
        auth_logger.warning("local_login_blocked", email=email, role=user.role, provider=user.auth_provider)
        raise HTTPException(status_code=403, detail="Use Google sign-in")

    if not verify_password(body.password, user.hashed_password):
        auth_logger.warning("local_login_failed", email=email, reason="bad_password")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Same JWT helper + identical claim shape as the Google callback flow.
    access_token = create_access_token(
        {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "org_id": user.organization_id,
        }
    )

    auth_logger.info("local_login_complete", email=email, role=user.role)
    return TokenOut(access_token=access_token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
async def get_me(user: User = Depends(get_current_user)):
    """Return the currently authenticated user."""
    auth_logger.info("me_requested", user_id=user.id)
    return user

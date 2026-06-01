"""
Onboarding gate middleware.

HARD GATE: when ONBOARDING_REQUIRED is True, an authenticated user whose
role == 'org_owner' AND whose Organization.onboarding_state != 'completed' is
blocked from every /api/* route EXCEPT the onboarding flow, auth (login/OAuth/
current-user/logout), and health. The frontend uses the 403 body to force the
owner into the onboarding wizard.

Design:
  - Implemented as a FastAPI HTTP middleware so it runs ahead of every router
    without touching individual route dependencies.
  - Token decoding mirrors get_current_user (same secret/algorithm). ANY auth
    or decoding miss is a NO-OP — normal route auth then handles the request,
    so unauthenticated routes and bad tokens are unaffected.
  - Only org_owner is gated. Missing org, missing flag, or any other role is a
    pass-through. Platform admins / org_admin / manager / user are never gated.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session
from app.core.security import decode_access_token
from app.models.organization import Organization
from app.models.user import User

# Prefixes / paths that an un-onboarded org_owner is ALWAYS allowed to reach.
# Everything else under /api is blocked for that user.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "/api/onboarding",  # the whole onboarding flow (incl. GET /state)
    "/api/auth",        # login, OAuth, token refresh, /me, logout
)
_ALLOWED_EXACT: frozenset[str] = frozenset(
    {
        "/api/health",
    }
)

_BLOCK_BODY = {"detail": "onboarding_required", "onboarding_required": True}


def _is_allowed_path(path: str) -> bool:
    if path in _ALLOWED_EXACT:
        return True
    return any(path == p or path.startswith(p + "/") or path == p for p in _ALLOWED_PREFIXES)


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def onboarding_gate_middleware(request: Request, call_next):
    """Block un-onboarded org_owners from non-onboarding API routes (403)."""
    settings = get_settings()

    # Flag off → total no-op.
    if not getattr(settings, "ONBOARDING_REQUIRED", False):
        return await call_next(request)

    path = request.url.path

    # Only gate /api/* routes; never the always-allowed set; never preflight.
    if request.method == "OPTIONS" or not path.startswith("/api/") or _is_allowed_path(path):
        return await call_next(request)

    # Decode the bearer token like get_current_user does. ANY miss → no-op
    # (let the route's own auth produce the normal 401/redirect behavior).
    token = _bearer_token(request)
    if not token:
        return await call_next(request)
    try:
        payload = decode_access_token(token)
    except Exception:  # noqa: BLE001 — invalid/expired token: defer to route auth
        return await call_next(request)
    user_id = payload.get("sub")
    if not user_id:
        return await call_next(request)

    try:
        session: AsyncSession
        async with async_session() as session:
            user = (
                await session.execute(select(User).where(User.id == user_id))
            ).scalar_one_or_none()
            # No user, not an owner, or no org → not gated.
            if user is None or user.role != "org_owner":
                return await call_next(request)
            org_id = getattr(user, "organization_id", None)
            org: Organization | None = None
            if org_id:
                org = await session.get(Organization, org_id)
            if org is None and getattr(user, "id", None):
                org = (
                    await session.execute(
                        select(Organization).where(Organization.owner_user_id == user.id)
                    )
                ).scalars().first()
            # Missing org → no-op (don't lock the owner out of nothing).
            if org is None:
                return await call_next(request)
            if (org.onboarding_state or "") == "completed":
                return await call_next(request)
    except Exception:  # noqa: BLE001 — any DB/decoding failure must not break routes
        return await call_next(request)

    # Owner with incomplete onboarding hitting a gated route → hard 403.
    return JSONResponse(status_code=403, content=_BLOCK_BODY)

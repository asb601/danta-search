"""
Access request lifecycle:
  1. User POSTs /api/access-requests/me  → creates a "pending" request
  2. Admin GETs  /api/access-requests    → list of pending requests
  3. Admin PATCHes /api/access-requests/{id}/approve or /decline
     → updates status, emails both parties
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.email import (
    access_declined_user_email,
    access_request_admin_email,
    send_email,
)
from app.core.logger import auth_logger
from app.dependencies import get_current_user, require_admin, require_platform_admin
from app.models.access_request import AccessRequest
from app.models.container import ContainerConfig
from app.models.organization import Organization
from app.models.platform_admin_grant import PlatformAdminGrant
from app.models.user import User

router = APIRouter(prefix="/access-requests", tags=["access"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class AccessRequestIn(BaseModel):
    org_name: str | None = None
    message: str | None = None


class AccessRequestOut(BaseModel):
    id: str
    user_id: str
    user_email: str
    user_name: str | None
    user_picture: str | None
    status: str
    message: str | None
    org_name: str | None
    requested_at: datetime

    model_config = {"from_attributes": True}


# ── User endpoints ─────────────────────────────────────────────────────────────

@router.post("/me", response_model=AccessRequestOut)
async def submit_access_request(
    body: AccessRequestIn,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit or re-fetch an access request for the current user.
    Idempotent: if a request already exists, returns it unchanged.
    """
    # Admins never need to request access
    if current_user.is_admin:
        raise HTTPException(400, detail="Admins do not need access requests.")

    existing = (
        await db.execute(
            select(AccessRequest).where(AccessRequest.user_id == current_user.id)
        )
    ).scalar_one_or_none()

    if existing:
        return _to_out(existing)

    req = AccessRequest(
        user_id=current_user.id,
        status="pending",
        message=body.message,
        org_name=(body.org_name or "").strip() or None,
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)

    auth_logger.info("access_request_created", user_id=current_user.id, email=current_user.email)

    # Email admin
    settings = get_settings()
    if settings.ADMIN_EMAIL:
        review_url = f"{settings.FRONTEND_URL}/profile"
        html = access_request_admin_email(
            user_name=current_user.name or "",
            user_email=current_user.email,
            message=body.message,
            review_url=review_url,
        )
        await send_email(
            to_email=settings.ADMIN_EMAIL,
            subject=f"Access request from {current_user.name or current_user.email}",
            html_body=html,
            smtp_host=settings.SMTP_HOST,
            smtp_port=settings.SMTP_PORT,
            smtp_user=settings.SMTP_USER,
            smtp_password=settings.SMTP_PASSWORD,
        )

    return _to_out(req)


@router.get("/me/status")
async def my_access_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Returns the current user's access request status.
    Possible values: "none" | "pending" | "approved" | "declined"
    """
    if current_user.is_admin:
        return {"status": "approved"}

    req = (
        await db.execute(
            select(AccessRequest).where(AccessRequest.user_id == current_user.id)
        )
    ).scalar_one_or_none()

    return {"status": req.status if req else "none"}


# ── Grants (platform-admin scoped org access) ───────────────────────────────────

class GrantedOrgOut(BaseModel):
    organization_id: str
    organization_name: str
    container_ids: list[str]


@router.get("/my-grants", response_model=list[GrantedOrgOut])
async def my_grants(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[GrantedOrgOut]:
    """Active organizations a platform admin has been granted, with each org's
    container ids (so the frontend can show that org's data).

    Frozen contract:
      [{organization_id, organization_name, container_ids:[string]}]

    Non-platform-admins receive an empty list (they reach data through their own
    org scope, not through grants).
    """
    is_platform_admin = (
        getattr(current_user, "is_platform_admin", False)
        or current_user.role == "platform_admin"
    )
    if not is_platform_admin:
        return []

    grants = (
        await db.execute(
            select(PlatformAdminGrant.organization_id).where(
                PlatformAdminGrant.platform_admin_user_id == current_user.id,
                PlatformAdminGrant.status == "active",
            )
        )
    ).scalars().all()

    org_ids = sorted({g for g in grants if g})
    if not org_ids:
        return []

    orgs = (
        await db.execute(
            select(Organization).where(Organization.id.in_(org_ids))
        )
    ).scalars().all()

    out: list[GrantedOrgOut] = []
    for org in orgs:
        container_ids = (
            await db.execute(
                select(ContainerConfig.id).where(
                    ContainerConfig.organization_id == org.id
                )
            )
        ).scalars().all()
        cids = [c for c in container_ids if c]
        # Legacy fallback: org's single bound container.
        if not cids and getattr(org, "container_id", None):
            cids = [org.container_id]
        out.append(
            GrantedOrgOut(
                organization_id=org.id,
                organization_name=org.name,
                container_ids=cids,
            )
        )
    return out


# ── Admin endpoints ────────────────────────────────────────────────────────────

@router.get("", response_model=list[AccessRequestOut])
async def list_access_requests(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all pending access requests (admin only)."""
    rows = (
        await db.execute(
            select(AccessRequest)
            .where(AccessRequest.status == "pending")
            .order_by(AccessRequest.requested_at)
        )
    ).scalars().all()
    return [_to_out(r) for r in rows]


@router.patch("/{request_id}/approve", deprecated=True)
async def approve_request(
    request_id: str,
    admin: User = Depends(require_platform_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """RETIRED. Granting now happens on the Users page via
    PATCH /api/users/{user_id}/grant. This endpoint is gone (410)."""
    raise HTTPException(
        status_code=410,
        detail="Approval moved to PATCH /api/users/{user_id}/grant on the Users page.",
    )


@router.patch("/{request_id}/decline")
async def decline_request(
    request_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    req = await _get_request(request_id, db)
    req.status       = "declined"
    req.reviewed_at  = datetime.now(timezone.utc)
    req.reviewed_by_id = admin.id
    await db.commit()

    auth_logger.info("access_declined", request_id=request_id,
                     user_id=req.user_id, by=admin.email)

    settings = get_settings()
    html = access_declined_user_email(user_name=req.user.name or "")
    await send_email(
        to_email=req.user.email,
        subject="Your access request was not approved",
        html_body=html,
        smtp_host=settings.SMTP_HOST,
        smtp_port=settings.SMTP_PORT,
        smtp_user=settings.SMTP_USER,
        smtp_password=settings.SMTP_PASSWORD,
    )
    return {"status": "declined"}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_request(request_id: str, db: AsyncSession) -> AccessRequest:
    req = (
        await db.execute(
            select(AccessRequest).where(AccessRequest.id == request_id)
        )
    ).scalar_one_or_none()
    if not req:
        raise HTTPException(404, detail="Access request not found")
    return req


def _to_out(req: AccessRequest) -> AccessRequestOut:
    return AccessRequestOut(
        id=req.id,
        user_id=req.user_id,
        user_email=req.user.email,
        user_name=req.user.name,
        user_picture=req.user.picture,
        status=req.status,
        message=req.message,
        org_name=req.org_name,
        requested_at=req.requested_at,
    )

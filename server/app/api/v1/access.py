"""
Access request lifecycle:
  1. User POSTs /api/access-requests/me  → creates a "pending" request
  2. Admin GETs  /api/access-requests    → list of pending requests
  3. Admin PATCHes /api/access-requests/{id}/approve or /decline
     → updates status, emails both parties
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.email import (
    access_approved_user_email,
    access_declined_user_email,
    access_request_admin_email,
    send_email,
)
from app.core.logger import auth_logger
from app.dependencies import get_current_user, require_admin, require_platform_admin
from app.models.access_request import AccessRequest
from app.models.organization import Organization
from app.models.user import User
from sqlalchemy import update


# Blob-/URL-safe slug (mirrors api/v1/onboarding.py::_slugify).
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or f"org-{uuid.uuid4().hex[:8]}"


async def _unique_org_slug(base_name: str, db: AsyncSession) -> str:
    base_slug = _slugify(base_name)
    slug = base_slug
    n = 1
    while (
        await db.execute(select(Organization.id).where(Organization.slug == slug))
    ).scalar_one_or_none() is not None:
        n += 1
        slug = f"{base_slug}-{n}"
    return slug

router = APIRouter(prefix="/access-requests", tags=["access"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class AccessRequestIn(BaseModel):
    message: str | None = None


class ApproveRequestIn(BaseModel):
    # "owner" → grant org_owner (creates an Organization; Google-SSO client).
    # "admin" → grant platform-side internal staff (no organization).
    role: str
    # Optional org name used when role == "owner"; falls back to the
    # requester's email local-part if omitted.
    org_name: str | None = None


class AccessRequestOut(BaseModel):
    id: str
    user_id: str
    user_email: str
    user_name: str | None
    user_picture: str | None
    status: str
    message: str | None
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


@router.patch("/{request_id}/approve")
async def approve_request(
    request_id: str,
    body: ApproveRequestIn,
    admin: User = Depends(require_platform_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Approve an access request and assign a role (platform admin only).

    body.role:
      "owner" → requester is a CLIENT. Promotes them to org_owner (Google-SSO),
                creates a new Organization (owner_user_id = user, slug from
                org_name or the user's email local-part, onboarding_state='created'),
                and binds user.organization_id to it. The owner is then forced
                through onboarding by the hard gate.
      "admin" → requester is internal DANTA staff. Promotes them to a platform
                admin (is_platform_admin = True). No organization is created.
    """
    role = (body.role or "").strip().lower()
    if role not in ("owner", "admin"):
        raise HTTPException(400, detail="role must be one of: 'owner', 'admin'")

    req = await _get_request(request_id, db)
    user = await db.get(User, req.user_id)
    if user is None:
        raise HTTPException(404, detail="Requesting user not found")

    new_org_id: str | None = None

    try:
        if role == "owner":
            # CLIENT: org owner, Google-SSO, gets a fresh organization.
            org_name = (body.org_name or "").strip() or (user.email or "").split("@")[0]
            slug = await _unique_org_slug(org_name, db)
            org = Organization(
                name=org_name,
                owner_user_id=user.id,
                slug=slug,
                onboarding_state="created",
            )
            db.add(org)
            await db.flush()  # populate org.id within the transaction
            new_org_id = org.id

            user.role = "org_owner"
            user.is_admin = True
            user.auth_provider = "google"
            user.organization_id = org.id
        else:  # role == "admin"
            # Internal DANTA staff: platform-side admin, no organization.
            user.role = "admin"
            user.is_platform_admin = True
            user.is_admin = True

        # Cleared through access: empty allowed_domains = unrestricted.
        user.allowed_domains = []

        req.status = "approved"
        req.reviewed_at = datetime.now(timezone.utc)
        req.reviewed_by_id = admin.id

        await db.commit()
    except Exception:
        await db.rollback()
        raise

    auth_logger.info("access_approved", request_id=request_id,
                     user_id=req.user_id, by=admin.email,
                     granted_role=role, org_id=new_org_id)

    settings = get_settings()
    html = access_approved_user_email(
        user_name=req.user.name or "",
        app_url=settings.FRONTEND_URL,
    )
    await send_email(
        to_email=req.user.email,
        subject="Your access request was approved",
        html_body=html,
        smtp_host=settings.SMTP_HOST,
        smtp_port=settings.SMTP_PORT,
        smtp_user=settings.SMTP_USER,
        smtp_password=settings.SMTP_PASSWORD,
    )
    return {"status": "approved"}


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
        requested_at=req.requested_at,
    )

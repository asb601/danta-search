import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import auth_logger
from app.dependencies import get_current_user, require_admin
from app.services.audit_log import record_audit_event_safe
from app.models.file import File
from app.models.folder import Folder
from app.models.organization import Organization
from app.models.user import User
from app.schemas.user import UserOut

router = APIRouter(prefix="/users", tags=["users"])

_VALID_ROLES = {"admin", "developer", "manager", "user"}


class _MeDomainsBody(BaseModel):
    allowed_domains: list[str] | None  # None or [] = clear restriction


class _RoleBody(BaseModel):
    role: str  # "admin" | "developer" | "manager" | "user"


class _OrgAssignBody(BaseModel):
    organization_id: str | None  # None = detach (turn into platform admin)


@router.get("/domains")
async def list_available_domains(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return all distinct domain tags any authenticated user can pick from."""
    rows = (
        await db.execute(
            select(Folder.domain_tag).where(Folder.domain_tag.isnot(None)).distinct()
        )
    ).scalars().all()
    return {"domains": sorted(rows)}


@router.patch("/me/domains")
async def set_my_domains(
    body: _MeDomainsBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Let the authenticated user choose which domains/departments they belong to."""
    domains = body.allowed_domains if body.allowed_domains else None
    await db.execute(
        update(User).where(User.id == current_user.id).values(allowed_domains=domains)
    )
    await db.commit()
    auth_logger.info("user_domains_updated", user_id=current_user.id, domains=domains)
    return {"allowed_domains": domains}


@router.get("", response_model=list[UserOut])
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    auth_logger.info("users_list_requested")

    result = await db.execute(
        select(
            User,
            func.count(File.id).label("file_count"),
        )
        .outerjoin(File, File.uploaded_by_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at)
    )
    rows = result.all()
    users = [
        UserOut(
            id=u.id,
            email=u.email,
            name=u.name,
            picture=u.picture,
            is_admin=u.is_admin,
            role=u.role if u.role else ("admin" if u.is_admin else "user"),
            created_at=u.created_at,
            file_count=file_count,
            allowed_domains=u.allowed_domains,
            organization_id=u.organization_id,
        )
        for u, file_count in rows
    ]
    auth_logger.info("users_list_complete", count=len(users), duration_ms=round((time.perf_counter() - start) * 1000, 2))
    return users


@router.patch("/{user_id}/toggle-admin")
async def toggle_admin(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own admin status")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    old_is_admin = user.is_admin
    user.is_admin = not user.is_admin
    await db.commit()

    auth_logger.info(
        "admin_toggled",
        user_id=user_id,
        admin_id=current_user.id,
        admin_email=current_user.email,
        old_is_admin=old_is_admin,
        is_admin=user.is_admin,
    )
    try:
        await record_audit_event_safe(
            actor=current_user,
            action="users.toggle_admin",
            event_type="action",
            status_code=200,
            path=f"/api/users/{user_id}/toggle-admin",
            route_template="/api/users/{user_id}/toggle-admin",
            target_user_id=user.id,
            target_user_email=user.email,
            target_user_name=user.name,
            details={"old_is_admin": old_is_admin, "new_is_admin": user.is_admin},
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        auth_logger.warning("admin_toggle_audit_failed", user_id=user_id, error=str(exc)[:300])
    return {"id": user.id, "is_admin": user.is_admin}


@router.patch("/{user_id}/role")
async def set_user_role(
    user_id: str,
    body: _RoleBody,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a user's role (admin / developer / manager / user). Syncs is_admin flag."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    if body.role not in _VALID_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(_VALID_ROLES)}")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    old_role = user.role
    old_is_admin = user.is_admin
    user.role = body.role
    user.is_admin = body.role == "admin"
    await db.commit()

    auth_logger.info(
        "role_changed",
        user_id=user_id,
        admin_id=current_user.id,
        admin_email=current_user.email,
        old_role=old_role,
        role=body.role,
        old_is_admin=old_is_admin,
        is_admin=user.is_admin,
    )
    try:
        await record_audit_event_safe(
            actor=current_user,
            action="users.set_role",
            event_type="action",
            status_code=200,
            path=f"/api/users/{user_id}/role",
            route_template="/api/users/{user_id}/role",
            target_user_id=user.id,
            target_user_email=user.email,
            target_user_name=user.name,
            details={
                "old_role": old_role,
                "new_role": body.role,
                "old_is_admin": old_is_admin,
                "new_is_admin": user.is_admin,
            },
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        auth_logger.warning("role_change_audit_failed", user_id=user_id, error=str(exc)[:300])
    return {"id": user.id, "role": user.role, "is_admin": user.is_admin}


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete a user account. Admins cannot delete themselves."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    deleted_email = user.email
    deleted_name = user.name
    await db.delete(user)
    await db.commit()

    auth_logger.info(
        "user_deleted",
        deleted_user_id=user_id,
        deleted_email=deleted_email,
        by_admin=current_user.id,
        admin_email=current_user.email,
    )
    try:
        await record_audit_event_safe(
            actor=current_user,
            action="users.delete",
            event_type="action",
            status_code=200,
            path=f"/api/users/{user_id}",
            route_template="/api/users/{user_id}",
            target_user_id=user_id,
            target_user_email=deleted_email,
            target_user_name=deleted_name,
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        auth_logger.warning("user_delete_audit_failed", user_id=user_id, error=str(exc)[:300])
    return {"deleted": True}


@router.patch("/{user_id}/organization")
async def assign_user_to_organization(
    user_id: str,
    body: _OrgAssignBody,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Attach (or detach) a user to an organization.

    Setting organization_id=None turns them back into a platform-level admin
    (no org scoping applied). Non-admin users without an organization will be
    unable to chat — the chat endpoints require either is_admin or an org.
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.organization_id is not None:
        org = await db.get(Organization, body.organization_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

    user.organization_id = body.organization_id
    await db.commit()

    auth_logger.info(
        "user_organization_changed",
        user_id=user_id,
        organization_id=body.organization_id,
        by=current_user.id,
    )
    return {"id": user.id, "organization_id": user.organization_id}

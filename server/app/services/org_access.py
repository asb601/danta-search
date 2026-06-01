"""
org_access — Org-RBAC v2 access decisions (Lane B).

Pure async helpers that answer two questions for the new org-scoped runtime:

  user_can_access_org(user, org_id, db)  -> bool
  effective_domains_for(user, org_id, db) -> list[str] | None

These are additive and only consulted when RBAC_V2 enforcement is active (or in
shadow logging). They never mutate state.

Access model
------------
  platform_admin : may access an org ONLY if an *active* PlatformAdminGrant row
                   exists for (org, user).
  org_owner /
  org_admin      : may access only their own organization (org_id == user.org).
  manager / user : may access only their own organization AND must hold at
                   least one ManagerDomainAssignment row in that org.

effective_domains_for returns:
  None                 -> all domains (org owners / org admins / granted platform admins)
  list[str]            -> the explicitly assigned domains (managers / users)
  []                   -> nothing (no access at all)
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.manager_domain_assignment import ManagerDomainAssignment
from app.models.platform_admin_grant import PlatformAdminGrant
from app.models.user import User

_ORG_ADMIN_ROLES = ("org_owner", "org_admin")


async def _has_active_platform_grant(
    user_id: str, org_id: str, db: AsyncSession
) -> bool:
    row = await db.execute(
        select(PlatformAdminGrant.id).where(
            PlatformAdminGrant.platform_admin_user_id == user_id,
            PlatformAdminGrant.organization_id == org_id,
            PlatformAdminGrant.status == "active",
        )
    )
    return row.scalar_one_or_none() is not None


async def _assigned_domains(
    user_id: str, org_id: str, db: AsyncSession
) -> list[str]:
    rows = await db.execute(
        select(ManagerDomainAssignment.domain_tag).where(
            ManagerDomainAssignment.user_id == user_id,
            ManagerDomainAssignment.organization_id == org_id,
        )
    )
    return sorted({d for d in rows.scalars().all() if d})


async def user_can_access_org(user: User, org_id: str, db: AsyncSession) -> bool:
    """Whether `user` may access organization `org_id` under RBAC v2."""
    if org_id is None:
        return False

    # Platform admins: only through an explicit active grant.
    if getattr(user, "is_platform_admin", False) or user.role == "platform_admin":
        return await _has_active_platform_grant(user.id, org_id, db)

    # Org owners / admins: only their own org.
    if user.role in _ORG_ADMIN_ROLES:
        return getattr(user, "organization_id", None) == org_id

    # Managers / regular users: own org AND at least one domain assignment.
    if getattr(user, "organization_id", None) != org_id:
        return False
    return len(await _assigned_domains(user.id, org_id, db)) > 0


async def platform_admin_has_container_grant(
    user: User, container_id: str | None, db: AsyncSession
) -> bool:
    """Whether `user` is a platform admin holding an active grant for the
    organization that owns `container_id`.

    Used by the file/folder guards to extend FULL (read+write) access to a
    granted org's containers, bypassing the per-domain filter.
    """
    if not container_id:
        return False
    if not (getattr(user, "is_platform_admin", False) or user.role == "platform_admin"):
        return False

    from app.models.container import ContainerConfig

    org_id = (
        await db.execute(
            select(ContainerConfig.organization_id).where(
                ContainerConfig.id == container_id
            )
        )
    ).scalar_one_or_none()
    if not org_id:
        return False
    return await _has_active_platform_grant(user.id, org_id, db)


async def can_write_org(user: User, org_id: str, db: AsyncSession) -> bool:
    """Whether `user` may WRITE (upload/delete/modify) within organization `org_id`.

    True when:
      * the user is a platform admin holding an *active* PlatformAdminGrant for
        that org, OR
      * the user is the org_owner / org_admin of that org (org_id == user.org).
    """
    if org_id is None:
        return False

    # Platform admins: only through an explicit active grant for this org.
    if getattr(user, "is_platform_admin", False) or user.role == "platform_admin":
        return await _has_active_platform_grant(user.id, org_id, db)

    # Org owners / admins: only their own org.
    if user.role in _ORG_ADMIN_ROLES:
        return getattr(user, "organization_id", None) == org_id

    return False


async def effective_domains_for(
    user: User, org_id: str, db: AsyncSession
) -> list[str] | None:
    """Resolve the domain sub-filter for `user` within `org_id`.

    Returns None for "all domains" (owners/admins/granted platform admins),
    the assigned domain list for managers/users, or [] when no access.
    """
    if org_id is None:
        return []

    # Platform admins with an active grant see all domains in that org.
    if getattr(user, "is_platform_admin", False) or user.role == "platform_admin":
        if await _has_active_platform_grant(user.id, org_id, db):
            return None
        return []

    # Org owners / admins see all domains in their own org.
    if user.role in _ORG_ADMIN_ROLES:
        return None if getattr(user, "organization_id", None) == org_id else []

    # Managers / users are limited to their assigned domains in their own org.
    if getattr(user, "organization_id", None) != org_id:
        return []
    return await _assigned_domains(user.id, org_id, db)

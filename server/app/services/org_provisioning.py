"""
org_provisioning — reusable organization provisioning for owner grants.

Extracted from api/v1/access.py so both the access-request approval flow and the
new Users-page grant flow can create an organization for a newly-promoted owner
without duplicating the slug + org-creation logic.

provision_owner_org(db, user, org_name):
  * creates an Organization (onboarding_state='created', unique slug),
  * promotes the user to org_owner (is_admin=True, auth_provider='google'),
  * binds user.organization_id to the new org,
  * clears allowed_domains (owners are unrestricted within their org).

It flushes (so org.id is populated) but does NOT commit — the caller owns the
transaction boundary so it can bundle the AccessRequest update atomically.
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import Organization
from app.models.user import User

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


async def provision_owner_org(
    db: AsyncSession, user: User, org_name: str | None
) -> Organization:
    """Create an Organization for `user` and promote them to org_owner.

    Flushes (org.id populated) but does not commit — caller commits.
    """
    # Idempotent: if this user already owns an organization, reuse it instead of
    # creating a duplicate (e.g. re-granting "owner" from the Users page).
    existing = (
        await db.execute(
            select(Organization).where(Organization.owner_user_id == user.id)
        )
    ).scalars().first()
    if existing is not None:
        user.role = "org_owner"
        user.is_admin = True
        user.auth_provider = "google"
        user.organization_id = existing.id
        user.allowed_domains = []
        return existing

    name = (
        (org_name or "").strip()
        or (user.email or "").split("@")[0]
        or "organization"
    )
    slug = await _unique_org_slug(name, db)
    org = Organization(
        name=name,
        owner_user_id=user.id,
        slug=slug,
        onboarding_state="created",
    )
    db.add(org)
    await db.flush()  # populate org.id within the transaction

    user.role = "org_owner"
    user.is_admin = True
    user.auth_provider = "google"
    user.organization_id = org.id
    # Cleared through access: empty allowed_domains = unrestricted within the org.
    user.allowed_domains = []

    return org

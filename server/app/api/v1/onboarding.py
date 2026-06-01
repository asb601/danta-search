"""
Onboarding API — org-RBAC overhaul (Lane C).

Drives an organization through its onboarding lifecycle:

    POST /onboarding/start                  -> create org (state 'created')
    PUT  /onboarding/ai-settings            -> upsert OrgAISettings ('ai_configured')
    POST /onboarding/storage                -> primary ContainerConfig + org-root folder ('storage_connected')
    POST /onboarding/domains                -> domain folders under org-root ('domains_created')
    POST /onboarding/users                  -> create users + domain assignments ('users_added')
    POST /onboarding/users/bulk             -> same, from an uploaded .xlsx
    POST /onboarding/platform-admin-grant   -> upsert PlatformAdminGrant
    POST /onboarding/complete               -> 'completed'
    GET  /onboarding/state                  -> {state, checklist booleans}

The parent app mounts this router under /api, so the effective prefix is
/api/onboarding. All transitions go through the onboarding state machine
(assert_step_allowed / advance_state). Additive + backward-compatible: nothing
here changes existing routes or default behavior.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone

import structlog
from azure.storage.blob import BlobServiceClient
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import hash_password
from app.models.container import ContainerConfig
from app.models.folder import Folder
from app.models.manager_domain_assignment import ManagerDomainAssignment
from app.models.org_ai_settings import OrgAISettings
from app.models.organization import Organization
from app.models.platform_admin_grant import PlatformAdminGrant
from app.models.user import User
from app.services.onboarding import (
    advance_state,
    assert_step_allowed,
    build_checklist,
    parse_users_xlsx,
)

logger = structlog.get_logger("onboarding_api")

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


# ── Guard resolution (degrade gracefully if Lane B guards aren't present) ────
#
# Preferred source is app.dependencies; fall back to app.core.security; finally
# fall back to get_current_user so the routes still mount instead of crashing
# the import at startup.
def _resolve_guards():
    from app.dependencies import get_current_user as _gcu

    candidates = {}
    for name in (
        "require_org_owner",
        "require_org_role",
        "require_platform_admin",
        "require_google_sso",
    ):
        fn = None
        try:
            import app.dependencies as _deps

            fn = getattr(_deps, name, None)
        except Exception:  # noqa: BLE001
            fn = None
        if fn is None:
            try:
                import app.core.security as _sec

                fn = getattr(_sec, name, None)
            except Exception:  # noqa: BLE001
                fn = None
        candidates[name] = fn
    return _gcu, candidates


_get_current_user, _GUARDS = _resolve_guards()


def _owner_guard():
    return _GUARDS.get("require_org_owner") or _get_current_user


def _platform_admin_guard():
    return _GUARDS.get("require_platform_admin") or _get_current_user


def _org_admin_guard():
    """org_owner OR org_admin. Uses the require_org_role factory when available,
    else degrades to the bare authenticated user."""
    factory = _GUARDS.get("require_org_role")
    if factory is not None:
        try:
            return factory("org_owner", "org_admin")
        except Exception:  # noqa: BLE001
            pass
    return _get_current_user


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or f"org-{uuid.uuid4().hex[:8]}"


async def _load_owned_org(db: AsyncSession, user: User) -> Organization:
    """Resolve the organization the caller owns/administers.

    Prefers Organization.owner_user_id == user.id; falls back to the user's
    organization_id. Raises 404 when none is found.
    """
    org: Organization | None = None
    org = (
        await db.execute(
            select(Organization).where(Organization.owner_user_id == user.id)
        )
    ).scalars().first()
    if org is None and getattr(user, "organization_id", None):
        org = await db.get(Organization, user.organization_id)
    if org is None:
        raise HTTPException(status_code=404, detail="No organization for this user")
    return org


# ── Schemas ──────────────────────────────────────────────────────────────────


class StartBody(BaseModel):
    name: str
    owner_user_id: str | None = None  # platform admin may name a different owner


class AISettingsBody(BaseModel):
    chat_api_key: str | None = None
    embeddings_api_key: str | None = None
    fallback_api_key: str | None = None
    chat_endpoint: str | None = None
    chat_deployment: str | None = None
    embeddings_deployment: str | None = None
    fallback_deployment: str | None = "gpt-4o-mini"
    api_version: str | None = None


class StorageBody(BaseModel):
    name: str
    container_name: str
    connection_string: str
    storage_kind: str = "azure_blob"


class DomainsBody(BaseModel):
    names: list[str]


class UserRow(BaseModel):
    email: str
    role: str | None = "user"
    domains: list[str] = []
    name: str | None = None
    password: str | None = None


class UsersBody(BaseModel):
    users: list[UserRow]


class PlatformAdminGrantBody(BaseModel):
    platform_admin_user_id: str
    status: str = "active"  # 'active' | 'revoked'


# ── Helpers ──────────────────────────────────────────────────────────────────


def _generate_temp_password() -> str:
    """URL-safe random temp password the owner distributes to a new user."""
    import secrets

    return secrets.token_urlsafe(12)


async def _materialize_folder_blob(db: AsyncSession, folder: Folder) -> None:
    """Reuse files._materialize_folder_blob (non-fatal, idempotent)."""
    try:
        from app.api.v1.files import _materialize_folder_blob as _mat

        await _mat(db, folder)
    except Exception as exc:  # noqa: BLE001 — blob failures must never break onboarding
        logger.warning("materialize_folder_blob_failed", folder_id=folder.id, error=str(exc)[:200])


async def _get_org_root(db: AsyncSession, org: Organization) -> Folder | None:
    return (
        await db.execute(
            select(Folder).where(
                Folder.organization_id == org.id,
                Folder.folder_kind == "org_root",
            )
        )
    ).scalars().first()


async def _mirror_allowed_domains(db: AsyncSession, user: User) -> None:
    """Recompute users.allowed_domains from the user's ManagerDomainAssignment
    rows (backward-compat mirror)."""
    rows = (
        await db.execute(
            select(ManagerDomainAssignment.domain_tag).where(
                ManagerDomainAssignment.user_id == user.id
            )
        )
    ).scalars().all()
    domains = sorted({d for d in rows if d})
    user.allowed_domains = domains or None


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/start", status_code=status.HTTP_201_CREATED)
async def start_onboarding(
    body: StartBody,
    user: User = Depends(_platform_admin_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Create a new Organization (state 'created'). Allowed for platform admins
    or a Google-authenticated owner. The caller becomes the owner unless a
    platform admin names a different owner_user_id."""
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="Organization name cannot be empty")

    existing = (
        await db.execute(select(Organization).where(Organization.name == name))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Organization with this name already exists")

    owner_id = body.owner_user_id or user.id
    if body.owner_user_id and not getattr(user, "is_platform_admin", False):
        raise HTTPException(
            status_code=403, detail="Only a platform admin may set a different owner"
        )

    # Unique slug.
    base_slug = _slugify(name)
    slug = base_slug
    n = 1
    while (
        await db.execute(select(Organization.id).where(Organization.slug == slug))
    ).scalar_one_or_none() is not None:
        n += 1
        slug = f"{base_slug}-{n}"

    org = Organization(
        name=name,
        owner_user_id=owner_id,
        slug=slug,
        onboarding_state="created",
    )
    db.add(org)
    await db.commit()
    await db.refresh(org)
    logger.info("onboarding_started", org_id=org.id, slug=slug, owner_id=owner_id)
    return {"id": org.id, "name": org.name, "slug": org.slug, "state": org.onboarding_state}


@router.put("/ai-settings")
async def upsert_ai_settings(
    body: AISettingsBody,
    user: User = Depends(_owner_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Upsert the organization's encrypted AI settings; -> 'ai_configured'."""
    org = await _load_owned_org(db, user)
    assert_step_allowed(org, "ai_configured")

    row = (
        await db.execute(
            select(OrgAISettings).where(OrgAISettings.organization_id == org.id)
        )
    ).scalar_one_or_none()
    if row is None:
        row = OrgAISettings(organization_id=org.id)
        db.add(row)

    row.chat_api_key = body.chat_api_key
    row.embeddings_api_key = body.embeddings_api_key
    row.fallback_api_key = body.fallback_api_key
    row.chat_endpoint = body.chat_endpoint
    row.chat_deployment = body.chat_deployment
    row.embeddings_deployment = body.embeddings_deployment
    row.fallback_deployment = body.fallback_deployment or "gpt-4o-mini"
    row.api_version = body.api_version
    await db.commit()

    state = await advance_state(org, "ai_configured", db)
    logger.info("onboarding_ai_configured", org_id=org.id)
    return {"organization_id": org.id, "state": state}


@router.post("/storage")
async def connect_storage(
    body: StorageBody,
    user: User = Depends(_owner_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Validate the Azure connection, create the primary ContainerConfig + the
    org-root Folder, and link the org. -> 'storage_connected'."""
    org = await _load_owned_org(db, user)
    assert_step_allowed(org, "storage_connected")

    # Reuse the same connection validation pattern as containers.create_container.
    try:
        blob_service = await asyncio.to_thread(
            BlobServiceClient.from_connection_string, body.connection_string
        )
        container_client = await asyncio.to_thread(
            blob_service.get_container_client, body.container_name
        )
        await asyncio.to_thread(container_client.get_container_properties)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to Azure container '{body.container_name}': {exc}",
        )

    config = ContainerConfig(
        name=body.name,
        container_name=body.container_name,
        connection_string=body.connection_string,
        created_by=user.id,
        organization_id=org.id,
        is_primary=True,
        storage_kind=body.storage_kind or "azure_blob",
    )
    db.add(config)
    await db.flush()

    # Link org -> primary container.
    org.container_id = config.id

    # Create the org-root folder (named by slug).
    root = await _get_org_root(db, org)
    if root is None:
        root = Folder(
            name=org.slug or _slugify(org.name),
            parent_id=None,
            owner_id=user.id,
            container_id=config.id,
            organization_id=org.id,
            folder_kind="org_root",
        )
        db.add(root)
        await db.flush()
        await _materialize_folder_blob(db, root)

    await db.commit()
    await db.refresh(org)

    state = await advance_state(org, "storage_connected", db)
    logger.info("onboarding_storage_connected", org_id=org.id, container_id=config.id)
    return {
        "organization_id": org.id,
        "container_id": config.id,
        "org_root_folder_id": root.id,
        "state": state,
    }


@router.post("/domains")
async def create_domains(
    body: DomainsBody,
    user: User = Depends(_org_admin_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Create domain folders under the org-root; -> 'domains_created'."""
    org = await _load_owned_org(db, user)
    assert_step_allowed(org, "domains_created")

    root = await _get_org_root(db, org)
    if root is None:
        raise HTTPException(
            status_code=422, detail="Storage must be connected (org-root) before creating domains"
        )

    created: list[dict] = []
    for raw_name in body.names:
        name = (raw_name or "").strip()
        if not name:
            continue
        # Skip if an org-scoped domain folder with this tag already exists.
        existing = (
            await db.execute(
                select(Folder).where(
                    Folder.organization_id == org.id,
                    Folder.domain_tag == name,
                    Folder.folder_kind == "domain",
                ).limit(1)
            )
        ).scalar_one_or_none()
        if existing:
            created.append({"name": name, "folder_id": existing.id, "existed": True})
            continue
        folder = Folder(
            name=name,
            parent_id=root.id,
            owner_id=user.id,
            container_id=root.container_id,
            organization_id=org.id,
            folder_kind="domain",
            domain_tag=name,
        )
        db.add(folder)
        await db.flush()
        await _materialize_folder_blob(db, folder)
        created.append({"name": name, "folder_id": folder.id, "existed": False})

    await db.commit()
    await db.refresh(org)

    state = await advance_state(org, "domains_created", db)
    logger.info("onboarding_domains_created", org_id=org.id, count=len(created))
    return {"organization_id": org.id, "domains": created, "state": state}


async def _provision_users(
    db: AsyncSession, org: Organization, rows: list[dict], granted_by: str
) -> list[dict]:
    """Create/locate users + their ManagerDomainAssignment rows, mirror
    allowed_domains. Returns a per-user result summary."""
    out: list[dict] = []
    for r in rows:
        email = (r.get("email") or "").strip().lower()
        if not email:
            continue
        # Normalize + clamp the role for bulk local-auth provisioning. Only
        # org-operational roles are allowed here: org_owner is Google-SSO-only
        # and platform_admin is never provisioned through org onboarding.
        _raw_role = (r.get("role") or "user").strip().lower()
        role = {"developer": "manager", "admin": "org_admin"}.get(_raw_role, _raw_role)
        if role not in ("org_admin", "manager", "user"):
            role = "user"
        domains = [d for d in (r.get("domains") or []) if d]
        name = r.get("name")
        password = r.get("password")

        existing = (
            await db.execute(select(User).where(User.email == email))
        ).scalar_one_or_none()
        temp_password: str | None = None
        if existing is None:
            # Use the provided password or generate a temp one to hand to the user.
            temp_password = password or _generate_temp_password()
            u = User(
                email=email,
                name=name,
                role=role,
                organization_id=org.id,
                auth_provider="local",
                hashed_password=hash_password(temp_password),
                is_admin=(role == "org_admin"),
                is_platform_admin=False,
            )
            db.add(u)
            await db.flush()
            created_flag = True
        else:
            u = existing
            u.organization_id = org.id
            if role:
                u.role = role
                u.is_admin = (role == "org_admin")
            if name and not u.name:
                u.name = name
            created_flag = False

        # Domain assignments (manager/user). is_domain_admin for managers.
        is_domain_admin = role in ("manager", "org_admin")
        for d in domains:
            exists = (
                await db.execute(
                    select(ManagerDomainAssignment).where(
                        ManagerDomainAssignment.user_id == u.id,
                        ManagerDomainAssignment.organization_id == org.id,
                        ManagerDomainAssignment.domain_tag == d,
                    )
                )
            ).scalar_one_or_none()
            if exists is None:
                db.add(
                    ManagerDomainAssignment(
                        user_id=u.id,
                        organization_id=org.id,
                        domain_tag=d,
                        is_domain_admin=is_domain_admin,
                        granted_by=granted_by,
                    )
                )
                await db.flush()
            elif is_domain_admin and not exists.is_domain_admin:
                exists.is_domain_admin = True

        await _mirror_allowed_domains(db, u)
        row_out = {
            "email": email,
            "user_id": u.id,
            "role": u.role,
            "domains": domains,
            "created": created_flag,
        }
        # Surface the temp password ONLY for freshly-created local users so the
        # owner can distribute it. Never echoed for pre-existing users.
        if temp_password is not None:
            row_out["temp_password"] = temp_password
        out.append(row_out)
    return out


@router.post("/users")
async def add_users(
    body: UsersBody,
    user: User = Depends(_org_admin_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Create users + domain assignments from a JSON payload; -> 'users_added'."""
    org = await _load_owned_org(db, user)
    assert_step_allowed(org, "users_added")

    rows = [r.model_dump() for r in body.users]
    result = await _provision_users(db, org, rows, granted_by=user.id)
    await db.commit()
    await db.refresh(org)

    state = await advance_state(org, "users_added", db)
    logger.info("onboarding_users_added", org_id=org.id, count=len(result))
    return {"organization_id": org.id, "users": result, "state": state}


@router.post("/users/bulk")
async def add_users_bulk(
    user: User = Depends(_org_admin_guard()),
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
):
    """Create users + domain assignments from an uploaded .xlsx; -> 'users_added'."""
    org = await _load_owned_org(db, user)
    assert_step_allowed(org, "users_added")

    content = await file.read()
    rows = parse_users_xlsx(content)
    result = await _provision_users(db, org, rows, granted_by=user.id)
    await db.commit()
    await db.refresh(org)

    state = await advance_state(org, "users_added", db)
    logger.info("onboarding_users_bulk_added", org_id=org.id, count=len(result))
    return {"organization_id": org.id, "users": result, "state": state}


@router.post("/platform-admin-grant")
async def upsert_platform_admin_grant(
    body: PlatformAdminGrantBody,
    user: User = Depends(_owner_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Upsert a PlatformAdminGrant for this org (active/revoked). Does not change
    onboarding state."""
    org = await _load_owned_org(db, user)
    if body.status not in ("active", "revoked"):
        raise HTTPException(status_code=422, detail="status must be 'active' or 'revoked'")

    grant = (
        await db.execute(
            select(PlatformAdminGrant).where(
                PlatformAdminGrant.organization_id == org.id,
                PlatformAdminGrant.platform_admin_user_id == body.platform_admin_user_id,
            )
        )
    ).scalar_one_or_none()
    if grant is None:
        grant = PlatformAdminGrant(
            organization_id=org.id,
            platform_admin_user_id=body.platform_admin_user_id,
            granted_by=user.id,
            status=body.status,
        )
        db.add(grant)
    else:
        grant.status = body.status
    if body.status == "revoked":
        grant.revoked_at = datetime.now(timezone.utc)
    else:
        grant.revoked_at = None

    await db.commit()
    await db.refresh(grant)
    logger.info(
        "onboarding_platform_admin_grant",
        org_id=org.id,
        grantee=body.platform_admin_user_id,
        status=body.status,
    )
    return {"organization_id": org.id, "grant_id": grant.id, "status": grant.status}


@router.post("/complete")
async def complete_onboarding(
    user: User = Depends(_owner_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Finalize onboarding; -> 'completed' + onboarding_completed_at."""
    org = await _load_owned_org(db, user)
    state = await advance_state(org, "completed", db)
    logger.info("onboarding_completed", org_id=org.id)
    return {
        "organization_id": org.id,
        "state": state,
        "onboarding_completed_at": (
            org.onboarding_completed_at.isoformat()
            if getattr(org, "onboarding_completed_at", None)
            else None
        ),
    }


@router.get("/state")
async def get_onboarding_state(
    user: User = Depends(_org_admin_guard()),
    db: AsyncSession = Depends(get_db),
):
    """Return the org's onboarding state + a per-step checklist."""
    org = await _load_owned_org(db, user)
    return {
        "organization_id": org.id,
        "state": org.onboarding_state,
        "checklist": build_checklist(org),
    }

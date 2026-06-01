"""Shared constants, schemas, and helpers for chat API modules."""
from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.database import async_session
from app.core.logger import chat_logger
from app.models.organization import Organization
from app.models.user import User
from app.services.context_service import maybe_generate_title, maybe_regenerate_summary

MAX_MESSAGES_PER_CONVERSATION = 200
WARN_MESSAGES_THRESHOLD = 180     # frontend shows "nearing limit" warning
MAX_STORED_DATA_ROWS = 50         # cap SQL result rows persisted in JSONB


class ChatMessageRequest(BaseModel):
    query: str
    conversation_id: str | None = None  # omit to start a new conversation
    # When set, retrieval is restricted to files belonging to this container.
    # Mirrors the behaviour of GitHub Copilot's model picker — the user
    # explicitly chooses which container to chat with.
    container_id: str | None = None


class IngestRequest(BaseModel):
    file_ids: list[str]
    force_preprocess: bool = False


class ConversationRenameRequest(BaseModel):
    title: str


async def _resolve_chat_scope_legacy(
    user: "User",
    requested_container_id: str | None,
    db,
) -> tuple[str | None, list[str] | None, str]:
    """The pre-RBAC-v2 scoping. Returns (container_id, allowed_domains, scope_source).

    Multi-tenancy hard-rule (Phase 16):
      - Platform admin (is_admin=True): may pass any container_id; allowed_domains
        ignored unless explicitly set on the user.
      - Org user (is_admin=False, organization_id set): container_id is FORCED
        to the org's container_id. The body.container_id field is IGNORED so a
        client cannot tamper with the JWT-bound scope.
      - Org-less non-admin: falls back to body.container_id (backward-compat for
        existing users not yet assigned to an org). Domain sub-filter still applies.
    """
    is_admin = bool(getattr(user, "is_admin", False))

    # Allowed-domains: same logic as before — None for admins or unset/empty list
    domains = getattr(user, "allowed_domains", None)
    allowed_domains: list[str] | None = list(domains) if (domains and not is_admin) else None

    effective_container_id: str | None
    scope_source: str
    if is_admin:
        effective_container_id = requested_container_id
        scope_source = "admin_body"
    else:
        org_id = getattr(user, "organization_id", None)
        if not org_id:
            effective_container_id = requested_container_id
            scope_source = "no_org_fallback_body"
        else:
            org = await db.get(Organization, org_id)
            if not org or not org.container_id:
                effective_container_id = requested_container_id
                scope_source = "org_no_container_fallback_body"
            else:
                effective_container_id = org.container_id
                scope_source = "org_forced"

    return effective_container_id, allowed_domains, scope_source


async def _resolve_org_container_ids(org_id: str, db) -> list[str]:
    """Return ALL container ids owned by an org, primary first.

    Multi-container support: an org may own many ContainerConfig rows. We resolve
    the full set so the chat scope covers every container the org can reach, with
    the primary container ordered first for single-container compatibility.

    Falls back to the org's legacy single `container_id` if no ContainerConfig
    rows are found (older orgs that predate multi-container).
    """
    from sqlalchemy import select

    from app.models.container import ContainerConfig

    container_ids: list[str] = []
    try:
        rows = (
            await db.execute(
                select(ContainerConfig.id, ContainerConfig.is_primary).where(
                    ContainerConfig.organization_id == org_id
                )
            )
        ).all()
        # Primary first, then the rest (stable) — primary is the compat fallback.
        rows = sorted(rows, key=lambda r: (not bool(r[1])))
        container_ids = [r[0] for r in rows if r[0]]
    except Exception as exc:  # never let scope resolution hard-fail
        chat_logger.warning("org_container_lookup_failed", organization_id=org_id, error=str(exc)[:200])

    if not container_ids:
        # Legacy fallback: the org's single bound container.
        org = await db.get(Organization, org_id)
        legacy = getattr(org, "container_id", None) if org else None
        if legacy:
            container_ids = [legacy]
    return container_ids


async def _resolve_chat_scope_v2(
    user: "User",
    requested_container_id: str | None,
    db,
) -> tuple[str | None, list[str] | None, str, list[str]]:
    """RBAC v2 org+domain scoping via org_access, over ALL org containers.

    Returns (primary_container_id, allowed_domains, scope_source, container_ids).

    - No org => empty scope (nothing).
    - Platform admin: default-deny — sees nothing without an active grant.
    - Org owner / admin: all org containers (all domains).
    - Manager / user: same org containers, limited to their assigned domains.
    """
    from app.services.org_access import effective_domains_for, user_can_access_org

    org_id = getattr(user, "organization_id", None)
    if not org_id:
        return None, [], "v2_no_org_empty", []

    if not await user_can_access_org(user, org_id, db):
        return None, [], "v2_no_access_empty", []

    # Resolve over ALL of the org's containers (not just org.container_id).
    container_ids = await _resolve_org_container_ids(org_id, db)
    primary_container_id = container_ids[0] if container_ids else None

    # Owner/org_admin -> all domains (None); manager/user -> assigned domains.
    allowed_domains = await effective_domains_for(user, org_id, db)
    return primary_container_id, allowed_domains, "v2_org_scoped", container_ids


async def resolve_chat_scope_full(
    user: "User",
    requested_container_id: str | None,
    db,
) -> tuple[str | None, list[str] | None, list[str]]:
    """Return (primary_container_id, allowed_domains, container_ids).

    Same flag-gating as resolve_chat_scope, but also exposes the FULL list of
    container ids the request is scoped to (multi-container support). The legacy
    path has a single container, so its list is just [container_id] (or []).
    """
    settings = get_settings()
    is_admin = bool(getattr(user, "is_admin", False))

    if settings.RBAC_V2_ENFORCE:
        effective_container_id, allowed_domains, scope_source, container_ids = (
            await _resolve_chat_scope_v2(user, requested_container_id, db)
        )
    else:
        effective_container_id, allowed_domains, scope_source = await _resolve_chat_scope_legacy(
            user, requested_container_id, db
        )
        container_ids = [effective_container_id] if effective_container_id else []

    # Audit trail — every chat request records the resolved scope so we can
    # confirm domain/container restrictions are actually being applied.
    chat_logger.info(
        "chat_scope_resolved",
        user_id=getattr(user, "id", None),
        is_admin=is_admin,
        organization_id=getattr(user, "organization_id", None),
        requested_container_id=requested_container_id,
        effective_container_id=effective_container_id,
        container_ids=container_ids,
        allowed_domains=allowed_domains,
        scope_source=scope_source,
    )

    # Shadow mode — only when NOT enforcing: compute and log what RBAC v2 WOULD
    # do, mirroring the chat_scope_resolved event. Never changes the return value.
    if not settings.RBAC_V2_ENFORCE and settings.RBAC_V2_SHADOW:
        try:
            v2_container, v2_domains, v2_source, v2_container_ids = await _resolve_chat_scope_v2(
                user, requested_container_id, db
            )
            chat_logger.info(
                "chat_scope_resolved_shadow_v2",
                user_id=getattr(user, "id", None),
                is_admin=is_admin,
                organization_id=getattr(user, "organization_id", None),
                requested_container_id=requested_container_id,
                effective_container_id=v2_container,
                container_ids=v2_container_ids,
                allowed_domains=v2_domains,
                scope_source=v2_source,
            )
        except Exception as exc:  # shadow must never affect the request
            chat_logger.warning(
                "chat_scope_shadow_v2_failed",
                user_id=getattr(user, "id", None),
                error=str(exc)[:200],
            )

    return effective_container_id, allowed_domains, container_ids


async def resolve_chat_scope(
    user: "User",
    requested_container_id: str | None,
    db,
) -> tuple[str | None, list[str] | None]:
    """Return (effective_container_id, allowed_domains) for a chat request.

    Compatibility wrapper over resolve_chat_scope_full — downstream paths that
    expect a single container receive the primary (is_primary) container id.
    """
    effective_container_id, allowed_domains, _container_ids = await resolve_chat_scope_full(
        user, requested_container_id, db
    )
    return effective_container_id, allowed_domains


async def bg_title_and_summary(conv_id: str) -> None:
    """Background task: generate title + regenerate summary if needed."""
    try:
        async with async_session() as db:
            await maybe_generate_title(conv_id, db)
            await maybe_regenerate_summary(conv_id, db)
    except Exception as exc:
        chat_logger.warning("bg_task_failed", conversation_id=conv_id, error=str(exc)[:200])

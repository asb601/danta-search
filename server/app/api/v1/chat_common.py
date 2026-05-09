"""Shared constants, schemas, and helpers for chat API modules."""
from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel

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


class ConversationRenameRequest(BaseModel):
    title: str


async def resolve_chat_scope(
    user: "User",
    requested_container_id: str | None,
    db,
) -> tuple[str | None, list[str] | None]:
    """Return (effective_container_id, allowed_domains) for a chat request.

    Multi-tenancy hard-rule (Phase 16):
      - Platform admin (is_admin=True): may pass any container_id; allowed_domains
        ignored unless explicitly set on the user.
      - Org user (is_admin=False, organization_id set): container_id is FORCED
        to the org's container_id. The body.container_id field is IGNORED so a
        client cannot tamper with the JWT-bound scope.
      - Org-less non-admin: chat is denied — they must be assigned to an org.
    """
    is_admin = bool(getattr(user, "is_admin", False))

    # Allowed-domains: same logic as before — None for admins or unset/empty list
    domains = getattr(user, "allowed_domains", None)
    allowed_domains: list[str] | None = list(domains) if (domains and not is_admin) else None

    if is_admin:
        return requested_container_id, allowed_domains

    org_id = getattr(user, "organization_id", None)
    if not org_id:
        raise HTTPException(
            status_code=403,
            detail="User is not assigned to an organization. Contact an admin.",
        )

    org = await db.get(Organization, org_id)
    if not org or not org.container_id:
        raise HTTPException(
            status_code=403,
            detail="Your organization has no container configured. Contact an admin.",
        )

    return org.container_id, allowed_domains


async def bg_title_and_summary(conv_id: str) -> None:
    """Background task: generate title + regenerate summary if needed."""
    try:
        async with async_session() as db:
            await maybe_generate_title(conv_id, db)
            await maybe_regenerate_summary(conv_id, db)
    except Exception as exc:
        chat_logger.warning("bg_task_failed", conversation_id=conv_id, error=str(exc)[:200])

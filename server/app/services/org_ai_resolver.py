"""
org_ai_resolver — resolve the effective Azure OpenAI credentials/deployments for
an organization.

Org-RBAC overhaul (Lane C). When ORG_AI_KEYS_ENABLED is True AND an OrgAISettings
row exists for the organization, the per-org keys/deployments win. Otherwise we
fall back to the global process-wide values from core.config.

This is a PURE resolver: it returns a plain dict and never mutates global state.
It is intentionally NOT wired into ai_client / openai_client yet — integration is
a later, separately flag-gated step. Until then the runtime behavior is unchanged.
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.org_ai_settings import OrgAISettings

logger = structlog.get_logger("org_ai_resolver")


def _global_settings_dict() -> dict[str, Any]:
    """Resolve the global (process-wide) AI settings from core.config.

    Mirrors the .env alias fallbacks already present in config.py
    (AZURE_OPENAI_API_BASE / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION).
    """
    settings = get_settings()
    endpoint = settings.AZURE_OPENAI_ENDPOINT or settings.AZURE_OPENAI_API_BASE or ""
    api_key = settings.AZURE_OPENAI_KEY or settings.AZURE_OPENAI_API_KEY or ""
    api_version = settings.AZURE_OPENAI_API_VERSION or ""
    return {
        "source": "global",
        "chat_endpoint": endpoint,
        "chat_api_key": api_key,
        "chat_deployment": settings.AZURE_OPENAI_DEPLOYMENT or "",
        "embeddings_api_key": api_key,
        "embeddings_deployment": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "",
        "fallback_api_key": api_key,
        "fallback_deployment": settings.AZURE_OPENAI_DEPLOYMENT_MINI or "gpt-4o-mini",
        "api_version": api_version,
    }


async def resolve_org_ai_settings(
    organization_id: str | None,
    db: AsyncSession,
) -> dict[str, Any]:
    """Return the effective AI settings dict for an organization.

    Per-org settings are used only when:
      * ORG_AI_KEYS_ENABLED is True, AND
      * organization_id is provided, AND
      * an OrgAISettings row exists for it.

    For each field, an empty/None per-org value transparently falls back to the
    corresponding global value, so a partially-filled OrgAISettings row never
    breaks resolution.
    """
    global_cfg = _global_settings_dict()

    flag = getattr(get_settings(), "ORG_AI_KEYS_ENABLED", False)
    if not flag or not organization_id:
        return global_cfg

    try:
        row = (
            await db.execute(
                select(OrgAISettings).where(
                    OrgAISettings.organization_id == organization_id
                )
            )
        ).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001 — never let resolution hard-fail
        logger.warning(
            "org_ai_settings_lookup_failed",
            organization_id=organization_id,
            error=str(exc)[:300],
        )
        return global_cfg

    if row is None:
        return global_cfg

    def _pick(value: Any, fallback_key: str) -> Any:
        return value if value else global_cfg.get(fallback_key)

    return {
        "source": "org",
        "organization_id": organization_id,
        "chat_endpoint": _pick(row.chat_endpoint, "chat_endpoint"),
        "chat_api_key": _pick(row.chat_api_key, "chat_api_key"),
        "chat_deployment": _pick(row.chat_deployment, "chat_deployment"),
        "embeddings_api_key": _pick(row.embeddings_api_key, "embeddings_api_key"),
        "embeddings_deployment": _pick(
            row.embeddings_deployment, "embeddings_deployment"
        ),
        "fallback_api_key": _pick(row.fallback_api_key, "fallback_api_key"),
        "fallback_deployment": _pick(row.fallback_deployment, "fallback_deployment")
        or "gpt-4o-mini",
        "api_version": _pick(row.api_version, "api_version"),
    }

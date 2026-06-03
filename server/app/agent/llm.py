"""Azure OpenAI LangChain clients — thread-safe singletons.

Two deployments:
  get_llm()       → gpt-4o      (primary, used on turn 1)
  get_llm_mini()  → gpt-4o-mini (cheaper, used on follow-up turns 2+;
                                   graph_builder falls back to get_llm() on RateLimitError)

Per-org AI keys (Lane C)
------------------------
When ORG_AI_KEYS_ENABLED and an org has OrgAISettings, the chat/agent path passes
a pre-resolved settings dict (from services.org_ai_resolver.resolve_org_ai_settings)
as `org_ai=...`. We then build a per-org client keyed by the org's effective
endpoint/deployment/api-version. When `org_ai` is None (ingestion + every other
path) we return the global process-wide singletons unchanged. Any failure building
an org client falls back to the global singleton — a missing/bad org key never
breaks the call.
"""
from __future__ import annotations

import threading
from typing import Any

import structlog
from langchain_openai import AzureChatOpenAI

from app.core.config import get_settings

logger = structlog.get_logger("agent_llm")

_llm: AzureChatOpenAI | None = None
_llm_mini: AzureChatOpenAI | None = None
_lock = threading.Lock()

# Per-org client cache, keyed by (kind, endpoint, deployment, api_version, max_tokens).
# Keys/endpoints differ per org; caching avoids rebuilding the client every request.
_org_clients: dict[tuple, AzureChatOpenAI] = {}


def _make_llm(deployment: str, max_tokens: int = 1500) -> AzureChatOpenAI:
    s = get_settings()
    endpoint = s.AZURE_OPENAI_ENDPOINT or s.AZURE_OPENAI_API_BASE
    api_key = s.AZURE_OPENAI_KEY or s.AZURE_OPENAI_API_KEY
    api_version = s.AZURE_OPENAI_API_VERSION
    return _build_client(endpoint, api_key, deployment, api_version, max_tokens)


def _build_client(
    endpoint: str | None,
    api_key: str | None,
    deployment: str | None,
    api_version: str | None,
    max_tokens: int,
) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0,
        max_completion_tokens=max_tokens,
        timeout=25,  # reduced from 60 — fail fast, free up quota faster
        max_retries=1,  # reduced from 2 — one retry is enough under load
    )


def _org_client(
    kind: str,
    org_ai: dict[str, Any],
    endpoint_key: str,
    deployment_key: str,
    api_key_key: str,
    max_tokens: int,
) -> AzureChatOpenAI | None:
    """Build (and cache) a per-org client from a resolved org_ai settings dict.

    Returns None on any problem so the caller falls back to the global singleton.
    The resolver already merges per-org values over global fallbacks, so partial
    OrgAISettings rows resolve to a complete, usable config.
    """
    try:
        endpoint = org_ai.get(endpoint_key) or ""
        deployment = org_ai.get(deployment_key) or ""
        api_key = org_ai.get(api_key_key) or ""
        api_version = org_ai.get("api_version") or ""
        if not (endpoint and deployment and api_key):
            return None  # incomplete — use global
        cache_key = (kind, endpoint, deployment, api_version, max_tokens)
        client = _org_clients.get(cache_key)
        if client is None:
            with _lock:
                client = _org_clients.get(cache_key)
                if client is None:
                    client = _build_client(endpoint, api_key, deployment, api_version, max_tokens)
                    _org_clients[cache_key] = client
        return client
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning("org_llm_build_failed", kind=kind, error=str(exc)[:200])
        return None


def get_llm(org_ai: dict[str, Any] | None = None) -> AzureChatOpenAI:
    """Return the gpt-4o client (primary model, turn 1).

    When `org_ai` is provided (chat path with org context + enabled flag) and
    yields a complete config, a per-org client is used; otherwise the global
    singleton is returned.
    """
    if org_ai and org_ai.get("source") == "org":
        # When gpt-4o is disabled, use the org's fallback (cheaper) lane.
        _disabled = get_settings().DISABLE_GPT4O
        client = _org_client(
            "primary", org_ai,
            endpoint_key="chat_endpoint",
            deployment_key="fallback_deployment" if _disabled else "chat_deployment",
            api_key_key="fallback_api_key" if _disabled else "chat_api_key",
            max_tokens=1500,
        )
        if client is not None:
            return client

    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                # chat_deployment() routes to gpt-4o-mini when DISABLE_GPT4O is
                # set, so this "primary" accessor never spins up gpt-4o.
                _llm = _make_llm(get_settings().chat_deployment())
    return _llm


def get_llm_mini(org_ai: dict[str, Any] | None = None) -> AzureChatOpenAI:
    """Return the gpt-4o-mini client (primary model for all turns).

    Per-org override applies when `org_ai` is provided and complete; otherwise the
    global singleton is used. The mini client maps to the org's fallback
    deployment/key (resolver default: gpt-4o-mini).
    """
    if org_ai and org_ai.get("source") == "org":
        client = _org_client(
            "mini", org_ai,
            endpoint_key="chat_endpoint",
            deployment_key="fallback_deployment",
            api_key_key="fallback_api_key",
            max_tokens=800,
        )
        if client is not None:
            return client

    global _llm_mini
    if _llm_mini is None:
        with _lock:
            if _llm_mini is None:
                # 800 max_tokens is sufficient for SQL (avg ~200 tokens) and
                # structured answers. Lower ceiling = faster streaming TTFT.
                _llm_mini = _make_llm(get_settings().AZURE_OPENAI_DEPLOYMENT_MINI, max_tokens=800)
    return _llm_mini

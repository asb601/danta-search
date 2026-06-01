"""Azure OpenAI client singleton for backend LLM tasks.

By default this is a single-deployment path identical to the legacy behaviour:
``get_client()`` returns one process-wide ``AzureOpenAI`` client + its deployment
name. When (and only when) operators configure more than one *chat* deployment
under ``model_pool.deployments`` in the ingestion policy, ``get_chat_client()``
becomes a SYNC, health-aware, weighted multi-lane picker with 429/timeout
failover — a synchronous mirror of ``app.core.model_pool.ModelPool``.

The callers of this module are SYNC (Azure OpenAI is invoked inside a worker
thread via ``asyncio.to_thread``); they must stay sync, so this picker is a
synchronous mirror and never touches the async ``ModelPool``.
"""
from __future__ import annotations

import threading
import time

from openai import AzureOpenAI

from app.core.config import get_settings

_ai_client: AzureOpenAI | None = None
_ai_deployment: str | None = None
_client_lock = threading.Lock()

# ── Multi-lane (model_pool) sync state ──────────────────────────────────────
# These are only ever touched when >1 chat deployment is configured. With the
# default empty deployment list they stay untouched and get_client() behaves
# byte-identically to the legacy single-deployment path.
_pool_lock = threading.Lock()
_pool_clients: dict[str, AzureOpenAI] = {}  # keyed "endpoint|deployment_id"
_pool_health: dict = {}  # name -> HealthState (model_pool.HealthState)


def get_client() -> tuple[AzureOpenAI, str]:
    """Get (or lazily create) a process-wide Azure OpenAI client and deployment name.

    Backward-compatible entry point. Delegates to ``get_chat_client()`` (default
    tier) so existing callers are unchanged; when no model pool is configured
    this is exactly the legacy single-deployment client.
    """
    return get_chat_client()


def _legacy_client() -> tuple[AzureOpenAI, str]:
    """The historical single-deployment client + deployment name (cached)."""
    global _ai_client, _ai_deployment
    if _ai_client is None:
        with _client_lock:
            if _ai_client is None:
                settings = get_settings()
                endpoint = settings.AZURE_OPENAI_ENDPOINT or settings.AZURE_OPENAI_API_BASE
                api_key = settings.AZURE_OPENAI_KEY or settings.AZURE_OPENAI_API_KEY
                deployment = (
                    settings.AZURE_OPENAI_DEPLOYMENT
                    if settings.AZURE_OPENAI_DEPLOYMENT != "gpt-4"
                    else settings.AZURE_OPENAI_MODEL
                ) or settings.AZURE_OPENAI_DEPLOYMENT

                _ai_client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                )
                _ai_deployment = deployment
    return _ai_client, _ai_deployment


def _chat_deployments():
    """Parse configured chat deployments from the ingestion policy.

    Returns the tuple of chat-kind ``Deployment`` records. With the default
    empty ``model_pool.deployments`` list this returns the single legacy lane,
    so the >1 multi-lane path below is never taken.
    """
    from app.core.model_pool import legacy_deployment_from_settings, load_deployments

    try:
        from app.services.ingestion_policy import get_ingestion_policy

        raw = get_ingestion_policy().lookup(("model_pool", "deployments"))
    except Exception:  # noqa: BLE001 — any policy failure degrades to legacy
        raw = None

    legacy = legacy_deployment_from_settings(get_settings(), kind="chat")
    deployments = load_deployments(raw, legacy=legacy)
    # Only chat lanes are eligible for chat selection.
    return tuple(d for d in deployments if d.kind == "chat")


def _pool_client_for(d) -> AzureOpenAI:
    """Lazily build + cache an AzureOpenAI client for a lane (sync mirror)."""
    key = f"{d.endpoint}|{d.deployment_id}"
    client = _pool_clients.get(key)
    if client is None:
        with _pool_lock:
            client = _pool_clients.get(key)
            if client is None:
                settings = get_settings()
                client = AzureOpenAI(
                    azure_endpoint=(
                        d.endpoint
                        or settings.AZURE_OPENAI_ENDPOINT
                        or settings.AZURE_OPENAI_API_BASE
                    ),
                    api_key=settings.AZURE_OPENAI_KEY or settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                )
                _pool_clients[key] = client
    return client


def _cooldown_seconds() -> float:
    try:
        from app.services.ingestion_policy import get_ingestion_policy

        value = get_ingestion_policy().lookup(("model_pool", "circuit_breaker_cooldown_seconds"))
        if value is not None:
            return max(0.0, float(value))
    except Exception:  # noqa: BLE001
        pass
    return 20.0


def get_chat_client(*, tier: str = "standard") -> tuple[AzureOpenAI, str]:
    """Return (sync AzureOpenAI client, deployment_id) for a chat completion.

    Default (no/one chat deployment configured) -> the legacy single-deployment
    client, byte-identical to today's behaviour. With >1 chat deployment this
    performs a weighted, health-aware lane pick (``select_deployment``) over a
    MODULE-LEVEL sync health map + per-lane client cache, tripping a lane that
    raises RateLimit/timeout into cooldown and returning the next healthy lane.

    NOTE: this returns a client+deployment pair (the caller makes the actual
    ``chat.completions.create`` call). True per-call failover requires retrying
    inside the caller; this picker excludes cooling lanes and rotates on each
    call so a tripped lane is avoided on subsequent picks within the cooldown.
    The trip itself is recorded via ``report_chat_failure``.
    """
    deployments = _chat_deployments()
    if len(deployments) <= 1:
        # Default path: zero behaviour change.
        return _legacy_client()

    import random

    from app.core.model_pool import HealthState, select_deployment

    now = time.monotonic()
    chosen = select_deployment(
        deployments,
        _pool_health,
        random.random(),
        kind="chat",
        tier=tier,
        now=now,
    )
    if chosen is None:
        return _legacy_client()
    return _pool_client_for(chosen), chosen.deployment_id


def chat_complete_with_failover(
    *,
    messages,
    tier: str = "standard",
    max_attempts: int | None = None,
    **create_kwargs,
):
    """Run a chat completion with intra-attempt multi-lane failover.

    Picks a (client, deployment) via ``get_chat_client(tier=tier)`` and calls
    ``client.chat.completions.create(model=deployment, messages=messages,
    **create_kwargs)``. On ``openai.RateLimitError`` / ``openai.APITimeoutError``
    it trips the picked lane via ``report_chat_failure`` and retries on the next
    pick; on any other exception (or success) it returns/propagates immediately.
    If every attempt is exhausted by rate-limit/timeout, the last error re-raises.

    SINGLE-LANE / DEFAULT (``model_pool.deployments == []``): there is exactly one
    chat lane, so ``max_attempts`` collapses to 1 and this makes exactly ONE
    ``create`` call with the supplied kwargs — byte-identical to calling
    ``get_chat_client(...)`` then ``client.chat.completions.create(...)`` directly.
    No extra retries are introduced for the single-lane case; the caller's own
    outer retry loop is unaffected.
    """
    import openai as _openai  # noqa: PLC0415 — local to avoid import cost/cycles

    if max_attempts is None:
        # One attempt per configured chat lane (min 1, capped to avoid runaway).
        max_attempts = min(4, max(1, len(_chat_deployments())))

    last_exc: Exception | None = None
    for _attempt in range(max_attempts):
        client, deployment = get_chat_client(tier=tier)
        try:
            return client.chat.completions.create(
                model=deployment, messages=messages, **create_kwargs
            )
        except (_openai.RateLimitError, _openai.APITimeoutError) as exc:
            last_exc = exc
            report_chat_failure(deployment)
            continue
    # All attempts exhausted by rate-limit/timeout — re-raise the last error.
    assert last_exc is not None
    raise last_exc


def report_chat_failure(deployment_id: str) -> None:
    """Trip the circuit breaker for a chat lane after a 429/timeout.

    No-op unless multi-lane is active. Callers that catch
    ``openai.RateLimitError`` / ``openai.APITimeoutError`` may call this with the
    deployment_id returned by ``get_chat_client`` so the next pick avoids the
    cooling lane.
    """
    deployments = _chat_deployments()
    if len(deployments) <= 1:
        return
    from app.core.model_pool import HealthState

    for d in deployments:
        if d.deployment_id == deployment_id:
            with _pool_lock:
                _pool_health[d.name] = HealthState(
                    cooling_until=time.monotonic() + _cooldown_seconds()
                )
            break

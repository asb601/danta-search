"""Grounded-synthesis LLM adapter (Stage 9). gpt-4o-mini ONLY by default.

Satisfies the agent's ``Llm`` protocol (``async def generate(system, user)``).
The model is resolved through the single model-selection seam
(``model_router.select_model``) — NOT directly — so per-container escalation,
budget caps, and the task→tier allowlist all apply. The synthesis task is
``TaskClass.QUERY_SYNTHESIS``; with no escalation signal (or no budget store
wired) the router fail-safes to the bulk gpt-4o-mini deployment.

The system prompt is sent as the first message AND a stable prompt-cache routing
hint (``user`` / ``prompt_cache_key``, keyed on the stable system-prompt version)
is passed so Azure OpenAI prompt caching can route repeat calls to the cached
instruction prefix. Guarded import: constructs without infra; raises only on CALL.
"""
from __future__ import annotations

import hashlib

from pdf_chat.config import azure_openai_credentials
from pdf_chat.model_router import select_model, TaskClass
from pdf_chat.observability.cost_tracker import get_cost_tracker

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - exercised only without infra
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _build_client():  # pragma: no cover - requires infra + env
    endpoint, api_key, api_version = azure_openai_credentials()
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


def _prompt_cache_key(system: str) -> str:
    """Stable routing hint keyed on the system-prompt version.

    Azure prompt caching routes on a stable ``prompt_cache_key`` / ``user`` hint;
    keying it on the system-prompt content means every query sharing the same
    stable instruction prefix routes to the same cache, while a prompt change
    rotates the key automatically.
    """
    return "pdf-sys-" + hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


class PdfLlm:
    """Grounded synthesis adapter; model chosen via the router, prompt-cached.

    Cost tracking: ``generate`` records the synthesis token usage into the
    per-tenant ``PdfCostTracker`` (Fix 10) so the /api/pdf/metrics cost surface is
    real for queries. cost_usd is best-effort 0.0 (no per-model price table is
    wired in pdf_chat yet). DEFERRED: extraction/vision cost call sites and the
    limiter→production-embed wiring remain unwired (sync Celery path, out of scope).
    """

    async def generate(
        self,
        system: str,
        user: str,
        *,
        container_id: str = "",
        signals: dict | None = None,
    ) -> str:
        if not _HAS_OPENAI:
            raise RuntimeError(
                "The OpenAI SDK is required for LLM synthesis but is not installed."
            )
        # Route the model through the single selection seam (contract C7). The
        # synthesis task can never reach Opus; with no signal/budget it fail-safes
        # to the bulk gpt-4o-mini deployment.
        choice = select_model(
            task=TaskClass.QUERY_SYNTHESIS,
            container_id=container_id,
            signals=signals or {},
        )
        client = _build_client()
        cache_key = _prompt_cache_key(system)
        resp = client.chat.completions.create(
            model=choice.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            # Prompt-cache routing hint: stable per system-prompt version so Azure
            # can route repeat calls to the cached instruction prefix.
            user=cache_key,
        )
        # Record the synthesis cost so GET /api/pdf/metrics reflects real query-time
        # usage (Fix 10 — track_llm previously had zero production callers). The
        # tenant scope in pdf_chat IS container_id; the phase is "synthesis"; the
        # SELECTED model id (choice.model_id) is what the router resolved, so a
        # gpt-4o regression is flagged as a policy_violation by the tracker.
        #
        # cost_usd is best-effort 0.0: pdf_chat has no per-model price table wired
        # yet, so token counts are tracked and the dollar figure stays 0.0 until a
        # price table is added. getattr-guards mean a fake/missing .usage never
        # raises and never changes generate()'s return value.
        #
        # DEFERRED productionization (documented per the task): the ingestion
        # extraction / vision cost call sites and the rate-limiter→production-embed
        # wiring are NOT done here — that path is sync Celery and out of scope; this
        # fix only makes the QUERY (synthesis) cost surface real.
        usage = getattr(resp, "usage", None)
        get_cost_tracker().track_llm(
            container_id,
            "synthesis",
            choice.model_id,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=0.0,  # no price table wired in pdf_chat yet (best-effort).
            document_id=None,
            trace_id=None,
        )
        return resp.choices[0].message.content or ""

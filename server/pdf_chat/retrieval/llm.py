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
import os

from pdf_chat.model_router import select_model, TaskClass

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - exercised only without infra
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _build_client():  # pragma: no cover - requires infra + env
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
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
    """Grounded synthesis adapter; model chosen via the router, prompt-cached."""

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
        return resp.choices[0].message.content or ""

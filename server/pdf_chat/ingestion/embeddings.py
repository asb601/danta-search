"""Stage 12 — Embedding API.

Generates chunk embeddings using the SAME model as query-time retrieval
(contract: ``text-embedding-3-small`` / 1536-dim / cosine). Mixing models
produces meaningless similarity scores, so both ingest and query call through
this single interface.

The Azure OpenAI client is imported behind a guard so the module imports and the
pure interface is usable with zero infra. Calling :func:`embed_texts` without the
SDK installed raises a clear ``RuntimeError`` at call time, never at import.
"""
from __future__ import annotations

import asyncio
import os

from ..config import get_pdf_settings

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - exercised only without infra
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _build_client():  # pragma: no cover - requires infra + env
    return AzureOpenAI(  # type: ignore[operator]
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts into 1536-dim vectors.

    Args:
        texts: chunk texts to embed.
        model: override the embedding deployment (defaults to the configured
            ``embedding_model`` — the same model used at query time).

    Returns:
        One vector per input text, in order. Empty input → empty list.

    Raises:
        RuntimeError: if the OpenAI SDK is not installed (raised on CALL).
    """
    if not texts:
        return []
    if not _HAS_OPENAI:
        raise RuntimeError(
            "The OpenAI SDK is required to generate embeddings but is not "
            "installed. Install it with `pip install openai` to enable embedding."
        )

    settings = get_pdf_settings()
    deployment = model or settings.embedding_model
    client = _build_client()
    resp = client.embeddings.create(input=texts, model=deployment)
    return [item.embedding for item in resp.data]


async def embed_texts_bounded(
    batches: list[list[str]],
    *,
    container_id: str,
    model: str | None = None,
) -> list[list[list[float]]]:
    """Embed many batches under exponential backoff + bounded concurrency.

    Each element of ``batches`` is one embedding request; results preserve input
    order. Used by the ingestion fan-out where N batches would otherwise burst
    Azure's rate limit. Reuses the synchronous :func:`embed_texts` per batch inside
    a thread (``asyncio.to_thread``) so the SDK's blocking call doesn't stall the
    event loop, and routes every call through :class:`BoundedBackoffExecutor` so a
    429/503 is retried and the in-flight count is capped per the tunables.

    The existing synchronous :func:`embed_texts` is left untouched so Phase 0/1
    callers are unaffected.
    """
    from .rate_limiter import BoundedBackoffExecutor

    executor = BoundedBackoffExecutor(container_id=container_id)

    def _factory(texts: list[str]) -> "Callable":  # noqa: F821 - local typing only
        async def _call() -> list[list[float]]:
            # Look up embed_texts on the module at call time so a test monkeypatch
            # of ``embeddings.embed_texts`` is respected (don't bind the symbol now).
            import pdf_chat.ingestion.embeddings as _emb

            return await asyncio.to_thread(_emb.embed_texts, texts, model=model)

        return _call

    return await executor.gather_bounded([_factory(b) for b in batches])

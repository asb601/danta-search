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

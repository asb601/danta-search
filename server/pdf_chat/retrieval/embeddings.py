"""Query + batch embedding adapters (token guards: batch + query cache).

Reuses the SAME embedding model as ingestion (text-embedding-3-small / 1536) via
the shared ``ingestion.embeddings.embed_texts`` call, and resolves the model id
through the single ``model_router.embedding_model`` seam so a per-container model
swap stays data-driven. Adds two cost-at-scale token guards:

  * ``embed_texts_batched`` — splits a large list into config-sized batches so a
    big document (millions of chunks across many tenants) never issues one
    oversized embedding request. Batch size is a tunable (per-container).
  * ``QueryEmbedder`` — async query embedder with an optional Redis
    query-embedding cache so a repeated query never re-embeds (a hot-query token
    saver). Cache key is model-scoped, so a model swap never serves a stale
    vector.

GOVERNING CRITERIA (cost-at-scale, multi-tenant, per-client tunable):
  * every threshold (batch size, cache TTL) resolves via ``get_tunable`` — no
    bare literal lives here;
  * every batch / cache-hit decision logs via ``log_gate_decision``;
  * everything is scoped by ``container_id`` (the tenant boundary).

Pure module — safe to import with zero infra. The embedding model id lookup and
the actual embedding call are both delegated lazily, so importing this never
touches Azure OpenAI or a database.
"""
from __future__ import annotations

import hashlib
from typing import Any

from pdf_chat.ingestion.embeddings import embed_texts
from pdf_chat.tunables import get_tunable, log_gate_decision


def embed_texts_batched(
    texts: list[str],
    *,
    container_id: str,
    batch_size: int | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in config-sized batches, preserving input order.

    The batch size is a per-container tunable (``embedding_batch_size``) so an
    operator can shrink the per-request FAN-OUT for a tenant whose documents
    issue many embedding requests — without a deploy. This caps the request
    fan-out (number of items per call), NOT a per-request token ceiling: each
    item's own size is bounded upstream by the chunker. Every batch is logged via
    ``log_gate_decision`` so the request fan-out is observable per container.
    """
    if not texts:
        return []
    if batch_size is None:
        batch_size = get_tunable(container_id, "embedding_batch_size")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        log_gate_decision(
            "embedding_batch",
            score=len(batch),
            threshold=batch_size,
            outcome="embed",
            container_id=container_id,
            batch_start=start,
        )
        out.extend(embed_texts(batch, model=model))
    return out


def query_embedding_cache_key(query: str, model: str, container_id: str) -> str:
    """Stable key for a (container, query, embedding-model) triple.

    Tenant-scoped (``container_id`` folded into the hash) so a query cache entry
    can NEVER be served across tenants — the query-embedding cache is per-tenant
    like every other pdf_chat surface. Model-scoped so a model swap never serves
    a stale vector. Prefixed so the namespace never collides with the response
    cache.
    """
    return "pdf:qemb:" + hashlib.sha256(
        f"{container_id}|{model}|{query}".encode("utf-8")
    ).hexdigest()


class QueryEmbedder:
    """Async query embedder with an optional Redis query-embedding cache.

    Satisfies the agent's ``Embedder`` protocol (``async def embed``). ``cache``
    is any object exposing ``get_vector(key) -> list[float] | None`` and
    ``set_vector(key, vec, ttl)`` (the retrieval ``RedisCache`` vector helpers);
    ``None`` disables caching but still embeds. The model id is resolved per
    container through ``model_router.embedding_model`` (the single model seam) so
    the query path embeds with exactly the model used at ingest time.
    """

    def __init__(self, cache: Any = None, model: str | None = None) -> None:
        self._cache = cache
        self._model = model

    def _resolve_model(self, container_id: str) -> str:
        if self._model is not None:
            return self._model
        from pdf_chat.model_router import embedding_model

        return embedding_model(container_id)

    async def embed(self, text: str, container_id: str = "") -> list[float]:
        model = self._resolve_model(container_id)
        key = query_embedding_cache_key(text, model, container_id)
        if self._cache is not None:
            hit = self._cache.get_vector(key)
            if hit is not None:
                log_gate_decision(
                    "query_embedding_cache",
                    score=1,
                    threshold=1,
                    outcome="hit",
                    container_id=container_id,
                )
                return hit
        vec = embed_texts([text], model=model)[0]
        if self._cache is not None:
            ttl = get_tunable(container_id, "query_embedding_cache_ttl")
            self._cache.set_vector(key, vec, ttl)
            log_gate_decision(
                "query_embedding_cache",
                score=0,
                threshold=1,
                outcome="miss_store",
                container_id=container_id,
            )
        return vec

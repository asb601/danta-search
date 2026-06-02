"""
Embedding service — generates and batches text embeddings via Azure OpenAI.

Uses `text-embedding-3-small` (1 536 dims) on the same Azure OpenAI resource
as the rest of the app.  The embedding deployment name is configured via
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` in .env (defaults to "text-embedding-3-small").

Cost reference (Azure OpenAI pricing, 2026):
  text-embedding-3-small  $0.02 / 1M tokens
  1M file descriptions × ~300 tok  ≈  $6 total one-time backfill
  Per user query                   ≈  $0.00001

Public functions
----------------
embed_text(text)          → list[float]           (1 536 dims, single string)
embed_batch(texts)        → list[list[float]]      (up to 100 strings)
build_search_text(obj)    → str                    (canonical text for a file)
"""
from __future__ import annotations

import threading
from typing import Any, Union

from openai import AsyncAzureOpenAI

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Lazy async client — created once, reused across the process lifetime.
# Protected by a lock because the lifespan event is the only writer.
# ---------------------------------------------------------------------------
_embedding_client: AsyncAzureOpenAI | None = None
_embedding_deployment: str | None = None
_lock = threading.Lock()

# ── Per-org embedding client cache ─────────────────────────────────────────
# Org-RBAC (Lane-Embed): when an org supplies its own embeddings key/endpoint/
# deployment we build a dedicated AsyncAzureOpenAI for it. Cached by the
# (endpoint, deployment, api_version) tuple — NOT by org_id — so distinct orgs
# pointing at the same Azure resource share one client, and the API key is
# never used as a cache key or logged.
_org_embedding_clients: dict[tuple[str, str, str], AsyncAzureOpenAI] = {}
_org_lock = threading.Lock()

_EMBED_DIMS = 1536

# ── Embedding pool (flag-gated, default OFF) ───────────────────────────────────
# Lazily built once. With model_pool.embedding_pool_enabled false (default), this
# stays None and embed_batch uses the existing single-client path unchanged.
_embedding_pool = None  # type: ignore[var-annotated]
_pool_lock = threading.Lock()


def _embedding_pool_enabled() -> bool:
    """True only when the flag is on AND ≥1 embedding deployment is configured.

    Returns today's behaviour (False) on any policy failure, so a misconfigured
    policy never changes the embedding path.
    """
    try:
        from app.services.ingestion_policy import get_ingestion_policy

        pol = get_ingestion_policy()
        if not bool(pol.lookup(("model_pool", "embedding_pool_enabled"))):
            return False
        raw = pol.lookup(("model_pool", "deployments")) or []
        from app.core.model_pool import load_deployments

        deployments = load_deployments(raw)
        return any(d.kind == "embedding" for d in deployments)
    except Exception:  # noqa: BLE001 — any failure keeps the default single-client path
        return False


def _get_embedding_pool():
    """Lazily construct the embedding ModelPool (embedding lanes only).

    Returns None when the pool is disabled or no embedding lanes exist, in which
    case callers use the legacy single-client path.
    """
    global _embedding_pool
    if not _embedding_pool_enabled():
        return None
    if _embedding_pool is None:
        with _pool_lock:
            if _embedding_pool is None:
                from app.services.ingestion_policy import get_ingestion_policy
                from app.core.model_pool import ModelPool, load_deployments

                pol = get_ingestion_policy()
                raw = pol.lookup(("model_pool", "deployments")) or []
                deployments = tuple(
                    d for d in load_deployments(raw) if d.kind == "embedding"
                )
                overrides = pol.lookup(("model_pool",))
                overrides = dict(overrides) if isinstance(overrides, dict) else None
                _embedding_pool = ModelPool(deployments, overrides=overrides)
    return _embedding_pool


def _get_embedding_client(
    org_ai: dict[str, Any] | None = None,
) -> tuple[AsyncAzureOpenAI, str]:
    """Return (client, deployment) for embeddings.

    When ``org_ai`` resolves to a fully-specified per-org configuration
    (source == "org" AND embeddings_api_key + chat_endpoint +
    embeddings_deployment all present) a dedicated AsyncAzureOpenAI is built
    and cached by (endpoint, deployment, api_version). Otherwise — org_ai is
    None, source != "org", or any required field is empty — the global
    process-wide client is returned (byte-identical to the legacy path).
    """
    if org_ai and org_ai.get("source") == "org":
        endpoint = org_ai.get("chat_endpoint")
        api_key = org_ai.get("embeddings_api_key")
        deployment = org_ai.get("embeddings_deployment")
        api_version = org_ai.get("api_version") or get_settings().AZURE_OPENAI_API_VERSION
        if endpoint and api_key and deployment:
            cache_key = (endpoint, deployment, api_version)
            client = _org_embedding_clients.get(cache_key)
            if client is None:
                with _org_lock:
                    client = _org_embedding_clients.get(cache_key)
                    if client is None:
                        client = AsyncAzureOpenAI(
                            azure_endpoint=endpoint,
                            api_key=api_key,
                            api_version=api_version,
                        )
                        _org_embedding_clients[cache_key] = client
            return client, deployment

    global _embedding_client, _embedding_deployment
    if _embedding_client is None:
        with _lock:
            if _embedding_client is None:
                s = get_settings()
                endpoint = s.AZURE_OPENAI_ENDPOINT or s.AZURE_OPENAI_API_BASE
                api_key = s.AZURE_OPENAI_KEY or s.AZURE_OPENAI_API_KEY
                _embedding_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=s.AZURE_OPENAI_API_VERSION,
                )
                _embedding_deployment = s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    return _embedding_client, _embedding_deployment  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def embed_text(text: str, org_ai: dict[str, Any] | None = None) -> list[float]:
    """Embed a single string.  Returns a 1 536-dim float list."""
    if not text or not text.strip():
        return [0.0] * _EMBED_DIMS
    results = await embed_batch([text], org_ai=org_ai)
    return results[0]


async def embed_batch(
    texts: list[str], org_ai: dict[str, Any] | None = None
) -> list[list[float]]:
    """Embed up to 100 strings in one API call.

    Returns embeddings in the same order as `texts`.
    Empty / whitespace strings are replaced with zero vectors without an API call.

    When ``org_ai`` resolves to a per-org embedding configuration, the non-pool
    branch routes through that org's dedicated client. NOTE: the flag-gated
    model-pool branch (embedding_pool_enabled, default OFF) is GLOBAL — org keys
    are bypassed while that operator flag is on.
    """
    if not texts:
        return []

    # Separate real texts from blanks (blanks get zero vectors, no API charges)
    indices_to_embed: list[int] = []
    cleaned: list[str] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            indices_to_embed.append(i)
            cleaned.append(t.strip())

    result: list[list[float]] = [[0.0] * _EMBED_DIMS for _ in texts]

    if cleaned:
        try:
            # Flag-gated pool path: route through ModelPool.aembed for weighted
            # selection + 429/timeout failover. Default OFF → legacy single client.
            # The pool is GLOBAL by design; org keys are bypassed when on.
            pool = _get_embedding_pool()
            if pool is not None:
                resp = await pool.aembed(inputs=cleaned)
                embeddings = [item.embedding for item in resp.data]
            else:
                client, deployment = _get_embedding_client(org_ai)
                resp = await client.embeddings.create(
                    model=deployment,
                    input=cleaned,
                )
                embeddings = [item.embedding for item in resp.data]

                # ── Dimension guard ──────────────────────────────────────────
                # An org could point embeddings_deployment at a non-1536-dim
                # model (e.g. text-embedding-3-large = 3072). A wrong-dimension
                # vector would corrupt pgvector / OpenSearch. If any returned
                # embedding is the wrong size when using an ORG client, fall
                # back and re-embed the whole batch with the GLOBAL client.
                used_org_client = bool(org_ai and org_ai.get("source") == "org") and (
                    client is not _embedding_client
                )
                if used_org_client and any(
                    len(e) != _EMBED_DIMS for e in embeddings
                ):
                    _log_embedding_dim_mismatch()
                    g_client, g_deployment = _get_embedding_client(None)
                    resp = await g_client.embeddings.create(
                        model=g_deployment,
                        input=cleaned,
                    )
                    embeddings = [item.embedding for item in resp.data]

            for pos, embedding in enumerate(embeddings):
                result[indices_to_embed[pos]] = embedding
        except Exception as exc:
            # DeploymentNotFound → deployment not yet created in Azure portal.
            # Other errors (rate limit, transient) also handled here.
            # Ingestion continues — embedding columns stay NULL until backfill.
            _log_embedding_failure(exc)

    return result


def _log_embedding_dim_mismatch() -> None:
    """Warn that an org embedding deployment returned a non-1536-dim vector."""
    try:
        from app.core.logger import ingest_logger  # lazy import avoids circular dep
        ingest_logger.warning(
            "embedding_dim_mismatch",
            hint="org embedding dim mismatch, falling back to global",
            expected_dims=_EMBED_DIMS,
        )
    except Exception:
        pass  # logging must never crash the ingestion pipeline


def _log_embedding_failure(exc: Exception) -> None:
    """Emit a single structured warning when embeddings are unavailable."""
    try:
        from app.core.logger import ingest_logger  # lazy import avoids circular dep
        ingest_logger.warning(
            "embedding_unavailable",
            error=str(exc)[:200],
            hint=(
                "Deploy 'text-embedding-3-small' in Azure OpenAI Studio "
                "and set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env"
            ),
        )
    except Exception:
        pass  # logging must never crash the ingestion pipeline


# ---------------------------------------------------------------------------
# Canonical search-text builder
# ---------------------------------------------------------------------------

# Accepted input types: SQLAlchemy FileMetadata row OR plain dict with same keys.
_AnyMetadata = Union["FileMetadata", dict]  # type: ignore[name-defined]


def build_search_text(metadata: _AnyMetadata) -> str:
    """Build the canonical text string indexed for BM25, trgm, and embeddings.

    Concatenates in order:
      file_name · ai_description · column names · good_for topics · key_metrics

    Works with both a FileMetadata ORM row and a dict (catalog entry format).
    Pure function — no I/O, no side effects.
    """
    def _get(key: str, default="") -> str:
        if isinstance(metadata, dict):
            return str(metadata.get(key, default) or default)
        return str(getattr(metadata, key, default) or default)

    def _get_list(key: str) -> list:
        if isinstance(metadata, dict):
            val = metadata.get(key) or []
        else:
            val = getattr(metadata, key, None) or []
        return val if isinstance(val, list) else []

    parts: list[str] = []

    # File name (strip blob-path prefix if present)
    blob = _get("blob_path")
    if blob:
        parts.append(blob.rsplit("/", 1)[-1])  # filename only
    file_name = _get("file_name") or _get("name")
    if file_name:
        parts.append(file_name)

    # AI description
    desc = _get("ai_description")
    if desc:
        parts.append(desc)

    # Column names — accept either the heavy `columns_info` shape (list of
    # {name, type, sample_values, ...} dicts) used by the FileMetadata ORM
    # row or the lean `column_names` list-of-strings shape used by the
    # cached catalog entries.
    col_names: list[str] = []
    for c in _get_list("columns_info"):
        if isinstance(c, dict):
            col_names.append(c.get("name", ""))
        elif isinstance(c, str):
            col_names.append(c)
    if not col_names:
        for c in _get_list("column_names"):
            if isinstance(c, str):
                col_names.append(c)
    if col_names:
        parts.append(" ".join(n for n in col_names if n))

    # good_for topics
    good_for = _get_list("good_for")
    if good_for:
        parts.append(" ".join(str(g) for g in good_for))

    # key_metrics
    key_metrics = _get_list("key_metrics")
    if key_metrics:
        parts.append(" ".join(str(m) for m in key_metrics))

    # key_dimensions
    key_dims = _get_list("key_dimensions")
    if key_dims:
        parts.append(" ".join(str(d) for d in key_dims))

    return " ".join(p.strip() for p in parts if p.strip())

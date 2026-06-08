"""pdf_chat configuration.

Reuses the main app settings where possible (DB, Redis, Azure OpenAI) and adds
PDF-pipeline-specific knobs. Everything is env-overridable so behaviour is never
hardcoded. Pure module — safe to import with no infra installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PdfSettings:
    # Stores
    neo4j_uri: str = _env("PDF_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _env("PDF_NEO4J_USER", "neo4j")
    neo4j_password: str = _env("PDF_NEO4J_PASSWORD", "")
    neo4j_database: str = _env("PDF_NEO4J_DATABASE", "neo4j")
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")

    # Embedding (DECISION: text-embedding-3-small / 1536 — same for ingest+query)
    embedding_model: str = _env("PDF_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = _env_int("PDF_EMBEDDING_DIM", 1536)

    # Chat synthesis + image VLM. POLICY: gpt-4o is NOT used anywhere — both the
    # answer-synthesis LLM and the image/vision (VLM) path default to gpt-4o-mini
    # (which supports vision) for cost control. Override via env only if needed.
    chat_model: str = _env("PDF_CHAT_MODEL", "gpt-4o-mini")
    vision_model: str = _env("PDF_VISION_MODEL", "gpt-4o-mini")

    # Chunking
    chunk_size: int = _env_int("PDF_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("PDF_CHUNK_OVERLAP", 100)

    # Retry / DLQ
    max_retries: int = _env_int("PDF_MAX_RETRIES", 3)
    retry_base_delay: int = _env_int("PDF_RETRY_BASE_DELAY", 60)

    # Retrieval
    vector_top_k: int = _env_int("PDF_VECTOR_TOP_K", 50)
    graph_top_k: int = _env_int("PDF_GRAPH_TOP_K", 20)
    rerank_top_n: int = _env_int("PDF_RERANK_TOP_N", 12)
    rrf_k: int = _env_int("PDF_RRF_K", 60)
    min_accessible_chunks: int = _env_int("PDF_MIN_ACCESSIBLE_CHUNKS", 1)

    # Cache
    cache_ttl_seconds: int = _env_int("PDF_CACHE_TTL", 3600)

    # Preflight reject thresholds (config-driven, not magic numbers in code)
    scanned_text_char_threshold: int = _env_int("PDF_SCANNED_CHAR_THRESHOLD", 10)
    image_entropy_vlm_threshold: float = _env_float("PDF_IMAGE_ENTROPY_THRESHOLD", 0.85)
    needs_review_confidence: float = _env_float("PDF_NEEDS_REVIEW_CONFIDENCE", 0.45)

    # Blob
    blob_container: str = _env("PDF_BLOB_CONTAINER", "pdf-documents")


@lru_cache(maxsize=1)
def get_pdf_settings() -> PdfSettings:
    return PdfSettings()


def azure_openai_credentials() -> tuple[str, str, str]:
    """Resolve ``(endpoint, api_key, api_version)`` for Azure OpenAI.

    pdf_chat reuses the SAME Azure OpenAI credentials the rest of the platform
    already has — there are no pdf-specific key/endpoint values to set. The
    resolution order makes that work no matter how the deployment supplies them
    (first non-empty wins, per field):

      1. process env, canonical names  (``AZURE_OPENAI_ENDPOINT`` / ``_KEY``)
      2. process env, ``.env`` alias names (``AZURE_OPENAI_API_BASE`` / ``_API_KEY``)
      3. the main app ``Settings``, which load ``server/.env`` via
         pydantic-settings — so values living ONLY in ``.env`` (never exported
         into ``os.environ``) are still found.

    This mirrors ``app/core/openai_client.py``'s
    ``endpoint = AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_BASE`` coalescing, so
    pdf_chat and the CSV pipeline always resolve to the same Azure resource.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE") or ""
    api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or ""
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or ""

    if not (endpoint and api_key and api_version):
        try:  # app Settings read server/.env directly — no os.environ export needed
            from app.core.config import get_settings

            s = get_settings()
            endpoint = endpoint or s.AZURE_OPENAI_ENDPOINT or s.AZURE_OPENAI_API_BASE or ""
            api_key = api_key or s.AZURE_OPENAI_KEY or s.AZURE_OPENAI_API_KEY or ""
            api_version = api_version or s.AZURE_OPENAI_API_VERSION or ""
        except Exception:  # pragma: no cover - standalone import with no app/infra
            pass

    return endpoint, api_key, api_version or "2024-02-01"

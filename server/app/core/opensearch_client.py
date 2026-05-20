"""OpenSearch client for metadata retrieval.

OpenSearch is optional. If OPENSEARCH_URL is not configured or a request fails,
callers fall back to PostgreSQL retrieval.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logger import chat_logger, ingest_logger

_DIMS = 1536
_INDEX_SAFE_RE = re.compile(r"[^a-z0-9_-]+")


def opensearch_enabled() -> bool:
    return bool(get_settings().OPENSEARCH_URL.strip())


def index_name_for_container(container_id: str) -> str:
    prefix = get_settings().OPENSEARCH_INDEX_PREFIX or "gchat-files"
    safe_container = _INDEX_SAFE_RE.sub("-", container_id.lower()).strip("-")
    return f"{prefix}-{safe_container}"


def _auth() -> tuple[str, str] | None:
    settings = get_settings()
    if settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
        return (settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD)
    return None


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = get_settings().OPENSEARCH_API_KEY
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    return headers


def _base_url() -> str:
    return get_settings().OPENSEARCH_URL.rstrip("/")


async def _request(method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=get_settings().OPENSEARCH_TIMEOUT_SECONDS) as client:
        response = await client.request(
            method,
            f"{_base_url()}{path}",
            headers=_headers(),
            auth=_auth(),
            **kwargs,
        )
        response.raise_for_status()
        return response


async def ensure_container_index(container_id: str) -> str:
    """Create the per-container index if missing and return its name."""
    index_name = index_name_for_container(container_id)
    if not opensearch_enabled():
        return index_name

    try:
        await _request("HEAD", f"/{index_name}")
        return index_name
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise

    body: dict[str, Any] = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": get_settings().OPENSEARCH_SHARDS,
                "number_of_replicas": get_settings().OPENSEARCH_REPLICAS,
            },
            "analysis": {
                "analyzer": {
                    "gchat_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "file_id": {"type": "keyword"},
                "container_id": {"type": "keyword"},
                "blob_path": {"type": "keyword"},
                "domain_tag": {"type": "keyword"},
                "ai_description": {"type": "text", "analyzer": "gchat_text"},
                "good_for": {"type": "text", "analyzer": "gchat_text"},
                "key_metrics": {"type": "text", "analyzer": "gchat_text"},
                "key_dimensions": {"type": "text", "analyzer": "gchat_text"},
                "column_names": {"type": "text", "analyzer": "gchat_text"},
                "search_text": {"type": "text", "analyzer": "gchat_text"},
                "date_range_start": {"type": "date"},
                "date_range_end": {"type": "date"},
                "description_embedding": {
                    "type": "knn_vector",
                    "dimension": _DIMS,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
    await _request("PUT", f"/{index_name}", json=body)
    ingest_logger.info("opensearch_index_created", index=index_name, container_id=container_id)
    return index_name


async def index_file_metadata(container_id: str, file_id: str, document: dict[str, Any]) -> None:
    if not opensearch_enabled():
        return
    try:
        index_name = await ensure_container_index(container_id)
        await _request("PUT", f"/{index_name}/_doc/{file_id}", json=document)
        ingest_logger.info("opensearch_document_indexed", index=index_name, file_id=file_id)
    except Exception as exc:
        # OpenSearch is optional; PostgreSQL remains the source of truth.
        ingest_logger.warning("opensearch_index_failed", file_id=file_id, error=str(exc)[:300])


async def search_index(container_id: str, body: dict[str, Any]) -> dict[str, Any]:
    index_name = index_name_for_container(container_id)
    response = await _request("POST", f"/{index_name}/_search", json=body)
    return response.json()


async def delete_file_metadata(container_id: str, file_id: str) -> None:
    if not opensearch_enabled():
        return
    index_name = index_name_for_container(container_id)
    try:
        await _request("DELETE", f"/{index_name}/_doc/{file_id}")
    except Exception as exc:
        chat_logger.warning("opensearch_delete_failed", file_id=file_id, error=str(exc)[:200])

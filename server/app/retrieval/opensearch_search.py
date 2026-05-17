"""OpenSearch-backed metadata retrieval.

This replaces PostgreSQL BM25/fuzzy/vector retrieval when OPENSEARCH_URL is set.
PostgreSQL remains the source of truth; OpenSearch returns ranked file_ids and
we hydrate FileMetadata rows from Postgres before returning to callers.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.core.opensearch_client import opensearch_enabled, search_index
from app.models.file_metadata import FileMetadata
from app.retrieval.embeddings import embed_text

_STAGE_LIMIT = 50


def _date_filter(date_from: date | None, date_to: date | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if date_from:
        filters.append({
            "bool": {
                "should": [
                    {"range": {"date_range_end": {"gte": date_from.isoformat()}}},
                    {"bool": {"must_not": {"exists": {"field": "date_range_end"}}}},
                ],
                "minimum_should_match": 1,
            }
        })
    if date_to:
        filters.append({
            "bool": {
                "should": [
                    {"range": {"date_range_start": {"lte": date_to.isoformat()}}},
                    {"bool": {"must_not": {"exists": {"field": "date_range_start"}}}},
                ],
                "minimum_should_match": 1,
            }
        })
    return filters


def _domain_filter(allowed_domains: list[str] | None) -> list[dict[str, Any]]:
    if not allowed_domains:
        return []
    return [{"terms": {"domain_tag": allowed_domains}}]


def _hits_to_ranked_ids(payload: dict[str, Any]) -> list[tuple[str, float]]:
    hits = payload.get("hits", {}).get("hits", []) or []
    ranked: list[tuple[str, float]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        file_id = source.get("file_id") or hit.get("_id")
        if file_id:
            ranked.append((str(file_id), float(hit.get("_score") or 0.0)))
    return ranked


async def _lexical_search(
    query: str,
    container_id: str,
    allowed_domains: list[str] | None,
    date_from: date | None,
    date_to: date | None,
    limit: int,
) -> list[tuple[str, float]]:
    body = {
        "size": limit,
        "query": {
            "bool": {
                "filter": _domain_filter(allowed_domains) + _date_filter(date_from, date_to),
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "ai_description^4",
                                "good_for^3",
                                "key_metrics^3",
                                "key_dimensions^2",
                                "column_names^2",
                                "search_text^1",
                            ],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["ai_description", "good_for", "column_names", "search_text"],
                            "fuzziness": "AUTO",
                            "prefix_length": 2,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }
    return _hits_to_ranked_ids(await search_index(container_id, body))


async def _vector_search(
    query: str,
    container_id: str,
    allowed_domains: list[str] | None,
    date_from: date | None,
    date_to: date | None,
    limit: int,
) -> list[tuple[str, float]]:
    vector = await embed_text(query)
    if not vector or all(v == 0.0 for v in vector):
        return []

    # OpenSearch supports filtering inside knn for modern engines. If a cluster
    # does not support filter-with-knn, the caller catches and falls back to PG.
    body = {
        "size": limit,
        "query": {
            "knn": {
                "description_embedding": {
                    "vector": vector,
                    "k": limit,
                    "filter": {
                        "bool": {
                            "filter": _domain_filter(allowed_domains) + _date_filter(date_from, date_to)
                        }
                    },
                }
            }
        },
    }
    return _hits_to_ranked_ids(await search_index(container_id, body))


async def opensearch_retrieve_with_scores(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
    allowed_domains: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[tuple[FileMetadata, float]]:
    """Return ranked FileMetadata rows via OpenSearch, or [] if disabled/error."""
    _ = user_id, is_admin  # Visibility is enforced by container/domain scope here.
    if not opensearch_enabled() or not container_id or not query.strip():
        return []

    try:
        lexical = await _lexical_search(query, container_id, allowed_domains, date_from, date_to, _STAGE_LIMIT)
    except Exception as exc:
        chat_logger.warning("opensearch_lexical_error", error=str(exc)[:300])
        lexical = []

    try:
        vector = await _vector_search(query, container_id, allowed_domains, date_from, date_to, _STAGE_LIMIT)
    except Exception as exc:
        chat_logger.warning("opensearch_vector_error", error=str(exc)[:300])
        vector = []

    if not lexical and not vector:
        return []

    # RRF expects FileMetadata objects, so fuse IDs first with the same formula.
    score_by_id: dict[str, float] = {}
    for ranked in (lexical, vector):
        for rank, (file_id, _score) in enumerate(ranked, start=1):
            score_by_id[file_id] = score_by_id.get(file_id, 0.0) + 1.0 / (60 + rank)

    ordered_ids = [file_id for file_id, _ in sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)[:top_k]]
    if not ordered_ids:
        return []

    rows = (await db.execute(
        select(FileMetadata).where(FileMetadata.file_id.in_(ordered_ids))
    )).scalars().all()
    by_id = {row.file_id: row for row in rows}

    results = [(by_id[file_id], score_by_id[file_id]) for file_id in ordered_ids if file_id in by_id]
    chat_logger.info(
        "opensearch_retrieval_complete",
        query_preview=query[:80],
        lexical=len(lexical),
        vector=len(vector),
        fused=len(results),
        container_id=container_id,
    )
    return results

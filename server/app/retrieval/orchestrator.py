"""
Retrieval orchestrator — the single entry point that wires together all 9
stages of the retrieval pipeline and returns a ranked top-K list of files.

Pipeline stages
---------------
  Stage 1  parse_temporal  — extract date bounds from query text (pure regex, <1ms)
  Stage 2  permission_clause — applied inside every DB-hitting stage via
                               build_base_query(); not an explicit function call here
  Stage 3  date_overlap     — same as above, baked into build_base_query()
  Stage 4  bm25_search      — tsvector keyword search (GIN index)
  Stage 5  fuzzy_search     — pg_trgm trigram similarity (GIN index)
  Stage 6  vector_search    — HNSW cosine similarity (pgvector)
           (stages 4-6 run sequentially on the shared AsyncSession — SQLAlchemy
            async sessions do not support concurrent operations on one connection)
    Stage 7  graph_expand     — one-hop expansion through approved semantic joins
    Stage 8  rrf_fuse         — Reciprocal Rank Fusion across all rank lists
    Stage 9  top-K            — return top_k FileMetadata rows (default 20)

Public API
----------
    async def retrieve(
        query: str,
        user_id: str,
        is_admin: bool,
        db: AsyncSession,
        top_k: int = 20,
    ) -> list[FileMetadata]
        Returns up to top_k FileMetadata rows, ranked by RRF score.
        Never raises — returns [] on any error.

    async def retrieve_with_scores(
        query: str,
        user_id: str,
        is_admin: bool,
        db: AsyncSession,
        top_k: int = 20,
    ) -> list[tuple[FileMetadata, float]]
        Same as retrieve() but also returns the RRF scores for debugging /
        AI Pipeline tab rendering.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.models.file_metadata import FileMetadata
from app.retrieval.bm25 import bm25_search
from app.retrieval.embeddings_search import vector_search
from app.retrieval.fuzzy import fuzzy_search
from app.retrieval.graph_expand import graph_expand
from app.retrieval.opensearch_search import opensearch_retrieve_with_scores
from app.retrieval.rrf import rrf_fuse
from app.retrieval.temporal import parse_temporal

# How many results to pull from each individual stage before fusion.
# More results per stage → better RRF coverage, but more DB work.
_STAGE_LIMIT = 50


async def retrieve_with_scores(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
) -> list[tuple[FileMetadata, float]]:
    """
    Full 9-stage retrieval pipeline.

    Returns list of (FileMetadata, rrf_score) sorted descending.
    Returns [] if no results or on error.

    container_id: when set, retrieval is restricted to files in that
    container. Used by the chat container picker.
    """
    if not query or not query.strip():
        return []

    # ── Load user's domain restrictions (PHASE 15) ───────────────────────────
    # Admin users are unrestricted. Regular users may have allowed_domains set.
    allowed_domains: list[str] | None = None
    if not is_admin and user_id:
        from app.models.user import User as _User
        user_row = await db.get(_User, user_id)
        if user_row and user_row.allowed_domains:
            allowed_domains = list(user_row.allowed_domains)

    # ── Stage 1: temporal parsing ─────────────────────────────────────────────
    date_from: date | None
    date_to: date | None
    date_from, date_to = parse_temporal(query)

    # ── Production path: OpenSearch metadata retrieval ───────────────────────
    # OpenSearch handles BM25 + fuzzy + vector over per-container indices. Use
    # it only when scope is explicit. Plain non-admin ownership filtering still
    # falls back to PostgreSQL because that logic joins files/folders in SQL.
    can_use_opensearch = bool(container_id) and (is_admin or bool(allowed_domains))
    if can_use_opensearch:
        os_results = await opensearch_retrieve_with_scores(
            query=query,
            user_id=user_id,
            is_admin=is_admin,
            db=db,
            top_k=top_k,
            container_id=container_id,
            allowed_domains=allowed_domains,
            date_from=date_from,
            date_to=date_to,
        )
        if os_results:
            try:
                graph_results = await graph_expand(
                    seed_file_ids=[meta.file_id for meta, _ in os_results],
                    user_id=user_id,
                    is_admin=is_admin,
                    db=db,
                    limit=_STAGE_LIMIT,
                    allowed_domains=allowed_domains,
                    container_id=container_id,
                )
            except Exception as exc:
                chat_logger.warning("retrieval_graph_expand_error", error=str(exc)[:200])
                graph_results = []
            return rrf_fuse([os_results, graph_results], top_k=top_k)

    # ── Stages 4-6: sequential retrieval ─────────────────────────────────────
    # SQLAlchemy async sessions share one connection and do not support
    # concurrent operations. Run BM25, fuzzy, vector sequentially.
    # Each stage is wrapped independently — a failing stage returns [] instead
    # of silently killing all retrieval (e.g. pg_trgm not installed on Azure).
    try:
        bm25_results = await bm25_search(
            query, user_id, is_admin, db,
            date_from=date_from, date_to=date_to,
            limit=_STAGE_LIMIT,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
    except Exception as exc:
        chat_logger.warning("retrieval_bm25_error", error=str(exc)[:200])
        bm25_results = []

    try:
        fuzzy_results = await fuzzy_search(
            query, user_id, is_admin, db,
            date_from=date_from, date_to=date_to,
            limit=_STAGE_LIMIT,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
    except Exception as exc:
        chat_logger.warning("retrieval_fuzzy_error", error=str(exc)[:200])
        fuzzy_results = []

    try:
        vector_results = await vector_search(
            query, user_id, is_admin, db,
            date_from=date_from, date_to=date_to,
            limit=_STAGE_LIMIT,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
    except Exception as exc:
        chat_logger.warning("retrieval_vector_error", error=str(exc)[:200])
        vector_results = []

    # ── Stage 7: approved semantic graph expansion ────────────────────────────
    try:
        seed_results = rrf_fuse(
            [bm25_results, fuzzy_results, vector_results],
            top_k=min(_STAGE_LIMIT, max(top_k, 1)),
        )
        seed_file_ids = [meta.file_id for meta, _ in seed_results]
        graph_results = await graph_expand(
            seed_file_ids=seed_file_ids,
            user_id=user_id,
            is_admin=is_admin,
            db=db,
            limit=_STAGE_LIMIT,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
    except Exception as exc:
        chat_logger.warning("retrieval_graph_expand_error", error=str(exc)[:200])
        graph_results = []

    # ── Stage 8: RRF fusion ───────────────────────────────────────────────────
    fused = rrf_fuse(
        [bm25_results, fuzzy_results, vector_results, graph_results],
        top_k=top_k,
    )

    chat_logger.info(
        "retrieval_complete",
        query_preview=query[:80],
        bm25=len(bm25_results),
        fuzzy=len(fuzzy_results),
        vector=len(vector_results),
        graph=len(graph_results),
        fused=len(fused),
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
    )

    return fused


async def retrieve(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
) -> list[FileMetadata]:
    """
    Full 9-stage retrieval pipeline — returns FileMetadata rows only.

    Convenience wrapper over retrieve_with_scores() that strips the scores.
    Use retrieve_with_scores() when you need scores for the AI Pipeline tab.
    """
    results = await retrieve_with_scores(query, user_id, is_admin, db, top_k=top_k, container_id=container_id)
    return [meta for meta, _ in results]

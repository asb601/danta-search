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

import contextvars
from datetime import date

from sqlalchemy import select
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
from app.policies.retrieval_policy import get_retrieval_policy as _get_retrieval_policy
from app.services.trust_propagation import retrieval_trust_weight as _rtw

# ── Retrieval telemetry context vars ────────────────────────────────────────────
# Populated as a side-effect of retrieve_with_scores() for the current async Task.
# Consumers (e.g., graph.py orchestration trace) read these after the await to get
# per-channel membership data without any signature change to the public API.
# Values are Task-local: safe under concurrent async workloads.
retrieval_channel_map: contextvars.ContextVar[dict[str, list[str]]] = (
    contextvars.ContextVar("retrieval_channel_map", default={})
)
retrieval_all_candidate_fids: contextvars.ContextVar[set[str]] = (
    contextvars.ContextVar("retrieval_all_candidate_fids", default=set())
)

# ── Retrieval stage bounds ─────────────────────────────────────────────────────
# All retrieval caps and score floors are governed by RetrievalPolicy.
# See server/app/policies/retrieval_policy.py for rationale on each value.
# Module-level aliases are kept so integration tests can monkeypatch them.
_rp = _get_retrieval_policy()
_STAGE_LIMIT           = _rp.stage_limit
_MAX_BM25_CANDIDATES   = _rp.bm25_candidates
_MAX_VECTOR_CANDIDATES = _rp.vector_candidates
_MAX_FUZZY_CANDIDATES  = _rp.fuzzy_candidates
_RETRIEVAL_MIN_SCORE   = _rp.min_score
# Phase 7: Scale hardening — cap caller-requested top_k and total candidate pool
_MAX_TOP_K             = _rp.max_top_k
_MAX_RRF_CANDIDATES    = _rp.max_rrf_candidates


# ── Phase 6: Post-RRF trust attenuation ──────────────────────────────────
# After RRF fusion, multiply each file's RRF score by its ingestion trust
# weight.  Files in weak-ingestion regions contribute less to orchestration
# without being hard-excluded from the shortlist.
# Non-raising: any error returns the original fused list unchanged.
def _trust_attenuate(
    fused: list[tuple[FileMetadata, float]],
) -> list[tuple[FileMetadata, float]]:
    """Apply ingestion-confidence trust weights to post-RRF scores.

    Multiplies each file's RRF score by retrieval_trust_weight(ingestion_score)
    and re-sorts descending.  Files with no ingestion score (pre-Phase-5) are
    neutral (weight 1.0) and are never penalised.
    """
    try:
        attenuated = [
            (meta, score * _rtw(meta.ingestion_confidence_score))
            for meta, score in fused
        ]
        attenuated.sort(key=lambda x: x[1], reverse=True)
        return attenuated
    except Exception:
        return fused  # never block retrieval


# ── SME quarantine hard-exclusion (flag-gated) ───────────────────────────────
# When SME mode + the quarantine flag are BOTH on, hard-EXCLUDE files whose
# persisted trust_state is QUARANTINED from the retrieval result set. This is
# ADDITIVE to the existing soft post-RRF trust attenuation (_trust_attenuate),
# which stays in place — attenuation down-weights, this removes.
#
# 100% data-driven: the decision reads the per-file `trust_state` column that
# Dev-A derives at ingestion from already-persisted confidence + audit signals
# (derive_trust_state). No literals, name lists, or score thresholds live here.
#
# Demo-safety net: if excluding quarantined files would empty the result set,
# fall back to the pre-filter list — quarantine must never cause zero results.
def _filter_quarantined(
    fused: list[tuple[FileMetadata, float]],
) -> list[tuple[FileMetadata, float]]:
    """Drop QUARANTINED files from a fused result list, flag-gated and safe.

    No-op (returns the input unchanged) when the SME quarantine flag is off,
    when Dev-A's trust_state contract is not yet importable, or on any error —
    so default-off behaviour is byte-identical to today.
    """
    try:
        from app.core.config import get_settings

        settings = get_settings()
        if not (settings.SME_MODE_ENABLED and settings.SME_QUARANTINE_ENABLED):
            return fused

        # Integration dependency: Dev-A owns trust_state.py + the trust_state
        # column. Import lazily so this module still loads if Dev-A has not
        # landed yet (the flag is default-off, so the off-path never reaches here).
        from app.services.trust_state import QUARANTINED

        kept = [
            (meta, score)
            for meta, score in fused
            if getattr(meta, "trust_state", None) != QUARANTINED
        ]
        excluded = len(fused) - len(kept)
        if excluded <= 0:
            return fused  # nothing quarantined — return as-is (no log noise)

        # Demo-safety net: never return zero results because of quarantine.
        if not kept:
            chat_logger.warning(
                "retrieval_quarantine_safety_fallback",
                excluded=excluded,
                kept=0,
                note="all candidates quarantined — falling back to pre-filter set",
            )
            return fused

        chat_logger.info(
            "retrieval_quarantine_excluded",
            excluded=excluded,
            kept=len(kept),
        )
        return kept
    except Exception as exc:
        # Never block retrieval on the trust gate (e.g. trust_state not yet wired).
        chat_logger.warning("retrieval_quarantine_skipped", error=str(exc)[:200])
        return fused


# ── Internal helpers ───────────────────────────────────────────────────────────

def _prune_and_dedup(
    results: list[tuple[FileMetadata, float]],
    min_score: float,
) -> list[tuple[FileMetadata, float]]:
    """Drop low-score entries and remove per-channel file_id duplicates.

    Keeps the first (highest-score) occurrence of each file_id, since results
    are already sorted descending by score from the DB/search layer.
    """
    seen: set[str] = set()
    pruned: list[tuple[FileMetadata, float]] = []
    for meta, score in results:
        if score < min_score:
            break   # sorted descending — no point continuing
        if meta.file_id in seen:
            continue
        seen.add(meta.file_id)
        pruned.append((meta, score))
    return pruned


async def _container_as_of(db: AsyncSession, container_id: str | None) -> date | None:
    """Robust data 'now' for a container — the high-percentile coverage end across
    its files, capped at the wall clock. Reuses ``data_as_of`` (the same anchor
    the prompt uses) so retrieval and the agent agree on what 'this year' means.
    Returns None (→ wall-clock fallback) when the container is unknown or no file
    carries a non-sentinel end-date. Never raises."""
    if not container_id:
        return None
    from app.services.erp.feasibility_gate import data_as_of  # noqa: PLC0415
    try:
        ends = (
            await db.execute(
                select(FileMetadata.date_range_end).where(
                    FileMetadata.container_id == container_id,
                    FileMetadata.date_range_end.isnot(None),
                )
            )
        ).scalars().all()
    except Exception:  # noqa: BLE001 — anchoring is best-effort; degrade to wall clock
        return None
    return data_as_of(list(ends))


async def retrieve_with_scores(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
    anchor_file_ids: list[str] | None = None,
) -> list[tuple[FileMetadata, float]]:
    """
    Full 9-stage retrieval pipeline.

    Returns list of (FileMetadata, rrf_score) sorted descending.
    Returns [] if no results or on error.

    container_id: when set, retrieval is restricted to files in that
    container. Used by the chat container picker.

    anchor_file_ids: resolver-pinned file IDs from EntityResolver. When
    provided, these IDs are merged into the graph_expand seed list so
    their relationship neighbors enter RRF fusion alongside the retrieval
    candidates. This makes relationship neighbors of high-confidence
    entity tables rank-eligible via RRF without bypassing retrieval.
    """
    if not query or not query.strip():
        return []

    # Phase 7: Clamp top_k to policy ceiling to prevent oversized result sets
    # from stressing RRF fusion, graph expansion, and trust propagation.
    top_k = min(top_k, _MAX_TOP_K)
    # Admin users are unrestricted. Regular users may have allowed_domains set.
    allowed_domains: list[str] | None = None
    if not is_admin and user_id:
        from app.models.user import User as _User
        user_row = await db.get(_User, user_id)
        if user_row and user_row.allowed_domains:
            allowed_domains = list(user_row.allowed_domains)

    # ── Stage 1: temporal parsing ─────────────────────────────────────────────
    # Anchor relative-time ("this year", "last month", "YTD") to the data's
    # effective latest coverage, NOT the wall clock. A stale dataset (ending
    # 2025-05 under a 2026 clock) would otherwise resolve "this year" to an empty
    # 2026 window, and the date-overlap filter below would exclude every
    # in-coverage table (the exact bug that surfaced AP/GL tables for a "cash
    # received" question). None → parse_temporal falls back to the wall clock.
    as_of = await _container_as_of(db, container_id)
    date_from: date | None
    date_to: date | None
    date_from, date_to = parse_temporal(query, today=as_of)

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
                _os_seed_ids = [meta.file_id for meta, _ in os_results]
                if anchor_file_ids:
                    # Merge resolver anchors so THEIR neighbors also enter RRF.
                    # dict.fromkeys preserves insertion order and deduplicates.
                    _os_seed_ids = list(dict.fromkeys(_os_seed_ids + anchor_file_ids))
                graph_results = await graph_expand(
                    seed_file_ids=_os_seed_ids,
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
            _os_fused = _trust_attenuate(rrf_fuse([os_results, graph_results], top_k=top_k))
            # ── Telemetry side-effects (OpenSearch path) ─────────────────────
            _os_cm: dict[str, list[str]] = {}
            for _m, _ in os_results:
                _os_cm.setdefault(_m.file_id, []).append("opensearch")
            for _m, _ in graph_results:
                _os_cm.setdefault(_m.file_id, []).append("graph")
            retrieval_channel_map.set(_os_cm)
            retrieval_all_candidate_fids.set(set(_os_cm))
            # SME quarantine hard-exclusion (flag-gated, additive to attenuation).
            return _filter_quarantined(_os_fused)

    # SQLAlchemy async sessions share one connection and do not support
    # concurrent operations. Run BM25, fuzzy, vector sequentially.
    # Each stage is wrapped independently — a failing stage returns [] instead
    # of silently killing all retrieval (e.g. pg_trgm not installed on Azure).
    try:
        bm25_results = await bm25_search(
            query, user_id, is_admin, db,
            date_from=date_from, date_to=date_to,
            limit=_MAX_BM25_CANDIDATES,
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
            limit=_MAX_FUZZY_CANDIDATES,
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
            limit=_MAX_VECTOR_CANDIDATES,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
    except Exception as exc:
        chat_logger.warning("retrieval_vector_error", error=str(exc)[:200])
        vector_results = []

    # ── Score floor + deduplication ───────────────────────────────────────────
    # Drop candidates below the minimum score floor to reduce RRF list noise.
    # Then deduplicate by file_id within each channel so one file can't rank
    # twice in the same channel due to multiple fuzzy/keyword matches.
    bm25_results   = _prune_and_dedup(bm25_results,   _RETRIEVAL_MIN_SCORE)
    fuzzy_results  = _prune_and_dedup(fuzzy_results,  _RETRIEVAL_MIN_SCORE)
    vector_results = _prune_and_dedup(vector_results, _RETRIEVAL_MIN_SCORE)

    # ── Stage 7: approved semantic graph expansion ────────────────────────────
    try:
        seed_results = rrf_fuse(
            [bm25_results, fuzzy_results, vector_results],
            top_k=min(_STAGE_LIMIT, max(top_k, 1)),
        )
        seed_file_ids = [meta.file_id for meta, _ in seed_results]
        if anchor_file_ids:
            # Merge resolver anchors into graph-expand seeds so relationship
            # neighbors of pinned entity tables participate in final RRF.
            seed_file_ids = list(dict.fromkeys(seed_file_ids + anchor_file_ids))
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
    # Phase 7: Cap total candidate pool before fusion to bound RRF complexity.
    # Each rank list is already bounded by stage caps; this trims the combined
    # pool when an unusually large catalog produces many near-tied candidates.
    _all_candidates = [bm25_results, fuzzy_results, vector_results, graph_results]
    _total = sum(len(r) for r in _all_candidates)
    if _total > _MAX_RRF_CANDIDATES:
        # Trim the noisiest/lowest-signal list (fuzzy) first, then vector, then BM25.
        for _trim_idx in (1, 2, 0, 3):  # fuzzy, vector, bm25, graph
            _excess = sum(len(r) for r in _all_candidates) - _MAX_RRF_CANDIDATES
            if _excess <= 0:
                break
            _trim = min(_excess, len(_all_candidates[_trim_idx]))
            _all_candidates[_trim_idx] = _all_candidates[_trim_idx][:-_trim]
    fused = _trust_attenuate(rrf_fuse(_all_candidates, top_k=top_k))
    # SME quarantine hard-exclusion (flag-gated, additive to attenuation above).
    fused = _filter_quarantined(fused)

    # ── Telemetry side-effects (Postgres path) ───────────────────────────────
    # Build channel membership map BEFORE _all_candidates is trimmed above;
    # the original channel list variables still hold pre-trim values here.
    _pg_cm: dict[str, list[str]] = {}
    for _m, _ in bm25_results:
        _pg_cm.setdefault(_m.file_id, []).append("bm25")
    for _m, _ in fuzzy_results:
        _pg_cm.setdefault(_m.file_id, []).append("fuzzy")
    for _m, _ in vector_results:
        _pg_cm.setdefault(_m.file_id, []).append("vector")
    for _m, _ in graph_results:
        _pg_cm.setdefault(_m.file_id, []).append("graph")
    retrieval_channel_map.set(_pg_cm)
    retrieval_all_candidate_fids.set(set(_pg_cm))

    chat_logger.info(
        "retrieval_complete",
        query_preview=query[:80],
        bm25=len(bm25_results),
        fuzzy=len(fuzzy_results),
        vector=len(vector_results),
        graph=len(graph_results),
        fused=len(fused),
        anchor_seeds=len(anchor_file_ids) if anchor_file_ids else 0,
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

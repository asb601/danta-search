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
from typing import Any

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
retrieval_stage_errors: contextvars.ContextVar[list[dict[str, str]]] = (
    contextvars.ContextVar("retrieval_stage_errors", default=[])
)
retrieval_discovery_evidence: contextvars.ContextVar[dict[str, dict[str, Any]]] = (
    contextvars.ContextVar("retrieval_discovery_evidence", default={})
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


def _brain_guidance_rescore(
    fused: list[tuple[FileMetadata, float]],
    brain_context: Any | None,
) -> list[tuple[FileMetadata, float]]:
    """Apply bounded semantic-memory authority as a small ranking modifier.

    This never adds files and never bypasses RBAC. It only reorders files that
    already survived the authorized retrieval pipeline.
    """
    if not fused or not brain_context:
        return fused
    try:
        guidance = getattr(brain_context, "retrieval_guidance", None)
        authority_by_file_id = dict(getattr(guidance, "authority_by_file_id", {}) or {})
        domain_authority_by_file_id = dict(getattr(guidance, "domain_authority_by_file_id", {}) or {})
        anchor_file_ids = set(getattr(guidance, "anchor_file_ids", []) or [])
        domain_anchor_file_ids = set(getattr(guidance, "domain_anchor_file_ids", []) or [])
        if not authority_by_file_id and not domain_authority_by_file_id and not anchor_file_ids and not domain_anchor_file_ids:
            return fused
        rescored: list[tuple[FileMetadata, float]] = []
        for meta, score in fused:
            authority = float(authority_by_file_id.get(meta.file_id, 0.0) or 0.0)
            domain_authority = float(domain_authority_by_file_id.get(meta.file_id, 0.0) or 0.0)
            anchor_bonus = 0.08 if meta.file_id in anchor_file_ids else 0.0
            domain_bonus = 0.1 if meta.file_id in domain_anchor_file_ids else 0.0
            multiplier = 1.0 + min(0.32, authority * 0.14 + domain_authority * 0.18 + anchor_bonus + domain_bonus)
            rescored.append((meta, score * multiplier))
        rescored.sort(key=lambda item: item[1], reverse=True)
        return rescored
    except Exception:
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


def _record_stage_error(stage: str, exc: Exception) -> None:
    """Attach optional retrieval-stage failures to request-local telemetry."""
    entry = {"stage": stage, "error": str(exc)[:200]}
    current = list(retrieval_stage_errors.get() or [])
    retrieval_stage_errors.set((current + [entry])[:12])


async def retrieve_with_scores(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
    anchor_file_ids: list[str] | None = None,
    brain_context: Any | None = None,
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
    retrieval_channel_map.set({})
    retrieval_all_candidate_fids.set(set())
    retrieval_stage_errors.set([])
    retrieval_discovery_evidence.set({})

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
    date_from: date | None
    date_to: date | None
    date_from, date_to = parse_temporal(query)

    # ── Production path: OpenSearch metadata retrieval ───────────────────────
    # OpenSearch handles BM25 + fuzzy + vector over per-container indices. Use
    # it only when scope is explicit. Plain non-admin ownership filtering still
    # falls back to PostgreSQL because that logic joins files/folders in SQL.
    can_use_opensearch = bool(container_id) and (is_admin or bool(allowed_domains))
    if can_use_opensearch:
        try:
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
        except Exception as exc:
            _record_stage_error("opensearch", exc)
            chat_logger.warning("retrieval_opensearch_error", error=str(exc)[:200])
            os_results = []
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
                _record_stage_error("opensearch_graph_expand", exc)
                chat_logger.warning("retrieval_graph_expand_error", error=str(exc)[:200])
                graph_results = []
            _os_fused = _brain_guidance_rescore(
                _trust_attenuate(rrf_fuse([os_results, graph_results], top_k=top_k)),
                brain_context,
            )
            # ── Telemetry side-effects (OpenSearch path) ─────────────────────
            _os_cm: dict[str, list[str]] = {}
            for _m, _ in os_results:
                _os_cm.setdefault(_m.file_id, []).append("opensearch")
            for _m, _ in graph_results:
                _os_cm.setdefault(_m.file_id, []).append("graph")
            retrieval_channel_map.set(_os_cm)
            retrieval_all_candidate_fids.set(set(_os_cm))
            return _os_fused

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
        _record_stage_error("bm25", exc)
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
        _record_stage_error("fuzzy", exc)
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
        _record_stage_error("vector", exc)
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
        _record_stage_error("graph_expand", exc)
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
    fused = _brain_guidance_rescore(
        _trust_attenuate(rrf_fuse(_all_candidates, top_k=top_k)),
        brain_context,
    )

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


def _work_order_query_variants(query: str, work_order: Any | None) -> list[str]:
    seen: set[str] = set()
    variants: list[str] = []
    for item in [query] + list(getattr(work_order, "candidate_search_queries", []) or []):
        value = str(item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        variants.append(value)
        if len(variants) >= 8:
            break
    return variants


async def retrieve_discovery_candidates(
    work_order: Any,
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    top_k: int = 20,
    container_id: str | None = None,
    anchor_file_ids: list[str] | None = None,
    brain_context: Any | None = None,
) -> list[tuple[FileMetadata, float]]:
    """Retrieve a broad, authorized discovery set from work-order variants.

    This keeps SQL execution unchanged. It only widens the retrieval evidence
    phase by trying the original query plus bounded source/output/filter search
    variants produced by QueryWorkOrder.
    """
    variants = _work_order_query_variants(query, work_order)
    if len(variants) <= 1:
        results = await retrieve_with_scores(
            query=query,
            user_id=user_id,
            is_admin=is_admin,
            db=db,
            top_k=top_k,
            container_id=container_id,
            anchor_file_ids=anchor_file_ids,
            brain_context=brain_context,
        )
        retrieval_discovery_evidence.set({
            getattr(meta, "file_id", ""): {
                "matched_queries": [query],
                "channels": retrieval_channel_map.get().get(getattr(meta, "file_id", ""), []),
                "variant_count": 1,
            }
            for meta, _ in results
            if getattr(meta, "file_id", None)
        })
        return results

    aggregated: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    stage_errors: list[dict[str, str]] = []
    channel_map: dict[str, list[str]] = {}
    max_variant_top_k = min(_MAX_TOP_K, max(top_k, min(top_k * 2, _MAX_TOP_K)))

    for index, variant in enumerate(variants):
        try:
            variant_results = await retrieve_with_scores(
                query=variant,
                user_id=user_id,
                is_admin=is_admin,
                db=db,
                top_k=max_variant_top_k,
                container_id=container_id,
                anchor_file_ids=anchor_file_ids,
                brain_context=brain_context,
            )
        except Exception as exc:
            _record_stage_error("discovery_variant", exc)
            variant_results = []

        current_channels = dict(retrieval_channel_map.get() or {})
        current_candidates = set(retrieval_all_candidate_fids.get() or set())
        current_errors = list(retrieval_stage_errors.get() or [])
        candidate_ids.update(current_candidates)
        stage_errors.extend(current_errors)

        variant_weight = 1.0 / (1.0 + index * 0.15)
        for meta, score in variant_results:
            file_id = meta.file_id
            item = aggregated.setdefault(file_id, {
                "meta": meta,
                "score": 0.0,
                "matched_queries": [],
                "channels": [],
            })
            item["score"] = float(item["score"]) + float(score) * variant_weight
            item["matched_queries"].append(variant)
            item["channels"].extend(current_channels.get(file_id, []))
            channel_map.setdefault(file_id, []).extend(current_channels.get(file_id, []))

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (float(item["score"]), len(set(item["matched_queries"]))),
        reverse=True,
    )[:top_k]
    final_results = [(item["meta"], float(item["score"])) for item in ranked]
    final_channel_map = {
        file_id: list(dict.fromkeys(channels))
        for file_id, channels in channel_map.items()
    }
    evidence = {
        file_id: {
            "matched_queries": list(dict.fromkeys(item["matched_queries"]))[:6],
            "channels": list(dict.fromkeys(item["channels"]))[:8],
            "variant_count": len(set(item["matched_queries"])),
        }
        for file_id, item in aggregated.items()
    }
    retrieval_channel_map.set(final_channel_map)
    retrieval_all_candidate_fids.set(candidate_ids | set(aggregated.keys()))
    retrieval_stage_errors.set(stage_errors[:12])
    retrieval_discovery_evidence.set(evidence)

    chat_logger.info(
        "retrieval_discovery_complete",
        variants=len(variants),
        fused=len(final_results),
        candidate_count=len(candidate_ids | set(aggregated.keys())),
        anchor_seeds=len(anchor_file_ids) if anchor_file_ids else 0,
    )
    return final_results


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

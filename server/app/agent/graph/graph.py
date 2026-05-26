"""
LangGraph agent — public entry points (sync + streaming).

This module orchestrates the pipeline:
  1. Load catalog (catalog_cache)
  2. Build tools
  3. Build system prompt (prompt_builder)
  4. Construct LangGraph (graph_builder)
  5. Run and extract results (response_helpers)
"""
from __future__ import annotations

import asyncio
import math
import re
import threading
import time
import uuid
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.catalog_cache import invalidate_catalog_cache, load_catalog  # re-export
from app.agent.catalog_hydration import hydrate_files, merge_hydrated
from app.agent.graph.graph_builder import build_graph
from app.agent.llm import get_llm_mini
from app.services.business_intent_planner import BusinessIntentPlan, build_business_intent_plan
from app.services.entity_resolver import EntityCandidate, resolve_entities
from app.services.sql_context_builder import build_sql_context
from app.services.execution_strategy import plan_execution_strategy
from app.agent.prompts.prompt_builder import build_system_prompt
from app.agent.search_normalization import tokenize_search_query
from app.retrieval.orchestrator import (
    retrieve_with_scores,
    retrieval_channel_map as _retrieval_channel_map,
    retrieval_all_candidate_fids as _retrieval_all_candidate_fids,
)
from app.agent.response_helpers import (
    extract_answer,
    extract_blob_paths,
    fallback_answer,
    fallback_answer_from_outputs,
    infer_chart,
)
from app.agent.state import AgentState
from app.agent.tools.catalog import build_catalog_tools
from app.agent.tools.column import build_column_tool
from app.agent.tools.definition_lookup import build_definition_lookup_tool, load_schema_registry
from app.agent.tools.sample import build_sample_tool
from app.agent.tools.relations import build_relations_tool
from app.agent.tools.sql import build_sql_tools, _execute as _sql_execute
from app.agent.tools.stats import build_stats_tool
from app.core.logger import chat_logger, pipeline_logger
from app.core import metrics
from app.core.orchestration_trace import OrchestrationTrace
import structlog as _structlog
import traceback as _traceback
from app.retrieval.embeddings import build_search_text
from app.services.semantic_policy import get_semantic_policy
from app.services.graph_health import score_graph_health
from app.services.trust_propagation import avg_ingestion_confidence as _avg_ing_conf
from app.services.query_confidence import compute_confidence

# Lazy import — planner is only loaded when first query arrives (avoids circular
# import at module level; semantic_planner imports FileMetadata / FileRelationship).
_semantic_plan = None

def _get_planner():
    global _semantic_plan
    if _semantic_plan is None:
        from app.services.semantic_planner import plan as _p
        _semantic_plan = _p
    return _semantic_plan

# Per-request mutable stores (keyed by request_id)
_request_stores: dict[str, dict] = {}
_stores_lock = threading.Lock()

_NO_FILES_MSG = "No files have been ingested yet. Please upload and ingest some files first."

# ── Explicit file-name extractor ─────────────────────────────────────────────
# When the user writes "on file_a.csv" or "use file_b.csv",
# we detect that mention and pin the matching catalog entry at the TOP of the
# shortlist so the retrieval ranking can never override the user's choice.
_HASH_PREFIX_RE = re.compile(r'^[0-9a-f]{8}_', re.IGNORECASE)


def _extract_mentioned_files(query: str, full_catalog: list[dict]) -> list[dict]:
    """Return catalog entries whose filename appears verbatim in the user query.

    Matching is case-insensitive. The 8-hex-char upload hash prefix
    (e.g. 'dba1285e_') is stripped before comparison so the user never has
    to know about internal storage names.
    """
    if not query:
        return []
    query_lower = query.lower()
    mentioned: list[dict] = []
    seen: set[str] = set()
    for entry in full_catalog:
        blob = entry.get("blob_path", "")
        filename = blob.rsplit("/", 1)[-1]                          # drop any folder prefix
        clean = _HASH_PREFIX_RE.sub("", filename).lower()           # drop hash prefix
        stem = clean.rsplit(".", 1)[0] if "." in clean else clean   # drop extension
        fid = entry.get("file_id", "")
        # Match full filename (with ext) OR stem — both case-insensitive.
        # Require stem length ≥ 4 to avoid false positives on short tokens.
        if fid not in seen and len(stem) >= 4 and (clean in query_lower or stem in query_lower):
            mentioned.append(entry)
            seen.add(fid)
    return mentioned


async def _get_approved_neighbor_ids(
    seed_file_ids: list[str],
    db: AsyncSession,
    container_id: str | None = None,
    cap: int = 50,
) -> set[str]:
    """Return file_ids of approved graph neighbors of the given seed files.

    Used to build domain-bounded candidate sets for:
      1. Lookup-slot injection (retrieval success path)
      2. Fallback candidate pool (when retrieve_with_scores returns 0)

    One DB query bounded by cap. Never raises — returns empty set on any error.
    """
    if not seed_file_ids:
        return set()
    try:
        from sqlalchemy import or_, select as _select  # noqa: PLC0415
        from app.models.semantic_layer import SemanticRelationship  # noqa: PLC0415

        seed_ids = seed_file_ids[:cap]
        q = (
            _select(
                SemanticRelationship.file_a_id,
                SemanticRelationship.file_b_id,
            )
            .where(
                or_(
                    SemanticRelationship.file_a_id.in_(seed_ids),
                    SemanticRelationship.file_b_id.in_(seed_ids),
                ),
                SemanticRelationship.status == "approved",
                SemanticRelationship.approval_status == "approved",
            )
            .limit(cap)
        )
        if container_id:
            q = q.where(SemanticRelationship.container_id == container_id)
        rows = (await db.execute(q)).all()
        seed_set = set(seed_ids)
        return {
            (fb if fa in seed_set else fa)
            for fa, fb in rows
        } - seed_set
    except Exception:
        return set()


async def _polish_answer(raw: str) -> str:
    """Polish pass DISABLED — it added a full LLM round-trip (~1500 tokens)
    for cosmetic rewriting with marginal value. Returns raw unchanged.
    Kept as a no-op so callers don't break."""
    return raw


# ── Semantic Planner fast path ─────────────────────────────────────────────────

async def _try_planner(
    query: str,
    ctx: dict,
    db,
    pipeline_start: float,
) -> dict | None:
    """Attempt the semantic planner path. Returns result dict on success, None on fallback.

    On success: SQL confidence passes the semantic policy threshold, executes via DataFusion,
    and synthesized via one LLM call (mini). Skips the full LangGraph agent.
    On fallback: returns None — caller continues with existing LangGraph path.

    The fallback guarantee: this function NEVER raises. Any exception returns None.
    """
    try:
        catalog = ctx["initial_state"]["catalog"]
        connection_string = ctx["initial_state"]["connection_string"]
        container_name = ctx["initial_state"]["container_name"]

        planner = _get_planner()
        ep = await planner(query, catalog, db, timeout_seconds=2.5)

        # Fallback conditions
        if ep.fallback_reason or not ep.sql or ep.confidence < get_semantic_policy().planner_fast_path_confidence:
            chat_logger.info(
                "planner_fallback_to_agent",
                reason=ep.fallback_reason or "low_confidence",
                confidence=round(ep.confidence, 2),
                planning_ms=ep.planning_ms,
            )
            return None

        # ── Execute the planner SQL ───────────────────────────────────────────
        import asyncio as _asyncio
        chat_logger.info(
            "planner_executing",
            sql_preview=ep.sql[:300],
            confidence=round(ep.confidence, 2),
            files=len(ep.files),
            joins=len(ep.joins),
        )
        loop = _asyncio.get_event_loop()
        rows, total = await loop.run_in_executor(
            None, lambda: _sql_execute(ep.sql, connection_string, container_name, max_rows=100)
        )

        if not rows and total == 0:
            # Planner generated valid SQL but data returned 0 rows — let agent try
            chat_logger.info("planner_zero_rows_fallback", sql_preview=ep.sql[:200])
            return None

        # ── Synthesize response with one mini LLM call ───────────────────────
        import json as _json
        data_preview = _json.dumps(rows[:25], default=str)
        synth_prompt = (
            f"The user asked: {query}\n\n"
            f"Query returned {total} row(s) (showing first {min(len(rows), 25)}):\n"
            f"{data_preview}\n\n"
            "Write a concise, precise analytical response. Include key totals, "
            "top values, and observations. Use numbers. Be direct."
        )
        llm_resp = await get_llm_mini().ainvoke([
            SystemMessage(content=(
                "You are an enterprise ERP data analyst. Answer the user's question "
                "based ONLY on the data provided. Be specific with numbers."
            )),
            HumanMessage(content=synth_prompt),
        ])
        answer = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)

        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        chat_logger.info(
            "planner_path_complete",
            total_ms=total_ms,
            rows=len(rows),
            planning_ms=ep.planning_ms,
            route="planner",
        )
        pipeline_logger.info(
            "final_answer",
            query=query,
            answer=answer,
            row_count=len(rows),
            tool_calls=0,
            route="planner",
            total_duration_ms=total_ms,
        )

        chart = infer_chart(answer, rows)
        return {
            "answer": answer,
            "data": rows,
            "chart": chart,
            "route": "planner",
            "row_count": total,
            "files_used": [f.blob_path for f in ep.files if f.blob_path],
            "tool_calls": 0,
        }

    except Exception as exc:
        chat_logger.warning(
            "planner_path_error",
            error=str(exc)[:300],
            fallback="agent",
        )
        return None

# ── Shortlist sizing (governed by RetrievalPolicy) ────────────────────────────────────────────
# See server/app/policies/retrieval_policy.py for rationale on each value.
from app.policies.retrieval_policy import get_retrieval_policy as _get_retrieval_policy  # noqa: E402
from app.policies.confidence_policy import get_confidence_policy as _get_confidence_policy  # noqa: E402
_rp = _get_retrieval_policy()
_SHORTLIST_TOP_K       = _rp.shortlist_top_k
_LOOKUP_RESERVED_SLOTS = _rp.lookup_reserved_slots

# Lookup-file detection lives in search_normalization so the search_catalog
# tool can apply the same heuristic.  Re-exported here under the old private
# name to keep call sites unchanged.
from app.agent.search_normalization import is_lookup_file as _is_lookup_file  # noqa: E402


# ── Shared context builder ────────────────────────────────────────────────────

async def _build_agent_context(
    query: str,
    db: AsyncSession,
    conversation_context: str = "",
    user_id: str = "",
    is_admin: bool = True,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
    prior_files: list[str] | None = None,
    request_trace_id: str | None = None,
) -> dict | None:
    """
    Shared setup for both streaming and non-streaming entry points.
    Returns None if no catalog data exists.
    """
    # ── STEP 1: USER QUERY RECEIVED ──────────────────────────────────────────
    pipeline_logger.info(
        "query_received",
        query=query,
        has_conversation_context=bool(conversation_context),
        conversation_context_preview=(conversation_context[:300] if conversation_context else ""),
    )

    cached = await load_catalog(db, allowed_domains=None if is_admin else allowed_domains, container_id=container_id)
    if not cached:
        metrics.inc("catalog_miss_count")
        pipeline_logger.warning("catalog_empty", query=query, reason="no files ingested yet")
        return None

    # ── STEP 2: CATALOG LOADED ───────────────────────────────────────────────
    pipeline_logger.info(
        "catalog_loaded",
        query=query,
        container=cached["container_name"],
        file_count=len(cached["catalog"]),
        parquet_count=len(cached["parquet_paths_all"]),
        files=[f.get("blob_path", "") for f in cached["catalog"]],
    )

    full_catalog = cached["catalog"]
    connection_string = cached["connection_string"]
    container_name = cached["container_name"]
    parquet_blob_path = cached["parquet_blob_path"]
    all_parquet_paths = cached["parquet_paths_all"]

    # ── STEP 2b: SCHEMA REGISTRY — load field definitions from any uploaded
    # data-dictionary files for this container.  Returns {} if none registered.
    # This is pre-loaded once here so all tool builders can share the same dict
    # without repeating SQL queries at each tool call.
    resolved_container_id = container_id or (
        full_catalog[0].get("container_id") if full_catalog else None
    )

    # ── STEP 2b + 2.4 (PARALLEL): Schema registry + Business intent plan ─────
    # These two are independent: one is an async DB query, the other is an LLM
    # call. Running in parallel saves the latency of whichever completes first.
    field_definitions, intent_plan = await asyncio.gather(
        load_schema_registry(db, resolved_container_id, connection_string, container_name),
        build_business_intent_plan(query),
    )
    intent_plan: BusinessIntentPlan  # type narrowing hint

    # ── Create per-request orchestration trace ────────────────────────────────
    # Trace accumulates decision telemetry at each stage and is emitted once
    # at pipeline completion. Created here so it can be passed through ctx.
    # Use the caller-supplied trace_id when available so all events share one id.
    req_id_for_trace = request_trace_id or uuid.uuid4().hex
    trace = OrchestrationTrace(request_id=req_id_for_trace)
    trace.set_planner(intent_plan)

    # ── STEP 2.45: ENTITY RESOLUTION ─────────────────────────────────────────
    # Deterministic, metadata-driven. Answers WHERE each planner entity lives.
    # One batch DB query (column_semantic_roles). No LLM. No schema-wide scans.
    # Phase 2 (future): entity_resolution will constrain retrieve_with_scores()
    # scoring so files matching resolved entities rank higher in the shortlist.
    entity_resolution: dict[str, list[EntityCandidate]] = await resolve_entities(
        intent_plan.entities, full_catalog, db
    )
    trace.set_entity_resolver(entity_resolution)

    # ── Resolver pins: files that MUST survive retrieval pruning ─────────────
    # Threshold governed by ConfidencePolicy.resolver_pin_threshold (0.85).
    # Only "semantic_role_match" and strong dimension matches qualify.
    # Below this, results are probabilistic overlaps that shouldn't override
    # retrieval ranking (avoids false positives from partial matches).
    _RESOLVER_PIN_THRESHOLD = _get_confidence_policy().resolver_pin_threshold
    resolver_pinned_blobs: set[str] = {
        c.table
        for candidates in entity_resolution.values()
        for c in candidates
        if c.confidence >= _RESOLVER_PIN_THRESHOLD
    }
    # file_ids for the same anchors — passed to retrieve_with_scores so
    # graph_expand uses them as extra seeds and their relationship neighbors
    # enter RRF fusion (depth-1 expansion via SemanticRelationship).
    resolver_pinned_file_ids: list[str] = [
        e["file_id"]
        for e in full_catalog
        if e.get("blob_path") in resolver_pinned_blobs and e.get("file_id")
    ] if resolver_pinned_blobs else []

    # ── STEP 2.5: RETRIEVAL — filter catalog to top-K relevant files ─────────
    # Run the 9-stage retrieval pipeline (temporal → BM25 → fuzzy → vector →
    # graph_expand → RRF). Only the relevant files go into the system prompt.
    # The full catalog is still passed to build_catalog_tools so search_catalog
    # can still scan all files if needed.
    retrieved_with_scores = []
    retrieval_error: str | None = None
    if user_id:
        try:
            retrieved_with_scores = await retrieve_with_scores(
                query, user_id, is_admin, db, top_k=_SHORTLIST_TOP_K,
                container_id=container_id,
                anchor_file_ids=resolver_pinned_file_ids or None,
            )
        except Exception as exc:
            retrieval_error = str(exc)[:200]
            chat_logger.warning("retrieval_error_fallback", error=retrieval_error)

    # ── In-memory keyword scorer (used for fallback AND for lookup-slot fill) ─
    q_words = tokenize_search_query(query)

    # IDF weights — rare tokens (those that appear in few catalog files)
    # are far more discriminative than common ones. Without this, a unique
    # filename token and a common descriptive token can receive the same weight.
    # Computed once per request: O(N_files × N_tokens), trivial.
    _N = max(1, len(full_catalog))
    _doc_freq: dict[str, int] = {}
    _file_blobs: list[str] = []
    _file_search_text: list[str] = []
    _file_col_text: list[str] = []
    for _e in full_catalog:
        st = build_search_text(_e).lower()
        bp = (_e.get("blob_path") or "").lower()
        cnames: list[str] = []
        for c in (_e.get("columns_info") or []):
            if isinstance(c, dict) and c.get("name"):
                cnames.append(c["name"])
        if not cnames:
            cnames = [c for c in (_e.get("column_names") or []) if isinstance(c, str)]
        ct = " ".join(cnames).lower()
        _file_search_text.append(st)
        _file_blobs.append(bp)
        _file_col_text.append(ct)
        # Only count each token once per file (document frequency).
        seen_in_file: set[str] = set()
        for w in q_words:
            if w in seen_in_file:
                continue
            if w in st or w in bp or w in ct:
                _doc_freq[w] = _doc_freq.get(w, 0) + 1
                seen_in_file.add(w)
    # Smoothed IDF: log((N + 1) / (df + 1)) + 1 — always positive, dampens
    # very common tokens, gives unique tokens (df=1) the largest weight.
    _idf = {w: math.log((_N + 1) / (_doc_freq.get(w, 0) + 1)) + 1.0 for w in q_words}

    # O(1) reverse index: blob_path → array position. Eliminates the O(N)
    # list.index() scan that _kw_score previously performed on every file lookup.
    _blob_to_idx: dict[str, int] = {bp: i for i, bp in enumerate(_file_blobs)}

    def _kw_score(e: dict) -> float:
        bp = (e.get("blob_path") or "").lower()
        idx = _blob_to_idx.get(bp)
        if idx is not None:
            search_text = _file_search_text[idx]
            column_text = _file_col_text[idx]
        else:
            search_text = build_search_text(e).lower()
            cnames: list[str] = []
            for c in (e.get("columns_info") or []):
                if isinstance(c, dict) and c.get("name"):
                    cnames.append(c["name"])
            if not cnames:
                cnames = [c for c in (e.get("column_names") or []) if isinstance(c, str)]
            column_text = " ".join(cnames).lower()

        score = 0.0
        for w in q_words:
            weight = _idf.get(w, 1.0)
            # Filename / blob_path hit is the strongest signal; a unique
            # filename token should dominate the ranking.
            if w in bp:
                score += 3.0 * weight
            if w in column_text:
                score += 2.0 * weight
            if w in search_text:
                score += 1.0 * weight
        return score

    if retrieved_with_scores:
        retrieved_ids = {meta.file_id for meta, _ in retrieved_with_scores}
        catalog = [e for e in full_catalog if e.get("file_id") in retrieved_ids]

        # ── Inject resolver-pinned files ──────────────────────────────────────
        # Entity tables with high resolver confidence (≥0.85) are prepended so
        # embeddings ranking cannot accidentally prune the authoritative source.
        # Deduplication: skip any file already present in the retrieval result.
        if resolver_pinned_blobs:
            _retrieved_file_ids = {e.get("file_id") for e in catalog}
            _resolver_injected = [
                e for e in full_catalog
                if e.get("blob_path") in resolver_pinned_blobs
                and e.get("file_id") not in _retrieved_file_ids
            ]
            if _resolver_injected:
                catalog = _resolver_injected + catalog
                pipeline_logger.info(
                    "resolver_pins_injected",
                    pinned=[e.get("blob_path") for e in _resolver_injected],
                    entities=list(entity_resolution.keys()),
                    path="retrieval",
                )

        # ── Pin files from the previous turn ─────────────────────────────────
        # If the user is asking a follow-up ("give me 20 rows", "filter by X"),
        # retrieval may rank unrelated files higher. Force any blob_path that
        # was actually queried in the last 3 assistant turns to stay in the
        # shortlist, avoiding context drift between turns.
        if prior_files:
            already_in_ids = {e.get("file_id") for e in catalog}
            for blob in prior_files:
                pinned = next((e for e in full_catalog if e.get("blob_path") == blob and e.get("file_id") not in already_in_ids), None)
                if pinned:
                    catalog.append(pinned)
                    already_in_ids.add(pinned.get("file_id"))
            pipeline_logger.info("prior_files_pinned", pinned=prior_files)

        # ── Reserve slots for master / lookup files ───────────────────────────
        # Lookup injection is scoped to the approved semantic neighborhood of
        # the retrieved files only. Prevents globally-visible lookup tables
        # (e.g. unrelated domain tables) from being injected into queries that
        # retrieved a domain-bounded result set via BM25/vector/graph.
        already_in = {e.get("blob_path") for e in catalog}
        _retrieved_fids = {meta.file_id for meta, _ in retrieved_with_scores}
        try:
            _lookup_neighbor_ids = await _get_approved_neighbor_ids(
                list(_retrieved_fids), db, container_id=container_id
            )
        except Exception:
            _lookup_neighbor_ids = set()
        _lookup_eligible_ids = _retrieved_fids | _lookup_neighbor_ids
        lookup_pool = [
            e for e in full_catalog
            if _is_lookup_file(e)
            and e.get("blob_path") not in already_in
            and e.get("file_id") in _lookup_eligible_ids
        ]
        # Rank lookup pool by keyword score (still query-aware: a "supplier
        # master" outranks "calendar lookup" when the query is about suppliers).
        lookup_pool.sort(key=_kw_score, reverse=True)
        injected_lookups = lookup_pool[:_LOOKUP_RESERVED_SLOTS]
        catalog = catalog + injected_lookups

        parquet_paths_all = {
            k: v for k, v in all_parquet_paths.items()
            if k in {e.get("blob_path") for e in catalog}
        }
        pipeline_logger.info(
            "retrieval_filtered",
            query=query,
            total_files=len(full_catalog),
            retrieved_files=len(catalog),
            top_scores=[(meta.file_id, round(s, 4)) for meta, s in retrieved_with_scores[:5]],
            lookup_eligible_neighbor_count=len(_lookup_neighbor_ids),
            lookup_slots_added=[e.get("blob_path") for e in injected_lookups],
        )
        trace.set_retrieval_fusion(
            retrieved_with_scores=retrieved_with_scores,
            shortlist=catalog,
            resolver_pins=list(resolver_pinned_blobs),
            fallback=False,
        )
        # ── Per-file retrieval decision telemetry ───────────────────────────────────
        # Read per-channel membership from the context vars set by retrieve_with_scores().
        # These are Task-local and safe under concurrent async workloads.
        _ch_map     = _retrieval_channel_map.get()
        _cand_fids  = _retrieval_all_candidate_fids.get()
        _short_fids = {meta.file_id for meta, _ in retrieved_with_scores}
        _inj_fids   = {e.get("file_id") for e in injected_lookups if e.get("file_id")}
        _prior_blobs = set(prior_files) if prior_files else set()
        _td_shortlisted = [
            {
                "file":            (getattr(meta, "blob_path", "") or meta.file_id)[:80],
                "channels":        _ch_map.get(meta.file_id, []),
                "rrf_score":       round(float(score), 5),
                "resolver_pin":    getattr(meta, "blob_path", "") in resolver_pinned_blobs,
                "lookup_injected": meta.file_id in _inj_fids,
                "prior_pin":       getattr(meta, "blob_path", "") in _prior_blobs,
            }
            for meta, score in retrieved_with_scores[:20]
        ]
        _low_rrf_rejected = [
            {
                "file_id":         fid[:8],
                "rejected_reason": "low_rrf_rank",
                "channels":        _ch_map.get(fid, []),
            }
            for fid in list(_cand_fids - _short_fids)[:10]
        ]
        _outside_boundary_rejected = [
            {
                "file":            (e.get("blob_path", "") or (e.get("file_id") or ""))[:80],
                "rejected_reason": "outside_graph_boundary",
            }
            for e in full_catalog
            if _is_lookup_file(e) and e.get("file_id") not in _lookup_eligible_ids
        ][:10]
        trace.set_retrieval_decision(
            shortlisted=_td_shortlisted,
            rejected=(_low_rrf_rejected + _outside_boundary_rejected)[:20],
        )
        _grounding_quality = "retrieved"  # RRF-based retrieval succeeded normally
    else:
        # ── Domain-bounded fallback ───────────────────────────────────────────
        # Retrieval returned 0 (or errored). Instead of starting from
        # full_catalog, use resolver-pinned file_ids as structural domain
        # anchors and expand to their approved graph neighbors to produce a
        # relationship-scoped candidate pool.
        # Only if no structural anchors exist does the pool degrade to the
        # full catalog (logged explicitly as full_catalog_degraded).
        _fb_anchor_ids: list[str] = resolver_pinned_file_ids or []
        _fb_neighbor_ids: set[str] = set()
        if _fb_anchor_ids:
            try:
                _fb_neighbor_ids = await _get_approved_neighbor_ids(
                    _fb_anchor_ids, db, container_id=container_id
                )
            except Exception:
                _fb_neighbor_ids = set()
        _fb_bounded_ids = set(_fb_anchor_ids) | _fb_neighbor_ids
        _fb_candidates = (
            [e for e in full_catalog if e.get("file_id") in _fb_bounded_ids]
            if _fb_bounded_ids
            else []
        )
        if _fb_candidates:
            _grounding_quality = "graph_bounded"
        else:
            # No structural anchors — degrade to full catalog with explicit signal.
            _fb_candidates = full_catalog
            _grounding_quality = "full_catalog_degraded"

        scored = sorted(_fb_candidates, key=_kw_score, reverse=True)
        # Take the top metric/transactional matches by keyword, then enrich
        # with lookup files from the same bounded candidate set.
        primary = scored[: _SHORTLIST_TOP_K - _LOOKUP_RESERVED_SLOTS]
        primary_blobs = {e.get("blob_path") for e in primary}
        # Lookup injection stays within the bounded candidate pool — never
        # expanded back to full_catalog regardless of grounding_quality.
        lookup_pool = [
            e for e in scored
            if _is_lookup_file(e) and e.get("blob_path") not in primary_blobs
        ]
        catalog = primary + lookup_pool[:_LOOKUP_RESERVED_SLOTS]

        # ── Inject resolver-pinned files (fallback path) ──────────────────────
        if resolver_pinned_blobs:
            _fallback_file_ids = {e.get("file_id") for e in catalog}
            _resolver_injected = [
                e for e in full_catalog
                if e.get("blob_path") in resolver_pinned_blobs
                and e.get("file_id") not in _fallback_file_ids
            ]
            if _resolver_injected:
                catalog = _resolver_injected + catalog
                pipeline_logger.info(
                    "resolver_pins_injected",
                    pinned=[e.get("blob_path") for e in _resolver_injected],
                    entities=list(entity_resolution.keys()),
                    path="fallback",
                )

        # Pin prior files in the fallback path too
        if prior_files:
            already_blobs = {e.get("blob_path") for e in catalog}
            for blob in prior_files:
                pinned = next((e for e in full_catalog if e.get("blob_path") == blob and blob not in already_blobs), None)
                if pinned:
                    catalog.append(pinned)
                    already_blobs.add(blob)

        parquet_paths_all = {
            k: v for k, v in all_parquet_paths.items()
            if k in {e.get("blob_path") for e in catalog}
        }
        if retrieval_error:
            reason = f"retrieval_error: {retrieval_error}"
        elif user_id:
            reason = "no retrieval results"
        else:
            reason = "no user_id"
        pipeline_logger.info(
            "retrieval_fallback",
            query=query,
            reason=reason,
            grounding_quality=_grounding_quality,
            anchor_count=len(_fb_anchor_ids),
            neighbor_count=len(_fb_neighbor_ids),
            candidate_pool=len(_fb_candidates),
            total_files=len(full_catalog),
            fallback_files=[e.get("blob_path") for e in catalog],
        )
        trace.set_retrieval_fusion(
            retrieved_with_scores=[],
            shortlist=catalog,
            resolver_pins=list(resolver_pinned_blobs),
            fallback=True,
        )
        metrics.inc("catalog_fallback_count")

    # ── STEP 2.55: PIN EXPLICITLY-MENTIONED FILES ────────────────────────────
    # If the user named a specific file,
    # force that file to the front of the shortlist regardless of retrieval rank.
    # This is done AFTER all retrieval/lookup logic so nothing can push it out.
    mentioned_entries = _extract_mentioned_files(query, full_catalog)
    mentioned_file_names: list[str] = []
    if mentioned_entries:
        mentioned_ids = {e.get("file_id") for e in mentioned_entries}
        catalog = mentioned_entries + [e for e in catalog if e.get("file_id") not in mentioned_ids]
        mentioned_file_names = [
            _HASH_PREFIX_RE.sub("", e.get("blob_path", "").rsplit("/", 1)[-1])
            for e in mentioned_entries
        ]
        pipeline_logger.info(
            "explicit_file_pinned",
            query=query[:200],
            pinned=mentioned_file_names,
        )

    # ── Top-ranked blobs for focused system-prompt context ───────────────────
    # Top-3 files by RRF retrieval score receive full column_stats context in
    # the prompt. All other shortlisted files get a compact summary to keep
    # total token load bounded without dropping any file from the shortlist.
    if retrieved_with_scores:
        _top_rrf_ids = {meta.file_id for meta, _ in retrieved_with_scores[:3]}
        top_blob_paths: set[str] = {
            e.get("blob_path") for e in catalog
            if e.get("file_id") in _top_rrf_ids and e.get("blob_path")
        }
    else:
        top_blob_paths = {e.get("blob_path") for e in catalog[:3] if e.get("blob_path")}

    # ── STEP 2.6: HYDRATE HEAVY FIELDS for the shortlist only ───────────────
    # The cached catalog is intentionally lean (no columns_info samples,
    # sample_rows, or column_stats). We now load those heavy fields ONLY for
    # the K shortlisted files. At ~10 KB per record this stays bounded
    # (≤300 KB per request) regardless of total catalog size.
    shortlist_ids = [e["file_id"] for e in catalog if e.get("file_id")]
    heavy_by_file = await hydrate_files(db, shortlist_ids)
    catalog = [merge_hydrated(e, heavy_by_file.get(e.get("file_id"))) for e in catalog]
    # The shortlist is what the system prompt shows the LLM. The FULL catalog
    # (with hydrated heavy fields where available) is what discovery tools
    # bind to, so the LLM can inspect any file it surfaces via search_catalog
    # without "File not found in catalog" failures.
    full_catalog = [
        merge_hydrated(e, heavy_by_file.get(e.get("file_id"))) for e in full_catalog
    ]
    sample_rows_by_blob = {
        e["blob_path"]: e.get("sample_rows") or []
        for e in catalog
        if e.get("blob_path") and e.get("sample_rows")
    }
    pipeline_logger.info(
        "catalog_hydrated",
        shortlist_size=len(catalog),
        hydrated_files=len(heavy_by_file),
        sample_rows_files=len(sample_rows_by_blob),
    )

    # ── STEP 2.7: VALIDATED SQL CONTEXT ──────────────────────────────────────
    # Two batch queries (approved joins + column_semantic_roles) scoped to the
    # final shortlisted files. Result is injected into the system prompt as a
    # read-only constraint block so the LLM uses validated join paths and
    # column bindings instead of free-form semantic guessing.
    # Non-fatal: an empty SQLContext produces no prompt section.
    sql_ctx = await build_sql_context(catalog, db)
    sql_context_note = sql_ctx.to_prompt_section()
    trace.set_approved_joins(sql_ctx)

    # ── STEP 2.8: EXECUTION STRATEGY ─────────────────────────────────────────
    # Pure graph-connectivity analysis — no LLM calls, no DB queries.
    # Runs Union-Find over approved_joins to find connected components in the
    # shortlist. Determines whether to execute as a single joined SQL, multiple
    # per-cluster SQLs, or fully independent analyses.
    # Prevents the LLM from hallucinating joins between unrelated domains.
    exec_strategy = plan_execution_strategy(catalog, sql_ctx)
    exec_strategy_note = exec_strategy.to_prompt_section()
    pipeline_logger.info(
        "execution_strategy_planned",
        mode=exec_strategy.mode,
        clusters=len(exec_strategy.clusters),
        cluster_sizes=[len(c.file_ids) for c in exec_strategy.clusters],
    )
    trace.set_execution_strategy(exec_strategy)
    # Both constraint sections are injected at the same prompt location
    # (before HOW TO WORK), so combine them into a single note.
    sql_context_note = "\n\n".join(filter(None, [sql_context_note, exec_strategy_note]))

    # ── STEP 2.9: GRAPH HEALTH + ORCHESTRATION CONFIDENCE ───────────────────
    # graph_health (pure CPU) and _avg_ing_conf (pure math) are independent of
    # each other — compute concurrently in a thread-pool executor. Both depend
    # only on data already fetched above. Non-blocking: failures produce neutral
    # scores and never prevent execution.
    _meta_list = [m for m, _ in retrieved_with_scores] if retrieved_with_scores else []
    _gh_loop = asyncio.get_running_loop()
    graph_health, _avg_ing = await asyncio.gather(
        _gh_loop.run_in_executor(None, score_graph_health, catalog, sql_ctx),
        _gh_loop.run_in_executor(None, _avg_ing_conf, _meta_list),
    )
    if graph_health.health_level in ("degraded", "poor"):
        metrics.inc("graph_health_degraded_count")
        pipeline_logger.warning(
            "graph_health_issue",
            health_level=graph_health.health_level,
            anomaly_flags=graph_health.anomaly_flags,
            edge_coverage=graph_health.edge_coverage,
            confidence_p50=graph_health.confidence_p50,
            weak_edge_ratio=graph_health.weak_edge_ratio,
        )

    confidence = compute_confidence(
        retrieved_with_scores=retrieved_with_scores,
        sql_ctx=sql_ctx,
        entity_resolution=entity_resolution,
        exec_strategy=exec_strategy,
        repair_attempts=0,   # initial pre-execution pass; sql.py increments this counter separately
        graph_health=graph_health,
    )
    if confidence.level == "low":
        metrics.inc("low_confidence_query_count")

    pipeline_logger.info(
        "orchestration_confidence",
        score=confidence.score,
        level=confidence.level,
        signals=confidence.signals,
        graph_health=graph_health.health_level,
    )

    # ── Policy snapshot: capture active thresholds for offline debugging ──────
    trace.set_policy_snapshot()

    # ── Grounding quality telemetry ───────────────────────────────────────────
    # Summarises how completely this request's shortlist was hydrated,
    # how many approved relationships back the joins, and whether the
    # retrieval path was normal (RRF) or degraded (fallback).
    trace.set_grounding_quality(
        hydrated_files=len(heavy_by_file),
        shortlist_size=len(catalog),
        sample_rows_available=len(sample_rows_by_blob),
        approved_relationships=len(getattr(sql_ctx, "approved_joins", []) or []),
        graph_health_level=graph_health.health_level,
        graph_edge_coverage=getattr(graph_health, "edge_coverage", None),
        retrieval_degraded=len(retrieved_with_scores) == 0,
        grounding_quality=_grounding_quality,
    )

    # ── Phase 6: Trust propagation trace ──────────────────────────────────
    # Summarise ingestion trust across the shortlisted files and emit the
    # degradation chain so operators can trace WHY confidence degraded.
    _cp_snap = _get_confidence_policy()
    _attenuated_count = sum(
        1 for m in _meta_list
        if m.ingestion_confidence_score is not None
        and m.ingestion_confidence_score < _cp_snap.ingestion_neutral
    )
    trace.set_trust_propagation(
        avg_ingestion_confidence=_avg_ing,
        attenuated_file_count=_attenuated_count,
        degradation_chain=confidence.degradation_chain,
    )
    if confidence.degradation_chain:
        pipeline_logger.info(
            "confidence_degradation",
            chain=confidence.degradation_chain,
            score=confidence.score,
            avg_ingestion=round(_avg_ing, 3),
        )

    # ── Phase 8: Calibration diagnostics ─────────────────────────────────────
    # Records modifier_breakdown + replay_inputs from the ConfidenceScore into
    # the orchestration trace.  Enables offline score reproduction and modifier
    # attribution without re-running the pipeline.
    trace.set_calibration_diagnostics(confidence)

    # ── Confidence attribution telemetry ──────────────────────────────────────
    # Human-readable per-component breakdown: answers "why did confidence drop?"
    # without requiring inspection of raw modifier_breakdown or replay_inputs.
    trace.set_confidence_attribution(confidence)

    # Per-request state store
    req_id = req_id_for_trace   # reuse the trace correlation ID as the request ID
    store: dict = {}
    with _stores_lock:
        _request_stores[req_id] = store

    # ── Request-local orchestration scratchpad ────────────────────────────────
    # Stores deterministic computations produced during context-build so that
    # tools and downstream code can reuse them without re-derivation.
    # Request-local only — never shared across users, never persisted.
    # Destroyed unconditionally in the `finally` block of the entry point.
    store["_scratchpad"] = {
        "blob_to_idx": _blob_to_idx,            # O(1) blob_path → array index
        "file_search_text": _file_search_text,  # precomputed per-file search text
        "file_col_text": _file_col_text,        # precomputed per-file column text
        "idf": _idf,                            # query-term IDF weights
        "intent_entities": list(intent_plan.entities) if intent_plan else [],
        "confidence_level": confidence.level,
        "confidence_score": confidence.score,
    }

    # Authorised blob paths for this request — all files visible to this user
    # in the full catalog (not just the retrieval shortlist).  search_catalog can
    # surface any of these files, so the ACL must cover the full set or the LLM
    # will get a false rejection when it tries to query a file it found via search.
    # Catalog entries only have blob_path (CSV); parquet paths live in all_parquet_paths.
    allowed_blob_paths: set[str] = set()
    for e in full_catalog:
        if e.get("blob_path"):
            allowed_blob_paths.add(f"az://{container_name}/{e['blob_path']}")
    for pq_path in all_parquet_paths.values():
        allowed_blob_paths.add(f"az://{container_name}/{pq_path}")

    # Build tools
    all_tools = []
    all_tools.extend(build_sql_tools(
        connection_string, container_name, parquet_blob_path, store,
        allowed_blob_paths=allowed_blob_paths,
        sql_ctx=sql_ctx,  # repair layer uses approved joins/columns as constraints
    ))
    # search_catalog uses the lean full catalog so it can find any file
    # without paying the heavy-field cost.
    # db is passed so get_file_schema can fetch real column types from Postgres
    # when the lean catalog entry lacks them (i.e. file not in hydration shortlist).
    all_tools.extend(build_catalog_tools(full_catalog, all_parquet_paths, container_name, db))
    # inspect_column — bound to full catalog with optional schema dict enrichment.
    # For non-hydrated files, falls back to a bounded SQL probe.
    # When field_definitions is non-empty, automatically appends business meaning
    # to the output (e.g. SHKZG → "Debit/Credit indicator: S=debit, H=credit").
    all_tools.extend(
        build_column_tool(
            full_catalog, all_parquet_paths, container_name, connection_string,
            field_definitions=field_definitions,
        )
    )
    # lookup_field_definition — standalone tool for explicit semantic lookups.
    # Uses the same pre-loaded dict as inspect_column — zero extra SQL calls.
    all_tools.extend(build_definition_lookup_tool(field_definitions))
    all_tools.extend(build_relations_tool(db, full_catalog))
    all_tools.extend(build_stats_tool(store))
    # inspect_data_format previews rows. Same full-catalog binding as
    # inspect_column; cached sample_rows for shortlist files, SQL probe
    # fallback for the rest.
    all_tools.extend(build_sample_tool(
        full_catalog, all_parquet_paths, container_name, connection_string,
    ))

    # Build graph and system prompt concurrently — both are pure CPU computation
    # with no shared mutable state. Overlapping them hides whichever is slower.
    _build_loop = asyncio.get_running_loop()
    graph, system_prompt = await asyncio.gather(
        _build_loop.run_in_executor(None, build_graph, all_tools),
        _build_loop.run_in_executor(
            None,
            lambda: build_system_prompt(
                catalog=catalog,
                parquet_paths_all=parquet_paths_all,
                parquet_blob_path=parquet_blob_path,
                container_name=container_name,
                sample_rows_by_blob=sample_rows_by_blob,
                conversation_context=conversation_context,
                total_file_count=len(full_catalog),
                mentioned_files=mentioned_file_names or None,
                sql_context_note=sql_context_note,
                top_blob_paths=top_blob_paths,
            ),
        ),
    )

    # ── Log the complete system prompt so we can audit exactly what the LLM sees ──
    pipeline_logger.info(
        "system_prompt_built",
        query=query,
        container=container_name,
        catalog_file_count=len(catalog),
        parquet_file_count=len(parquet_paths_all),
        has_conversation_context=bool(conversation_context),
        system_prompt=system_prompt,  # full prompt, no truncation
    )

    initial_state: AgentState = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)],
        "catalog": catalog,
        "connection_string": connection_string,
        "container_name": container_name,
        "parquet_blob_path": parquet_blob_path,
        "tool_call_count": 0,
        "request_id": req_id,
        "broaden_nudges": 0,
        # Model selection is now error-driven (see _should_escalate_to_primary in
        # graph_builder.py). is_first_turn is kept for backward state-shape compat
        # but no longer controls which LLM gets called.
        "is_first_turn": False,
    }

    return {
        "graph": graph,
        "initial_state": initial_state,
        "store": store,
        "req_id": req_id,
        "trace": trace,
        "catalog_len": len(catalog),
        "total_files": len(full_catalog),
        "container_name": container_name,
        "parquet_blob_path": parquet_blob_path,
        "intent_plan": intent_plan,
        "entity_resolution": entity_resolution,
        "graph_health": graph_health,
        "confidence": confidence,
    }


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agent_query(
    query: str,
    db: AsyncSession,
    *,
    conversation_context: str = "",
    user_id: str = "",
    is_admin: bool = True,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
    prior_files: list[str] | None = None,
    actor_email: str = "",
    actor_role: str = "",
) -> dict:
    """
    Main entry point for the agentic query pipeline.
    Returns {answer, data, chart, route, row_count, files_used, tool_calls}.
    """
    pipeline_start = time.perf_counter()
    _req_trace_id = uuid.uuid4().hex
    _structlog.contextvars.bind_contextvars(
        trace_id=_req_trace_id,
        actor_user_id=user_id or None,
        actor_email=actor_email or None,
        actor_role=actor_role or None,
    )

    try:
        ctx = await _build_agent_context(query, db, conversation_context, user_id, is_admin, allowed_domains, container_id, prior_files, request_trace_id=_req_trace_id)
    except Exception as exc:
        chat_logger.exception("agent_context_error", error=str(exc)[:400], query=query[:200])
        return {
            "answer": "An error occurred while preparing your query. Please try again.",
            "data": [], "chart": None,
        }
    if not ctx:
        return {"answer": _NO_FILES_MSG, "data": [], "chart": None}

    # Pull req_id early so we can clean up the request store on every exit path
    # (planner fast-path, agent path, and exception paths all converge here).
    req_id = ctx["req_id"]

    # ── Semantic Planner fast path ────────────────────────────────────────────
    # For deterministic structured queries (aggregations, time-filtered analytics),
    # the planner resolves join paths from the ontology layer and generates SQL
    # directly — bypassing the full LangGraph agent. Falls back automatically.
    planner_result = await _try_planner(query, ctx, db, pipeline_start)
    if planner_result:
        # Store was registered during _build_agent_context; clean it up here
        # since the agent try/finally block below is never reached on this path.
        with _stores_lock:
            _request_stores.pop(req_id, None)
        return planner_result

    graph = ctx["graph"]
    initial_state = ctx["initial_state"]
    store = ctx["store"]
    trace: OrchestrationTrace = ctx["trace"]

    chat_logger.info("agent_start",
                     query=query[:200],
                     file_count=ctx["catalog_len"],
                     container=ctx["container_name"],
                     has_parquet=ctx["parquet_blob_path"] is not None)

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        chat_logger.exception("agent_error", error=str(exc)[:400])
        trace.set_execution_outcome(rows=0, total=0, duration_ms=0.0, error=str(exc)[:200])
        trace.emit()
        return {
            "answer": "An error occurred while processing your query. Please try again.",
            "data": [], "chart": None,
        }
    finally:
        with _stores_lock:
            _request_stores.pop(req_id, None)

    # Extract results
    final_msgs = final_state["messages"]
    answer = extract_answer(final_msgs)
    sql_results = store.get("sql_results", [])
    tool_calls_made = final_state.get("tool_call_count", 0)
    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    chat_logger.info("agent_complete",
                     tool_calls=tool_calls_made,
                     row_count=len(sql_results),
                     total_duration_ms=total_ms,
                     answer_preview=answer[:200])

    # ── FINAL STEP: ANSWER READY ─────────────────────────────────────────────
    pipeline_logger.info(
        "final_answer",
        query=query,
        answer=answer,
        row_count=len(sql_results),
        tool_calls=tool_calls_made,
        total_duration_ms=total_ms,
    )

    if not answer and sql_results:
        answer = "Here are the results:"
    elif not answer and not sql_results:
        answer = fallback_answer(final_msgs)

    # Polish pass — gpt-4o-mini makes the answer warmer and more client-friendly
    answer = await _polish_answer(answer)

    chart = infer_chart(answer, sql_results)
    sql_total_rows = store.get("sql_total_rows", len(sql_results))

    # Emit orchestration trace for this request
    trace.set_execution_outcome(
        rows=len(sql_results),
        total=sql_total_rows,
        duration_ms=total_ms,
    )
    trace.emit()

    return {
        "answer": answer,
        "data": sql_results,
        "chart": chart,
        "route": "agent",
        "row_count": sql_total_rows,
        "files_used": list({
            blob
            for msg in final_msgs
            if isinstance(msg, ToolMessage)
            for blob in extract_blob_paths(msg.content)
        }),
        "tool_calls": tool_calls_made,
    }


# ── Streaming entry point ────────────────────────────────────────────────────

async def run_agent_query_stream(
    query: str,
    db: AsyncSession,
    *,
    conversation_context: str = "",
    user_id: str = "",
    is_admin: bool = True,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
    prior_files: list[str] | None = None,
    actor_email: str = "",
    actor_role: str = "",
) -> AsyncIterator[dict]:
    """
    Streaming variant of run_agent_query.

    Yields dicts:
      {"type": "thinking", "tool": tool_name}
      {"type": "token", "content": str}
      {"type": "tool_result", "tool": name, "preview": str}
      {"type": "done", "payload": {answer, data, chart, ...}}
    """
    pipeline_start = time.perf_counter()
    _req_trace_id = uuid.uuid4().hex
    _structlog.contextvars.bind_contextvars(
        trace_id=_req_trace_id,
        actor_user_id=user_id or None,
        actor_email=actor_email or None,
        actor_role=actor_role or None,
    )

    try:
        ctx = await _build_agent_context(query, db, conversation_context, user_id, is_admin, allowed_domains, container_id, prior_files, request_trace_id=_req_trace_id)
    except Exception as exc:
        chat_logger.exception("agent_context_error", error=str(exc)[:400], query=query[:200])
        yield {
            "type": "done",
            "payload": {
                "answer": "An error occurred while preparing your query. Please try again.",
                "data": [], "chart": None, "route": "agent", "row_count": 0,
                "files_used": [], "tool_calls": 0,
            },
        }
        return
    if not ctx:
        yield {
            "type": "done",
            "payload": {
                "answer": _NO_FILES_MSG,
                "data": [], "chart": None, "route": "agent", "row_count": 0,
                "files_used": [], "tool_calls": 0,
            },
        }
        return

    # Emit retrieval summary so the frontend can show "Searching N files…"
    yield {
        "type": "pipeline_step",
        "step": "retrieval",
        "retrieved_files": ctx["catalog_len"],
        "total_files": ctx["total_files"],
    }

    # Pull req_id early so we can clean up the request store on every exit path.
    req_id = ctx["req_id"]

    # ── Semantic Planner fast path (streaming) ────────────────────────────────
    # Try the planner before spinning up the LangGraph agent. If it produces a
    # high-confidence plan, we execute + synthesize without any agent tool calls.
    yield {"type": "thinking", "tool": "semantic_planner"}
    planner_result = await _try_planner(query, ctx, db, pipeline_start)
    if planner_result:
        # Store was registered during _build_agent_context; clean it up here
        # since the agent try/finally block below is never reached on this path.
        with _stores_lock:
            _request_stores.pop(req_id, None)
        # Stream the answer tokens then emit done
        for ch in planner_result["answer"].split():
            yield {"type": "token", "content": ch + " "}
        yield {"type": "done", "payload": planner_result}
        return

    graph = ctx["graph"]
    initial_state = ctx["initial_state"]
    store = ctx["store"]
    trace: OrchestrationTrace = ctx["trace"]

    chat_logger.info("agent_stream_start", query=query[:200], file_count=ctx["catalog_len"])

    answer_tokens: list[str] = []
    tool_calls_made = 0
    files_used: set[str] = set()
    tool_outputs: list[str] = []
    # Buffer chunks for the CURRENT LLM call. We only flush them to the user
    # at on_chat_model_end *if* the response has no tool_calls (i.e. it is the
    # final user-facing answer). Intermediate planning / "let me check the
    # schema next" narration is discarded so the user never sees it.
    pending_chunks: list[str] = []

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk:
                    tool_calls = getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None)
                    if tool_calls:
                        continue
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content and isinstance(content, str):
                        # Buffer only — do NOT yield yet. We don't know whether
                        # this LLM call is the final answer or an intermediate
                        # reasoning turn until on_chat_model_end fires.
                        pending_chunks.append(content)

            elif kind == "on_chat_model_start":
                # New LLM call starting — reset the buffer.
                # NOTE: We do NOT log the input here. agent_node already emits
                # `llm_input` for the same invocation; logging again here would
                # duplicate every LLM Input/Decision panel in the UI and made
                # the pipeline trace look like the model was being called twice.
                pending_chunks = []

            elif kind == "on_chat_model_end":
                # Determine whether this LLM turn produced tool calls so we know
                # whether to flush the buffered chunks. Do NOT log the response
                # here — agent_node already emits `llm_output` for the same call.
                resp = event["data"].get("output")
                resp_tool_calls = getattr(resp, "tool_calls", None) if resp else None

                # Flush buffered chunks ONLY if this LLM turn produced no tool
                # calls — i.e. it is the final answer the user should see.
                # Intermediate planning / "now I'll check the schema" narration
                # is dropped on the floor so the user only sees the result.
                if pending_chunks and not resp_tool_calls:
                    for piece in pending_chunks:
                        answer_tokens.append(piece)
                        yield {"type": "token", "content": piece}
                pending_chunks = []

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                tool_input = event["data"].get("input", {})
                tool_calls_made += 1
                pipeline_logger.info(
                    "tool_call_start",
                    tool=tool_name,
                    iteration=tool_calls_made,
                    input=tool_input,  # full args, no truncation
                )
                yield {"type": "thinking", "tool": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                tool_output = event["data"].get("output", "")
                pipeline_logger.info(
                    "tool_call_end",
                    tool=tool_name,
                    iteration=tool_calls_made,
                    output=str(tool_output),  # full output, no truncation
                )
                tool_output_str = tool_output if isinstance(tool_output, str) else str(tool_output)
                if isinstance(tool_output, str):
                    files_used.update(extract_blob_paths(tool_output))
                if tool_output_str:
                    tool_outputs.append(tool_output_str)

    except Exception as exc:
        chat_logger.exception("agent_stream_error", error=str(exc)[:400])
        trace.set_execution_outcome(rows=0, total=0, duration_ms=0.0, error=str(exc)[:200])
        trace.emit()
        yield {
            "type": "done",
            "payload": {
                "answer": "An error occurred while processing your query. Please try again.",
                "data": [], "chart": None, "route": "agent",
                "row_count": 0, "files_used": [], "tool_calls": 0,
            },
        }
        return
    finally:
        with _stores_lock:
            _request_stores.pop(req_id, None)

    final_answer = "".join(answer_tokens) if answer_tokens else ""
    sql_results = store.get("sql_results", [])
    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    chat_logger.info("agent_stream_complete",
                     tool_calls=tool_calls_made,
                     row_count=len(sql_results),
                     total_duration_ms=total_ms,
                     answer_len=len(final_answer))

    # ── FINAL STEP: ANSWER READY ─────────────────────────────────────────────
    pipeline_logger.info(
        "final_answer",
        query=query,
        answer=final_answer,
        row_count=len(sql_results),
        tool_calls=tool_calls_made,
        total_duration_ms=total_ms,
    )

    # ── Silent-model recovery ─────────────────────────────────────────────────
    # gpt-4o-mini sometimes produces zero text tokens in its final turn after a
    # multi-tool chain (it just stops without writing the answer). Detect this
    # and do one focused synthesis call from the SQL results / tool outputs so
    # the user always gets a real answer rather than a hollow fallback.
    _HOLLOW = {
        "", "Here are the results:",
        "I've gathered enough data. Let me summarise.",
    }
    if final_answer.strip() in _HOLLOW and tool_calls_made > 0:
        _context_parts: list[str] = []
        if sql_results:
            import json as _j
            _context_parts.append(
                f"Query returned {len(sql_results)} rows (first 15 shown):\n"
                + _j.dumps(sql_results[:15], default=str)
            )
        if tool_outputs:
            _context_parts.append("Tool outputs:\n" + "\n---\n".join(tool_outputs[-3:]))
        if not _context_parts:
            _context_parts.append(
                f"The system executed {tool_calls_made} tool call(s) but the queries returned 0 rows. "
                "Explain that the data was not found and describe what was searched."
            )
        if _context_parts:
            try:
                _synth_resp = await get_llm_mini().ainvoke([
                    SystemMessage(content=(
                        "You are an ERP data analyst. The user asked a question and the "
                        "system executed queries to find the answer. Write a clear, direct "
                        "response based on the query results below. Include key numbers, "
                        "totals, and observations. If 0 rows were returned, say the data "
                        "was not found and briefly explain what was searched."
                    )),
                    HumanMessage(content=(
                        f"User question: {query}\n\n"
                        + "\n\n".join(_context_parts)
                    )),
                ])
                if _synth_resp.content:
                    final_answer = _synth_resp.content
                    # Stream the synthesized answer so clients see tokens (not blank)
                    yield {"type": "token", "content": final_answer}
                    chat_logger.info("synthesis_fallback_used",
                                     query=query[:100],
                                     tool_calls=tool_calls_made,
                                     sql_rows=len(sql_results))
            except Exception as _synth_exc:
                chat_logger.warning("synthesis_fallback_error", error=str(_synth_exc)[:200])

    if not final_answer and sql_results:
        final_answer = "Here are the results:"
    elif not final_answer:
        final_answer = fallback_answer_from_outputs(tool_outputs)

    # Polish pass — no-op currently, kept for future use
    final_answer = await _polish_answer(final_answer)

    chart = infer_chart(final_answer, sql_results)
    sql_total_rows = store.get("sql_total_rows", len(sql_results))

    # Emit orchestration trace for this request (streaming path)
    trace.set_execution_outcome(
        rows=len(sql_results),
        total=sql_total_rows,
        duration_ms=total_ms,
    )
    trace.emit()

    yield {
        "type": "done",
        "payload": {
            "answer": final_answer,
            "data": sql_results,
            "chart": chart,
            "route": "agent",
            "row_count": sql_total_rows,
            "files_used": list(files_used),
            "tool_calls": tool_calls_made,
            "retrieved_files": ctx["catalog_len"],
            "total_files": ctx["total_files"],
        },
    }

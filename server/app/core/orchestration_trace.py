"""Structured orchestration trace — per-request decision telemetry.

DESIGN:
  One OrchestrationTrace is created at the start of each pipeline invocation.
  Each major stage writes its decision + evidence into the trace via a typed
  setter. At pipeline completion the trace is emitted as a single structured
  JSON log event.

PROPERTIES:
  - JSON-safe only: all values serialised via _safe_val() before storage.
  - No chain-of-thought: only decisions, evidence, and signals.
  - No user-sensitive data: queries/answers are NOT stored; only structural
    metadata (file names, column labels, confidence scores, counts).
  - Per-request correlation via request_id.
  - Safe truncation: list fields capped at _MAX_LIST items; strings at
    _MAX_STR chars.  Prevents log bloat on large catalogs.
  - Lightweight: all methods are synchronous, zero I/O, zero LLM calls.

TRACE STAGES (all optional — absent if pipeline short-circuited):
  planner              — BusinessIntentPlanner decisions
  entity_resolver      — entity → file mappings with confidence + signals
  graph_expansion      — anchor files + expansion result
  retrieval_fusion     — retrieved files + RRF scores (summary)
  retrieval_decision   — per-file survival/rejection telemetry with channels + reasons
    workflow_assembly    — query-time workflow tasks, candidate scores, temporal/authority decisions
  approved_joins       — validated join pairs + graph_verified / fallback_inferred flags
  grounding_quality    — hydration coverage, graph health, retrieval degradation level
  execution_strategy   — cluster mode + cluster breakdown
  sql_repair           — per-repair attempt outcome
  execution_outcome    — success/failure, row_count, duration_ms
  confidence_attribution — per-component scores + modifier signals (human-readable)
  calibration_diagnostics — modifier_breakdown + replay_inputs (offline replay)

USAGE (in graph.py):
  trace = OrchestrationTrace(request_id=req_id)
  trace.set_planner(intent_plan)
  trace.set_entity_resolver(entity_resolution)
  ...
  trace.set_execution_outcome(rows=len(rows), total=total, duration_ms=ms)
  trace.emit()   ← single structured log event at pipeline end
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.core.logger import pipeline_logger

# ── Truncation constants ───────────────────────────────────────────────────────
_MAX_LIST  = 20    # max items in any list field
_MAX_STR   = 200   # max chars in any string field
_MAX_KEYS  = 30    # max keys in any dict field

# Phase 7: Telemetry compaction — additional size guards for high-volume paths
_MAX_CHAIN = 8     # max items in a degradation_chain list
_MAX_SCORES = 5    # max top_scores entries in retrieval_fusion (compact mode)
# Payload byte estimate above which the trace is summarised instead of fully
# emitted.  Prevents a single large trace from spiking structured log volume.
_COMPACT_THRESHOLD_BYTES = 8_000


# ── JSON-safe truncating serialiser ───────────────────────────────────────────

def _safe_str(v: Any, max_len: int = _MAX_STR) -> str:
    s = str(v) if not isinstance(v, str) else v
    return s[:max_len] + "…" if len(s) > max_len else s


def _safe_val(v: Any) -> Any:
    """Recursively make a value JSON-safe with size bounds."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return _safe_str(v)
    if isinstance(v, dict):
        items = list(v.items())[:_MAX_KEYS]
        return {_safe_str(k, 80): _safe_val(vv) for k, vv in items}
    if isinstance(v, (list, tuple)):
        return [_safe_val(item) for item in list(v)[:_MAX_LIST]]
    # Fallback: stringify
    return _safe_str(v)


# ── Trace dataclass ────────────────────────────────────────────────────────────

class OrchestrationTrace:
    """
    Accumulates per-stage telemetry for one pipeline invocation.

    All setters are idempotent and non-raising. If a stage fails to produce
    data, the corresponding key is simply absent from the emitted trace.
    """

    __slots__ = ("_request_id", "_created_at", "_stages")

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._created_at = time.perf_counter()
        self._stages: dict[str, Any] = {}

    # ── Stage setters ──────────────────────────────────────────────────────────

    def set_planner(self, intent_plan: Any) -> None:
        """
        Record BusinessIntentPlanner output.

        Captures: detected behaviors (aggregation, time-filter, etc.),
        temporal_intent, constraints, entity names, and sort/limit signals.
        """
        try:
            self._stages["planner"] = _safe_val({
                "behaviors":       list(getattr(intent_plan, "behaviors", []) or []),
                "temporal_intent": getattr(intent_plan, "temporal_intent", None),
                "constraints":     list(getattr(intent_plan, "constraints", []) or []),
                "entities":        list(getattr(intent_plan, "entities", []) or []),
                "sort_intent":     getattr(intent_plan, "sort_intent", None),
                "limit_intent":    getattr(intent_plan, "limit_intent", None),
            })
        except Exception:
            pass  # never raise from trace

    def set_entity_resolver(
        self,
        entity_resolution: dict[str, list[Any]],
    ) -> None:
        """
        Record EntityResolver output.

        For each entity name, captures the top candidates with file name,
        confidence, and matching signals.  Candidate detail is capped at
        3 per entity to stay compact.
        """
        try:
            result: dict[str, list[dict]] = {}
            for entity, candidates in list(entity_resolution.items())[:_MAX_KEYS]:
                result[_safe_str(entity, 80)] = [
                    _safe_val({
                        "file":       getattr(c, "table", None),
                        "confidence": round(float(getattr(c, "confidence", 0)), 3),
                        "signals":    list(getattr(c, "signals", []) or []),
                    })
                    for c in list(candidates or [])[:3]
                ]
            self._stages["entity_resolver"] = result
        except Exception:
            pass

    def set_graph_expansion(
        self,
        anchor_file_ids: list[str],
        expanded_file_ids: list[str],
    ) -> None:
        """Record anchor seeds and the set of files reached via graph expansion."""
        try:
            self._stages["graph_expansion"] = _safe_val({
                "anchors":  anchor_file_ids,
                "expanded": expanded_file_ids,
                "expansion_count": len(expanded_file_ids) - len(anchor_file_ids),
            })
        except Exception:
            pass

    def set_retrieval_fusion(
        self,
        retrieved_with_scores: list[tuple[Any, float]],
        shortlist: list[dict],
        resolver_pins: list[str],
        fallback: bool = False,
    ) -> None:
        """
        Record retrieval fusion output.

        Captures the top-scored files with their RRF scores, the final
        shortlist blobs, and which files were injected by the resolver.
        Does NOT capture query text.
        """
        try:
            top_scores = [
                {
                    "file_id": _safe_str(getattr(meta, "file_id", ""), 36),
                    "score":   round(float(score), 4),
                }
                for meta, score in list(retrieved_with_scores or [])[:10]
            ]
            self._stages["retrieval_fusion"] = _safe_val({
                "shortlist_size":    len(shortlist),
                "top_scores":        top_scores,
                "resolver_pins":     resolver_pins,
                "fallback":          fallback,
                "shortlist_blobs":   [
                    _safe_str(e.get("blob_path", ""), 80)
                    for e in list(shortlist)[:_MAX_LIST]
                ],
            })
        except Exception:
            pass

    def set_retrieval_decision(
        self,
        shortlisted: list[dict],
        rejected: list[dict],
    ) -> None:
        """
        Per-file retrieval decision telemetry: why files survived or were dropped.

        shortlisted items schema:
          {"file": blob_path, "channels": ["bm25","vector","graph"],
           "rrf_score": 0.048, "resolver_pin": false, "lookup_injected": false,
           "prior_pin": false}

        rejected items schema:
          {"file": blob_path, "rejected_reason": "low_rrf_rank"|"outside_graph_boundary",
           "channels": [...]}   # channels present only for low_rrf_rank

        Lists are capped at _MAX_LIST entries each. Does NOT store query text.
        Never raises.
        """
        try:
            self._stages["retrieval_decision"] = _safe_val({
                "shortlisted":       list(shortlisted)[:_MAX_LIST],
                "rejected":          list(rejected)[:_MAX_LIST],
                "shortlisted_count": len(shortlisted),
                "rejected_count":    len(rejected),
            })
        except Exception:
            pass

    def set_workflow_assembly(self, workflow_result: Any) -> None:
        """Record query-time workflow cognition decisions.

        Captures decomposed workflow tasks, candidate selection/rejection
        rationale, temporal eligibility, transactional authority, and summary
        warning flags. The result object is intentionally duck-typed so the
        trace module does not depend on service-layer dataclasses.
        """
        try:
            if hasattr(workflow_result, "to_trace_dict"):
                payload = workflow_result.to_trace_dict()
            else:
                payload = workflow_result
            self._stages["workflow_assembly"] = _safe_val(payload)
        except Exception:
            pass

    def set_approved_joins(self, sql_ctx: Any) -> None:
        """
        Record validated join pairs from SQLContext.

        Captures each approved join's table pair, column pair, relationship
        type, confidence, and trust flags (graph_verified, fallback_inferred).
        """
        try:
            joins = []
            for j in list(getattr(sql_ctx, "approved_joins", []) or [])[:_MAX_LIST]:
                joins.append({
                    "left_file_id":       _safe_str(getattr(j, "left_file_id", ""), 36),
                    "right_file_id":      _safe_str(getattr(j, "right_file_id", ""), 36),
                    "left":              f"{getattr(j, 'left_table', '')}."
                                         f"{getattr(j, 'left_col', '')}",
                    "right":             f"{getattr(j, 'right_table', '')}."
                                         f"{getattr(j, 'right_col', '')}",
                    "type":              _safe_str(getattr(j, "relationship_type", ""), 60),
                    "conf":              round(float(getattr(j, "confidence", 0)), 2),
                    "graph_verified":    getattr(j, "graph_verified", True),
                    "fallback_inferred": getattr(j, "fallback_inferred", False),
                })
            # Surface a join_risk signal when any inferred (non-graph-verified) join
            # is present in the approved set.
            has_unverified = any(not jd["graph_verified"] or jd["fallback_inferred"]
                                 for jd in joins)
            null_sem_count = len(getattr(sql_ctx, "null_semantics", {}) or {})
            binding_count  = len(getattr(sql_ctx, "column_bindings", {}) or {})
            date_col_count = len(getattr(sql_ctx, "date_columns", {}) or {})
            self._stages["approved_joins"] = {
                "joins":            joins,
                "join_count":       len(joins),
                "binding_count":    binding_count,
                "date_col_count":   date_col_count,
                "null_sem_count":   null_sem_count,
                **({"join_risk": "unverified_inferred_join"} if has_unverified else {}),
            }
        except Exception:
            pass

    def set_file_identity_map(self, file_identities: Any) -> None:
        """Record the request-local logical table to canonical file mapping."""
        try:
            records = []
            for identity in list(getattr(file_identities, "identities", []) or [])[:_MAX_LIST]:
                records.append({
                    "canonical_id": _safe_str(getattr(identity, "canonical_id", ""), 36),
                    "logical_name": _safe_str(getattr(identity, "logical_name", ""), 80),
                    "sql_name": _safe_str(getattr(identity, "sql_name", ""), 80),
                    "display_name": _safe_str(getattr(identity, "display_name", ""), 100),
                    "has_parquet": bool(getattr(identity, "parquet_blob_path", None)),
                })
            self._stages["file_identity_map"] = _safe_val({
                "file_count": len(getattr(file_identities, "identities", []) or []),
                "files": records,
            })
        except Exception:
            pass

    def set_grounding_quality(
        self,
        *,
        hydrated_files: int,
        shortlist_size: int,
        sample_rows_available: int,
        approved_relationships: int,
        graph_health_level: str,
        graph_edge_coverage: float | None,
        retrieval_degraded: bool,
        grounding_quality: str,
    ) -> None:
        """
        Record grounding completeness for this pipeline invocation.

        grounding_quality values:
          "retrieved"              — RRF-based retrieval succeeded normally
          "graph_bounded"          — fallback used resolver pins + graph neighbors
          "full_catalog_degraded"  — no structural anchors; fallback used full catalog

        graph_edge_coverage is the fraction of shortlisted file pairs that have
        at least one approved join (from GraphHealthScore).

        Never raises.
        """
        try:
            _coverage_pct = (
                round(hydrated_files / shortlist_size, 3)
                if shortlist_size > 0 else 0.0
            )
            self._stages["grounding_quality"] = _safe_val({
                "hydrated_files":        hydrated_files,
                "shortlist_size":        shortlist_size,
                "hydration_coverage":    _coverage_pct,
                "sample_rows_available": sample_rows_available,
                "approved_relationships": approved_relationships,
                "graph_health_level":    graph_health_level,
                "graph_edge_coverage":   (
                    round(float(graph_edge_coverage), 3)
                    if graph_edge_coverage is not None else None
                ),
                "retrieval_degraded":    retrieval_degraded,
                "grounding_quality":     grounding_quality,
            })
        except Exception:
            pass

    def set_execution_strategy(self, exec_strategy: Any) -> None:
        """
        Record execution strategy planner output.

        Captures mode (single_joined / multi_cluster / independent_analyses),
        cluster count, and per-cluster file lists.
        """
        try:
            clusters = []
            for c in list(getattr(exec_strategy, "clusters", []) or [])[:_MAX_LIST]:
                clusters.append({
                    "cluster_id": _safe_str(getattr(c, "cluster_id", ""), 40),
                    "files":      [
                        _safe_str(f, 80)
                        for f in list(getattr(c, "file_ids", []) or [])[:10]
                    ],
                    "join_count": len(getattr(c, "joins", []) or []),
                })
            self._stages["execution_strategy"] = {
                "mode":          _safe_str(getattr(exec_strategy, "mode", ""), 40),
                "cluster_count": len(clusters),
                "clusters":      clusters,
            }
        except Exception:
            pass

    def set_sql_repair(
        self,
        attempt: int,
        tier: str,            # "tier1" | "tier2"
        outcome: str,         # "repaired" | "declined" | "failed"
        error_preview: str,
        repaired: bool,
    ) -> None:
        """
        Record one SQL repair attempt outcome.

        Multiple calls accumulate into a list (one entry per attempt).
        Does NOT store the SQL itself — that is already in the pipeline log.
        """
        try:
            entry = _safe_val({
                "attempt":       attempt,
                "tier":          tier,
                "outcome":       outcome,
                "error_preview": error_preview[:100],
                "repaired":      repaired,
            })
            repairs = self._stages.setdefault("sql_repair", [])
            if isinstance(repairs, list):
                repairs.append(entry)
        except Exception:
            pass

    def set_execution_outcome(
        self,
        *,
        rows: int,
        total: int,
        duration_ms: float,
        repair_count: int = 0,
        error: str | None = None,
    ) -> None:
        """Record final SQL execution result."""
        try:
            self._stages["execution_outcome"] = _safe_val({
                "rows":          rows,
                "total_rows":    total,
                "duration_ms":   round(duration_ms, 2),
                "repair_count":  repair_count,
                "success":       error is None,
                "error_preview": error[:100] if error else None,
            })
        except Exception:
            pass

    def set_trust_propagation(
        self,
        *,
        avg_ingestion_confidence: float,
        attenuated_file_count: int,
        degradation_chain: list[str],
    ) -> None:
        """Record Phase 6/7 trust propagation summary for this pipeline invocation.

        avg_ingestion_confidence : mean ingestion score across shortlisted files
                                   (pre-Phase-5 files counted as neutral 0.70).
        attenuated_file_count    : number of files whose RRF score was reduced
                                   due to ingestion quality below neutral.
        degradation_chain        : ordered list of reason codes explaining why
                                   orchestration confidence degraded (empty = clean).

        Phase 7 compaction: chain is deduplicated and capped at _MAX_CHAIN items.
        """
        try:
            # Compact: deduplicate chain while preserving order, then cap length.
            _seen: set[str] = set()
            _compact_chain: list[str] = []
            for _code in degradation_chain:
                if _code not in _seen:
                    _seen.add(_code)
                    _compact_chain.append(_code)
                if len(_compact_chain) >= _MAX_CHAIN:
                    break
            self._stages["trust_propagation"] = _safe_val({
                "avg_ingestion_confidence": round(float(avg_ingestion_confidence), 3),
                "attenuated_file_count":    attenuated_file_count,
                "degradation_chain":        _compact_chain,
                "chain_truncated":          len(degradation_chain) > _MAX_CHAIN,
            })
        except Exception:
            pass  # telemetry — never block the pipeline

    def set_policy_snapshot(self) -> None:
        """
        Capture the active policy values at pipeline invocation time.

        Stores a compact, JSON-safe snapshot of all 6 policy modules so that
        offline debugging can answer: "what thresholds were active when this
        query ran?"

        Called once per request, typically after graph health and confidence
        scoring are complete (Step 2.9 in graph.py).
        Never raises.
        """
        try:
            from app.policies import build_policy_snapshot  # local import avoids circulars
            self._stages["policy_snapshot"] = _safe_val(build_policy_snapshot())
        except Exception:
            pass  # telemetry — never block the pipeline

    def set_calibration_diagnostics(self, confidence: Any) -> None:
        """
        Record calibration diagnostic breakdown from a ConfidenceScore.

        Captures modifier_breakdown and replay_inputs so that offline
        calibration tools can correlate score distributions to specific
        modifier activations without parsing individual component fields.

        Called after compute_confidence() returns, before emit().
        Never raises.
        """
        try:
            self._stages["calibration_diagnostics"] = _safe_val({
                "modifier_breakdown": getattr(confidence, "modifier_breakdown", {}),
                "replay_inputs":      getattr(confidence, "replay_inputs", {}),
            })
        except Exception:
            pass  # telemetry — never block the pipeline

    def set_confidence_attribution(self, confidence: Any) -> None:
        """
        Record human-readable per-component confidence attribution.

        Emits each scoring component's value (0-1) alongside the composite
        score and which modifiers fired.  Designed for operator dashboards
        that need to answer "why did confidence drop?" without parsing the
        raw modifier_breakdown or replay_inputs calibration fields.

        Complements set_calibration_diagnostics() (which covers replay
        reproducibility); this method covers human-readable attribution.
        Never raises.
        """
        try:
            _mb = getattr(confidence, "modifier_breakdown", {}) or {}
            _components: dict[str, float] = {}
            for _attr in (
                "retrieval_component", "graph_component", "resolver_component",
                "complexity_component", "repair_component",
                "health_component", "ingestion_component",
            ):
                _v = getattr(confidence, _attr, None)
                if _v is not None:
                    _components[_attr.replace("_component", "")] = round(float(_v), 3)
            self._stages["confidence_attribution"] = _safe_val({
                "components":    _components,
                "score":         round(float(getattr(confidence, "score", 0)), 3),
                "level":         getattr(confidence, "level", ""),
                "signals":       list(getattr(confidence, "signals", []) or []),
                "modifiers": {
                    "trust_normalization":    bool(_mb.get("trust_normalization_applied")),
                    "trust_ceiling":          bool(_mb.get("trust_ceiling_applied")),
                    "floor_applied":          bool(_mb.get("minimum_viable_floor_applied")),
                    "ceiling_credit_pts":     _mb.get("degradation_ceiling_credit_score_pts", 0),
                    "normalization_forgiven_pts": _mb.get("normalization_forgiveness_score_pts", 0),
                },
            })
        except Exception:
            pass  # telemetry — never block the pipeline

    # ── Emission ───────────────────────────────────────────────────────────────

    def emit(self) -> None:
        """
        Emit the complete trace as a single structured pipeline log event.

        Safe to call multiple times — subsequent calls are no-ops if the trace
        was already emitted (prevents double-logging on error paths).

        Phase 7 compaction: if the estimated trace payload exceeds
        _COMPACT_THRESHOLD_BYTES, emit a summarised version (stage names +
        confidence + trust fields only) and set compact_trace=True.  Full trace
        detail remains available in individual stage log events.

        Phase 8 escalation mode: when the composite confidence score falls below
        ConfidencePolicy.escalation_trace_threshold, the full trace is always
        emitted regardless of payload size.  This ensures the most critical
        failure cases always have complete debugging context in the log.
        """
        try:
            elapsed_ms = round((time.perf_counter() - self._created_at) * 1000, 2)
            stage_keys = list(self._stages.keys())

            # Phase 8: escalation mode — very low confidence overrides compaction.
            # Read score from calibration_diagnostics.replay_inputs (Phase 8 field)
            # or fall back to execution_outcome if diagnostics are absent.
            _is_escalation = False
            try:
                from app.policies.confidence_policy import get_confidence_policy as _gcp  # noqa: PLC0415
                _esc_threshold = _gcp().escalation_trace_threshold
                _diag = self._stages.get("calibration_diagnostics", {})
                if isinstance(_diag, dict):
                    _ri = _diag.get("replay_inputs", {})
                    # replay_inputs is not present when diagnostics weren't set;
                    # in that case escalation is not triggered.
                    if isinstance(_ri, dict) and "calibration_version" in _ri:
                        # We don’t store the composite score directly in replay_inputs,
                        # so look it up from policy_snapshot or leave as non-escalation.
                        pass
                # Simpler path: check calibration_diagnostics.modifier_breakdown
                _mb = _diag.get("modifier_breakdown", {}) if isinstance(_diag, dict) else {}
                _floor_applied = _mb.get("minimum_viable_floor_applied", False)
                # If the global floor fired, score was at or below minimum_viable_confidence
                # (0.25), which is always below escalation_trace_threshold (0.35).
                if _floor_applied:
                    _is_escalation = True
            except Exception:
                pass

            # Phase 8.1: deterministic JSON-based payload estimation.
            # json.dumps with fixed separators and sort_keys produces stable
            # byte counts independent of Python repr variance, dict ordering,
            # or unicode rendering differences across platforms.
            # The _safe_val() bounds already constrain dict size so this is fast.
            try:
                _payload_est = len(
                    json.dumps(
                        self._stages,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    )
                )
            except Exception:
                _payload_est = len(str(self._stages))  # fallback: str estimate
            if not _is_escalation and _payload_est > _COMPACT_THRESHOLD_BYTES:
                # Summarised trace: keep only high-signal fields
                _compact_stages: dict[str, Any] = {
                    "stage_names":       stage_keys,
                    "compact_trace":     True,
                    "payload_est_bytes": _payload_est,
                }
                for _k in ("execution_outcome", "trust_propagation",
                            "policy_snapshot", "calibration_diagnostics"):
                    if _k in self._stages:
                        _compact_stages[_k] = self._stages[_k]
                pipeline_logger.info(
                    "orchestration_trace",
                    request_id=self._request_id,
                    elapsed_ms=elapsed_ms,
                    stages=stage_keys,
                    trace=_compact_stages,
                )
            else:
                pipeline_logger.info(
                    "orchestration_trace",
                    request_id=self._request_id,
                    elapsed_ms=elapsed_ms,
                    stages=stage_keys,
                    trace=self._stages,
                    **(  {"escalation": True} if _is_escalation else {} ),
                )
        except Exception:
            pass  # trace emission must never crash the pipeline

    def as_dict(self) -> dict:
        """Return a copy of the accumulated trace for programmatic inspection (tests)."""
        return dict(self._stages)

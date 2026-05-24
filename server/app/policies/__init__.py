"""Orchestration policy modules — deterministic governance layer.

This package centralises every magic number, threshold, and cap that governs
orchestration behavior across the analytical pipeline.

ARCHITECTURE INTENT:
  Before this layer, policy was implicit: thresholds lived as module-level
  constants scattered across 8+ files with no traceability between them.

  After this layer, every orchestration decision has a named policy attribute
  with an explicit rationale comment. All policies are:
    - Immutable frozen dataclasses (no runtime mutation)
    - In-memory only (no env-var loading, no Redis, no config servers)
    - Synchronous (zero I/O, zero async)
    - Singleton-accessed (lru_cache — one instance per process)

FUTURE READINESS:
  This architecture is shaped for — but does not yet implement:
    - Deployment-level overrides (e.g. high-RAM VM → relax caps)
    - Tenant-level calibration (per-container policy variants)
    - Telemetry-driven threshold adjustment (confidence scoring feedback)
  When those are added, only the singleton accessors need to change.
  All consuming code already goes through the accessor — zero call-site changes.

RE-EXPORTS:
  from app.policies import (
      get_retrieval_policy,
      get_graph_policy,
      get_execution_policy,
      get_confidence_policy,
      get_repair_policy,
      get_ingestion_policy,
      build_policy_snapshot,
  )
"""
from __future__ import annotations

from app.policies.retrieval_policy import RetrievalPolicy, get_retrieval_policy
from app.policies.graph_policy import GraphPolicy, get_graph_policy
from app.policies.execution_policy import ExecutionPolicy, get_execution_policy
from app.policies.confidence_policy import ConfidencePolicy, get_confidence_policy
from app.policies.repair_policy import RepairPolicy, get_repair_policy
from app.policies.ingestion_policy import IngestionPolicy, get_ingestion_policy
from app.services.calibration_manifest import get_calibration_manifest  # noqa: E402


def build_policy_snapshot() -> dict:
    """
    Return a compact, JSON-safe snapshot of all active policy values.

    Captured once per pipeline request and stored in the OrchestrationTrace.
    Enables offline debugging: "what policy was active when this query ran?"

    Does NOT capture every field — only the thresholds most likely to affect
    observable orchestration behavior.
    """
    r = get_retrieval_policy()
    g = get_graph_policy()
    e = get_execution_policy()
    c = get_confidence_policy()
    rp = get_repair_policy()
    i = get_ingestion_policy()

    # Scoring formula version — separate lineage from calibration_version.
    # Local import: query_confidence imports confidence_policy (not __init__),
    # so no circular dependency.
    try:
        from app.services.query_confidence import _FORMULA_VERSION as _fv  # noqa: PLC0415
    except Exception:
        _fv = "unknown"

    return {
        "retrieval": {
            "shortlist_top_k":        r.shortlist_top_k,
            "bm25_candidates":        r.bm25_candidates,
            "vector_candidates":      r.vector_candidates,
            "fuzzy_candidates":       r.fuzzy_candidates,
            "min_score":              r.min_score,
            "rrf_k":                  r.rrf_k,
            "lookup_reserved_slots":  r.lookup_reserved_slots,
            "max_top_k":              r.max_top_k,          # Phase 7
            "max_rrf_candidates":     r.max_rrf_candidates,  # Phase 7
        },
        "graph": {
            "max_seed_ids":              g.max_seed_ids,
            "max_neighbors_per_node":    g.max_neighbors_per_node,
            "expansion_conf_floor":      g.expansion_conf_floor,
            "join_hard_floor":           g.join_hard_floor,
            "join_soft_floor":           g.join_soft_floor,
            "top_n_approved_joins":      g.top_n_approved_joins,
            "supernode_degree_threshold":    g.supernode_degree_threshold,   # Phase 7
            "supernode_confidence_penalty":  g.supernode_confidence_penalty, # Phase 7
            "max_neighbor_influence_ratio":  g.max_neighbor_influence_ratio, # Phase 7
            "min_supernode_participation_slots": g.min_supernode_participation_slots,  # Phase 8
            "graph_density_mode":                g.graph_density_mode,                 # Phase 8
        },
        "execution": {
            "max_sql_length":         e.max_sql_length,
            "max_joins":              e.max_joins,
            "max_scan_files":         e.max_scan_files,
            "max_concurrent":         e.max_concurrent,
            "default_timeout_s":      e.default_timeout_seconds,
            "max_tool_calls":         e.max_tool_calls,
        },
        "confidence": {
            "low_threshold":               c.low_confidence_threshold,
            "repair_penalty":              c.repair_penalty_per_attempt,
            "resolver_pin_threshold":      c.resolver_pin_threshold,
            "edge_attenuation_strength":      c.edge_attenuation_strength,        # Phase 7
            "retrieval_attenuation_strength": c.retrieval_attenuation_strength,   # Phase 7
            "trust_normalization_factor":     c.trust_normalization_factor,       # Phase 7
            "max_trust_degradation":          c.max_trust_degradation,            # Phase 7
            "minimum_viable_confidence":      c.minimum_viable_confidence,        # Phase 8
            "escalation_trace_threshold":     c.escalation_trace_threshold,       # Phase 8
            "calibration_version":            c.calibration_version,              # Phase 7+
            # formula_version tracks scoring-semantic changes (formula redesigns).
            # Distinct from calibration_version (threshold changes).
            # Both are required for deterministic offline replay.
            "formula_version":                _fv,
        },
        "repair": {
            "max_attempts":           rp.max_attempts,
            "tier2_output_tokens":    rp.tier2_output_tokens,
        },
        "ingestion": {
            "weak_confidence_warn":   i.weak_confidence_warn,
            "high_density_ratio":     i.high_density_ratio,
            "min_role_coverage":      i.min_role_coverage,
        },
    }


__all__ = [
    "RetrievalPolicy",   "get_retrieval_policy",
    "GraphPolicy",       "get_graph_policy",
    "ExecutionPolicy",   "get_execution_policy",
    "ConfidencePolicy",  "get_confidence_policy",
    "RepairPolicy",      "get_repair_policy",
    "IngestionPolicy",   "get_ingestion_policy",
    "build_policy_snapshot",
    "get_calibration_manifest",
]

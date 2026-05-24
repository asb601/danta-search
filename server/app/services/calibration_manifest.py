"""Calibration governance manifest — deterministic trust flow documentation.

PURPOSE
=======
This module is the single authoritative record of:
  1. Which policy owns each trust-modifier parameter.
  2. How trust modifiers interact across policy boundaries.
  3. The confidence flow from raw ingestion quality to final composite score.
  4. Calibration version lineage for offline replay infrastructure.

ARCHITECTURE RULES
==================
* Parameter ownership: each modifier belongs to exactly ONE policy.
  Cross-policy reads are permitted; cross-policy writes are never allowed.
* All policies are frozen at startup (dataclasses with frozen=True).
  No runtime mutation. No adaptive calibration.
* Policy version lineage: ConfidencePolicy.calibration_version is incremented
  on each calibration change.  Every orchestration trace records this version
  via the policy_snapshot stage.
* Offline replay: given a trace's replay_inputs + calibration_version, the
  exact confidence score can be reproduced by loading the matching policy
  version and calling compute_confidence() with those inputs.

CONFIDENCE FLOW (ordered pipeline stages)
==========================================
Stage 1 — Ingestion Pipeline (owned by IngestionPolicy)
  column_role_resolver + relationship_detector
  → ingestion_confidence.py → FileMetadata.ingestion_confidence_score

Stage 2 — Graph Traversal (cross-owned: GraphPolicy + ConfidencePolicy)
  SemanticRelationship.confidence_score
  × supernode_degree_penalty  (GraphPolicy: supernode_confidence_penalty)
  → penalized_conf
  → effective_edge_confidence(rel_conf=penalized_conf, ing_a, ing_b)
    [ConfidencePolicy: edge_attenuation_strength, trust_floor, ingestion_mod_floor]
  → neighbor traversal score in graph_expand.py
  Supernode participation floor: GraphPolicy.min_supernode_participation_slots
  guarantees even hub files contribute minimum expansion slots.

Stage 3 — Retrieval Fusion (cross-owned: RetrievalPolicy + ConfidencePolicy)
  RRF rank-list fusion (rank-only, Cormack 2009)
  → _trust_attenuate: score × retrieval_trust_weight(ingestion_score)
    [ConfidencePolicy: retrieval_attenuation_strength, retrieval_trust_floor]
  → trust-attenuated fused result list in orchestrator.py
  Scale caps: RetrievalPolicy.max_top_k, max_rrf_candidates

Stage 4 — Orchestration Confidence (owned by ConfidencePolicy)
  Components (weights sum = 1.0):
    retrieval  (w=0.25) — top RRF score via sigmoid normalisation
    graph      (w=0.20) — approved-join p50 confidence
    resolver   (w=0.15) — entity resolution top confidence
    complexity (w=0.15) — execution mode score
    repair     (w=0.10) — SQL repair penalty per attempt
    health     (w=0.05) — graph health level (good / degraded / poor)
    ingestion  (w=0.10) — avg shortlist ingestion, trust-normalised
  Trust normalization (ConfidencePolicy.trust_normalization_factor = 0.65):
    deficit-below-neutral is discounted to prevent retrieval + ingestion
    from double-penalising the same catalog weakness.
  Trust ceiling (ConfidencePolicy.max_trust_degradation = 0.12):
    total trust impact (ingestion + health) cannot exceed this value.
  Global floor (ConfidencePolicy.minimum_viable_confidence = 0.25):
    absolute minimum — prevents total orchestration collapse even when all
    trust signals are simultaneously weak.

CROSS-POLICY INTERACTION MATRIX
================================
Interaction                                    Owner Policy   Consumer
─────────────────────────────────────────────────────────────────────────
ingestion_neutral (fallback for pre-P5 files)  Confidence     trust_propagation.py
ingestion_mod_floor (min edge modifier)        Confidence     trust_propagation.py
edge_attenuation_strength (soft curve α)       Confidence     trust_propagation.py
retrieval_attenuation_strength (soft curve β)  Confidence     trust_propagation.py
trust_floor (min edge confidence)              Confidence     trust_propagation.py
retrieval_trust_floor (min RRF multiplier)     Confidence     trust_propagation.py
supernode_confidence_penalty (hub damping)     Graph          graph_expand.py
supernode_degree_threshold (hub classifier)    Graph          graph_expand.py
max_neighbor_influence_ratio (diversity cap)   Graph          graph_expand.py
min_supernode_participation_slots (min slots)  Graph          graph_expand.py
max_top_k (retrieval cap)                      Retrieval      orchestrator.py
max_rrf_candidates (pool cap)                  Retrieval      orchestrator.py
"""
from __future__ import annotations

from app.policies.confidence_policy import get_confidence_policy
from app.policies.graph_policy import get_graph_policy
from app.policies.retrieval_policy import get_retrieval_policy

# ── Parameter ownership table ─────────────────────────────────────────────────
# Maps parameter name → owning policy + consumer paths.
# Used by offline calibration tools to verify no cross-policy writes occur.
_PARAMETER_OWNERSHIP: dict[str, dict] = {
    # ConfidencePolicy-owned parameters
    "ingestion_neutral":              {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.avg_ingestion_confidence",
                      "trust_propagation.effective_edge_confidence"],
    },
    "ingestion_mod_floor":            {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.effective_edge_confidence"],
    },
    "edge_attenuation_strength":      {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.effective_edge_confidence"],
    },
    "retrieval_attenuation_strength": {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.retrieval_trust_weight"],
    },
    "trust_floor":                    {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.effective_edge_confidence"],
    },
    "retrieval_trust_floor":          {
        "policy": "ConfidencePolicy",
        "consumers": ["trust_propagation.retrieval_trust_weight"],
    },
    "trust_normalization_factor":     {
        "policy": "ConfidencePolicy",
        "consumers": ["query_confidence.compute_confidence"],
    },
    "max_trust_degradation":          {
        "policy": "ConfidencePolicy",
        "consumers": ["query_confidence.compute_confidence"],
    },
    "minimum_viable_confidence":      {
        "policy": "ConfidencePolicy",
        "consumers": ["query_confidence.compute_confidence"],
    },
    "escalation_trace_threshold":     {
        "policy": "ConfidencePolicy",
        "consumers": ["orchestration_trace.emit"],
    },
    # GraphPolicy-owned parameters
    "supernode_degree_threshold":        {
        "policy": "GraphPolicy",
        "consumers": ["graph_expand.graph_expand"],
    },
    "supernode_confidence_penalty":      {
        "policy": "GraphPolicy",
        "consumers": ["graph_expand.graph_expand"],
    },
    "max_neighbor_influence_ratio":      {
        "policy": "GraphPolicy",
        "consumers": ["graph_expand.graph_expand"],
    },
    "min_supernode_participation_slots": {
        "policy": "GraphPolicy",
        "consumers": ["graph_expand.graph_expand"],
    },
    # RetrievalPolicy-owned parameters
    "max_top_k":          {
        "policy": "RetrievalPolicy",
        "consumers": ["orchestrator.retrieve_with_scores"],
    },
    "max_rrf_candidates": {
        "policy": "RetrievalPolicy",
        "consumers": ["orchestrator.retrieve_with_scores"],
    },
}

_TRUST_FLOW_STAGES = [
    "ingestion_pipeline",
    "graph_traversal",
    "retrieval_fusion",
    "orchestration_confidence",
]


def get_calibration_manifest() -> dict:
    """Return the authoritative calibration governance record.

    Combines policy-version lineage, trust flow stages, parameter ownership
    matrix, and active parameter values into a single structured document.
    Deterministic: same output on every call for a given process.

    Used by:
      - build_policy_snapshot() — recorded in every orchestration trace stage.
      - Offline calibration pipelines — correlate score distributions to
        specific calibration_version values without parsing individual fields.
      - Replay infrastructure — active_values + calibration_version are the
        minimal inputs needed to reproduce any historical confidence score.
    """
    cp = get_confidence_policy()
    gp = get_graph_policy()
    rp = get_retrieval_policy()

    return {
        "calibration_version": cp.calibration_version,
        "trust_flow_stages":   _TRUST_FLOW_STAGES,
        "parameter_ownership": _PARAMETER_OWNERSHIP,
        "active_values": {
            # ConfidencePolicy
            "edge_attenuation_strength":        cp.edge_attenuation_strength,
            "retrieval_attenuation_strength":   cp.retrieval_attenuation_strength,
            "trust_normalization_factor":       cp.trust_normalization_factor,
            "max_trust_degradation":            cp.max_trust_degradation,
            "minimum_viable_confidence":        cp.minimum_viable_confidence,
            "escalation_trace_threshold":       cp.escalation_trace_threshold,
            # GraphPolicy
            "supernode_confidence_penalty":       gp.supernode_confidence_penalty,
            "supernode_degree_threshold":         gp.supernode_degree_threshold,
            "min_supernode_participation_slots":  gp.min_supernode_participation_slots,
            "max_neighbor_influence_ratio":       gp.max_neighbor_influence_ratio,
            # RetrievalPolicy
            "max_top_k":          rp.max_top_k,
            "max_rrf_candidates": rp.max_rrf_candidates,
        },
        "confidence_floors": {
            "edge_confidence":       cp.trust_floor,
            "retrieval_weight":      cp.retrieval_trust_floor,
            "trust_degradation_cap": cp.max_trust_degradation,
            "composite_score":       cp.minimum_viable_confidence,
        },
    }

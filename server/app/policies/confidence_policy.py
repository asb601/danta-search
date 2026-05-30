"""Pipeline confidence scoring policy.

Governs:
  - Component weights for the composite confidence score
  - Penalty curves for SQL repair attempts
  - Alert thresholds and resolver pin thresholds

WHY CONFIDENCE SCORING EXISTS:
  Confidence is a telemetry signal, not a gating signal. Queries always run
  regardless of confidence level. The score feeds:
    1. Orchestration trace (offline debugging)
    2. Metrics (low_confidence_query_count gauge)
    3. LLM context: informational flag in the prompt ("retrieval may be weak")

  This policy defines the constants that shape that signal.

WEIGHT SUM CONSTRAINT:
  w_retrieval + w_graph + w_resolver + w_complexity + w_repair + w_health == 1.0
  This is a logical constraint, not runtime-enforced. Violations produce a
  score outside [0,1] which will immediately show up in metrics.

FUTURE READINESS:
  - Tenant override: a tenant that only uses single-table queries could
    set w_complexity → 0.05 and redistribute to w_retrieval/w_graph.
  - Telemetry-driven: if repair penalty is too aggressive, raise it;
    if low_confidence_threshold is too sensitive, lower it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ConfidencePolicy:
    """
    All confidence scoring coefficients in one typed, immutable config.

    ── COMPONENT WEIGHTS ────────────────────────────────────────────────────
    Each weight is the fractional contribution of that component to the
    final composite confidence score. Sum must equal 1.0.

    w_retrieval : float  (0.25)
        How well the retrieval stage found relevant files.
        Highest single contributor — retrieval quality drives everything.

    w_graph : float  (0.25)
        How many of the retrieved files have approved join edges between them.
        Equal weight to retrieval — join quality is as important as recall.

    w_resolver : float  (0.20)
        Entity resolution score. Did the identified entities resolve cleanly?

    w_complexity : float  (0.15)
        Execution strategy complexity score.
        Single-file → 1.0; independent analyses → 0.30.
        Simple queries have inherently higher confidence than multi-cluster ones.

    w_repair : float  (0.10)
        Penalty for SQL repair invocations (each attempt reduces score).
        Lower weight — repair is a soft signal, not a primary quality indicator.

    w_health : float  (0.05)
        Graph health score (graph_health.py result).
        Smallest weight — health reflects catalog quality, not query quality.

    w_ingestion : float  (0.10)  ← Phase 6
        Average ingestion confidence of the shortlisted files.
        Captures how trustworthy the underlying metadata is for this request.
        w_graph reduced 0.25→0.20 and w_resolver reduced 0.20→0.15 to make room.

    ── PENALTY CURVE ─────────────────────────────────────────────────────────
    repair_penalty_per_attempt : float
        Score reduction per repair attempt: repair_component = 1.0 - N * penalty.
        0.20 per attempt → 0 attempts = 1.0, 1 attempt = 0.80, 2 attempts = 0.60.
        If clamped to 0 on 3+ attempts, the repair component floors at 0.

    ── THRESHOLDS ────────────────────────────────────────────────────────────
    low_confidence_threshold : float
        Score below this → emit low_confidence_query_count metric + trace flag.
        DOES NOT block execution.

    resolver_pin_threshold : float
        Entity resolution confidence required to "pin" a file into the shortlist
        (bypassing rank order). High threshold ensures only strongly-resolved
        entities get a guaranteed slot.

    ── COMPLEXITY SCORES ─────────────────────────────────────────────────────
    complexity_single_joined : float  (1.0)
        One SQL query across all files in one join plan.

    complexity_multi_cluster : float  (0.65)
        Multiple sub-queries across file clusters — higher coordination risk.

    complexity_independent : float  (0.30)
        Fully independent sub-queries with no join path between them.
        Lowest confidence because the planner can't cross-reference results.
    """
    # Component weights (must sum to 1.0)
    # Phase 6: w_ingestion added (0.10); w_graph reduced (0.25→0.20);
    # w_resolver reduced (0.20→0.15). Ingestion trust is now a first-class dimension.
    w_retrieval:              float = 0.25
    w_graph:                  float = 0.20
    w_resolver:               float = 0.15
    w_complexity:             float = 0.15
    w_repair:                 float = 0.10
    w_health:                 float = 0.05
    w_ingestion:              float = 0.10  # Phase 6: avg shortlist ingestion quality

    # Penalty / thresholds
    repair_penalty_per_attempt: float = 0.20
    low_confidence_threshold:   float = 0.50
    resolver_pin_threshold:     float = 0.85
    resolver_seed_threshold:    float = 0.18   # min confidence to seed graph expansion and append to shortlist

    # Complexity level scores
    complexity_single_joined:   float = 1.00
    complexity_multi_cluster:   float = 0.65
    complexity_independent:     float = 0.30

    # ── Phase 6: Trust propagation parameters ─────────────────────────────────
    # ingestion_neutral : score assigned to pre-Phase-5 files with no stored
    #   ingestion_confidence_score.  0.70 = moderate trust, no heavy penalty.
    ingestion_neutral:          float = 0.70

    # ingestion_mod_floor : minimum ingestion modifier for effective_edge_confidence.
    #   Floors at 0.50 so even two very weak endpoints still allow some traversal.
    ingestion_mod_floor:        float = 0.50

    # Health-level modifiers applied to effective_edge_confidence.
    health_mod_good:            float = 1.00
    health_mod_degraded:        float = 0.90
    health_mod_poor:            float = 0.75

    # trust_floor : minimum effective_edge_confidence after all modifiers.
    #   Prevents complete traversal suppression even in worst-case regions.
    trust_floor:                float = 0.40

    # retrieval_trust_floor : minimum multiplier for post-RRF trust attenuation.
    #   Weakly-ingested files still contribute to retrieval; they are just ranked lower.
    retrieval_trust_floor:      float = 0.60

    # weak_ingestion_warn : avg shortlist ingestion below this threshold appends
    #   "weak_ingestion_region" to the orchestration confidence degradation_chain.
    weak_ingestion_warn:        float = 0.55

    # ── OWNERSHIP: TRUST NORMALIZATION + DEGRADATION CEILING ──────────────────
    # Consumer: query_confidence.compute_confidence() — Component 7 ingestion
    # and the post-composite ceiling check.
    # These prevent double-penalization (same catalog weakness hitting both
    # retrieval attenuation AND ingestion component) and cap total trust damage.
    # Modifying these changes score ceilings and normalization strength only.

    # Phase 7: Soft attenuation, trust normalization, calibration governance

    # edge_attenuation_strength : scales the compound trust penalty in
    #   effective_edge_confidence.
    #   Formula: effective = rel_conf × (1 − α × (1 − ing_mod × health_mod))
    #   1.0 = full multiplicative penalty (Phase 6 behavior).
    #   0.60 = only 60% of the theoretical compound penalty is applied —
    #   prevents aggressive collapse from compounded weak-ingestion + health hit.
    #   Explainer: a region with ing_mod=0.5, health_mod=0.75 produced a
    #   combined modifier of 0.375 before (62.5% penalty); with α=0.60 the
    #   effective modifier is 1−0.60×0.625 = 0.625 (37.5% penalty, above floor).
    edge_attenuation_strength:      float = 0.60

    # retrieval_attenuation_strength : scales the ingestion penalty in
    #   retrieval_trust_weight (post-RRF score multiplier).
    #   Formula: weight = max(floor, 1 − β × (1 − ingestion_score))
    #   0.50 = only half the penalty is applied to RRF scores, since graph
    #   traversal already absorbed part of the same signal via
    #   effective_edge_confidence.  Prevents same weakness from fully
    #   collapsing both graph scores AND retrieval scores simultaneously.
    retrieval_attenuation_strength: float = 0.50

    # trust_normalization_factor : discount applied to the ingestion component's
    #   deficit-below-neutral when retrieval is already trust-attenuated.
    #   0.0 = no normalization (full ingestion penalty).
    #   1.0 = full normalization (ingestion always reads as neutral).
    #   0.65 = 65% of the deficit below ingestion_neutral is preserved; 35%
    #   is forgiven to prevent the same underlying catalog weakness from
    #   penalizing ingestion_component AND (indirectly) retrieval_component
    #   simultaneously at full strength.
    trust_normalization_factor:     float = 0.65

    # max_trust_degradation : absolute ceiling on how much trust-sourced signals
    #   (ingestion quality + graph health) can reduce the composite confidence
    #   score.  Any combined trust penalty above this value is credited back.
    #   Prevents correlated catalog-wide weakness (poor ingestion AND poor graph
    #   health) from collapsing the composite score even when retrieval and
    #   resolver signals are strong.
    #   Derivation: max possible trust impact =
    #     w_ingestion×1.0 + w_health×1.0 = 0.10 + 0.05 = 0.15
    #   Setting 0.12 caps at ~80% of theoretical maximum, preserving a floor.
    max_trust_degradation:          float = 0.12

    # ── OWNERSHIP: CALIBRATION GOVERNANCE + TELEMETRY ESCALATION ──────────────
    # Consumer: build_policy_snapshot(), orchestration_trace.emit()
    # These fields control trace fidelity, calibration version lineage,
    # and the global confidence floor.  They do NOT alter retrieval or
    # graph traversal.  Changes here affect telemetry and replay correctness.

    # Phase 8: Calibration stabilization

    # minimum_viable_confidence : absolute floor applied to the composite score
    #   AFTER all weighted components and trust modifiers.  Even if every trust
    #   signal is simultaneously degraded, the orchestration layer still reports
    #   a non-zero confidence so callers can distinguish "answered with low
    #   confidence" from "could not compute confidence".
    #   0.25 = one standard deviation below random, but well above a no-signal
    #   baseline.  Preserves correct ordering: a query touching only
    #   well-ingested files will always score above 0.25 due to retrieval and
    #   resolver components; only catalog-wide failure triggers this floor.
    minimum_viable_confidence:      float = 0.25

    # escalation_trace_threshold : when the composite score falls below this
    #   value the orchestration trace is emitted at full fidelity even if the
    #   normal compaction heuristic would have triggered a summarised trace.
    #   Trades log volume for complete debugging context in the most critical
    #   failure cases.  Distinct from minimum_viable_confidence — escalation
    #   is a telemetry behaviour only; it never alters execution.
    escalation_trace_threshold:     float = 0.35

    # calibration_version : policy-generation tag emitted into every
    #   policy_snapshot trace event.  Lets offline calibration pipelines
    #   correlate observed score distributions to specific policy generations
    #   without parsing individual thresholds.
    #   Increment this string when calibration parameters change.
    calibration_version:            str   = "8.0"


@lru_cache(maxsize=1)
def get_confidence_policy() -> ConfidencePolicy:
    """Return the module-level singleton ConfidencePolicy."""
    return ConfidencePolicy()

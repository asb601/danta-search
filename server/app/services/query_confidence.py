"""Orchestration confidence scoring — deterministic, not LLM.

PURPOSE:
  After all pipeline stages complete, compute a single composite score that
  answers: "How trustworthy is this analytical result?"

  This is NOT:
    - LLM output confidence
    - Model probability / perplexity
    - Retrieval relevance score

  This IS:
    - Structural trustworthiness of the orchestration pipeline
    - A signal for telemetry, UI warnings, and evaluation pipelines
    - A bounded, deterministic, explainable score

DIMENSIONS (each contributes to the 0.0–1.0 composite):
  1. retrieval_confidence  — quality of the top-ranked retrieved file.
                             High RRF score = clear signal; low = noisy.
  2. graph_confidence      — quality of approved joins used (p50 confidence).
                             No joins available = independent analysis penalty.
  3. resolver_confidence   — top entity resolution candidate confidence.
                             Low resolver hits = uncertain entity mapping.
  4. execution_complexity  — penalty for multi-cluster or independent execution.
                             single_joined = max; independent_analyses = penalty.
  5. repair_penalty        — deduction per SQL repair attempt.
                             Repairs = the original SQL was wrong.
  6. graph_health_penalty  — deduction when graph_health is "degraded" or "poor".

SCORING:
  score = Σ (weight_i × component_i), clamped to [0.0, 1.0]

  Low confidence:
    - Emits telemetry (pipeline_logger warning)
    - Stored in orchestration trace
    - Does NOT block execution

DESIGN CONSTRAINTS:
  - Zero I/O, zero LLM calls, pure computation.
  - All inputs come from already-computed pipeline state.
  - Weights and thresholds are in a single config block.
  - Score is reproducible: same inputs → same score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logger import pipeline_logger
from app.policies.confidence_policy import get_confidence_policy as _get_confidence_policy

# ── Scoring weights and thresholds (governed by ConfidencePolicy) ───────────────────
# See server/app/policies/confidence_policy.py for rationale on each value.
_cp = _get_confidence_policy()
_W_RETRIEVAL   = _cp.w_retrieval
_W_GRAPH       = _cp.w_graph
_W_RESOLVER    = _cp.w_resolver
_W_COMPLEXITY  = _cp.w_complexity
_W_REPAIR      = _cp.w_repair
_W_HEALTH      = _cp.w_health
_W_INGESTION   = _cp.w_ingestion   # Phase 6

assert abs(
    _W_RETRIEVAL + _W_GRAPH + _W_RESOLVER + _W_COMPLEXITY
    + _W_REPAIR + _W_HEALTH + _W_INGESTION - 1.0
) < 1e-9

# ── Low-confidence threshold ──────────────────────────────────────────────────
_LOW_CONFIDENCE_THRESHOLD   = _cp.low_confidence_threshold

# ── Repair penalty per attempt ──────────────────────────────────────────────────
_REPAIR_PENALTY_PER_ATTEMPT = _cp.repair_penalty_per_attempt

# ── Execution strategy complexity scores ──────────────────────────────────────────
_COMPLEXITY_SCORES = {
    "single_joined":        _cp.complexity_single_joined,
    "multi_cluster":        _cp.complexity_multi_cluster,
    "independent_analyses": _cp.complexity_independent,
    "schema_driven":        1.0,
}

# ── Scoring formula version ───────────────────────────────────────────────────
# Tracks scoring SEMANTICS — changes ONLY when the formula changes (e.g.,
# new component added, aggregation scheme redesigned, normalization approach
# replaced).  Distinct from ConfidencePolicy.calibration_version, which
# tracks threshold/parameter changes.  BOTH are required for deterministic
# offline replay: same formula_version + same calibration_version + same
# replay_inputs → identical score.
_FORMULA_VERSION: str = "8.0-stable"


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class ConfidenceScore:
    """
    Composite orchestration confidence for one pipeline invocation.

    All component scores are 0.0–1.0. Final score is weighted sum clamped
    to [0.0, 1.0].
    """
    # Component scores
    retrieval_component:   float = 0.0
    graph_component:       float = 0.0
    resolver_component:    float = 0.0
    complexity_component:  float = 0.0
    repair_component:      float = 0.0
    health_component:      float = 0.0

    # Composite
    score:        float = 0.0    # weighted sum, [0.0, 1.0]
    level:        str   = "high"  # "high" | "medium" | "low"

    # Phase 6: ingestion trust dimension
    ingestion_component: float = 0.0

    # Phase 6: ordered list of reason codes explaining confidence degradation.
    # Empty when the pipeline is operating cleanly.
    degradation_chain: list[str] = field(default_factory=list)

    # Evidence (for trace)
    signals: list[str] = field(default_factory=list)

    # Phase 8: which trust modifiers were active and their combined impact.
    # Telemetry only — never used for routing or execution decisions.
    modifier_breakdown: dict = field(default_factory=dict)

    # Phase 8: exact numerical inputs to compute_confidence() that, together
    # with calibration_version, deterministically reproduce this score.
    replay_inputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score":                 round(self.score, 3),
            "level":                 self.level,
            "components": {
                "retrieval":   round(self.retrieval_component, 3),
                "graph":       round(self.graph_component, 3),
                "resolver":    round(self.resolver_component, 3),
                "complexity":  round(self.complexity_component, 3),
                "repair":      round(self.repair_component, 3),
                "health":      round(self.health_component, 3),
                "ingestion":   round(self.ingestion_component, 3),
            },
            "signals":           self.signals,
            "degradation_chain": self.degradation_chain,
            "modifier_breakdown": self.modifier_breakdown,
            "replay_inputs":      self.replay_inputs,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_confidence(
    *,
    retrieved_with_scores: list[tuple[Any, float]],
    sql_ctx: Any,           # SQLContext
    entity_resolution: dict[str, list[Any]],
    exec_strategy: Any,     # ExecutionStrategy
    repair_attempts: int,   # total SQL repair attempts made (0 = no repairs)
    graph_health: Any,      # GraphHealthScore (from graph_health.py)
) -> ConfidenceScore:
    """
    Compute orchestration confidence from pipeline-stage outputs.

    All arguments are already computed — this function does zero I/O.
    Never raises.
    """
    cs = ConfidenceScore()
    signals: list[str] = []

    try:
        # Scalar inputs stored for replay_inputs at the end of the try block.
        # Initialized to sentinel values so they are always defined even when
        # the corresponding branch is not taken.
        _top_rrf:   float        = 0.0
        _join_p50:  float | None = None
        _exec_mode: str          = "independent_analyses"

        # ── Component 1: Retrieval confidence ───────────────────────────────────
        # Top RRF score gives the signal strength. Normalise against a
        # “perfect” RRF score: if the same doc appears at rank-1 across 4 lists
        # with k=60: 4 × 1/(60+1) ≈ 0.066. We normalise so top score maps to 1.0.
        # In practice most top scores are 0.02–0.07; scale so 0.05 → ~0.75.
        if retrieved_with_scores:
            top_rrf = retrieved_with_scores[0][1]
            _top_rrf = top_rrf
            # Sigmoid-like normalisation: score/(score + 0.02) gives ~0.71 at 0.05
            cs.retrieval_component = min(1.0, top_rrf / (top_rrf + 0.02))
        else:
            cs.retrieval_component = 0.3    # fallback path fired — weaker signal
            signals.append("retrieval_fallback")

        # ── Component 2: Graph confidence ─────────────────────────────────────
        # Use approved-join confidence p50 as the graph signal.
        # If no approved joins: execution is independent — graph contributes 0.4
        # (not zero, because independent execution is legitimate, just weaker).
        approved_joins = list(getattr(sql_ctx, "approved_joins", []) or [])
        if approved_joins:
            confs = [j.confidence for j in approved_joins]
            p50 = sorted(confs)[len(confs) // 2]
            _join_p50 = p50
            cs.graph_component = min(1.0, p50)  # confidence is already 0–1
        else:
            cs.graph_component = 0.40
            signals.append("no_approved_joins")

        # ── Component 3: Resolver confidence ──────────────────────────────────
        # Take the max confidence across all resolver candidates.
        # If no entities to resolve (non-entity query), score is neutral (0.8).
        # If entities were requested but nothing found, score is low (0.3).
        if not entity_resolution:
            cs.resolver_component = 0.80    # no entity query — not penalised
        else:
            top_conf: float = 0.0
            miss_count = 0
            for candidates in entity_resolution.values():
                if not candidates:
                    miss_count += 1
                else:
                    top_conf = max(top_conf, max(
                        getattr(c, "confidence", 0.0) for c in candidates
                    ))
            if miss_count > 0:
                signals.append(f"resolver_misses:{miss_count}")
            cs.resolver_component = top_conf if top_conf > 0 else 0.30

        # ── Component 4: Execution strategy complexity ─────────────────────────
        mode = getattr(exec_strategy, "mode", "independent_analyses")
        _exec_mode = mode
        cs.complexity_component = _COMPLEXITY_SCORES.get(mode, 0.50)
        if mode == "independent_analyses":
            signals.append("independent_analyses")
        elif mode == "multi_cluster":
            signals.append("multi_cluster")

        # ── Component 5: Repair penalty ───────────────────────────────────────
        # Each repair attempt = the query engine rejected the original SQL.
        # Start at 1.0 and deduct per attempt.
        cs.repair_component = max(0.0, 1.0 - repair_attempts * _REPAIR_PENALTY_PER_ATTEMPT)
        if repair_attempts > 0:
            signals.append(f"sql_repairs:{repair_attempts}")

        # ── Component 6: Graph health ─────────────────────────────────────────
        health_level = getattr(graph_health, "health_level", "good")
        if health_level == "good":
            cs.health_component = 1.0
        elif health_level == "degraded":
            cs.health_component = 0.60
            signals.append("graph_health_degraded")
        else:
            cs.health_component = 0.20
            signals.append("graph_health_poor")

        # ── Component 7: Ingestion trust (Phase 6 / 7) ─────────────────────────
        # Average ingestion_confidence_score across shortlisted files.
        # Pre-Phase-5 files (None score) are neutral at ingestion_neutral (0.70).
        #
        # Phase 7 trust normalization: the ingestion signal is already partially
        # embedded in retrieval scores (via trust-attenuated RRF) and in graph
        # traversal scores (via effective_edge_confidence).  To prevent the same
        # underlying catalog weakness from penalizing the composite through two
        # correlated channels simultaneously, normalize the deficit-below-neutral:
        #   if avg_ing < ingestion_neutral:
        #     effective_ingestion = neutral − (neutral − avg_ing) × norm_factor
        # Above-neutral scores are applied without discount (good ingestion counts).
        from app.services.trust_propagation import (  # noqa: PLC0415
            avg_ingestion_confidence as _avg_ing,
            build_degradation_chain as _build_chain,
        )
        meta_list = [
            m for m, _ in (retrieved_with_scores or [])
            if hasattr(m, "ingestion_confidence_score")
        ]
        avg_ing = _avg_ing(meta_list)
        if avg_ing < _cp.ingestion_neutral:
            # Normalize: apply trust_normalization_factor of the deficit so that
            # the same catalog weakness does not penalize retrieval AND ingestion
            # simultaneously at full strength.
            _deficit   = _cp.ingestion_neutral - avg_ing
            _norm_ing  = _cp.ingestion_neutral - _deficit * _cp.trust_normalization_factor
        else:
            _norm_ing = avg_ing  # above-neutral: full credit, no discount
        cs.ingestion_component = _norm_ing
        if avg_ing < _cp.weak_ingestion_warn:
            signals.append(f"weak_ingestion_region:{round(avg_ing, 2)}")

        # ── Composite ─────────────────────────────────────────────────────────
        cs.score = (
            _W_RETRIEVAL    * cs.retrieval_component
            + _W_GRAPH      * cs.graph_component
            + _W_RESOLVER   * cs.resolver_component
            + _W_COMPLEXITY * cs.complexity_component
            + _W_REPAIR     * cs.repair_component
            + _W_HEALTH     * cs.health_component
            + _W_INGESTION  * cs.ingestion_component
        )

        # ── Phase 7: Trust degradation ceiling ────────────────────────────────
        # After normalization, a correlated catalog failure (poor ingestion +
        # poor graph health simultaneously) can still compound.  Cap total
        # trust-sourced penalty so non-trust factors (retrieval recall, join
        # quality, resolver) are fully preserved.
        # Trust-sourced channels: ingestion_component + health_component.
        _trust_impact = (
            _W_INGESTION * max(0.0, 1.0 - cs.ingestion_component)
            + _W_HEALTH  * max(0.0, 1.0 - cs.health_component)
        )
        _trust_ceiling_fired = _trust_impact > _cp.max_trust_degradation
        _ceiling_credit_pts  = 0.0
        if _trust_ceiling_fired:
            _ceiling_credit_pts = _trust_impact - _cp.max_trust_degradation
            cs.score = cs.score + _ceiling_credit_pts  # credit back the excess
            signals.append(f"trust_ceiling:{round(_trust_impact, 3)}")

        cs.score = round(max(0.0, min(1.0, cs.score)), 3)
        # ── Phase 8: Global confidence floor ────────────────────────────────────
        # Prevents total orchestration collapse.  Even simultaneous catalog-wide
        # failures (poor ingestion + poor graph health + degraded retrieval) will
        # not push the composite below this floor.  Non-trust components (resolver,
        # complexity) are preserved as long as they contribute any positive signal.
        # Applied AFTER trust ceiling so the two bounds are additive-safe.
        _floor_applied = cs.score < _cp.minimum_viable_confidence
        if _floor_applied:
            cs.score = round(_cp.minimum_viable_confidence, 3)
            signals.append(f"minimum_viable_floor:{_cp.minimum_viable_confidence}")

        # ── Phase 8: Modifier breakdown (calibration observability) ────────────
        # Captures which modifiers fired and their interaction magnitudes so
        # operators can answer "why did confidence change?" without tracing
        # through all scoring stages manually.
        _norm_forgiveness_pts = (
            round(
                _W_INGESTION
                * (_cp.ingestion_neutral - avg_ing)
                * (1.0 - _cp.trust_normalization_factor),
                4,
            )
            if avg_ing < _cp.ingestion_neutral else 0.0
        )
        cs.modifier_breakdown = {
            "trust_normalization_applied":         avg_ing < _cp.ingestion_neutral,
            "trust_ceiling_applied":               _trust_ceiling_fired,
            "minimum_viable_floor_applied":        _floor_applied,
            "raw_avg_ingestion":                   round(avg_ing, 3),
            "normalized_ingestion":                round(_norm_ing, 3),
            "trust_impact":                        round(_trust_impact, 3),
            # Interaction amounts: score points affected by each modifier.
            # Enables offline attribution: sum of these explains composite delta.
            "normalization_forgiveness_score_pts":  _norm_forgiveness_pts,
            "degradation_ceiling_credit_score_pts": round(_ceiling_credit_pts, 4),
        }

        # ── Phase 8: Replay inputs (calibration replay readiness) ────────────
        # The exact numerical inputs needed, together with calibration_version,
        # to reproduce this confidence score without re-running the pipeline.
        cs.replay_inputs = {
            "avg_ingestion_raw":   round(avg_ing, 4),
            "top_rrf_score":       round(_top_rrf, 6) if retrieved_with_scores else None,
            "repair_attempts":     repair_attempts,
            "health_level":        health_level,
            "execution_mode":      _exec_mode,
            "has_approved_joins":  bool(approved_joins),
            "join_p50":            round(_join_p50, 4) if _join_p50 is not None else None,
            "shortlist_size":      len(retrieved_with_scores) if retrieved_with_scores else 0,
            # Version lineage: BOTH are needed for deterministic offline replay.
            # calibration_version tracks threshold/parameter changes.
            # formula_version tracks scoring-semantic changes (formula redesigns).
            "formula_version":     _FORMULA_VERSION,
            "calibration_version": _cp.calibration_version,
        }
        # ── Level ─────────────────────────────────────────────────────────────
        if cs.score >= 0.70:
            cs.level = "high"
        elif cs.score >= 0.50:
            cs.level = "medium"
        else:
            cs.level = "low"
            signals.append("low_orchestration_confidence")

        cs.signals = signals

        # ── Degradation chain (Phase 6) ────────────────────────────────────────
        # Ordered list of reason codes explaining WHY confidence degraded.
        cs.degradation_chain = _build_chain(
            avg_ingestion=avg_ing,
            graph_component=cs.graph_component,
            repair_attempts=repair_attempts,
            health_level=health_level,
            retrieval_component=cs.retrieval_component,
        )

        # ── Telemetry ─────────────────────────────────────────────────────────
        if cs.score < _LOW_CONFIDENCE_THRESHOLD:
            pipeline_logger.warning(
                "low_orchestration_confidence",
                score=cs.score,
                level=cs.level,
                signals=signals,
                degradation_chain=cs.degradation_chain,
            )

    except Exception:
        pass  # never raise — confidence is telemetry

    return cs

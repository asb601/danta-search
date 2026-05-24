"""Deterministic trust propagation formulas — zero I/O, zero LLM.

PURPOSE
=======
Propagate ingestion-level and graph-level trust signals through three
orchestration layers so that weak metadata regions naturally yield lower
confidence rather than failing silently.

  1. Graph expansion  — effective_edge_confidence()
     Reduces traversal score when either endpoint was weakly ingested or
     the overall graph health is degraded.

  2. Retrieval fusion — retrieval_trust_weight()
     Post-RRF multiplier that attenuates scores for files whose ingestion
     confidence is below the neutral baseline.

  3. Orchestration confidence — avg_ingestion_confidence()
     Computes the mean ingestion quality across shortlisted files for use
     as the `ingestion_component` dimension in the composite score.

DESIGN CONSTRAINTS
==================
- Deterministic: same inputs → same output.
- Bounded:       all outputs are clamped to [0.0, 1.0].
- Explainable:   each formula is a simple product / average — no hidden state.
- Lightweight:   zero DB queries, zero LLM calls, pure Python math.
- Policy-governed: all floors, neutrals, and modifiers come from ConfidencePolicy.
  Tuning a threshold → change one field in ConfidencePolicy, nothing else.

FORMULAS
========
effective_edge_confidence (Phase 7 — soft attenuation):
  ingestion_mod = max(ingestion_mod_floor, min(ing_a, ing_b))
              (weakest endpoint dominates — chain is only as strong as
               its weakest link; pre-Phase-5 files use ingestion_neutral)
  health_mod    = { good→1.00, degraded→0.90, poor→0.75 }  (default: good)

  Phase 6 (full product):
    effective = clamp(rel_conf × ing_mod × h_mod, trust_floor, 1.0)
    Risk: triple product could collapse to trust_floor for borderline inputs.

  Phase 7 (soft attenuation):
    combined_penalty = 1.0 − ing_mod × h_mod           (0 = perfect, ~0.625 = worst)
    softened_mod     = 1.0 − α × combined_penalty       (α = edge_attenuation_strength)
    effective        = clamp(rel_conf × softened_mod, trust_floor, 1.0)
    At α=0.60: worst-case modifier rises from 0.375 → 0.625 (prevents collapse).

retrieval_trust_weight (Phase 7 — soft curve):
  weight = max(retrieval_trust_floor, 1.0 − β × (1.0 − ingestion_score))
           (β = retrieval_attenuation_strength; pre-Phase-5 files → 1.0 no penalty)
  At β=0.50: ingestion_score=0.40 yields weight 0.80 instead of the hard
  retrieval_trust_floor=0.60 — prevents the same weakness from fully
  collapsing both graph traversal scores AND post-RRF retrieval scores.

avg_ingestion_confidence:
  mean of ingestion_confidence_score across shortlist;
  None values treated as ingestion_neutral so legacy files are not penalised.

build_degradation_chain:
  Produces a list of human-readable reason codes that explain WHY the
  orchestration confidence degraded.  Used in `ConfidenceScore.degradation_chain`
  and emitted into the orchestration trace.

  Reason codes (all additive, never replace one another):
    weak_ingestion_region       avg shortlist ingestion < weak_ingestion_warn
    low_edge_confidence         graph_component < 0.50 (weak join evidence)
    repair_applied:<n>          SQL engine required n repairs to run the query
    degraded_graph_health       graph health level = "degraded"
    poor_graph_health           graph health level = "poor"
    poor_retrieval_signal       retrieval_component < 0.40 (noisy retrieval)
"""
from __future__ import annotations

from typing import Any

from app.policies.confidence_policy import get_confidence_policy as _get_cp


# ── Public API ─────────────────────────────────────────────────────────────────

def effective_edge_confidence(
    rel_conf: float,
    ing_a: float | None,
    ing_b: float | None,
    health_level: str = "good",
) -> float:
    """
    Compute trust-adjusted edge confidence for one graph hop.

    Parameters
    ----------
    rel_conf     : raw relationship confidence score (0.0–1.0)
    ing_a        : ingestion_confidence_score for file_a endpoint (None = pre-Phase-5)
    ing_b        : ingestion_confidence_score for file_b endpoint (None = pre-Phase-5)
    health_level : "good" | "degraded" | "poor"  (from GraphHealthScore.health_level)

    Returns
    -------
    float in [trust_floor, 1.0] — attenuated by ingestion quality and graph health.
    """
    cp = _get_cp()
    neutral = cp.ingestion_neutral

    a = ing_a if ing_a is not None else neutral
    b = ing_b if ing_b is not None else neutral
    ing_mod = max(cp.ingestion_mod_floor, min(a, b))

    if health_level == "good":
        h_mod = cp.health_mod_good
    elif health_level == "degraded":
        h_mod = cp.health_mod_degraded
    else:
        h_mod = cp.health_mod_poor

    # Phase 7: Soft attenuation — apply only `edge_attenuation_strength`
    # fraction of the theoretical compound penalty, preventing cascading
    # multiplication from collapsing healthy edges near the trust_floor.
    # Formula: effective = rel_conf × (1 − α × (1 − ing_mod × h_mod))
    # α=0.60: worst-case modifier rises from 0.375 → 0.625 (above trust_floor).
    combined_penalty = 1.0 - ing_mod * h_mod
    softened_mod = 1.0 - cp.edge_attenuation_strength * combined_penalty
    effective = rel_conf * softened_mod
    return max(cp.trust_floor, min(1.0, effective))


def retrieval_trust_weight(ingestion_score: float | None) -> float:
    """
    Post-RRF score multiplier for one file based on its ingestion quality.

    A file with perfect ingestion (1.0) → weight 1.0  (no attenuation).
    A file with poor ingestion (0.40)  → weight max(floor, 0.80) at β=0.50.
    A file with no ingestion score     → weight 1.0   (neutral; legacy file).

    Phase 7 — soft curve (prevents the same weakness from fully collapsing
    both graph traversal scores AND retrieval scores simultaneously):
      weight = max(retrieval_trust_floor, 1.0 − β × (1.0 − ingestion_score))
      β = retrieval_attenuation_strength (default 0.50)

    Parameters
    ----------
    ingestion_score : FileMetadata.ingestion_confidence_score (None = pre-Phase-5)

    Returns
    -------
    float in [retrieval_trust_floor, 1.0]
    """
    if ingestion_score is None:
        return 1.0  # no penalty for pre-Phase-5 files
    cp = _get_cp()
    # Soft curve: only apply a fraction of the raw penalty so that the same
    # ingestion weakness does not simultaneously crush both graph traversal
    # (already absorbed via effective_edge_confidence) and retrieval scores.
    soft_weight = 1.0 - cp.retrieval_attenuation_strength * (1.0 - ingestion_score)
    return max(cp.retrieval_trust_floor, min(1.0, soft_weight))


def avg_ingestion_confidence(
    file_metas: list[Any],
    *,
    neutral: float | None = None,
) -> float:
    """
    Mean ingestion_confidence_score across a list of FileMetadata objects.

    Pre-Phase-5 files (None score) are treated as `neutral` to avoid
    artificially deflating the score for mixed catalogs.

    Parameters
    ----------
    file_metas : list of FileMetadata (or any object with
                 `.ingestion_confidence_score` attribute)
    neutral    : fallback score for None values.
                 Defaults to ConfidencePolicy.ingestion_neutral (0.70).

    Returns
    -------
    float in [0.0, 1.0]  — 0.70 if list is empty (neutral assumption).
    """
    cp = _get_cp()
    _neutral = neutral if neutral is not None else cp.ingestion_neutral

    if not file_metas:
        return _neutral

    scores = [
        (getattr(m, "ingestion_confidence_score", None) or _neutral)
        for m in file_metas
    ]
    return sum(scores) / len(scores)


def build_degradation_chain(
    *,
    avg_ingestion: float,
    graph_component: float,
    repair_attempts: int,
    health_level: str,
    retrieval_component: float,
) -> list[str]:
    """
    Produce an ordered list of reason codes explaining confidence degradation.

    Each code is a human-readable token that maps to a specific pipeline
    issue.  The list is empty when all signals are healthy.  Used in the
    orchestration trace and emitted as structured telemetry.

    Parameters
    ----------
    avg_ingestion      : mean ingestion_confidence_score of shortlisted files
    graph_component    : graph confidence component from ConfidenceScore (0–1)
    repair_attempts    : number of SQL repair attempts made (0 = clean run)
    health_level       : GraphHealthScore.health_level string
    retrieval_component: retrieval component from ConfidenceScore (0–1)

    Returns
    -------
    list[str] — empty when pipeline is operating cleanly.
    """
    cp = _get_cp()
    chain: list[str] = []

    if avg_ingestion < cp.weak_ingestion_warn:
        chain.append("weak_ingestion_region")
    if graph_component < 0.50:
        chain.append("low_edge_confidence")
    if repair_attempts > 0:
        chain.append(f"repair_applied:{repair_attempts}")
    if health_level == "poor":
        chain.append("poor_graph_health")
    elif health_level == "degraded":
        chain.append("degraded_graph_health")
    if retrieval_component < 0.40:
        chain.append("poor_retrieval_signal")

    return chain

"""v2 RESOLVE contract — pure, deterministic query contract + fallback condition.

This is the v2 RESOLVE contract surface. It is DISTINCT from the GATE-B
governed-join `app/services/contract/` package: that one governs approved-join
enforcement; this one models the per-question analytical CONTRACT (the slots a
query must fill — source / grain / measure / filters / joins) and the
deterministic condition under which the runtime must DROP TO FALLBACK (the
LangGraph agent path) instead of trusting a high-confidence plan.

Design properties (enforced, not aspirational):
  * Pure Python only. No LLM, no DB, no I/O of any kind in this module — every
    function here is a deterministic function of its inputs.
  * No hardcoded business heuristics. There are no column-name lists, ERP
    dictionaries, or dataset-fitted literals. The two scoring knobs — the
    unverified-twin penalty (`Contract.twin_penalty`) and the fallback
    confidence cutoff (`needs_fallback(tau_conf=...)`) — are injectable
    parameters whose defaults the caller overrides from `SemanticPolicy`
    (`unverified_twin_penalty` / `resolve_contract_tau`) at wiring time, so no
    tuned constant is baked into the scoring logic.
  * Additive. Nothing here is wired into `graph.py`. Activation is gated by the
    `RESOLVE_CONTRACT_ENABLED` flag (config.py), default False, so the runtime
    is byte-identical until a caller opts in.

A `Contract` is built upstream from precomputed ingestion artifacts (semantic
roles, the relationship graph, the column-key registry, verification results);
this module only SCORES it (`Contract.confidence`) and DECIDES whether the plan
is trustworthy (`needs_fallback`) or whether a returned result set is
implausible enough to retry (`needs_post_exec_fallback`).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class SlotState(str, Enum):
    BOUND = "bound"        # resolved to a concrete artifact AND verified on data
    PROPOSED = "proposed"  # resolved but verify_* not yet run / not passed
    UNFILLED = "unfilled"  # no candidate above threshold


@dataclass(frozen=True)
class JoinSlot:
    left_table: str; left_col: str
    right_table: str; right_col: str
    relationship_id: str | None
    role_ok: bool
    containment: float
    fanout: float
    verified: bool


@dataclass(frozen=True)
class Contract:
    question: str
    entity: str
    source_table: str | None
    source_state: SlotState
    grain: str | None
    grain_pk: tuple[str, ...]
    grain_state: SlotState
    measure: str | None
    measure_expr: str | None
    measure_state: SlotState
    filter_preds: tuple[str, ...]
    filter_state: SlotState
    joins: tuple[JoinSlot, ...] = ()
    twin_cluster: tuple[str, ...] = ()
    canonical_verified: bool = False
    candidate_tables: tuple[str, ...] = ()
    facts: dict = field(default_factory=dict)
    # Scoring knob, not a tuned constant: the caller injects this from
    # SemanticPolicy.unverified_twin_penalty at construction time. Default kept
    # here only so the dataclass is usable standalone in tests.
    twin_penalty: float = 0.5

    @property
    def confidence(self) -> float:
        slots = [self.source_state, self.grain_state, self.measure_state, self.filter_state]
        bound = sum(s == SlotState.BOUND for s in slots)
        base = bound / 4.0
        if self.joins:
            base *= (sum(j.verified for j in self.joins) / len(self.joins))
        if self.twin_cluster and not self.canonical_verified:
            base *= self.twin_penalty
        return round(base, 4)


def needs_fallback(c: Contract, *, tau_conf: float = 0.75) -> tuple[bool, str]:
    if c.measure_state == SlotState.UNFILLED or c.measure is None:
        return True, "no_governed_metric_match"
    if len(c.twin_cluster) >= 2 and not c.canonical_verified:
        return True, "twin_cluster_no_verified_master"
    if any(not j.verified for j in c.joins):
        return True, "required_join_unverified"
    if any(s != SlotState.BOUND for s in
           (c.source_state, c.grain_state, c.measure_state, c.filter_state)) \
       or c.confidence < tau_conf:
        return True, "contract_slot_below_threshold"
    return False, "complete_verified_contract"


def needs_post_exec_fallback(c: Contract, row_count: int, has_coverage: bool) -> tuple[bool, str]:
    # Trigger 5: 0 rows where the data demonstrably has coverage for the requested scope.
    if row_count == 0 and has_coverage:
        return True, "zero_rows_with_coverage"
    return False, "ok"

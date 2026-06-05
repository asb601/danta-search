"""Phase 4 Task 5 — grain alignment + numeric reconciliation.

A PDF *fact* (e.g. a contract line: ``rate`` × ``volume`` for a given period and
unit) only proves a CSV *aggregate* (a precomputed total) when three things line
up: the **period** matches, the **unit** matches, and the implied amount
(``rate × volume``) reconciles the aggregate ``total`` within a tunable numeric
tolerance. This is the join-proof step: it never asserts equality on names — only
on aligned, value-evidenced numbers.

Pure module — no DB, no infra. Every threshold flows through
``pdf_chat.tunables`` and the numeric check is emitted via ``log_gate_decision``
so a score is never compared-and-discarded silently (Spec §3 invariant 4). No
score-comparison literal lives here; the tolerance is the named tunable
``grain.numeric_tolerance_pct``.

Imported directly as ``pdf_chat.bridge.grain`` (the ``bridge`` package's
``__init__`` exports are owned by the bridge-core worker; this module does not
depend on them).
"""
from __future__ import annotations

from dataclasses import dataclass

from pdf_chat.tunables import get_tunable, log_gate_decision

# Tunable key (single source: TUNABLE_DEFAULTS["grain.numeric_tolerance_pct"]).
TUN_GRAIN_NUMERIC_TOLERANCE_PCT = "grain.numeric_tolerance_pct"

# Denominator floor for the relative-difference test, so a zero/near-zero
# aggregate total can never divide-by-zero (mirrors the plan's max(1, total)).
_DENOMINATOR_FLOOR = 1.0


@dataclass(frozen=True)
class GrainFact:
    """A PDF-side fact whose implied amount is ``rate * volume``."""

    rate: float
    volume: float
    period: str
    unit: str

    @property
    def amount(self) -> float:
        return self.rate * self.volume


@dataclass(frozen=True)
class GrainAggregate:
    """A CSV-side precomputed aggregate total for a period + unit."""

    total: float
    period: str
    unit: str


@dataclass(frozen=True)
class GrainResult:
    reconciled: bool
    reason: str
    relative_diff: float = 0.0
    tolerance: float = 0.0


def _normalize(token: str) -> str:
    """Case/whitespace-insensitive normalization for period + unit comparison."""
    return (token or "").strip().casefold()


def reconcile_grain(
    *,
    tenant_id: str,
    fact: GrainFact,
    aggregate: GrainAggregate,
) -> GrainResult:
    """Prove a PDF fact reconciles a CSV aggregate at the same grain.

    Steps (all must hold to reconcile):
      1. ``fact.period`` == ``aggregate.period`` (normalized) — else refuse.
      2. ``fact.unit``   == ``aggregate.unit``   (normalized) — else refuse.
      3. ``abs(fact.amount - aggregate.total) / max(1, aggregate.total)``
         <= ``grain.numeric_tolerance_pct`` — logged via ``log_gate_decision``.
    """
    # 1) Period alignment — a fact for a different period proves nothing here.
    if _normalize(fact.period) != _normalize(aggregate.period):
        return GrainResult(
            reconciled=False,
            reason=(
                f"period mismatch: fact period {fact.period!r} "
                f"!= aggregate period {aggregate.period!r}"
            ),
        )

    # 2) Unit alignment — apples-to-oranges currencies/units never reconcile.
    if _normalize(fact.unit) != _normalize(aggregate.unit):
        return GrainResult(
            reconciled=False,
            reason=(
                f"unit mismatch: fact unit {fact.unit!r} "
                f"!= aggregate unit {aggregate.unit!r}"
            ),
        )

    # 3) Numeric reconciliation against the tunable tolerance band.
    tolerance = float(get_tunable(tenant_id, TUN_GRAIN_NUMERIC_TOLERANCE_PCT))
    denominator = max(_DENOMINATOR_FLOOR, abs(aggregate.total))
    relative_diff = abs(fact.amount - aggregate.total) / denominator

    # log_gate_decision treats passed as score >= threshold; we want the SMALL
    # relative_diff to be within tolerance, so negate both sides to keep the
    # canonical comparison (-relative_diff >= -tolerance  <=>  diff <= tolerance).
    decision = log_gate_decision(
        "grain.numeric_reconciliation",
        score=-relative_diff,
        threshold=-tolerance,
        outcome="reconciled" if relative_diff <= tolerance else "refused",
        tenant_id=tenant_id,
        relative_diff=relative_diff,
        tolerance=tolerance,
        fact_amount=fact.amount,
        aggregate_total=aggregate.total,
        period=_normalize(fact.period),
        unit=_normalize(fact.unit),
    )

    if not decision["passed"]:
        return GrainResult(
            reconciled=False,
            reason=(
                f"numeric reconciliation outside tolerance: "
                f"relative_diff {relative_diff:.4f} > tolerance {tolerance:.4f}"
            ),
            relative_diff=relative_diff,
            tolerance=tolerance,
        )

    return GrainResult(
        reconciled=True,
        reason=(
            f"reconciled: relative_diff {relative_diff:.4f} "
            f"<= tolerance {tolerance:.4f}"
        ),
        relative_diff=relative_diff,
        tolerance=tolerance,
    )

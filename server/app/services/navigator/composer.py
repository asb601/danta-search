"""[5] COMPOSE — the ONLY place cross-step arithmetic happens (I9 / I2).

A COMPOSE step does not own a table. It reads PROMOTED scalars from the ledger
(``ledger.get_scalar(step_id)``) and combines them with a deterministic Python
operation. The LLM is NOT involved: I9 says COMPOSE does ALL cross-step math
deterministically, and I2 forbids the LLM from computing any number. So a
"overdue-to-total ratio" is ``overdue / total`` computed HERE, in Python — never
emitted by mini, never relabeled by mini.

Supported ops (operands are step ids; ``left`` = left_step, ``right`` = right_step):

    ratio   left / right
    diff    left - right
    growth  (left - right) / right        # period-over-period change
    share   left / (left + right)         # left's share of the two-part whole

Safety (I12-aligned — never confidently wrong, never crash):
  * a missing upstream step, or an upstream step that produced no scalar
    (``scalar is None``), yields a safe ``StepResult`` with ``scalar=None``.
  * a divide-by-zero (ratio/growth/share denominator == 0) yields ``scalar=None``
    and an ``error_marker`` of ``undefined`` — NEVER raises.
  * an unknown op yields ``scalar=None``.

The navigator is self-contained — this module imports nothing from
``app.services.resolve.*``.
"""
from __future__ import annotations

from typing import Optional

from app.services.navigator.types import ComposePlan, StepLedger, StepResult


def _ratio(left: float, right: float) -> Optional[float]:
    if right == 0:
        return None
    return left / right


def _diff(left: float, right: float) -> Optional[float]:
    return left - right


def _growth(left: float, right: float) -> Optional[float]:
    if right == 0:
        return None
    return (left - right) / right


def _share(left: float, right: float) -> Optional[float]:
    denom = left + right
    if denom == 0:
        return None
    return left / denom


# op name -> (pure binary function, measure-label stem). Adding an op is a row
# here, not a control-flow change. SQL/Python arithmetic only — no LLM.
_OPS = {
    "ratio": (_ratio, "ratio"),
    "diff": (_diff, "diff"),
    "growth": (_growth, "growth"),
    "share": (_share, "share"),
}


def _compose_step_id(plan: ComposePlan) -> str:
    """A synthetic, deterministic id for the compose result node, derived from the
    plan operands so it is stable and traceable."""
    return f"compose:{plan.op}:{plan.left_step}:{plan.right_step}"


def compose(ledger: StepLedger, plan: ComposePlan) -> StepResult:
    """Deterministic cross-step arithmetic over PROMOTED scalars (I9). PURE: reads
    ``ledger.get_scalar`` for both operands and computes in Python. Divide-by-zero,
    a missing/None upstream scalar, or an unknown op all yield a safe
    ``StepResult`` (scalar=None) — NEVER raises."""
    step_id = _compose_step_id(plan)
    op_entry = _OPS.get(plan.op)
    label = f"{(op_entry[1] if op_entry else plan.op)}({plan.left_step},{plan.right_step})"

    if op_entry is None:
        return StepResult(
            step_id=step_id,
            measure_label=label,
            scalar=None,
            error_marker=f"unknown_op: {plan.op}",
        )

    left = ledger.get_scalar(plan.left_step)
    right = ledger.get_scalar(plan.right_step)
    if left is None or right is None:
        missing = [
            sid
            for sid, val in ((plan.left_step, left), (plan.right_step, right))
            if val is None
        ]
        return StepResult(
            step_id=step_id,
            measure_label=label,
            scalar=None,
            error_marker=f"missing_upstream_scalar: {','.join(missing)}",
        )

    fn = op_entry[0]
    value = fn(float(left), float(right))
    if value is None:
        return StepResult(
            step_id=step_id,
            measure_label=label,
            scalar=None,
            error_marker="undefined: divide_by_zero",
        )

    return StepResult(step_id=step_id, measure_label=label, scalar=value)

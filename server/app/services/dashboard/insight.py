"""Phase 4 — deterministic per-widget insight (no LLM).

A pure one-liner computed ONLY from the returned rows + the bound config, gated so
it never states a wrong number:
  - a %-share claim requires an ADDITIVE measure (`summable`) — summing ratios is
    meaningless, so a non-additive measure gets the top contributor without a share;
  - a "top" requires a strict winner (a tie -> no claim);
  - a trend requires >= 2 points.
Returns None whenever it cannot compute an honest headline. Deterministic: a pure
function of (component_type, dataset, config, summable).
"""
from __future__ import annotations

import math


def _num(v) -> bool:
    # Exclude bool (True == 1) and non-finite (inf/nan would crash _fmt's int()).
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _numbers(dataset, col) -> list:
    if not col:
        return []
    return [row[col] for row in dataset if isinstance(row, dict) and _num(row.get(col))]


def _fmt(v, fmt) -> str:
    if fmt == "currency":
        return f"${v:,.0f}"
    if float(v) == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _top(dataset, label, measure):
    """The (name, value) with the strictly-largest measure, or None on tie/empty."""
    pairs = [(row.get(label), row.get(measure)) for row in dataset
             if isinstance(row, dict) and _num(row.get(measure))]
    if not pairs:
        return None
    ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return None  # tie -> no honest "top"
    return ranked[0]


def compute_insight(component_type: str, dataset: list, config: dict, *, summable: bool = False) -> str | None:
    if not dataset or not isinstance(config, dict):
        return None
    fmt = config.get("format")

    if component_type in ("kpi_card", "metric_tile"):
        vals = _numbers(dataset, config.get("value"))
        if not vals:
            return None
        total = sum(vals) if len(vals) > 1 else vals[0]
        return f"Total: {_fmt(total, fmt)}"

    if component_type in ("pie_chart", "bar_chart"):
        label = config.get("label") or config.get("x")
        measure = config.get("value") or config.get("y")
        if isinstance(measure, list):
            measure = measure[0] if measure else None
        top = _top(dataset, label, measure)
        if not top:
            return None
        name, val = top
        # %-share is honest ONLY for an additive measure, and only for the
        # part-of-whole (pie) intent — a bar top-N share would be a partial total.
        if component_type == "pie_chart" and summable:
            total = sum(_numbers(dataset, measure))
            if total > 0:
                return f"{name} is {val / total:.0%} of the total ({_fmt(val, fmt)})."
        return f"{name} leads at {_fmt(val, fmt)}."

    if component_type in ("line_chart", "area_chart"):
        y = config.get("y")
        col = (y[0] if y else None) if isinstance(y, list) else y
        x = config.get("x")
        # Pair (x, y) and SORT BY THE TIME AXIS before taking first/last — the rows
        # are not guaranteed time-ordered, and a DESC/unordered result would invert
        # the direction. Fail closed to None if the x axis isn't orderable.
        pairs = [(row.get(x), row[col]) for row in dataset
                 if isinstance(row, dict) and _num(row.get(col))]
        if len(pairs) < 2:
            return None
        if x is not None:
            try:
                pairs.sort(key=lambda p: p[0])
            except TypeError:
                return None
        delta = pairs[-1][1] - pairs[0][1]
        if delta == 0:
            return "Flat over the period."
        return f"{'Up' if delta > 0 else 'Down'} {_fmt(abs(delta), fmt)} over the period."

    return None

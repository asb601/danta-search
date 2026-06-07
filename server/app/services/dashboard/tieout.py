"""Phase 5 — tie-out reconciliation (the binding number-level catch).

P2's join gate + additivity directive are ADVISORY (the agent can ignore them).
P5 catches the symptom from ALREADY-RETURNED widget data — no second query, no LLM:
for the same ADDITIVE measure + scope, the sum of a breakdown (bar/pie) can never
exceed the KPI total, so `sum(parts) > whole` (beyond tolerance) is a definite
double-count / fan-out symptom (G1) or a SUMmed-ratio symptom (G2).

Demo-safe by construction — ZERO false positives:
- warn ONLY on parts>whole. `parts < whole` is the NORMAL top-N breakdown and is
  never flagged; no positive "reconciled" badge is emitted (it can't be proven).
- reconcile ONLY additive measures (provenance.summable is True), same
  (planned.measure, planned.table), no local comparison filter, KPI-vs-bar/pie.
  Every unprovable case fails closed to silence.
"""
from __future__ import annotations

from app.services.dashboard.insight import _numbers

_TOL = 0.01        # relative: parts may exceed whole by up to 1% (float/rounding drift)
_ABS_FLOOR = 1e-9  # near-zero whole -> not reconcilable (avoid div/ratio noise)

_WHOLE = ("kpi_card", "metric_tile")
_PARTS = ("bar_chart", "pie_chart")


def _measure_total(widget):
    """Extract (key, measure, kind, total) for a reconcilable widget, else None.
    Applies all gates: additivity, stable measure key, no local filter, KPI/bar/pie
    only. Reads the number by the bound config key — never by name-guessing."""
    prov = getattr(widget, "provenance", None) or {}
    if prov.get("summable") is not True:          # additivity gate (fail-closed on absent)
        return None
    planned = (prov.get("spec") or {}).get("planned") or {}
    measure = planned.get("measure")
    if not measure:                                # need a stable grouping key
        return None
    if planned.get("comparison"):                  # local comparison filter -> not board-scope
        return None
    ct = getattr(widget, "component_type", None)
    cfg = getattr(widget, "config", None) or {}
    rows = getattr(widget, "dataset", None) or []

    if ct in _WHOLE:
        vals = _numbers(rows, cfg.get("value"))
        if not vals:
            return None
        total, kind = (vals[0] if len(vals) == 1 else sum(vals)), "whole"
    elif ct == "bar_chart":
        col = cfg.get("y")
        col = (col[0] if col else None) if isinstance(col, list) else col
        vals = _numbers(rows, col)
        if not vals:
            return None
        total, kind = sum(vals), "parts"
    elif ct == "pie_chart":
        vals = _numbers(rows, cfg.get("value"))
        if not vals:
            return None
        total, kind = sum(vals), "parts"
    else:                                           # line/area/table/heatmap/funnel
        return None

    return {"key": (measure, planned.get("table")), "measure": measure, "kind": kind, "total": total}


def reconcile_tieout(widgets, *, tol: float = _TOL) -> tuple[list[str], dict[str, str]]:
    """Return (warnings, badges). badges maps widget_id -> "over" ONLY for breakdowns
    whose parts exceed their KPI total. Pure and deterministic; never raises."""
    warnings: list[str] = []
    badges: dict[str, str] = {}

    groups: dict = {}
    for w in widgets or []:
        info = _measure_total(w)
        if info is not None:
            groups.setdefault(info["key"], []).append((w, info))

    for members in groups.values():
        wholes = [(w, i) for w, i in members if i["kind"] == "whole"]
        parts = [(w, i) for w, i in members if i["kind"] == "parts"]
        if not wholes or not parts:                 # need both a KPI and a breakdown
            continue
        whole = max(i["total"] for _, i in wholes)   # the grand total upper-bounds the parts
        if abs(whole) <= _ABS_FLOOR:
            continue
        for w, i in parts:
            if i["total"] > whole * (1 + tol):
                badges[w.widget_id] = "over"
                warnings.append(
                    f"Widget '{w.title}': the breakdown sums to {i['total']:,.0f}, which exceeds "
                    f"the total {whole:,.0f} for measure '{i['measure']}' — a double-counting symptom "
                    f"(the parts of an additive measure cannot exceed the whole)."
                )
    return warnings, badges

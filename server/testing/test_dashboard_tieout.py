"""Phase 5 — tie-out reconciliation (pure, no DB/LLM). ZERO false positives.

For the SAME additive measure + scope, the sum of a breakdown (bar/pie) can NEVER
exceed the KPI total — so parts>whole (beyond tolerance) is a definite double-count
symptom (warn). parts<whole is the NORMAL top-N case and must never warn. Only
additive (summable) measures, same (measure, table), no local comparison filter,
KPI-vs-bar/pie are reconciled; everything else is left unchecked (no warn).

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_tieout.py -q
"""
from __future__ import annotations

from app.services.dashboard.recommendation_engine import ResolvedWidget
from app.services.dashboard.tieout import reconcile_tieout


def _w(wid, ctype, dataset, config, *, summable=True, measure="revenue",
       table="sales", comparison=None):
    planned = {"measure": measure, "table": table}
    if comparison:
        planned["comparison"] = comparison
    prov: dict = {"spec": {"planned": planned}}
    if summable is not None:
        prov["summable"] = summable
    return ResolvedWidget(widget_id=wid, component_id="c", component_type=ctype,
                          title=f"{ctype}-{wid}", dataset=dataset, config=config,
                          score=1.0, rationale="", provenance=prov)


def test_parts_exceed_whole_warns():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 500}], {"x": "r", "y": "v"})
    warns, badges = reconcile_tieout([kpi, bar])
    assert badges.get("b") == "over"
    assert len(warns) == 1 and "exceeds" in warns[0].lower()


def test_topn_partial_does_not_warn():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 300}], {"x": "r", "y": "v"})
    warns, badges = reconcile_tieout([kpi, bar])
    assert warns == [] and badges.get("b") != "over"


def test_non_additive_not_reconciled():
    kpi = _w("k", "kpi_card", [{"v": 0.85}], {"value": "v"}, summable=False)
    bar = _w("b", "bar_chart", [{"r": "A", "v": 0.9}, {"r": "B", "v": 0.8}], {"x": "r", "y": "v"}, summable=False)
    warns, _ = reconcile_tieout([kpi, bar])   # 1.7 > 0.85 numerically, but non-additive
    assert warns == []


def test_summable_absent_not_reconciled():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"}, summable=None)
    bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 700}], {"x": "r", "y": "v"}, summable=None)
    warns, _ = reconcile_tieout([kpi, bar])   # unproven additivity -> never warn
    assert warns == []


def test_exact_within_tolerance_ok():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart", [{"r": "A", "v": 1000}], {"x": "r", "y": "v"})
    warns, badges = reconcile_tieout([kpi, bar])
    assert warns == [] and badges.get("b") != "over"


def test_tolerance_boundary():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    within, _ = reconcile_tieout([kpi, _w("b1", "bar_chart", [{"r": "A", "v": 1009}], {"x": "r", "y": "v"})])
    assert within == []                                # 1009 <= 1010 (1%)
    over, badges = reconcile_tieout([kpi, _w("b2", "bar_chart", [{"r": "A", "v": 1020}], {"x": "r", "y": "v"})])
    assert len(over) == 1 and badges.get("b2") == "over"   # 1020 > 1010


def test_different_measure_not_grouped():
    rev_kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"}, measure="revenue")
    cost_bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 700}], {"x": "r", "y": "v"}, measure="cost")
    warns, _ = reconcile_tieout([rev_kpi, cost_bar])   # cost bar has no cost KPI peer
    assert warns == []


def test_no_kpi_peer_unchecked():
    bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 700}], {"x": "r", "y": "v"})
    warns, badges = reconcile_tieout([bar])
    assert warns == [] and badges.get("b") != "over"


def test_comparison_widget_excluded():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart", [{"r": "A", "v": 600}, {"r": "B", "v": 700}], {"x": "r", "y": "v"},
             comparison="vs previous period")
    warns, _ = reconcile_tieout([kpi, bar])   # divergent scope -> not a comparable pair
    assert warns == []


def test_line_and_table_excluded():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    line = _w("l", "line_chart", [{"m": "1", "v": 600}, {"m": "2", "v": 700}], {"x": "m", "y": ["v"]})
    table = _w("t", "table", [{"v": 600}, {"v": 700}], {"columns": ["v"]})
    warns, _ = reconcile_tieout([kpi, line, table])   # sums 1300 > 1000 but both excluded
    assert warns == []


def test_pie_parts_exceed_whole_warns():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    pie = _w("p", "pie_chart", [{"l": "A", "v": 600}, {"l": "B", "v": 700}], {"label": "l", "value": "v"})
    warns, badges = reconcile_tieout([kpi, pie])
    assert badges.get("p") == "over" and len(warns) == 1


def test_nan_inf_bool_excluded_from_sum():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart",
             [{"r": "A", "v": 600}, {"r": "B", "v": float("inf")}, {"r": "C", "v": True}],
             {"x": "r", "y": "v"})
    warns, _ = reconcile_tieout([kpi, bar])   # only 600 counts -> 600 < 1000 -> no warn
    assert warns == []


def test_deterministic_and_malformed_never_raises():
    kpi = _w("k", "kpi_card", [{"v": 1000}], {"value": "v"})
    bar = _w("b", "bar_chart", [{"r": "A", "v": 1100}], {"x": "r", "y": "v"})
    assert reconcile_tieout([kpi, bar]) == reconcile_tieout([kpi, bar])
    bad = ResolvedWidget(widget_id="x", component_id="c", component_type="kpi_card",
                         title="bad", dataset=None, config=None, score=0.0, rationale="",
                         provenance={})
    reconcile_tieout([bad])   # must not raise

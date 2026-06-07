"""Phase 4 — honest empty states (G5) + deterministic per-widget insight (pure).

classify_empty distinguishes ERROR (errored/SQL-failed) vs MISSING (no table
resolved) vs EMPTY (a table resolved but 0 rows matched), using the REAL signals
(`error` OR `execution_error`; table resolution — NOT bare files_used), and shows
the table's real loaded coverage range on an EMPTY tile. compute_insight produces a
deterministic one-liner from the returned rows only (no LLM), gating any %-share
claim on the additive `summable` flag, and returns None when it can't compute.

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_empty_and_insight.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.empty_state import classify_empty, EMPTY, MISSING, ERROR
from app.services.dashboard.insight import compute_insight
from app.services.dashboard.query_engine import WidgetIntent


def _intent():
    return WidgetIntent(title="t", nl_query="q")


def _table(coverage=None):
    return SimpleNamespace(date_coverage=lambda: (coverage or []))


# --- classify_empty (corrected signals) -------------------------------------

def test_error_from_error_key():
    state, _ = classify_empty({"error": "boom", "files_used": ["x"]}, _table(), _intent())
    assert state == ERROR


def test_error_from_execution_error_key():
    # SQL-failed-before-materialization surfaces as execution_error, NOT error.
    state, _ = classify_empty(
        {"execution_error": {"status": "failed", "message": "bad sql"}, "files_used": ["x"]},
        _table(), _intent(),
    )
    assert state == ERROR


def test_missing_when_no_table_and_no_files():
    state, msg = classify_empty({"files_used": []}, None, _intent())
    assert state == MISSING
    assert "not available" in msg.lower()


def test_empty_not_missing_when_table_resolved_even_if_files_empty():
    # A resolved table with 0 rows (and no scanned blobs) is EMPTY, never MISSING.
    state, _ = classify_empty({"files_used": []}, _table(), _intent())
    assert state == EMPTY


def test_empty_shows_real_coverage_range():
    state, msg = classify_empty(
        {"files_used": ["x"]}, _table(["posting_date: 2026-04-01 .. 2026-04-30"]), _intent()
    )
    assert state == EMPTY
    assert "2026-04-01" in msg and "2026-04-30" in msg


def test_empty_generic_when_no_coverage():
    state, msg = classify_empty({"files_used": ["x"]}, _table([]), _intent())
    assert state == EMPTY
    assert "no matching data" in msg.lower()


def test_failclosed_on_malformed_result_never_raises():
    state, _ = classify_empty({}, None, _intent())   # {} -> no error, no files, no table
    assert state == MISSING


# --- compute_insight (deterministic, faithful, summable-gated) --------------

def test_kpi_total_currency():
    ins = compute_insight("kpi_card", [{"total_revenue": 12345}],
                          {"value": "total_revenue", "format": "currency"})
    assert "$" in ins and "12,345" in ins


def test_kpi_number_format_has_no_dollar():
    ins = compute_insight("metric_tile", [{"orders": 42}],
                          {"value": "orders", "format": "number"})
    assert "$" not in ins and "42" in ins


def test_pie_share_only_when_summable():
    rows = [{"region": "APAC", "sales": 300}, {"region": "EU", "sales": 700}]
    cfg = {"label": "region", "value": "sales", "format": "number"}
    ins = compute_insight("pie_chart", rows, cfg, summable=True)
    assert "EU" in ins and "70%" in ins


def test_pie_no_share_when_not_summable():
    # A ratio measure: never claim "% of total" (sum of ratios is meaningless).
    rows = [{"region": "APAC", "rate": 0.3}, {"region": "EU", "rate": 0.7}]
    cfg = {"label": "region", "value": "rate", "format": "number"}
    ins = compute_insight("pie_chart", rows, cfg, summable=False)
    assert ins is None or "of the total" not in ins.lower()


def test_bar_top_contributor_no_share():
    rows = [{"region": "APAC", "sales": 300}, {"region": "EU", "sales": 700}]
    cfg = {"x": "region", "y": "sales", "format": "currency"}
    ins = compute_insight("bar_chart", rows, cfg, summable=True)
    assert "EU" in ins and "$700" in ins and "of the total" not in ins.lower()


def test_line_direction_and_delta():
    rows = [{"m": "2026-01", "rev": 100}, {"m": "2026-02", "rev": 150}]
    cfg = {"x": "m", "y": ["rev"], "format": "currency"}
    ins = compute_insight("line_chart", rows, cfg)
    assert "Up" in ins and "50" in ins


def test_insight_none_for_table_and_uncomputable():
    assert compute_insight("table", [{"a": 1}], {"columns": ["a"]}) is None
    assert compute_insight("kpi_card", [], {"value": "x"}) is None          # empty rows
    assert compute_insight("line_chart", [{"m": "1", "rev": 5}], {"x": "m", "y": ["rev"]}) is None  # 1 point
    # bound measure absent from rows -> None (never substitute a different column)
    assert compute_insight("kpi_card", [{"a": 1}], {"value": "missing_col"}) is None


def test_insight_none_on_tie_for_top():
    rows = [{"region": "A", "sales": 500}, {"region": "B", "sales": 500}]
    cfg = {"x": "region", "y": "sales"}
    assert compute_insight("bar_chart", rows, cfg, summable=True) is None


def test_insight_is_deterministic():
    rows = [{"region": "APAC", "sales": 300}, {"region": "EU", "sales": 700}]
    cfg = {"label": "region", "value": "sales", "format": "number"}
    a = compute_insight("pie_chart", rows, cfg, summable=True)
    b = compute_insight("pie_chart", rows, cfg, summable=True)
    assert a == b


def test_insight_number_equals_actual_sum():
    rows = [{"x": "a", "y": 10}, {"x": "b", "y": 20}, {"x": "c", "y": 30}]
    ins = compute_insight("kpi_card", rows, {"value": "y", "format": "number"})
    assert "60" in ins   # exactly sum(10,20,30)


def test_line_sorts_by_x_so_direction_is_correct_when_rows_unordered():
    # Rows returned out of order: after sorting by the time axis the direction must
    # be Up (100 -> 300), not Down from the raw row order.
    rows = [{"m": "2026-03", "rev": 300}, {"m": "2026-01", "rev": 100}, {"m": "2026-02", "rev": 200}]
    ins = compute_insight("line_chart", rows, {"x": "m", "y": ["rev"], "format": "currency"})
    assert "Up" in ins and "200" in ins   # 300 - 100, ascending by month


def test_line_none_when_x_axis_unorderable():
    rows = [{"m": 1, "rev": 100}, {"m": "two", "rev": 200}]   # mixed-type x -> can't order
    assert compute_insight("line_chart", rows, {"x": "m", "y": ["rev"]}) is None


def test_insight_none_on_inf_or_nan_measure():
    assert compute_insight("kpi_card", [{"v": float("inf")}], {"value": "v"}) is None
    assert compute_insight("kpi_card", [{"v": float("nan")}], {"value": "v"}) is None

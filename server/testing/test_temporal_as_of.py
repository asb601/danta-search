"""F1 — temporal as-of anchor + widened date extraction.

The reference 'now' for relative-time resolution must be the data's latest
coverage date (capped at the wall clock), NOT date.today(). Otherwise a correct
parser resolves 'this year' to a period the data does not cover (the OEBS demo:
data ends 2025-05-31, clock is 2026-06).
"""
from datetime import date

from app.services.erp.feasibility_gate import (
    resolve_as_of_date,
    parse_requested_window,
)
from app.services.business_intent_planner import _extract_constraints


# ── resolve_as_of_date ─────────────────────────────────────────────────────────

def test_as_of_anchors_to_latest_data_when_clock_is_ahead():
    catalog = [
        {"date_range_start": "2023-01-01", "date_range_end": "2025-05-31"},
        {"date_range_start": "2023-06-01", "date_range_end": "2024-12-31"},
    ]
    assert resolve_as_of_date(catalog, today=date(2026, 6, 7)) == date(2025, 5, 31)


def test_as_of_capped_at_today_when_data_has_future_dates():
    catalog = [{"date_range_start": "2023-01-01", "date_range_end": "2030-01-01"}]
    assert resolve_as_of_date(catalog, today=date(2026, 6, 7)) == date(2026, 6, 7)


def test_as_of_ignores_open_ended_sentinel_windows():
    catalog = [
        {"date_range_start": "2023-01-01", "date_range_end": "2025-05-31"},
        {"date_range_start": "2023-01-01", "date_range_end": "9999-12-31"},  # sentinel
    ]
    assert resolve_as_of_date(catalog, today=date(2026, 6, 7)) == date(2025, 5, 31)


def test_as_of_none_when_no_dated_files():
    assert resolve_as_of_date([{"display_name": "x"}], today=date(2026, 6, 7)) is None
    assert resolve_as_of_date([], today=date(2026, 6, 7)) is None


# ── relative windows resolve against the as-of date ────────────────────────────

def test_this_year_resolves_against_as_of_not_clock():
    # as_of = 2025-05-31 → "this year" is 2025-01-01 .. 2025-05-31 (has data)
    win = parse_requested_window({"date_range": "this_year"}, today=date(2025, 5, 31))
    assert win == (date(2025, 1, 1), date(2025, 5, 31))


def test_this_quarter_window():
    # 2025-05-31 is in Q2 (Apr-Jun) → 2025-04-01 .. as_of
    win = parse_requested_window({"date_range": "this_quarter"}, today=date(2025, 5, 31))
    assert win == (date(2025, 4, 1), date(2025, 5, 31))


def test_last_quarter_window():
    # as_of in Q2 2025 → last quarter = Q1 2025
    win = parse_requested_window({"date_range": "last_quarter"}, today=date(2025, 5, 31))
    assert win == (date(2025, 1, 1), date(2025, 3, 31))


# ── widened extraction: YTD + quarter phrasings the gate must catch ────────────

def test_ytd_extracted_as_this_year():
    assert _extract_constraints("show YTD debit balances")["date_range"] == "this_year"
    assert _extract_constraints("revenue year to date")["date_range"] == "this_year"


def test_quarter_phrasings_extracted():
    assert _extract_constraints("sales this quarter")["date_range"] == "this_quarter"
    assert _extract_constraints("orders last quarter")["date_range"] == "last_quarter"
    assert _extract_constraints("orders in the previous quarter")["date_range"] == "last_quarter"

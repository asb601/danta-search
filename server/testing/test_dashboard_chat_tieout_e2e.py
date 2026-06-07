"""Phase 5 — G4: dashboard == chat equality harness (the demo's trust anchor).

The strongest proof for a hands-on Databricks reviewer: ask the same question in
chat and as a 1-widget dashboard, and get the SAME number. This requires live
DB + LLM + the curated demo container, so the end-to-end test is SKIPPED unless
`DASHBOARD_E2E_LIVE=1` — it is committed as discoverable documentation-as-code and
runs the moment infra is present (mirrors the pdf_chat eval CI gate pattern).

The pure number-extractor below IS unit-tested in CI.

Run unit parts:  cd server && uv run --with pytest python -m pytest testing/test_dashboard_chat_tieout_e2e.py -q
Run live:        DASHBOARD_E2E_LIVE=1 DASHBOARD_E2E_CONTAINER=<id> uv run --with pytest python -m pytest testing/test_dashboard_chat_tieout_e2e.py -q
"""
from __future__ import annotations

import os
import re

import pytest

# Curated headline KPIs for the demo container — business-analyst fills these in
# (each a single-number question that BOTH chat and a 1-widget dashboard answer).
# Example: {"prompt": "What was total April 2026 revenue?", "tolerance": 0.005}
CURATED_KPIS: list[dict] = []

_REL_TOL = 0.005  # dashboard and chat numbers must agree within 0.5%
_NUM_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*([kmb])?", re.I)
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


def extract_number(value):
    """Pull a single comparable number from a chat answer string or a KPI value.
    Handles $, thousands commas, and K/M/B suffixes. None when no number is found."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    m = _NUM_RE.search(value)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    return num * _SUFFIX.get((m.group(2) or "").lower(), 1.0)


# --- pure unit tests (run in CI) --------------------------------------------

def test_extract_number_currency_and_commas():
    assert extract_number("Total revenue was $4,234,567.") == 4234567.0


def test_extract_number_suffixes():
    assert extract_number("about 4.2M") == 4_200_000.0
    assert extract_number("12K orders") == 12_000.0


def test_extract_number_passthrough_and_none():
    assert extract_number(4234567) == 4234567.0
    assert extract_number("no number here") is None
    assert extract_number(True) is None


def test_extract_number_percent_is_bare():
    assert extract_number("conversion was 42%") == 42.0   # no suffix multiplier


# --- live end-to-end (skipped unless infra present) -------------------------

@pytest.mark.skipif(
    not os.getenv("DASHBOARD_E2E_LIVE"),
    reason="requires live DB + LLM + curated container (set DASHBOARD_E2E_LIVE=1)",
)
def test_dashboard_equals_chat_on_curated_kpis():
    """For each curated KPI, the dashboard KPI number must equal the chat answer
    within _REL_TOL. Wire `run_chat(prompt)` and `run_dashboard_kpi(prompt)` to the
    live agent + dashboard generate path for the configured container, then:

        for kpi in CURATED_KPIS:
            chat_n = extract_number(run_chat(kpi["prompt"]))
            dash_n = extract_number(run_dashboard_kpi(kpi["prompt"]))
            assert chat_n is not None and dash_n is not None
            assert abs(chat_n - dash_n) <= kpi.get("tolerance", _REL_TOL) * max(abs(chat_n), abs(dash_n))
    """
    if not CURATED_KPIS:
        pytest.skip("CURATED_KPIS not yet defined for the demo container")
    pytest.fail("wire run_chat/run_dashboard_kpi to the live paths before enabling")

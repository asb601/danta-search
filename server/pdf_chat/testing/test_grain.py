"""Phase 4 Task 5 — grain alignment + numeric reconciliation.

Imports the module directly (not via ``pdf_chat.bridge`` package exports) so this
test does not depend on ``bridge/__init__.py`` (owned by the bridge-core worker).
"""
from __future__ import annotations

from pdf_chat.bridge.grain import (
    GrainAggregate,
    GrainFact,
    GrainResult,
    reconcile_grain,
)


def test_within_tolerance_reconciles():
    # rate*volume = 10*100 = 1000; aggregate total 1000 → exact match.
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1000.0, period="2026-04", unit="USD"),
    )
    assert isinstance(result, GrainResult)
    assert result.reconciled is True


def test_within_tolerance_band_reconciles():
    # default tolerance 0.05: 1000 vs 1040 → 4% diff, within band.
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1040.0, period="2026-04", unit="USD"),
    )
    assert result.reconciled is True


def test_outside_tolerance_refuses():
    # 1000 vs 1200 → 20% diff, outside the 5% band.
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1200.0, period="2026-04", unit="USD"),
    )
    assert result.reconciled is False
    assert result.reason and (
        "tolerance" in result.reason.lower() or "numeric" in result.reason.lower()
    )


def test_mismatched_period_not_reconciled():
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1000.0, period="2026-05", unit="USD"),
    )
    assert result.reconciled is False
    assert "period" in result.reason.lower()


def test_mismatched_unit_not_reconciled():
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1000.0, period="2026-04", unit="EUR"),
    )
    assert result.reconciled is False
    assert "unit" in result.reason.lower()


def test_period_and_unit_normalized_before_compare():
    # Whitespace/case differences in period+unit must not block reconciliation.
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period=" 2026-04 ", unit="usd"),
        aggregate=GrainAggregate(total=1000.0, period="2026-04", unit="USD"),
    )
    assert result.reconciled is True


def test_zero_aggregate_uses_denominator_floor():
    # max(1, total) guards divide-by-zero: fact 0 vs total 0 → diff 0 → reconciled.
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=0.0, volume=0.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=0.0, period="2026-04", unit="USD"),
    )
    assert result.reconciled is True


def test_tenant_override_widens_tolerance(monkeypatch):
    # A per-tenant env override of the numeric tolerance is honored.
    monkeypatch.setenv("PDF_TUNABLE_GRAIN.NUMERIC_TOLERANCE_PCT", "0.50")
    result = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, period="2026-04", unit="USD"),
        aggregate=GrainAggregate(total=1200.0, period="2026-04", unit="USD"),
    )
    # 20% diff now within the widened 50% band.
    assert result.reconciled is True

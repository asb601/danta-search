"""Pure tests for value-overlap join scoring (Fix 2, no DB).

Proves the edge-creation gate + confidence are DATA-DRIVEN: real master keys
(Vendor_ID high-card ~80%, Region 100%@8) get approved; non-referential document
keys (PO_Number ~0%) and tiny-domain coincidences are rejected — by VALUE, never
by column name.

Run: cd server && uv run --with pytest python -m pytest testing/test_join_confidence.py -q
"""
from __future__ import annotations

from app.services.relationship_detector import join_confidence
from app.services.semantic_policy import get_semantic_policy

P = get_semantic_policy()


def _passes_gate(overlap_pct: float, min_card: int) -> bool:
    return overlap_pct >= P.min_join_overlap and min_card >= P.min_join_cardinality


def test_vendor_id_high_card_is_approved():
    # Vendor_ID: ~80% overlap, high cardinality
    assert _passes_gate(0.80, 1200)
    assert join_confidence(0.80, 1200, P) >= P.approved_join_confidence


def test_region_lowcard_full_overlap_is_approved():
    # Region: 100% overlap on 8 distinct values — passes (card floor == 8)
    assert _passes_gate(1.0, 8)
    assert join_confidence(1.0, 8, P) >= P.approved_join_confidence


def test_po_number_zero_overlap_rejected():
    # PO_Number: non-referential across tables (~0% overlap) — rejected at gate
    assert not _passes_gate(0.004, 2500)


def test_tiny_domain_coincidence_rejected():
    # 100% overlap but only 4 distinct values — coincidence, rejected by card guard
    assert not _passes_gate(1.0, 4)


def test_confidence_monotonic_and_clamped():
    lo = join_confidence(0.55, 10, P)
    hi = join_confidence(0.95, 5000, P)
    assert hi > lo
    assert P.fingerprint_min_confidence <= lo <= P.fingerprint_max_confidence
    assert P.fingerprint_min_confidence <= hi <= P.fingerprint_max_confidence

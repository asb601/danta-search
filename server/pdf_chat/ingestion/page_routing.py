"""Data-driven per-page digital-vs-scanned routing (Spec §2 L1a).

Routes on MEASURED per-page extractable-text coverage ratio (text-span area /
page area), NOT ``if text == ""``. The threshold is a tunable and every routing
decision is logged with its score (Spec §3 invariant 4).
"""
from __future__ import annotations

from pdf_chat.tunables import get_tunable, log_gate_decision


def text_coverage_ratio(*, text_area: float, page_area: float) -> float:
    """Fraction of the page covered by extractable text spans, clamped to [0,1]."""
    if page_area <= 0:
        return 0.0
    return max(0.0, min(1.0, text_area / page_area))


def route_page_extractor(*, coverage: float, container_id: str, page_num: int) -> str:
    """Return ``"digital"`` (PyMuPDF) or ``"scanned"`` (OCR) for one page."""
    threshold = get_tunable(container_id, "digital_text_coverage", 0.70)
    decision = log_gate_decision(
        "digital_vs_scanned",
        score=coverage,
        threshold=threshold,
        outcome="digital" if coverage >= threshold else "scanned",
        container_id=container_id,
        page_num=page_num,
    )
    return decision["outcome"]

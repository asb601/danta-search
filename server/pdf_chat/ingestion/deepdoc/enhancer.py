"""Hard-page gate + ``enhance_page`` orchestrator (DeepDoc, optional).

The gate mirrors the pure routing style of ``parser_router.py``: a page is
enhanced ONLY when it is flagged complex OR low-confidence. Simple pages take the
Phase-1 PyMuPDF/OCR fast path untouched. The orchestrator ALWAYS returns a valid
``UnifiedElement[]``: on any gate-skip, missing-dependency, or enhancer failure it
returns the input fast-path unchanged — ingestion is never blocked on an
optional enhancer.

All thresholds resolve via :func:`get_tunable`; every gate/skip/degrade decision
logs via :func:`log_gate_decision`. No bare literals.
"""
from __future__ import annotations

from ..ton_schema import UnifiedElement
from ...tunables import get_tunable, log_gate_decision
from ._deps import deepdoc_available
from .column_order import assign_columns
from .table_structure import recognize_table_structure

TUN_DD_COMPLEXITY_THRESHOLD = "deepdoc.complexity_threshold"  # default 0.60
TUN_DD_CONFIDENCE_FLOOR = "deepdoc.confidence_floor"          # default 0.55
_DEFAULT_COMPLEXITY_THRESHOLD = 0.60
_DEFAULT_CONFIDENCE_FLOOR = 0.55


def should_enhance(*, container_id: str, complexity_score: float,
                   extract_confidence: float) -> bool:
    """Hard-page gate: enhance only when the page is complex OR low-confidence."""
    cx_thr = get_tunable(container_id, TUN_DD_COMPLEXITY_THRESHOLD,
                         _DEFAULT_COMPLEXITY_THRESHOLD)
    conf_floor = get_tunable(container_id, TUN_DD_CONFIDENCE_FLOOR,
                             _DEFAULT_CONFIDENCE_FLOOR)
    fire = complexity_score >= cx_thr or extract_confidence < conf_floor
    log_gate_decision(
        "deepdoc.gate", score=complexity_score, threshold=cx_thr,
        outcome="enhance" if fire else "skip",
        container_id=container_id, extract_confidence=extract_confidence,
    )
    return fire


def enhance_page(elements: list[UnifiedElement], *, container_id: str,
                 complexity_score: float, extract_confidence: float,
                 page_image=None, session_factory=None) -> list[UnifiedElement]:
    """Optional ONNX enhancement for a single page.

    Always returns a valid ``UnifiedElement[]``. On gate-skip / missing-dep /
    failure it returns the input fast-path unchanged. When the gate fires and deps
    are present it (1) reorders elements into reading order via KMeans columns and
    (2) — if a page image + ONNX session are supplied — recognizes table structure
    (spanning cells). The enhancer is invisible to downstream phases: they consume
    the same schema regardless of whether enhancement ran.
    """
    if not should_enhance(container_id=container_id, complexity_score=complexity_score,
                          extract_confidence=extract_confidence):
        return elements

    if not deepdoc_available():
        log_gate_decision(
            "deepdoc.unavailable", score=0.0, threshold=0.0,
            outcome="fastpath", container_id=container_id,
        )
        return elements

    try:
        out = assign_columns(elements, container_id=container_id)
        if page_image is not None and session_factory is not None:
            table = recognize_table_structure(
                page_image, container_id=container_id,
                session_factory=session_factory,
            )
            if table is not None:
                out = out + [table]
        return out
    except Exception as exc:  # never block ingestion on an enhancer failure
        log_gate_decision(
            "deepdoc.error", score=0.0, threshold=0.0,
            outcome="fastpath", container_id=container_id, error=str(exc),
        )
        return elements

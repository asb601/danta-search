"""Page-extraction orchestrator (Spec §2 L1a): route → digital/OCR → enhance → elements.

The single seam the worker's ``extract_fn`` calls per page. Routing is data-
driven (page_routing.route_page_extractor on measured coverage); the two backends
are imported at module scope so tests can monkeypatch them. After the fast-path
extraction the optional DeepDoc ``enhance_page`` runs (hard-page gate, always
degrades to the fast-path output when the gate skips, deps are missing, or the
enhancer fails — ingestion is never blocked on the optional enhancer).
"""
from __future__ import annotations

from typing import Any

from .deepdoc import enhance_page
from .digital_extractor import extract_digital_page
from .ocr_extractor import extract_scanned_page
from .page_routing import route_page_extractor
from .ton_schema import UnifiedElement


def _mean_confidence(elements: list[UnifiedElement]) -> float:
    """Mean per-element extraction confidence (1.0 when no elements)."""
    if not elements:
        return 1.0
    return sum(el.confidence for el in elements) / len(elements)


def extract_page_elements(
    *,
    page: Any,
    page_image_bytes: bytes,
    coverage: float,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
    complexity_score: float = 0.0,
    page_image: Any = None,
    session_factory: Any = None,
) -> list[UnifiedElement]:
    """Route one page to the digital or OCR extractor and return its elements.

    A hard-page DeepDoc enhancement pass runs over the fast-path output; it is a
    no-op (returns the fast-path unchanged) unless the page is complex or
    low-confidence AND the ONNX deps are present (Spec §2 L1a addendum B).
    """
    route = route_page_extractor(
        coverage=coverage, container_id=tenant_id, page_num=page_num
    )
    if route == "digital":
        elements = extract_digital_page(
            page, doc_id=doc_id, page_num=page_num, tenant_id=tenant_id, acl=acl
        )
    else:
        elements = extract_scanned_page(
            page_image_bytes, doc_id=doc_id, page_num=page_num,
            tenant_id=tenant_id, acl=acl,
        )
    if not elements or not all(isinstance(e, UnifiedElement) for e in elements):
        return elements
    return enhance_page(
        elements,
        container_id=tenant_id,
        complexity_score=complexity_score,
        extract_confidence=_mean_confidence(elements),
        page_image=page_image,
        session_factory=session_factory,
    )

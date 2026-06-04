"""Digital page extraction via PyMuPDF (Spec §2 L1a).

Reads a digital page's text blocks with their bounding boxes and emits
``UnifiedElement`` objects (text full-confidence; bbox retained for click-to-
highlight citations). The fitz dependency lives upstream (page_reader streams the
``fitz.Page``); this function only consumes the page's ``get_text("dict")`` +
``.rect`` surface, so it is pure-testable with a fake page.
"""
from __future__ import annotations

from typing import Any

from .ton_schema import BBox, ElementType, UnifiedElement


def extract_digital_page(
    page: Any,
    *,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
) -> list[UnifiedElement]:
    """Extract text elements (with bbox) from a digital ``fitz.Page``-like object."""
    elements: list[UnifiedElement] = []
    data = page.get_text("dict")
    order = 0
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 == text block in PyMuPDF
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if not text:
            continue
        bx = block.get("bbox", [0, 0, 0, 0])
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:b{order}",
                doc_id=doc_id,
                page_num=page_num,
                element_type=ElementType.TEXT,
                content=text,
                reading_order=order,
                tenant_id=tenant_id,
                bbox=BBox(x1=bx[0], y1=bx[1], x2=bx[2], y2=bx[3]),
                confidence=1.0,
                parser_version="pymupdf-digital-1",
                acl=dict(acl or {}),
            )
        )
        order += 1
    return elements

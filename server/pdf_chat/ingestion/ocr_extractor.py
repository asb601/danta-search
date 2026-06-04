"""Scanned-page extraction via Azure Document Intelligence (Spec §2 L1a + open-Q #2).

OCR + table extraction + OCR-native region typing with confidence — no
font-size/whitespace rule literals. ``parse_di_result`` is PURE over a DI
``analyzeResult`` dict (unit-testable). ``extract_scanned_page`` issues the live
call behind a guarded import (constructs/raises only on call without infra).
"""
from __future__ import annotations

import os
from typing import Any

from .ton_schema import BBox, ElementType, UnifiedElement

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore

    _HAS_DI = True
except ImportError:  # pragma: no cover - exercised only without infra
    DocumentIntelligenceClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore
    _HAS_DI = False


def _polygon_bbox(polygon: list[float]) -> BBox:
    """Reduce a DI polygon (x0,y0,x1,y1,...) to an axis-aligned BBox."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def _table_to_markdown(cells: list[dict], row_count: int, col_count: int) -> str:
    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        grid[cell["rowIndex"]][cell["columnIndex"]] = str(cell.get("content", ""))
    lines = ["| " + " | ".join(grid[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_di_result(
    result: dict, *, doc_id: str, tenant_id: str, acl: dict
) -> list[UnifiedElement]:
    """Convert an Azure DI analyzeResult dict into UnifiedElements (pure)."""
    elements: list[UnifiedElement] = []
    order = 0
    acl = dict(acl or {})

    for page in result.get("pages", []):
        words = page.get("words", [])
        if not words:
            continue
        page_num = int(page.get("pageNumber", 1)) - 1
        text = " ".join(w.get("content", "") for w in words).strip()
        if not text:
            continue
        confs = [float(w.get("confidence", 1.0)) for w in words]
        mean_conf = sum(confs) / len(confs)
        polys = [w["polygon"] for w in words if w.get("polygon")]
        bbox = _polygon_bbox([c for p in polys for c in p]) if polys else None
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:ocr{order}",
                doc_id=doc_id, page_num=page_num,
                element_type=ElementType.TEXT, content=text,
                reading_order=order, tenant_id=tenant_id, bbox=bbox,
                confidence=mean_conf, parser_version="azure-di-1", acl=acl,
            )
        )
        order += 1

    for table in result.get("tables", []):
        cells = table.get("cells", [])
        if not cells:
            continue
        md = _table_to_markdown(cells, table["rowCount"], table["columnCount"])
        cell_confs = [float(c.get("confidence", 1.0)) for c in cells]
        region = (table.get("boundingRegions") or [{}])[0]
        poly = region.get("polygon")
        page_num = int(region.get("pageNumber", 1)) - 1
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:tbl{order}",
                doc_id=doc_id, page_num=page_num,
                element_type=ElementType.TABLE, content=md,
                reading_order=order, tenant_id=tenant_id,
                bbox=_polygon_bbox(poly) if poly else None,
                confidence=min(cell_confs) if cell_confs else 1.0,
                parser_version="azure-di-1", acl=acl,
            )
        )
        order += 1

    return elements


def extract_scanned_page(  # pragma: no cover - requires infra + env
    page_image_bytes: bytes, *, doc_id: str, page_num: int, tenant_id: str, acl: dict
) -> list[UnifiedElement]:
    """Run Azure DI prebuilt-layout over one rendered page image."""
    if not _HAS_DI:
        raise RuntimeError(
            "azure-ai-documentintelligence is required for OCR but is not installed."
        )
    client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
        credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")),
    )
    poller = client.begin_analyze_document("prebuilt-layout", body=page_image_bytes)
    result = poller.result().as_dict()
    return parse_di_result(result, doc_id=doc_id, tenant_id=tenant_id, acl=acl)

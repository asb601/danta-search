"""Pure tests for Phase-1 page extraction (routing, digital extract, confidence)."""
from __future__ import annotations

from pdf_chat.ingestion.page_routing import text_coverage_ratio, route_page_extractor


def test_text_coverage_ratio_full_text():
    # All page area covered by text spans → ratio ~1.0.
    assert text_coverage_ratio(text_area=1000.0, page_area=1000.0) == 1.0


def test_text_coverage_ratio_scanned_page():
    # A scanned page has near-zero extractable text area.
    assert text_coverage_ratio(text_area=2.0, page_area=1000.0) < 0.01


def test_text_coverage_ratio_zero_page_area_is_zero():
    assert text_coverage_ratio(text_area=5.0, page_area=0.0) == 0.0


def test_route_page_extractor_digital_above_threshold():
    route = route_page_extractor(coverage=0.92, container_id="c-1", page_num=0)
    assert route == "digital"


def test_route_page_extractor_scanned_below_threshold():
    route = route_page_extractor(coverage=0.05, container_id="c-1", page_num=1)
    assert route == "scanned"


def test_route_page_extractor_threshold_is_tunable(monkeypatch):
    # Lower the digital threshold so 0.4 coverage now routes digital.
    monkeypatch.setenv("PDF_TUNABLE_DIGITAL_TEXT_COVERAGE", "0.3")
    assert route_page_extractor(coverage=0.4, container_id="c-1", page_num=2) == "digital"


# --------------------------------------------------------------------------- #
# Task 9 — extraction confidence propagation
# --------------------------------------------------------------------------- #
from pdf_chat.ingestion.extraction_confidence import propagate_confidence
from pdf_chat.ingestion.ton_schema import Chunk, ElementType, UnifiedElement


def _el(conf):
    return UnifiedElement(
        element_id="e1", doc_id="d", page_num=0, element_type=ElementType.TEXT,
        content="hi", reading_order=0, tenant_id="t", confidence=conf,
    )


def _chunk():
    return Chunk(
        chunk_id="e1::c0", doc_id="d", page_num=0, element_type=ElementType.TEXT,
        text="hi", reading_order=0, tenant_id="t", source_element_id="e1",
    )


def test_propagate_confidence_sets_low_flag_below_threshold(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_LOW_CONFIDENCE_FLAG_BELOW", "0.60")
    chunks = propagate_confidence([_chunk()], {"e1": 0.4}, container_id="t")
    assert chunks[0].confidence == 0.4
    assert chunks[0].low_confidence is True


def test_propagate_confidence_high_confidence_not_flagged(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_LOW_CONFIDENCE_FLAG_BELOW", "0.60")
    chunks = propagate_confidence([_chunk()], {"e1": 0.95}, container_id="t")
    assert chunks[0].low_confidence is False


# --------------------------------------------------------------------------- #
# Task 10 — digital page extractor
# --------------------------------------------------------------------------- #
from pdf_chat.ingestion.digital_extractor import extract_digital_page


class _FakeRect:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class _FakePage:
    """Minimal stand-in for fitz.Page used by extract_digital_page."""
    def __init__(self):
        self.rect = _FakeRect(100.0, 100.0)

    def get_text(self, kind):
        assert kind == "dict"
        return {
            "blocks": [
                {"type": 0, "bbox": [0, 0, 100, 20],
                 "lines": [{"spans": [{"text": "Hello world"}]}]},
            ]
        }


def test_extract_digital_page_emits_text_element_with_bbox():
    els = extract_digital_page(
        _FakePage(), doc_id="d1", page_num=2, tenant_id="t1", acl={"public": True},
    )
    assert len(els) == 1
    el = els[0]
    assert el.element_type == ElementType.TEXT
    assert el.text == "Hello world" or el.content == "Hello world"
    assert el.bbox is not None
    assert el.bbox.x2 == 100.0
    assert el.confidence == 1.0          # digital text is full-confidence
    assert el.tenant_id == "t1"


def test_extract_digital_page_skips_empty_blocks():
    class _Empty(_FakePage):
        def get_text(self, kind):
            return {"blocks": [{"type": 0, "bbox": [0, 0, 1, 1],
                                "lines": [{"spans": [{"text": "   "}]}]}]}
    assert extract_digital_page(_Empty(), doc_id="d", page_num=0,
                                tenant_id="t", acl={}) == []


# --------------------------------------------------------------------------- #
# Task 11 — OCR / scanned extractor (Azure Document Intelligence)
# --------------------------------------------------------------------------- #
from pdf_chat.ingestion.ocr_extractor import parse_di_result


def _di_result():
    # Minimal Azure Document Intelligence "analyzeResult"-shaped dict.
    return {
        "pages": [{
            "pageNumber": 1,
            "words": [
                {"content": "Invoice", "confidence": 0.99,
                 "polygon": [0, 0, 50, 0, 50, 10, 0, 10]},
                {"content": "Total", "confidence": 0.40,
                 "polygon": [0, 12, 40, 12, 40, 22, 0, 22]},
            ],
        }],
        "tables": [{
            "rowCount": 2, "columnCount": 1,
            "cells": [
                {"rowIndex": 0, "columnIndex": 0, "content": "Amount", "confidence": 0.9},
                {"rowIndex": 1, "columnIndex": 0, "content": "100",   "confidence": 0.3},
            ],
            "boundingRegions": [{"pageNumber": 1,
                                 "polygon": [0, 30, 60, 30, 60, 60, 0, 60]}],
        }],
    }


def test_parse_di_result_emits_text_and_table_with_confidence():
    els = parse_di_result(
        _di_result(), doc_id="d1", tenant_id="t1", acl={"public": True},
    )
    types = {e.element_type for e in els}
    assert ElementType.TEXT in types
    assert ElementType.TABLE in types
    text_el = next(e for e in els if e.element_type == ElementType.TEXT)
    # Page-level OCR text confidence = mean word confidence (0.99, 0.40) ≈ 0.695.
    assert 0.69 <= text_el.confidence <= 0.70
    assert text_el.bbox is not None
    table_el = next(e for e in els if e.element_type == ElementType.TABLE)
    assert "| Amount |" in table_el.content   # markdown header row
    # Table confidence = min cell confidence (worst cell governs).
    assert table_el.confidence == 0.3


def test_parse_di_result_empty():
    assert parse_di_result({"pages": [], "tables": []},
                           doc_id="d", tenant_id="t", acl={}) == []


# --------------------------------------------------------------------------- #
# Task 12 — page-extraction orchestrator
# --------------------------------------------------------------------------- #
from pdf_chat.ingestion.page_extraction import extract_page_elements


def test_extract_page_elements_digital_route(monkeypatch):
    captured = {}

    def _digital(page, *, doc_id, page_num, tenant_id, acl):
        captured["route"] = "digital"
        return [_el_text(doc_id, page_num, tenant_id)]

    def _ocr(image, *, doc_id, page_num, tenant_id, acl):
        captured["route"] = "ocr"
        return []

    from pdf_chat.ingestion import page_extraction as pe
    monkeypatch.setattr(pe, "extract_digital_page", _digital)
    monkeypatch.setattr(pe, "extract_scanned_page", _ocr)

    els = extract_page_elements(
        page=object(), page_image_bytes=b"", coverage=0.9,
        doc_id="d", page_num=0, tenant_id="c-1", acl={},
    )
    assert captured["route"] == "digital"
    assert els[0].element_type == ElementType.TEXT


def test_extract_page_elements_scanned_route(monkeypatch):
    captured = {}
    from pdf_chat.ingestion import page_extraction as pe
    monkeypatch.setattr(pe, "extract_digital_page",
                        lambda *a, **k: captured.setdefault("route", "digital") or [])
    monkeypatch.setattr(pe, "extract_scanned_page",
                        lambda *a, **k: captured.setdefault("route", "ocr") or [])
    extract_page_elements(page=object(), page_image_bytes=b"img", coverage=0.02,
                          doc_id="d", page_num=1, tenant_id="c-1", acl={})
    assert captured["route"] == "ocr"


def _el_text(doc_id, page_num, tenant_id):
    from pdf_chat.ingestion.ton_schema import UnifiedElement
    return UnifiedElement(
        element_id="e", doc_id=doc_id, page_num=page_num,
        element_type=ElementType.TEXT, content="x", reading_order=0,
        tenant_id=tenant_id,
    )

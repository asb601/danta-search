"""Pure unit tests for Team B (Ingestion).

Every test here runs with ZERO infra installed — only pure logic is exercised.
Covers: fingerprint, preflight decision matrix, page classification, parser
routing (incl. entropy override), chunking (text overlap, table row-per-chunk
with headers, image caption), and chunk_id determinism.
"""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from pdf_chat.ingestion import (
    Chunk,
    ElementType,
    UnifiedElement,
    chunk_elements,
    classify_page,
    compute_sha256,
    dlq_key,
    evaluate_preflight,
    retry_countdown,
    route_parser,
)
from pdf_chat.ingestion.preflight import PDF_MIME
from pdf_chat.models.enums import ParserHint, ParserName


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def test_compute_sha256_empty_known_vector():
    # Well-known SHA-256 of empty input.
    assert (
        compute_sha256(b"")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_compute_sha256_matches_hashlib():
    data = b"enterprise pdf rag"
    assert compute_sha256(data) == hashlib.sha256(data).hexdigest()


def test_compute_sha256_deterministic():
    assert compute_sha256(b"abc") == compute_sha256(b"abc")


def test_compute_sha256_rejects_non_bytes():
    with pytest.raises(TypeError):
        compute_sha256("not-bytes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# preflight — reject matrix
# --------------------------------------------------------------------------- #
def test_evaluate_preflight_accepts_clean_pdf():
    rejected, reason = evaluate_preflight(
        mime_type=PDF_MIME, is_encrypted=False, page_count=12
    )
    assert rejected is False
    assert reason is None


def test_evaluate_preflight_rejects_encrypted():
    rejected, reason = evaluate_preflight(
        mime_type=PDF_MIME, is_encrypted=True, page_count=12
    )
    assert rejected is True
    assert reason == "encrypted"


def test_evaluate_preflight_rejects_non_pdf_mime():
    rejected, reason = evaluate_preflight(
        mime_type="application/zip", is_encrypted=False, page_count=5
    )
    assert rejected is True
    assert reason.startswith("not_a_pdf")


def test_evaluate_preflight_rejects_zero_pages():
    rejected, reason = evaluate_preflight(
        mime_type=PDF_MIME, is_encrypted=False, page_count=0
    )
    assert rejected is True
    assert reason == "zero_pages"


def test_evaluate_preflight_encryption_takes_precedence():
    # Encrypted is checked first even if mime/page also fail.
    rejected, reason = evaluate_preflight(
        mime_type="text/plain", is_encrypted=True, page_count=0
    )
    assert rejected is True
    assert reason == "encrypted"


# --------------------------------------------------------------------------- #
# classify_page
# --------------------------------------------------------------------------- #
def test_classify_page_native():
    assert classify_page(500) == ParserHint.NATIVE


def test_classify_page_scanned_below_threshold():
    # Default scanned threshold is 10 chars.
    assert classify_page(3) == ParserHint.SCANNED
    assert classify_page(0) == ParserHint.SCANNED


def test_classify_page_threshold_boundary():
    # Exactly at threshold (10) is NOT scanned (strict <).
    assert classify_page(10, scanned_char_threshold=10) == ParserHint.NATIVE
    assert classify_page(9, scanned_char_threshold=10) == ParserHint.SCANNED


def test_classify_page_complex_layout():
    assert classify_page(500, is_complex_layout=True) == ParserHint.COMPLEX_LAYOUT


def test_classify_page_high_entropy_overrides():
    # High image entropy wins even over plentiful text.
    hint = classify_page(500, image_entropy=0.95, entropy_threshold=0.85)
    assert hint == ParserHint.HIGH_IMAGE_ENTROPY


# --------------------------------------------------------------------------- #
# parser router
# --------------------------------------------------------------------------- #
def test_route_parser_native():
    assert route_parser("native") == ParserName.PYMUPDF


def test_route_parser_complex_layout():
    assert route_parser("complex_layout") == ParserName.DOCLING


def test_route_parser_scanned():
    assert route_parser("scanned") == ParserName.UNSTRUCTURED


def test_route_parser_high_image_entropy_hint():
    assert route_parser("high_image_entropy") == ParserName.VLM


def test_route_parser_accepts_enum_hint():
    assert route_parser(ParserHint.COMPLEX_LAYOUT) == ParserName.DOCLING


def test_route_parser_entropy_override_to_vlm():
    # Entropy above threshold escalates even a "native" page to the VLM.
    assert route_parser("native", image_entropy=0.99) == ParserName.VLM


def test_route_parser_entropy_below_threshold_no_override():
    assert route_parser("native", image_entropy=0.1) == ParserName.PYMUPDF


def test_route_parser_unknown_hint_defaults_native():
    assert route_parser("garbage") == ParserName.PYMUPDF


# --------------------------------------------------------------------------- #
# chunker — helpers
# --------------------------------------------------------------------------- #
def _text_el(content: str, element_id: str = "el-text-1") -> UnifiedElement:
    return UnifiedElement(
        element_id=element_id,
        doc_id="doc-1",
        page_num=0,
        element_type=ElementType.TEXT,
        content=content,
        reading_order=0,
        tenant_id="tenant-1",
        acl={"groups": ["finance"]},
    )


def _table_el(content: str, element_id: str = "el-table-1") -> UnifiedElement:
    return UnifiedElement(
        element_id=element_id,
        doc_id="doc-1",
        page_num=1,
        element_type=ElementType.TABLE,
        content=content,
        reading_order=1,
        tenant_id="tenant-1",
    )


def _image_el(content: str, element_id: str = "el-img-1") -> UnifiedElement:
    return UnifiedElement(
        element_id=element_id,
        doc_id="doc-1",
        page_num=2,
        element_type=ElementType.IMAGE,
        content=content,
        reading_order=2,
        tenant_id="tenant-1",
    )


# --------------------------------------------------------------------------- #
# chunker — text
# --------------------------------------------------------------------------- #
def test_chunk_text_single_short_chunk():
    chunks = chunk_elements([_text_el("One sentence only.")])
    assert len(chunks) == 1
    assert chunks[0].element_type == ElementType.TEXT
    assert chunks[0].text == "One sentence only."


def test_chunk_text_splits_with_overlap():
    # 6 sentences of 5 tokens each (= 30 tokens). chunk_size=10, overlap=5.
    sentences = [f"word word word word s{i}." for i in range(6)]
    text = " ".join(sentences)
    chunks = chunk_elements([_text_el(text)], chunk_size=10, overlap=5)
    assert len(chunks) > 1
    # Overlap: the last sentence of chunk N reappears at the start of chunk N+1.
    first_words = chunks[0].text
    second = chunks[1].text
    # The trailing sentence of the first chunk should seed the second chunk.
    last_sentence_of_first = first_words.split(". ")[-1]
    assert second.startswith(last_sentence_of_first.split()[0])


def test_chunk_text_preserves_acl_and_tenant():
    chunks = chunk_elements([_text_el("Hello world.")])
    assert chunks[0].tenant_id == "tenant-1"
    assert chunks[0].acl == {"groups": ["finance"]}
    assert chunks[0].source_element_id == "el-text-1"


def test_chunk_text_empty_yields_nothing():
    assert chunk_elements([_text_el("   ")]) == []


def test_chunk_long_single_sentence_not_dropped():
    long_sentence = " ".join(["tok"] * 50) + "."
    chunks = chunk_elements([_text_el(long_sentence)], chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == long_sentence


# --------------------------------------------------------------------------- #
# chunker — table (1 row = 1 chunk, header prepended)
# --------------------------------------------------------------------------- #
def test_chunk_table_row_per_chunk_with_headers():
    table_md = (
        "| Name | Amount |\n"
        "| --- | --- |\n"
        "| Acme | 100 |\n"
        "| Globex | 200 |\n"
        "| Initech | 300 |"
    )
    chunks = chunk_elements([_table_el(table_md)])
    # 3 data rows → 3 chunks (separator dropped, header not its own chunk).
    assert len(chunks) == 3
    for ch in chunks:
        assert ch.element_type == ElementType.TABLE
        # Every row chunk carries the header.
        assert "| Name | Amount |" in ch.text
    assert "Acme" in chunks[0].text
    assert "Globex" in chunks[1].text
    assert "Initech" in chunks[2].text


def test_chunk_table_header_only():
    chunks = chunk_elements([_table_el("| A | B |")])
    assert len(chunks) == 1
    assert "| A | B |" in chunks[0].text


def test_chunk_table_empty():
    assert chunk_elements([_table_el("")]) == []


# --------------------------------------------------------------------------- #
# chunker — image (caption = 1 chunk)
# --------------------------------------------------------------------------- #
def test_chunk_image_caption_single_chunk():
    chunks = chunk_elements([_image_el("Bar chart of quarterly revenue.")])
    assert len(chunks) == 1
    assert chunks[0].element_type == ElementType.IMAGE
    assert chunks[0].text == "Bar chart of quarterly revenue."


def test_chunk_image_empty_caption_skipped():
    assert chunk_elements([_image_el("")]) == []


# --------------------------------------------------------------------------- #
# chunker — chunk_id determinism
# --------------------------------------------------------------------------- #
def test_chunk_id_deterministic_across_runs():
    el = _text_el("First sentence. Second sentence.", element_id="stable-el")
    run1 = chunk_elements([el], chunk_size=800, overlap=100)
    run2 = chunk_elements([el], chunk_size=800, overlap=100)
    assert [c.chunk_id for c in run1] == [c.chunk_id for c in run2]


def test_chunk_id_format_and_ordinals():
    table_md = "| H |\n| --- |\n| r1 |\n| r2 |"
    chunks = chunk_elements([_table_el(table_md, element_id="tbl")])
    assert [c.chunk_id for c in chunks] == ["tbl::c0", "tbl::c1"]


def test_chunk_id_unique_per_element():
    els = [_text_el("A.", "ea"), _text_el("B.", "eb")]
    chunks = chunk_elements(els)
    ids = {c.chunk_id for c in chunks}
    assert ids == {"ea::c0", "eb::c0"}


# --------------------------------------------------------------------------- #
# tasks — pure helpers
# --------------------------------------------------------------------------- #
def test_retry_countdown_exponential():
    assert retry_countdown(0, base_delay=60) == 60
    assert retry_countdown(1, base_delay=60) == 120
    assert retry_countdown(2, base_delay=60) == 240


def test_dlq_key_per_tenant():
    assert dlq_key("tenant-42") == "dlq:ingestion:tenant-42"


# --------------------------------------------------------------------------- #
# tasks — _run_page_extraction control flow (no Celery/Redis needed)
# --------------------------------------------------------------------------- #
class _FakeRepo:
    def __init__(self):
        self.statuses: list[tuple] = []
        self.retries = 0

    def set_status(self, task_id, status, error=None):
        self.statuses.append((task_id, status, error))

    def increment_retry(self, task_id):
        self.retries += 1


class _FakeRedis:
    def __init__(self):
        self.pushed: list[tuple] = []

    def lpush(self, key, value):
        self.pushed.append((key, value))


def test_run_page_extraction_success():
    from pdf_chat.ingestion.tasks import _run_page_extraction

    repo = _FakeRepo()
    result = asyncio.run(
        _run_page_extraction(
            "pg-1",
            tenant_id="t1",
            extract_fn=lambda tid: None,
            page_repo=repo,
        )
    )
    assert result == "succeeded"
    assert ("pg-1", "running", None) == repo.statuses[0]
    assert ("pg-1", "succeeded", None) == repo.statuses[-1]


def test_run_page_extraction_transient_retries():
    from pdf_chat.ingestion.tasks import TransientError, _run_page_extraction

    repo = _FakeRepo()
    scheduled = []

    def _boom(tid):
        raise TransientError("network")

    async def _go():
        with pytest.raises(TransientError):
            await _run_page_extraction(
                "pg-2",
                tenant_id="t1",
                extract_fn=_boom,
                page_repo=repo,
                retries=0,
                max_retries=3,
                on_retry=lambda cd: scheduled.append(cd),
            )

    asyncio.run(_go())
    assert repo.retries == 1
    assert scheduled == [retry_countdown(0)]


def test_run_page_extraction_transient_exhausted_to_dlq():
    from pdf_chat.ingestion.tasks import TransientError, _run_page_extraction

    repo = _FakeRepo()
    rds = _FakeRedis()

    def _boom(tid):
        raise TransientError("still failing")

    result = asyncio.run(
        _run_page_extraction(
            "pg-3",
            tenant_id="t1",
            extract_fn=_boom,
            page_repo=repo,
            redis_client=rds,
            retries=3,
            max_retries=3,
        )
    )
    assert result == "failed_terminal"
    assert rds.pushed == [("dlq:ingestion:t1", "pg-3")]


class _FakeAsyncRepo:
    """Mirrors the real PageManifestRepo surface: async set_page_status / increment_retry."""

    def __init__(self):
        self.statuses: list[tuple] = []
        self.retries = 0

    async def set_page_status(self, task_id, status, **fields):
        self.statuses.append((task_id, status, fields.get("error_message")))

    async def increment_retry(self, task_id):
        self.retries += 1


def test_run_page_extraction_drives_async_repo_set_page_status():
    # C11: _run_page_extraction must work against the REAL async repo signature.
    from pdf_chat.ingestion.tasks import _run_page_extraction

    repo = _FakeAsyncRepo()
    result = asyncio.run(
        _run_page_extraction(
            "pg-async",
            tenant_id="t1",
            extract_fn=lambda tid: None,
            page_repo=repo,
        )
    )
    assert result == "succeeded"
    assert ("pg-async", "running", None) == repo.statuses[0]
    assert ("pg-async", "succeeded", None) == repo.statuses[-1]


def test_run_page_extraction_async_repo_terminal_uses_error_message():
    from pdf_chat.ingestion.tasks import PermanentError, _run_page_extraction

    repo = _FakeAsyncRepo()
    rds = _FakeRedis()

    def _boom(tid):
        raise PermanentError("corrupt")

    result = asyncio.run(
        _run_page_extraction(
            "pg-async2",
            tenant_id="t1",
            extract_fn=_boom,
            page_repo=repo,
            redis_client=rds,
            retries=0,
        )
    )
    assert result == "failed_terminal"
    # error routed through the real `error_message` kwarg
    assert repo.statuses[-1] == ("pg-async2", "failed_terminal", "corrupt")
    assert rds.pushed == [("dlq:ingestion:t1", "pg-async2")]


def test_run_page_extraction_permanent_to_dlq():
    from pdf_chat.ingestion.tasks import PermanentError, _run_page_extraction

    repo = _FakeRepo()
    rds = _FakeRedis()

    def _boom(tid):
        raise PermanentError("corrupt bytes")

    result = asyncio.run(
        _run_page_extraction(
            "pg-4",
            tenant_id="t1",
            extract_fn=_boom,
            page_repo=repo,
            redis_client=rds,
            retries=0,
        )
    )
    assert result == "failed_terminal"
    assert rds.pushed == [("dlq:ingestion:t1", "pg-4")]


# --------------------------------------------------------------------------- #
# guarded imports — module loads with zero infra
# --------------------------------------------------------------------------- #
def test_chunk_to_neo4j_props_carries_tenant_and_acl():
    el = _text_el("Hi.", "e1")
    chunk = chunk_elements([el])[0]
    props = chunk.to_neo4j_props()
    assert props["tenant_id"] == "tenant-1"
    assert isinstance(props["acl"], str)  # serialized JSON
    assert props["embedding"] == []


def test_neo4j_writer_constructible_without_driver():
    from pdf_chat.ingestion import Neo4jWriter

    writer = Neo4jWriter("bolt://x", "neo4j", "pw")
    # Empty write short-circuits before touching the driver.
    assert writer.write_chunks([]) == 0
    # A real write (or index op) requires the driver, which is absent here.
    with pytest.raises(RuntimeError):
        writer.ensure_vector_index(1536)

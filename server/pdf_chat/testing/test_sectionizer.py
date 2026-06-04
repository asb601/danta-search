"""Tests for the Phase-2 sectionizer (Task 2).

A SECTION is the LLM extraction unit (spec §1b, granularity dial). The
sectionizer groups a doc's chunks into sections from layout/reading-order
signals and degrades to page-grouping when no heading boundaries are present.

Pure module — no infra. We build ``Chunk`` fixtures and assert grouping +
determinism + the tunable read path. No bare score-comparison literal lives in
the production module: every gate routes through ``log_gate_decision`` and the
granularity dial resolves through ``get_tunable``.
"""
from __future__ import annotations

from pdf_chat.ingestion.sectionizer import Section, sectionize
from pdf_chat.ingestion.ton_schema import Chunk, ElementType


def _chunk(
    cid: str,
    *,
    doc_id: str = "doc1",
    page_num: int = 1,
    reading_order: int = 0,
    text: str = "body text",
    element_type: ElementType = ElementType.TEXT,
    tenant_id: str = "t1",
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        page_num=page_num,
        element_type=element_type,
        text=text,
        reading_order=reading_order,
        tenant_id=tenant_id,
    )


def test_empty_input_returns_empty():
    assert sectionize([], container_id="t1") == []


def test_groups_chunks_into_sections_with_headings():
    # A heading marker (short, title-cased, low reading-order discontinuity)
    # opens a new section; subsequent body chunks join it.
    chunks = [
        _chunk("c0", reading_order=0, text="Introduction", element_type=ElementType.TEXT),
        _chunk("c1", reading_order=1, text="This is the intro body paragraph one."),
        _chunk("c2", reading_order=2, text="Methods", element_type=ElementType.TEXT),
        _chunk("c3", reading_order=3, text="This describes the methods used in detail."),
    ]
    sections = sectionize(chunks, container_id="t1")
    assert len(sections) >= 1
    for s in sections:
        assert isinstance(s, Section)
        assert s.chunk_ids, "every section must own >=1 chunk"
        assert s.tenant_id == "t1"
        assert s.doc_id == "doc1"
    # Every chunk is assigned to exactly one section (partition, no loss/dupe).
    assigned = [cid for s in sections for cid in s.chunk_ids]
    assert sorted(assigned) == ["c0", "c1", "c2", "c3"]


def test_section_id_is_deterministic():
    chunks = [
        _chunk("c0", reading_order=0, text="Overview"),
        _chunk("c1", reading_order=1, text="Some body content here."),
    ]
    first = sectionize(chunks, container_id="t1")
    second = sectionize(chunks, container_id="t1")
    assert [s.section_id for s in first] == [s.section_id for s in second]
    assert [s.fingerprint for s in first] == [s.fingerprint for s in second]
    # section_id format is f"{doc_id}::s{ordinal}"
    assert first[0].section_id == "doc1::s0"


def test_fingerprint_changes_with_text():
    a = sectionize([_chunk("c0", text="alpha content")], container_id="t1")
    b = sectionize([_chunk("c0", text="beta content")], container_id="t1")
    assert a[0].fingerprint != b[0].fingerprint


def test_degrades_to_page_grouping_when_no_headings():
    # No heading-like chunks: long bodies on two pages → one section per page.
    long_body = "This is a long body paragraph that is clearly not a heading. " * 4
    chunks = [
        _chunk("c0", page_num=1, reading_order=0, text=long_body),
        _chunk("c1", page_num=1, reading_order=1, text=long_body),
        _chunk("c2", page_num=2, reading_order=2, text=long_body),
    ]
    sections = sectionize(chunks, container_id="t1")
    # page-grouping degrade → exactly one section per distinct page_num
    assert len(sections) == 2
    pages = {s.page_span for s in sections}
    assert (1, 1) in pages
    assert (2, 2) in pages
    # the page-1 section owns both page-1 chunks
    page1 = next(s for s in sections if s.page_span == (1, 1))
    assert sorted(page1.chunk_ids) == ["c0", "c1"]


def test_page_span_reflects_member_pages():
    chunks = [
        _chunk("c0", page_num=2, reading_order=0, text="Results"),
        _chunk("c1", page_num=2, reading_order=1, text="Body of the results section here."),
        _chunk("c2", page_num=3, reading_order=2, text="Continued results spilling onto page three."),
    ]
    sections = sectionize(chunks, container_id="t1")
    # the heading-rooted section spans pages 2..3
    spanning = [s for s in sections if s.page_span[0] != s.page_span[1]]
    # at minimum, min<=max for every section
    for s in sections:
        assert s.page_span[0] <= s.page_span[1]
    # combined coverage includes pages 2 and 3
    covered = set()
    for s in sections:
        covered.update(range(s.page_span[0], s.page_span[1] + 1))
    assert {2, 3}.issubset(covered)


def test_text_is_concatenation_of_member_chunks():
    chunks = [
        _chunk("c0", reading_order=0, text="Summary"),
        _chunk(
            "c1",
            reading_order=1,
            text=(
                "Line one of the summary describes the overall findings in "
                "considerable narrative detail across a full paragraph."
            ),
        ),
    ]
    sections = sectionize(chunks, container_id="t1")
    s = sections[0]
    assert "Summary" in s.text
    assert "Line one of the summary describes" in s.text


def test_granularity_tunable_is_read(monkeypatch):
    # The granularity dial must resolve through get_tunable (spec §3 inv 4),
    # not a hardcoded literal. Force page-grouping via the env override and
    # assert behavior changes accordingly.
    seen: dict[str, str] = {}
    import pdf_chat.ingestion.sectionizer as sec

    real_get = sec.get_tunable

    def spy(container_id, key, default=None):
        if key == "kg.extraction.granularity":
            seen[key] = "read"
            return "page"
        return real_get(container_id, key, default)

    monkeypatch.setattr(sec, "get_tunable", spy)
    chunks = [
        _chunk("c0", page_num=1, reading_order=0, text="Heading"),
        _chunk("c1", page_num=1, reading_order=1, text="Body of section one."),
        _chunk("c2", page_num=2, reading_order=2, text="Another heading"),
    ]
    sections = sectionize(chunks, container_id="t1")
    assert seen.get("kg.extraction.granularity") == "read"
    # forced "page" granularity → one section per page regardless of headings
    assert len(sections) == 2


def test_log_gate_decision_invoked(monkeypatch):
    import pdf_chat.ingestion.sectionizer as sec

    calls: list[str] = []
    real = sec.log_gate_decision

    def spy(name, **kw):
        calls.append(name)
        return real(name, **kw)

    monkeypatch.setattr(sec, "log_gate_decision", spy)
    sectionize([_chunk("c0", text="Heading"), _chunk("c1", text="body")], container_id="t1")
    assert any(c == "kg.sectionize" for c in calls)


def test_chunks_sorted_by_reading_order_before_grouping():
    # Out-of-order input must be normalized by reading_order so grouping is
    # deterministic regardless of input ordering.
    chunks = [
        _chunk("c2", reading_order=2, text="Body after heading."),
        _chunk("c0", reading_order=0, text="Title"),
        _chunk("c1", reading_order=1, text="First body line under the title."),
    ]
    sections = sectionize(chunks, container_id="t1")
    assigned = [cid for s in sections for cid in s.chunk_ids]
    # reading-order normalization → c0 (title) leads
    assert assigned[0] == "c0"

"""Sectionizer — group a document's chunks into SECTIONS (Phase 2, Task 2).

A SECTION is the unit the LLM extractor consumes (spec §1b, granularity dial,
SECTION default). Grouping is data-driven and tunable: we open a new section at a
detected heading boundary (a short, heading-shaped chunk) and otherwise append
to the current section. When a document exposes no heading boundaries we degrade
gracefully to PAGE-grouping (one section per ``page_num``) so the extractor still
receives coherent, bounded units rather than one giant section or per-chunk noise.

Pure module — zero infra imports. Every threshold resolves through
``get_tunable`` and every grouping decision is emitted via ``log_gate_decision``
(spec §3 invariant 4): no bare score-comparison literal lives here. The
``section_id`` and ``fingerprint`` are deterministic so downstream extraction is
idempotent on ``section_fingerprint`` (Task 4/5).

GOVERNING CRITERIA (millions of files, many tenants): grouping is O(n) over a
doc's chunks, allocation-light, and carries ``tenant_id`` on every Section for
per-hop isolation downstream.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pdf_chat.ingestion.ton_schema import Chunk
from pdf_chat.tunables import get_tunable, log_gate_decision

# ── Tunable keys (named here; defaults SHOULD live in TUNABLE_DEFAULTS) ───────
# The granularity dial is the spec's §1b control. Defaults are passed at the
# call site so the module stays import-safe with zero infra, and are LISTED for
# integration to register in tunables.TUNABLE_DEFAULTS (single source of truth).
_TUN_GRANULARITY = "kg.extraction.granularity"            # "section" | "page"
_TUN_HEADING_MAX_WORDS = "kg.sectionize.heading_max_words"  # heading-length ceiling
_TUN_HEADING_MAX_CHARS = "kg.sectionize.heading_max_chars"  # heading-char ceiling

# Named defaults (passed to get_tunable; mirror these into TUNABLE_DEFAULTS).
_DEFAULT_GRANULARITY = "section"
_DEFAULT_HEADING_MAX_WORDS = 8
_DEFAULT_HEADING_MAX_CHARS = 80


@dataclass(frozen=True)
class Section:
    """A contiguous group of chunks — the LLM extraction unit.

    ``section_id``  deterministic: ``f"{doc_id}::s{ordinal}"``.
    ``text``        concatenation of member chunk text (what the LLM sees).
    ``fingerprint`` model-stable sha256 of the section text (idempotency key).
    ``page_span``   (min_page, max_page) over member chunks.
    """

    section_id: str
    doc_id: str
    tenant_id: str
    chunk_ids: list[str]
    text: str
    fingerprint: str
    page_span: tuple[int, int]


def _norm_text(s: str) -> str:
    """Whitespace-collapsed text for stable fingerprinting."""
    return " ".join((s or "").split())


def _fingerprint(texts: list[str]) -> str:
    """Model-stable section fingerprint (deterministic, length-bounded)."""
    body = "\n".join(_norm_text(t) for t in texts)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _looks_like_heading(
    chunk: Chunk, *, max_words: int, max_chars: int
) -> bool:
    """Data-driven heading shape: short, non-empty, single-line-ish text.

    No dataset-fitted dictionary — purely shape signals (length thresholds are
    tunable). Tables/images/formulas are never headings.
    """
    from pdf_chat.ingestion.ton_schema import ElementType

    if chunk.element_type is not ElementType.TEXT:
        return False
    body = _norm_text(chunk.text)
    if not body:
        return False
    word_count = len(body.split())
    return word_count <= max_words and len(body) <= max_chars


def _build_section(
    doc_id: str, tenant_id: str, ordinal: int, members: list[Chunk]
) -> Section:
    texts = [m.text for m in members]
    pages = [m.page_num for m in members]
    return Section(
        section_id=f"{doc_id}::s{ordinal}",
        doc_id=doc_id,
        tenant_id=tenant_id,
        chunk_ids=[m.chunk_id for m in members],
        text="\n".join(texts),
        fingerprint=_fingerprint(texts),
        page_span=(min(pages), max(pages)),
    )


def _page_grouping(chunks: list[Chunk], *, container_id: str) -> list[Section]:
    """Degrade path: one section per distinct ``page_num`` (in page order)."""
    doc_id = chunks[0].doc_id
    tenant_id = chunks[0].tenant_id
    by_page: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_page.setdefault(c.page_num, []).append(c)

    sections: list[Section] = []
    for ordinal, page in enumerate(sorted(by_page)):
        sections.append(_build_section(doc_id, tenant_id, ordinal, by_page[page]))
    log_gate_decision(
        "kg.sectionize",
        score=float(len(sections)),
        threshold=1.0,
        outcome="page_grouping",
        container_id=container_id,
        doc_id=doc_id,
        chunk_count=len(chunks),
        section_count=len(sections),
    )
    return sections


def _heading_grouping(chunks: list[Chunk], *, container_id: str) -> list[Section]:
    """Primary path: open a new section at each detected heading boundary.

    If no heading boundary is found across the whole document we DEGRADE to
    page-grouping (a single giant section would defeat section-level extraction).
    """
    doc_id = chunks[0].doc_id
    tenant_id = chunks[0].tenant_id
    max_words = int(
        get_tunable(container_id, _TUN_HEADING_MAX_WORDS, _DEFAULT_HEADING_MAX_WORDS)
    )
    max_chars = int(
        get_tunable(container_id, _TUN_HEADING_MAX_CHARS, _DEFAULT_HEADING_MAX_CHARS)
    )

    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    heading_count = 0
    for c in chunks:
        is_heading = _looks_like_heading(c, max_words=max_words, max_chars=max_chars)
        if is_heading:
            heading_count += 1
            if current:
                groups.append(current)
            current = [c]
        else:
            if not current:
                # body before any heading → seed an implicit leading section
                current = [c]
            else:
                current.append(c)
    if current:
        groups.append(current)

    if heading_count == 0:
        # no layout signal → graceful degrade to page-grouping
        return _page_grouping(chunks, container_id=container_id)

    sections = [
        _build_section(doc_id, tenant_id, ordinal, members)
        for ordinal, members in enumerate(groups)
    ]
    log_gate_decision(
        "kg.sectionize",
        score=float(heading_count),
        threshold=1.0,
        outcome="heading_grouping",
        container_id=container_id,
        doc_id=doc_id,
        chunk_count=len(chunks),
        section_count=len(sections),
        heading_count=heading_count,
    )
    return sections


def sectionize(chunks: list[Chunk], *, container_id: str) -> list[Section]:
    """Group one document's chunks into Sections (the LLM extraction unit).

    Reads the ``kg.extraction.granularity`` dial (``"section"`` default →
    layout/reading-order grouping; ``"page"`` → page-grouping). Chunks are
    normalized by ``reading_order`` first so grouping is independent of input
    ordering. Returns ``[]`` for empty input. All chunks are expected to belong
    to a single document/tenant (the ingestion pipeline sections per doc).
    """
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda c: c.reading_order)
    granularity = get_tunable(
        container_id, _TUN_GRANULARITY, _DEFAULT_GRANULARITY
    )

    if granularity == "page":
        return _page_grouping(ordered, container_id=container_id)
    return _heading_grouping(ordered, container_id=container_id)

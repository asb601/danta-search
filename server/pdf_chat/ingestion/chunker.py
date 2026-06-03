"""Stage 11 — Chunk Extraction.

Normalizes a list of :class:`UnifiedElement` (any parser path) into retrievable
:class:`Chunk` units:

  * **text**  — sentence-boundary splitting at ``chunk_size`` tokens with
                ``overlap`` tokens of overlap. llama-index ``SentenceSplitter``
                is the production choice (guarded import below); a pure,
                dependency-free splitter is always available as the fallback so
                the logic + tests run with zero infra.
  * **table** — 1 row == 1 chunk, with the column-header row prepended to every
                row chunk (so each row is self-describing for retrieval).
  * **image** — the caption / description is stored as a single chunk (the full
                description is extracted lazily at retrieval time per spec).

All ``chunk_id`` values are DETERMINISTIC (derived from the source element id +
ordinal) so re-ingesting the same document is idempotent.
"""
from __future__ import annotations

import re

from ..config import get_pdf_settings
from .ton_schema import Chunk, ElementType, UnifiedElement

# Guarded production splitter. Pure fallback is used when it is unavailable.
try:  # pragma: no cover - import shape only
    from llama_index.core.node_parser import SentenceSplitter  # type: ignore

    _HAS_SENTENCE_SPLITTER = True
except ImportError:
    SentenceSplitter = None  # type: ignore
    _HAS_SENTENCE_SPLITTER = False


# Sentence boundary: end punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Token approximation: whitespace-delimited words (dependency-free, deterministic).
_WORD_RE = re.compile(r"\S+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]


def _token_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _pure_sentence_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Dependency-free sentence-aware splitter with token-ish overlap.

    Greedily packs whole sentences until adding the next one would exceed
    ``chunk_size`` tokens, then starts a new chunk that re-includes trailing
    sentences from the previous chunk totalling up to ``overlap`` tokens. A
    single sentence longer than ``chunk_size`` becomes its own chunk (never
    dropped or mid-sentence split).
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)
        if current and current_tokens + sent_tokens > chunk_size:
            chunks.append(" ".join(current))
            # Build overlap tail from the end of the just-emitted chunk.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                ptoks = _token_count(prev)
                if tail_tokens + ptoks > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += ptoks
            current = tail
            current_tokens = tail_tokens

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _text_segments(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Return text segments using the prod splitter if present, else the pure one."""
    if _HAS_SENTENCE_SPLITTER:  # pragma: no cover - requires infra
        try:
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            return [s for s in splitter.split_text(text) if s.strip()]
        except Exception:
            pass
    return _pure_sentence_chunks(text, chunk_size, overlap)


def _chunk_id(element_id: str, ordinal: int) -> str:
    """Deterministic chunk id from source element + ordinal."""
    return f"{element_id}::c{ordinal}"


def _new_chunk(el: UnifiedElement, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(el.element_id, ordinal),
        doc_id=el.doc_id,
        page_num=el.page_num,
        element_type=el.element_type,
        text=text,
        reading_order=el.reading_order,
        tenant_id=el.tenant_id,
        acl=dict(el.acl or {}),
        source_element_id=el.element_id,
    )


def _split_table_rows(content: str) -> list[str]:
    """Split table markdown into rows, dropping blank and separator rows.

    Handles standard markdown tables where the second line is a ``---|---``
    separator. Each remaining line is one logical row.
    """
    lines = [ln for ln in (content or "").splitlines() if ln.strip()]
    rows: list[str] = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # Drop markdown separator rows like |---|:--:|
        if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            if any(c for c in cells):
                continue
        rows.append(ln.strip())
    return rows


def _chunk_table(el: UnifiedElement) -> list[Chunk]:
    """1 row == 1 chunk; the header row is prepended to every data row."""
    rows = _split_table_rows(el.content)
    if not rows:
        return []
    if len(rows) == 1:
        # Header only (or single row) — emit it as one chunk.
        return [_new_chunk(el, 0, rows[0])]

    header = rows[0]
    chunks: list[Chunk] = []
    for i, row in enumerate(rows[1:]):
        text = f"{header}\n{row}"
        chunks.append(_new_chunk(el, i, text))
    return chunks


def chunk_elements(
    elements: list[UnifiedElement],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Turn unified elements into retrievable chunks (Stage 11).

    Args:
        elements: parser-normalized :class:`UnifiedElement` objects.
        chunk_size: text chunk size in tokens (defaults to config ``chunk_size``).
        overlap: text overlap in tokens (defaults to config ``chunk_overlap``).

    Returns:
        A flat list of :class:`Chunk` in input order. Deterministic chunk ids.
    """
    settings = get_pdf_settings()
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    out: list[Chunk] = []
    for el in elements:
        etype = el.element_type
        if etype == ElementType.TABLE:
            out.extend(_chunk_table(el))
        elif etype == ElementType.IMAGE:
            # Caption / description is one chunk; full description is lazy.
            caption = (el.content or "").strip()
            if caption:
                out.append(_new_chunk(el, 0, caption))
        else:  # TEXT, FORMULA → sentence-style splitting
            segments = _text_segments(el.content or "", chunk_size, overlap)
            for i, seg in enumerate(segments):
                out.append(_new_chunk(el, i, seg))
    return out

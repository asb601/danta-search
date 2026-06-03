"""Stage 5 — Streaming Page Reader.

Loads exactly one page at a time via PyMuPDF (``fitz``). Memory footprint stays
constant regardless of document size (Hard rule #1: never load the entire PDF
into memory).

``fitz`` is imported behind a guard so this module imports with zero infra. The
clear ``RuntimeError`` is raised only when :func:`stream_pages` is actually
CALLED without the library present — not at import time.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

try:
    import fitz  # type: ignore  # PyMuPDF

    _HAS_FITZ = True
except ImportError:  # pragma: no cover - exercised only without infra
    fitz = None  # type: ignore
    _HAS_FITZ = False


def stream_pages(blob_bytes: bytes) -> Iterator[tuple[int, Any]]:
    """Yield ``(page_num, page)`` one page at a time.

    Args:
        blob_bytes: raw PDF bytes (downloaded from blob storage). Only one
            decoded page is resident at a time; the generator keeps the memory
            profile flat for arbitrarily large documents.

    Yields:
        ``(page_num, fitz.Page)`` tuples in document order.

    Raises:
        RuntimeError: if PyMuPDF (``fitz``) is not installed. Raised on CALL,
            never at import, so pure logic and tests load without infra.
    """
    if not _HAS_FITZ:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to stream PDF pages but is not installed. "
            "Install it with `pip install pymupdf` to enable page streaming."
        )

    pdf = fitz.open(stream=blob_bytes, filetype="pdf")  # type: ignore[union-attr]
    try:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            yield page_num, page
            # page is freed from memory as the loop advances
    finally:
        pdf.close()

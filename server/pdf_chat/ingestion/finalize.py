"""Finalization (Spec §2 L1a): a document is 'ready' only after ALL pages settle.

Pure reducer over page statuses. While any page is non-terminal the document
stays PROCESSING; once every page is settled it becomes INDEXED (all succeeded)
or PARTIALLY_INDEXED (some terminally failed but at least one succeeded) or FAILED
(none succeeded). The worker calls this on each page-settle; the orchestrator's
``reconcile`` persists the result.
"""
from __future__ import annotations

from pdf_chat.models.enums import DocStatus, PageStatus, SETTLED_PAGE_STATES

_SETTLED = {s.value for s in SETTLED_PAGE_STATES}


def finalize_document(page_statuses: list[str]) -> str:
    """Reduce per-page statuses to a document status."""
    if not page_statuses:
        return DocStatus.PROCESSING.value
    if any(s not in _SETTLED for s in page_statuses):
        return DocStatus.PROCESSING.value
    succeeded = sum(1 for s in page_statuses if s == PageStatus.SUCCEEDED.value)
    if succeeded == len(page_statuses):
        return DocStatus.INDEXED.value
    if succeeded == 0:
        return DocStatus.FAILED.value
    return DocStatus.PARTIALLY_INDEXED.value

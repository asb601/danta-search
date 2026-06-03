"""Page + document state machines — pure logic, no I/O.

Implements the page task state machine and the partial-success document
reconciliation from enterprise-pdf spec 4.4 / section 9.

Two responsibilities:

* ``next_page_status`` — given the outcome of a single page extraction attempt,
  decide the page's next ``PageStatus`` (the per-page state machine).
* ``reconcile_document_status`` — given the *settled* statuses of all pages,
  decide the document-level ``DocStatus`` (partial-success reconciliation).

Everything here is deterministic and config-thresholds are passed in by the
caller (sourced from ``get_pdf_settings()``), so there are no magic numbers and
the functions are trivially unit-testable with zero infra.
"""
from __future__ import annotations

from pdf_chat.models.enums import (
    DocStatus,
    PageStatus,
    SETTLED_PAGE_STATES,
)

__all__ = ["reconcile_document_status", "next_page_status"]


def reconcile_document_status(page_statuses: list[str]) -> str:
    """Reduce per-page statuses to a single document status (spec 4.4 / §9).

    Rules:
      * every page succeeded                 -> ``indexed``
      * no page succeeded                    -> ``failed``
      * some succeeded, some did not         -> ``partially_indexed``

    A page "succeeded" only if its status is exactly ``PageStatus.SUCCEEDED``.
    ``FAILED_TERMINAL`` and ``NEEDS_HUMAN_REVIEW`` are *settled* (no more work)
    but explicitly count as **not succeeded** — a doc whose only non-succeeded
    pages are awaiting human review is still partially indexed, not done.

    Parameters
    ----------
    page_statuses:
        Raw ``PageStatus`` value strings for every page of the document.

    Returns
    -------
    str
        A ``DocStatus`` value string: ``indexed`` / ``partially_indexed`` / ``failed``.
    """
    total = len(page_statuses)

    # No pages registered at all -> nothing could possibly be indexed.
    if total == 0:
        return DocStatus.FAILED.value

    succeeded = sum(1 for s in page_statuses if s == PageStatus.SUCCEEDED.value)

    if succeeded == total:
        # All pages indexed cleanly.
        return DocStatus.INDEXED.value
    if succeeded == 0:
        # Not a single page made it — the document is unusable.
        return DocStatus.FAILED.value
    # Mixed: e.g. 3 bad pages out of 10,000. Still a useful document.
    return DocStatus.PARTIALLY_INDEXED.value


def next_page_status(
    current: str,
    error_kind: str | None,
    retry_count: int,
    max_retries: int,
    confidence: float | None,
    needs_review_threshold: float,
) -> PageStatus:
    """Compute the next state of a page after one processing attempt (spec 4.4).

    State machine::

        running ──(success, conf ok)──────────────► succeeded
                ──(success, conf < threshold)──────► needs_human_review
                ──(transient err, retries left)────► failed_retryable
                ──(transient err, exhausted)───────► failed_terminal
                ──(permanent err)──────────────────► failed_terminal

    Parameters
    ----------
    current:
        The page's current status value (typically ``running``). Settled states
        are returned unchanged — this function never resurrects a finished page.
    error_kind:
        ``None`` on success; otherwise ``"transient"`` (network/OCR timeout —
        retryable) or anything else, which is treated as ``"permanent"``
        (corrupt page bytes — not retryable).
    retry_count:
        How many retries have already been consumed for this page.
    max_retries:
        Retry budget (from ``PdfSettings.max_retries``).
    confidence:
        Extraction confidence in ``[0, 1]`` on success, else ``None``.
    needs_review_threshold:
        Below this confidence a *successful* extraction is routed to human
        review instead of being marked succeeded
        (from ``PdfSettings.needs_review_confidence``).

    Returns
    -------
    PageStatus
        The next page status.
    """
    # Never transition a page that is already settled (idempotent / safe under
    # at-least-once Celery delivery and reconciler re-runs).
    if current in {s.value for s in SETTLED_PAGE_STATES}:
        return PageStatus(current)

    # --- Failure paths --------------------------------------------------------
    if error_kind is not None:
        if error_kind == "transient":
            # Retryable error: stay retryable while budget remains, else give up.
            if retry_count < max_retries:
                return PageStatus.FAILED_RETRYABLE
            return PageStatus.FAILED_TERMINAL
        # Permanent error (corrupt bytes, unsupported content): no point retrying.
        return PageStatus.FAILED_TERMINAL

    # --- Success paths --------------------------------------------------------
    # Low-confidence extraction succeeded mechanically but is not trustworthy;
    # park it for a human rather than indexing questionable content.
    if confidence is not None and confidence < needs_review_threshold:
        return PageStatus.NEEDS_HUMAN_REVIEW

    return PageStatus.SUCCEEDED

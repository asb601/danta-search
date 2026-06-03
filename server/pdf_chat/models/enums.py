"""Canonical enums shared across the PDF pipeline (control plane + workers).

Plain str-enums so they serialize cleanly to PostgreSQL TEXT columns and JSON.
"""
from __future__ import annotations

from enum import Enum


class DocStatus(str, Enum):
    """upload_manifest.status — document-level lifecycle."""
    UPLOADED = "uploaded"
    SPLITTING = "splitting"
    PROCESSING = "processing"
    INDEXED = "indexed"
    PARTIALLY_INDEXED = "partially_indexed"
    FAILED = "failed"


class PageStatus(str, Enum):
    """page_manifest.status — per-page task lifecycle."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class ParserHint(str, Enum):
    """Routing hint computed at preflight, stored per page."""
    NATIVE = "native"
    COMPLEX_LAYOUT = "complex_layout"
    SCANNED = "scanned"
    HIGH_IMAGE_ENTROPY = "high_image_entropy"


class ParserName(str, Enum):
    """Concrete parser selected by the router."""
    PYMUPDF = "pymupdf"
    DOCLING = "docling"
    UNSTRUCTURED = "unstructured"
    VLM = "vlm"


class DedupDecision(str, Enum):
    """Outcome of the SHA-256 fingerprint check."""
    SKIP = "skip"                # same hash+tenant, already indexed
    REPROCESS = "reprocess"      # same hash+tenant, previous attempt failed
    NEW = "new"                  # unseen, or same hash different tenant
    NEW_VERSION = "new_version"  # same logical doc, different bytes


# Terminal page states (no further processing).
SETTLED_PAGE_STATES = frozenset({
    PageStatus.SUCCEEDED,
    PageStatus.FAILED_TERMINAL,
    PageStatus.NEEDS_HUMAN_REVIEW,
})

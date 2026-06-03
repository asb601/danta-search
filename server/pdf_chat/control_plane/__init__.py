"""Control Plane — the PostgreSQL-backed management layer for PDF ingestion.

Public surface (per shared CONTRACTS.md, Team A):

Pure logic (no I/O, unit-tested):
    decide_dedup(existing_status, same_tenant) -> DedupDecision
    reconcile_document_status(page_statuses) -> str
    next_page_status(...) -> PageStatus

Async repositories (persistence only, no business logic):
    UploadManifestRepo, PageManifestRepo
"""
from __future__ import annotations

from pdf_chat.control_plane.dedup import decide_dedup
from pdf_chat.control_plane.orchestrator import (
    IngestDeps,
    IngestResult,
    ingest_document,
    reconcile,
)
from pdf_chat.control_plane.repositories import (
    PageManifestRepo,
    UploadManifestRepo,
)
from pdf_chat.control_plane.state_machine import (
    next_page_status,
    reconcile_document_status,
)

__all__ = [
    "decide_dedup",
    "reconcile_document_status",
    "next_page_status",
    "UploadManifestRepo",
    "PageManifestRepo",
    "IngestDeps",
    "IngestResult",
    "ingest_document",
    "reconcile",
]

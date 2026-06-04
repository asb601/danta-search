"""Pure unit tests for the Control Plane logic (no DB, no infra).

Covers:
  * decide_dedup            — full decision matrix (spec 4.2)
  * reconcile_document_status — all / none / mixed + review/terminal states (spec 4.4)
  * next_page_status        — every transition of the page state machine (spec 4.4)

Repositories require a live AsyncSession and are intentionally NOT unit-tested
here (they contain no logic — they persist only).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from pdf_chat.config import get_pdf_settings
from pdf_chat.control_plane import (
    IngestDeps,
    decide_dedup,
    ingest_document,
    next_page_status,
    reconcile,
    reconcile_document_status,
)
from pdf_chat.models.enums import DedupDecision, DocStatus, PageStatus, ParserHint

_SETTINGS = get_pdf_settings()
_THRESH = _SETTINGS.needs_review_confidence
_MAX_RETRIES = _SETTINGS.max_retries


# --------------------------------------------------------------------------- #
# decide_dedup — full matrix
# --------------------------------------------------------------------------- #
def test_dedup_no_existing_is_new():
    assert decide_dedup(None, same_tenant=False) is DedupDecision.NEW
    assert decide_dedup(None, same_tenant=True) is DedupDecision.NEW


def test_dedup_same_tenant_indexed_skips():
    assert (
        decide_dedup(DocStatus.INDEXED.value, same_tenant=True)
        is DedupDecision.SKIP
    )


def test_dedup_same_tenant_failed_reprocesses():
    assert (
        decide_dedup(DocStatus.FAILED.value, same_tenant=True)
        is DedupDecision.REPROCESS
    )


def test_dedup_different_tenant_is_new_even_if_indexed():
    # Different tenant -> always NEW regardless of the other tenant's status.
    assert (
        decide_dedup(DocStatus.INDEXED.value, same_tenant=False)
        is DedupDecision.NEW
    )
    assert (
        decide_dedup(DocStatus.FAILED.value, same_tenant=False)
        is DedupDecision.NEW
    )


@pytest.mark.parametrize(
    "status",
    [
        DocStatus.UPLOADED.value,
        DocStatus.SPLITTING.value,
        DocStatus.PROCESSING.value,
        DocStatus.PARTIALLY_INDEXED.value,
    ],
)
def test_dedup_same_tenant_in_flight_or_partial_skips(status):
    # In-flight / partial duplicate uploads must not spawn a competing pipeline.
    assert decide_dedup(status, same_tenant=True) is DedupDecision.SKIP


# --------------------------------------------------------------------------- #
# reconcile_document_status
# --------------------------------------------------------------------------- #
def test_reconcile_empty_is_failed():
    assert reconcile_document_status([]) == DocStatus.FAILED.value


def test_reconcile_all_succeeded_is_indexed():
    statuses = [PageStatus.SUCCEEDED.value] * 5
    assert reconcile_document_status(statuses) == DocStatus.INDEXED.value


def test_reconcile_none_succeeded_is_failed():
    statuses = [PageStatus.FAILED_TERMINAL.value] * 3
    assert reconcile_document_status(statuses) == DocStatus.FAILED.value


def test_reconcile_mixed_is_partial():
    statuses = [
        PageStatus.SUCCEEDED.value,
        PageStatus.SUCCEEDED.value,
        PageStatus.FAILED_TERMINAL.value,
    ]
    assert (
        reconcile_document_status(statuses) == DocStatus.PARTIALLY_INDEXED.value
    )


def test_reconcile_review_states_count_as_not_succeeded():
    # needs_human_review is settled but NOT a success.
    only_review = [PageStatus.NEEDS_HUMAN_REVIEW.value] * 4
    assert reconcile_document_status(only_review) == DocStatus.FAILED.value

    mixed_review = [
        PageStatus.SUCCEEDED.value,
        PageStatus.NEEDS_HUMAN_REVIEW.value,
    ]
    assert (
        reconcile_document_status(mixed_review)
        == DocStatus.PARTIALLY_INDEXED.value
    )


def test_reconcile_single_page_documents():
    assert (
        reconcile_document_status([PageStatus.SUCCEEDED.value])
        == DocStatus.INDEXED.value
    )
    assert (
        reconcile_document_status([PageStatus.FAILED_TERMINAL.value])
        == DocStatus.FAILED.value
    )


# --------------------------------------------------------------------------- #
# next_page_status — every transition
# --------------------------------------------------------------------------- #
def test_page_success_high_confidence_succeeds():
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind=None,
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=0.95,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.SUCCEEDED
    )


def test_page_success_no_confidence_succeeds():
    # confidence None on success means "not measured" -> succeed.
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind=None,
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=None,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.SUCCEEDED
    )


def test_page_success_low_confidence_needs_review():
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind=None,
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=_THRESH - 0.01,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.NEEDS_HUMAN_REVIEW
    )


def test_page_confidence_exactly_threshold_succeeds():
    # Boundary: at the threshold (not below) it succeeds.
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind=None,
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=_THRESH,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.SUCCEEDED
    )


def test_page_transient_error_with_retries_left_is_retryable():
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind="transient",
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=None,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.FAILED_RETRYABLE
    )


def test_page_transient_error_exhausted_is_terminal():
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind="transient",
            retry_count=_MAX_RETRIES,
            max_retries=_MAX_RETRIES,
            confidence=None,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.FAILED_TERMINAL
    )


def test_page_permanent_error_is_terminal_even_with_retries_left():
    assert (
        next_page_status(
            current=PageStatus.RUNNING.value,
            error_kind="permanent",
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=None,
            needs_review_threshold=_THRESH,
        )
        is PageStatus.FAILED_TERMINAL
    )


@pytest.mark.parametrize(
    "settled",
    [
        PageStatus.SUCCEEDED,
        PageStatus.FAILED_TERMINAL,
        PageStatus.NEEDS_HUMAN_REVIEW,
    ],
)
def test_page_settled_states_are_never_resurrected(settled):
    # Idempotency under at-least-once delivery: a settled page stays put even if
    # the function is called again with a "success" outcome.
    assert (
        next_page_status(
            current=settled.value,
            error_kind=None,
            retry_count=0,
            max_retries=_MAX_RETRIES,
            confidence=0.99,
            needs_review_threshold=_THRESH,
        )
        is settled
    )


# --------------------------------------------------------------------------- #
# Ingestion orchestrator — decision flow with fakes (zero infra)
# --------------------------------------------------------------------------- #
@dataclass
class _Row:
    upload_id: str
    status: str = DocStatus.UPLOADED.value


class _FakeUploadRepo:
    def __init__(self, existing=None):
        self.existing = existing
        self.created: list[dict] = []
        self.status_calls: list[tuple] = []
        self._seq = 0

    async def find_by_sha256(self, sha256, tenant_id):
        return self.existing

    async def create_upload(self, **fields):
        self._seq += 1
        uid = f"up-{self._seq}"
        self.created.append({"upload_id": uid, **fields})
        return _Row(upload_id=uid, status=fields.get("status", DocStatus.UPLOADED.value))

    async def set_status(self, upload_id, status, **fields):
        self.status_calls.append((upload_id, status, fields))


class _FakePageRepo:
    def __init__(self, pages=None):
        self.created_specs: list[dict] = []
        self._pages = pages or []

    async def create_pages(self, pages):
        specs = list(pages)
        self.created_specs.extend(specs)
        return specs

    async def get_all_pages(self, upload_id):
        return self._pages


@dataclass
class _Preflight:
    rejected: bool = False
    reject_reason: str | None = None
    page_count: int = 0
    mime_type: str = "application/pdf"
    scanned_pages: list = field(default_factory=list)
    complex_layout_pages: list = field(default_factory=list)

    def to_dict(self):
        return {"rejected": self.rejected, "page_count": self.page_count}


def _deps(upload_repo, page_repo, preflight):
    enqueued: list[tuple] = []

    async def _blob(**kw):
        return f"blob://{kw['tenant_id']}/{kw['sha256']}"

    async def _enqueue(task_id, tenant_id):
        enqueued.append((task_id, tenant_id))

    async def _commit():
        return None

    deps = IngestDeps(
        upload_repo=upload_repo,
        page_repo=page_repo,
        hash_fn=lambda b: "sha-fixed",
        preflight_fn=lambda b: preflight,
        blob_writer=_blob,
        enqueue_fn=_enqueue,
        commit=_commit,
    )
    return deps, enqueued


def test_orchestrator_dedup_skip_queues_nothing():
    existing = _Row(upload_id="up-existing", status=DocStatus.INDEXED.value)
    up = _FakeUploadRepo(existing=existing)
    pg = _FakePageRepo()
    deps, enqueued = _deps(up, pg, _Preflight(page_count=5))

    res = asyncio.run(
        ingest_document(b"%PDF-1.4", "t1", "u1", {"public": True}, deps=deps)
    )

    assert res.deduplicated is True
    assert res.decision is DedupDecision.SKIP
    assert res.upload_id == "up-existing"
    assert enqueued == []          # nothing queued
    assert up.created == []         # no new manifest
    assert pg.created_specs == []   # no pages


def test_orchestrator_reject_marks_failed_and_queues_nothing():
    up = _FakeUploadRepo(existing=None)
    pg = _FakePageRepo()
    deps, enqueued = _deps(up, pg, _Preflight(rejected=True, reject_reason="encrypted"))

    res = asyncio.run(
        ingest_document(b"junk", "t1", "u1", {}, deps=deps)
    )

    assert res.rejected is True
    assert res.reject_reason == "encrypted"
    assert res.status == DocStatus.FAILED.value
    # a failed manifest was persisted, but no pages and nothing enqueued
    assert len(up.created) == 1
    assert up.created[0]["status"] == DocStatus.FAILED.value
    assert enqueued == []
    assert pg.created_specs == []


def test_orchestrator_happy_fanout_creates_pages_and_enqueues():
    up = _FakeUploadRepo(existing=None)
    pg = _FakePageRepo()
    # 3 pages; page 1 scanned, page 2 complex_layout, page 0 native.
    pf = _Preflight(page_count=3, scanned_pages=[1], complex_layout_pages=[2])
    deps, enqueued = _deps(up, pg, pf)

    res = asyncio.run(
        ingest_document(b"%PDF-1.4", "t1", "u1", {"public": True}, deps=deps)
    )

    assert res.decision is DedupDecision.NEW
    assert res.rejected is False
    assert res.pages_enqueued == 3
    assert res.status == DocStatus.PROCESSING.value
    # one PageManifest per page, with per-page parser_hint from preflight
    hints = {s["page_num"]: s["parser_hint"] for s in pg.created_specs}
    assert hints == {
        0: ParserHint.NATIVE.value,
        1: ParserHint.SCANNED.value,
        2: ParserHint.COMPLEX_LAYOUT.value,
    }
    # exactly one enqueue per page, scoped to the tenant
    assert len(enqueued) == 3
    assert all(t == "t1" for _, t in enqueued)
    # document was moved to processing
    assert (res.upload_id, DocStatus.PROCESSING.value, {}) in up.status_calls


def test_orchestrator_reprocess_reuses_existing_upload_id():
    existing = _Row(upload_id="up-old", status=DocStatus.FAILED.value)
    up = _FakeUploadRepo(existing=existing)
    pg = _FakePageRepo()
    deps, enqueued = _deps(up, pg, _Preflight(page_count=1))

    res = asyncio.run(
        ingest_document(b"%PDF-1.4", "t1", "u1", {}, deps=deps)
    )

    assert res.decision is DedupDecision.REPROCESS
    assert res.upload_id == "up-old"   # reused, not a fresh id
    assert up.created == []             # no new manifest row
    assert res.pages_enqueued == 1


def test_orchestrator_reconcile_persists_reduced_status():
    pages = [
        {"status": PageStatus.SUCCEEDED.value},
        {"status": PageStatus.FAILED_TERMINAL.value},
    ]
    up = _FakeUploadRepo()
    pg = _FakePageRepo(pages=pages)
    deps, _ = _deps(up, pg, _Preflight())

    status = asyncio.run(reconcile("up-1", deps))

    assert status == DocStatus.PARTIALLY_INDEXED.value
    assert ("up-1", DocStatus.PARTIALLY_INDEXED.value, {}) in up.status_calls


# --------------------------------------------------------------------------- #
# Task 13b — PageManifestRepo.load_page_inputs (page-input loader)
# --------------------------------------------------------------------------- #
def test_load_page_inputs_returns_pipeline_tuple():
    from pdf_chat.control_plane.repositories import PageManifestRepo

    class _FakeRow:
        page_blob_path = "az://x/p7.png"
        text_coverage_ratio = 0.92
        doc_id = "doc-1"
        acl_snapshot = {"public": True}
        page_num = 7                                   # real (non-zero) page number

    class _FakeSession:
        async def get(self, *a, **k):
            return None

    seen = {}

    def _fetch(tid, *, tenant_id=None):                # tenant_id threaded through
        seen["tid"] = tid
        seen["tenant_id"] = tenant_id
        return _FakeRow()

    repo = PageManifestRepo(_FakeSession())
    repo._fetch_row = _fetch                            # injected for the pure test
    repo._download = lambda path: b"PNGBYTES"          # injected blob fetch
    repo._render_page = lambda blob: ("page-obj", b"PNGBYTES")

    page, image, coverage, doc_id, acl, page_num = asyncio.run(
        repo.load_page_inputs("pg-1", tenant_id="t9")
    )
    assert coverage == 0.92
    assert doc_id == "doc-1"
    assert acl == {"public": True}
    assert image == b"PNGBYTES"
    assert page_num == 7                               # real page_num returned
    assert seen["tenant_id"] == "t9"                   # tenant scope threaded

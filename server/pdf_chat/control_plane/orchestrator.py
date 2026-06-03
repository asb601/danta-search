"""Ingestion orchestrator — composes the control-plane + ingestion steps that
take a raw upload all the way to a fanned-out set of per-page Celery tasks.

This is the missing seam between the API route and the workers (Spec §5 Stage
0–7). It is deliberately a thin COORDINATOR: every side-effectful capability
(hashing, preflight, persistence, page-count derivation, task enqueue) is an
injected adapter on :class:`IngestDeps`, so the decision flow is unit-testable
with in-memory fakes and ZERO infra. The orchestrator itself contains only the
sequencing + the dedup branch.

Flow (``ingest_document``)::

    compute_sha256
      → find_by_sha256 + decide_dedup
          SKIP      → return existing upload_id (queue nothing)
          REPROCESS → reuse the existing upload_id, re-run the pipeline
          NEW       → fresh upload_id
      → run_preflight  (rejected → mark failed, return; queue nothing)
      → create_upload manifest (status=uploaded, preflight_json, acl_snapshot)
      → derive page_count (from preflight)
      → create_pages   (one PageManifest per page, parser_hint per page)
      → enqueue process_page_task per page  (status → processing)

``reconcile`` reduces the settled page statuses to a document status via
``reconcile_document_status`` and persists it.

The repos do NOT commit (the caller owns the transaction); ``ingest_document``
calls ``deps.commit()`` at the end so a route can run it as one atomic unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from pdf_chat.control_plane.dedup import decide_dedup
from pdf_chat.control_plane.state_machine import reconcile_document_status
from pdf_chat.models.enums import DedupDecision, DocStatus, ParserHint
from pdf_chat.models.manifests import PageManifest

__all__ = ["IngestDeps", "IngestResult", "ingest_document", "reconcile"]


# --------------------------------------------------------------------------- #
# Injected-adapter protocols. Fakes in tests and the real repos/services both
# satisfy these. Everything that touches infra lives behind one of these.
# --------------------------------------------------------------------------- #
class UploadRepoProto(Protocol):
    async def create_upload(self, **fields: Any) -> Any: ...
    async def set_status(self, upload_id: str, status: str, **fields: Any) -> None: ...
    async def find_by_sha256(self, sha256: str, tenant_id: str) -> Any | None: ...


class PageRepoProto(Protocol):
    async def create_pages(self, pages: Any) -> Any: ...
    async def get_all_pages(self, upload_id: str) -> Any: ...


@dataclass
class IngestDeps:
    """Adapters injected into the orchestrator (testable with fakes).

    Attributes:
        upload_repo / page_repo: control-plane persistence (Team A repos).
        hash_fn: bytes → sha256 hex (``compute_sha256``).
        preflight_fn: bytes → object with ``.rejected``/``.reject_reason``/
            ``.page_count``/``.scanned_pages``/``.complex_layout_pages``/
            ``.to_dict()`` (``run_preflight`` → ``PreflightResult``).
        blob_writer: persists raw bytes, returns the blob uri (async).
        enqueue_fn: enqueues one page task ``(task_id, tenant_id)`` (async).
        commit: commits the transaction the orchestrator built (async, optional).
    """

    upload_repo: UploadRepoProto
    page_repo: PageRepoProto
    hash_fn: Callable[[bytes], str]
    preflight_fn: Callable[[bytes], Any]
    blob_writer: Callable[..., Awaitable[str]]
    enqueue_fn: Callable[[str, str], Awaitable[None]]
    commit: Callable[[], Awaitable[None]] | None = None


@dataclass
class IngestResult:
    """Outcome of an ``ingest_document`` call."""

    upload_id: str
    status: str
    deduplicated: bool = False
    rejected: bool = False
    reject_reason: str | None = None
    pages_enqueued: int = 0
    decision: DedupDecision = DedupDecision.NEW


def _upload_id_of(row: Any) -> str:
    return str(row.get("upload_id") if isinstance(row, dict) else getattr(row, "upload_id"))


def _parser_hint_for_page(
    page_num: int, scanned: set[int], complex_layout: set[int]
) -> str:
    """Per-page routing hint derived from preflight signals (Stage 6).

    Precedence mirrors the preflight classifier: scanned (needs OCR) →
    complex_layout → native. Entropy-based VLM escalation is decided later by the
    parser router at extraction time, not here.
    """
    if page_num in scanned:
        return ParserHint.SCANNED.value
    if page_num in complex_layout:
        return ParserHint.COMPLEX_LAYOUT.value
    return ParserHint.NATIVE.value


async def ingest_document(
    file_bytes: bytes,
    tenant_id: str,
    user_id: str,
    acl: dict,
    *,
    deps: IngestDeps,
    filename: str = "document.pdf",
    content_type: str | None = None,
) -> IngestResult:
    """Run the upload → preflight → manifest → page fan-out pipeline.

    Returns an :class:`IngestResult`. Queues nothing on a dedup SKIP or a
    preflight rejection. On REPROCESS the existing ``upload_id`` is reused.
    """
    sha256 = deps.hash_fn(file_bytes)

    # --- Dedup decision (pure) over the tenant-scoped fingerprint lookup. -----
    existing = await deps.upload_repo.find_by_sha256(sha256, tenant_id)
    existing_status = None
    if existing is not None:
        existing_status = (
            existing.get("status") if isinstance(existing, dict) else getattr(existing, "status", None)
        )
    decision = decide_dedup(existing_status, same_tenant=existing is not None)

    if decision is DedupDecision.SKIP:
        # Pure duplicate (already indexed or in-flight) — return the existing id.
        return IngestResult(
            upload_id=_upload_id_of(existing),
            status=str(existing_status),
            deduplicated=True,
            decision=decision,
        )

    # --- Preflight gate (the "bouncer"). Reject before any expensive work. ----
    preflight = deps.preflight_fn(file_bytes)
    if getattr(preflight, "rejected", False):
        reason = getattr(preflight, "reject_reason", None)
        # Persist a failed manifest so the rejection is observable/auditable.
        blob_uri = await deps.blob_writer(
            file_bytes=file_bytes, tenant_id=tenant_id, sha256=sha256, filename=filename
        )
        preflight_json = preflight.to_dict() if hasattr(preflight, "to_dict") else None
        if decision is DedupDecision.REPROCESS and existing is not None:
            upload_id = _upload_id_of(existing)
            await deps.upload_repo.set_status(
                upload_id, DocStatus.FAILED.value, error_message=reason,
                preflight_json=preflight_json,
            )
        else:
            row = await deps.upload_repo.create_upload(
                blob_uri=blob_uri,
                sha256=sha256,
                content_length=len(file_bytes),
                mime_type=getattr(preflight, "mime_type", content_type),
                page_count=getattr(preflight, "page_count", 0),
                tenant_id=tenant_id,
                user_id=user_id,
                acl_snapshot=acl or {},
                preflight_json=preflight_json,
                status=DocStatus.FAILED.value,
                error_message=reason,
            )
            upload_id = _upload_id_of(row)
        if deps.commit is not None:
            await deps.commit()
        return IngestResult(
            upload_id=upload_id,
            status=DocStatus.FAILED.value,
            rejected=True,
            reject_reason=reason,
            decision=decision,
        )

    # --- Accepted: persist raw bytes + the upload manifest. -------------------
    blob_uri = await deps.blob_writer(
        file_bytes=file_bytes, tenant_id=tenant_id, sha256=sha256, filename=filename
    )
    preflight_json = preflight.to_dict() if hasattr(preflight, "to_dict") else None
    page_count = int(getattr(preflight, "page_count", 0) or 0)

    if decision is DedupDecision.REPROCESS and existing is not None:
        upload_id = _upload_id_of(existing)
        await deps.upload_repo.set_status(
            upload_id, DocStatus.UPLOADED.value, error_message=None,
            preflight_json=preflight_json, page_count=page_count, blob_uri=blob_uri,
        )
    else:
        row = await deps.upload_repo.create_upload(
            blob_uri=blob_uri,
            sha256=sha256,
            content_length=len(file_bytes),
            mime_type=getattr(preflight, "mime_type", content_type),
            page_count=page_count,
            tenant_id=tenant_id,
            user_id=user_id,
            acl_snapshot=acl or {},
            preflight_json=preflight_json,
            status=DocStatus.UPLOADED.value,
        )
        upload_id = _upload_id_of(row)

    # --- Page manifest fan-out (created BEFORE enqueue → crash recovery). -----
    scanned = set(getattr(preflight, "scanned_pages", []) or [])
    complex_layout = set(getattr(preflight, "complex_layout_pages", []) or [])
    page_specs = [
        {
            "task_id": PageManifest.make_task_id(upload_id, n),
            "upload_id": upload_id,
            "page_num": n,
            "parser_hint": _parser_hint_for_page(n, scanned, complex_layout),
        }
        for n in range(page_count)
    ]
    if page_specs:
        await deps.page_repo.create_pages(page_specs)

    # Mark the document as processing, then enqueue one task per page.
    await deps.upload_repo.set_status(upload_id, DocStatus.PROCESSING.value)
    enqueued = 0
    for spec in page_specs:
        await deps.enqueue_fn(spec["task_id"], tenant_id)
        enqueued += 1

    if deps.commit is not None:
        await deps.commit()

    return IngestResult(
        upload_id=upload_id,
        status=DocStatus.PROCESSING.value,
        deduplicated=False,
        pages_enqueued=enqueued,
        decision=decision,
    )


async def reconcile(upload_id: str, deps: IngestDeps) -> str:
    """Reduce the document's settled page statuses to a document status.

    Reads all page rows, applies the pure ``reconcile_document_status``, and
    persists the result on the upload manifest. Returns the new document status.
    """
    pages = await deps.page_repo.get_all_pages(upload_id)
    statuses = [
        (p.get("status") if isinstance(p, dict) else getattr(p, "status")) for p in pages
    ]
    doc_status = reconcile_document_status(statuses)
    await deps.upload_repo.set_status(upload_id, doc_status)
    if deps.commit is not None:
        await deps.commit()
    return doc_status

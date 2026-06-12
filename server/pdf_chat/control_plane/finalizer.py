"""Document reconciliation trigger — runs after each page settles.

The page tasks write per-page outcomes, but nothing moved the DOCUMENT from
``processing`` to a terminal status, so a fully-extracted PDF stayed "processing"
forever and a bridged File row stayed "running". :func:`finalize_if_complete`
closes that loop: once EVERY page of a document has settled it reduces them to a
document status (reusing the pure :func:`reconcile_document_status`), persists it,
and — for documents that arrived via the file-manager bridge — mirrors the result
back onto the originating ``files.ingest_status`` (the file-manager badge).

It is idempotent and safe under Celery's at-least-once delivery: re-running after
the document already settled simply re-writes the same terminal status. It only
acts when all pages are settled, so it never marks a doc terminal prematurely.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from pdf_chat.control_plane.state_machine import reconcile_document_status
from pdf_chat.models.enums import DocStatus, SETTLED_PAGE_STATES
from pdf_chat.models.manifests import PageManifest, UploadManifest

_SETTLED = frozenset(s.value for s in SETTLED_PAGE_STATES)
_DOC_OK = frozenset({DocStatus.INDEXED.value, DocStatus.PARTIALLY_INDEXED.value})


async def finalize_if_complete(upload_id: str) -> "str | None":
    """Reconcile + persist the document status iff all its pages have settled.

    Returns the new document status when finalized, else ``None`` (still in
    flight, or the document has no pages). Best-effort callers should ignore
    exceptions — a missed finalize is recovered on the next page's call.
    """
    from app.core.database import async_session  # type: ignore

    async with async_session() as session:
        statuses = (
            await session.execute(
                select(PageManifest.status).where(PageManifest.upload_id == upload_id)
            )
        ).scalars().all()

        if not statuses or any(s not in _SETTLED for s in statuses):
            return None  # no pages yet, or at least one still in flight

        doc_status = reconcile_document_status(list(statuses))

        manifest = await session.get(UploadManifest, upload_id)
        if manifest is None:
            return None
        # Capture the prior status BEFORE the overwrite so the expensive Phase-2/
        # Phase-5 graph build is triggered exactly once — on the FIRST transition
        # into a terminal indexed state, not on every idempotent re-finalize.
        was_indexed = manifest.status in _DOC_OK
        manifest.status = doc_status

        # Mirror onto the originating File row when this came via the bridge.
        if manifest.source_file_id:
            await _sync_file_status(session, manifest.source_file_id, doc_status)

        tenant_id = manifest.tenant_id
        container_id = manifest.container_id
        await session.commit()

    # First-time settle into an indexed state → build the knowledge graph +
    # tenant comprehension out-of-band (Celery in production). Best-effort: a
    # failed enqueue never fails the page, and both phases are idempotent so a
    # later reconciler can re-trigger them.
    if doc_status in _DOC_OK and not was_indexed:
        _trigger_graph_build(upload_id, tenant_id, container_id)

    return doc_status


def _trigger_graph_build(upload_id: str, tenant_id: str, container_id: "str | None") -> None:
    """Enqueue the Phase-2/Phase-5 graph build on the dedicated PDF queue.

    Mirrors ``upload_service.enqueue_page`` routing (``apply_async(queue=...)`` is
    authoritative so the build can never leak onto the CSV queue). Swallows every
    error — graph build is an enhancement over the already-retrievable vector
    chunks, never a blocker.
    """
    try:
        from pdf_chat.config import get_pdf_settings
        from pdf_chat.ingestion.tasks import build_document_graph_task

        build_document_graph_task.apply_async(
            args=[upload_id],
            kwargs={"tenant_id": tenant_id, "container_id": container_id},
            queue=get_pdf_settings().ingest_queue,
        )
    except Exception:  # pragma: no cover - enqueue is best-effort
        pass


async def _sync_file_status(session: Any, file_id: str, doc_status: str) -> None:
    """Map a document status onto ``files.ingest_status`` (best-effort)."""
    from app.models.file import File  # type: ignore
    from app.services.ingestion_config import IngestStatus  # type: ignore

    file_status = (
        IngestStatus.INGESTED.value if doc_status in _DOC_OK else IngestStatus.FAILED.value
    )
    f = await session.get(File, file_id)
    if f is not None:
        f.ingest_status = file_status

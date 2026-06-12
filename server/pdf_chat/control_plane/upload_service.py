"""upload_service — shared wiring for the PDF ingestion control plane.

Two upload entry points converge here:

  * ``/api/pdf/upload``         → :func:`handle_upload` (raw bytes in the request;
                                  this module uploads them to the tenant container
                                  under a ``pdfs/`` prefix), and
  * the file-manager bridge     → :mod:`pdf_chat.control_plane.file_manager_bridge`
                                  (bytes already in the tenant container; it reuses
                                  :func:`build_ingest_deps` with a no-op writer).

Both reuse :func:`build_ingest_deps` and :func:`enqueue_page` so the orchestrator
contract (hash → preflight → manifest → page fan-out) is defined ONCE. Imported
late (inside route bodies) so the module stays importable with zero infra.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..config import get_pdf_settings


def enqueue_page(task_id: str, tenant_id: str) -> Awaitable[None]:
    """Dispatch ONE per-page extraction task onto the dedicated PDF queue.

    This is the real implementation of the orchestrator's ``enqueue_fn`` seam
    (previously a stub). The queue name comes from config (``PDF_INGEST_QUEUE``)
    so routing is never hardcoded; the isolated PDF worker (Option B) consumes it
    with ``-Q <that queue>``. ``apply_async(queue=...)`` is authoritative — it
    overrides any task-default/route so a page can NEVER leak onto the CSV queue.

    Returns an already-resolved awaitable so the async orchestrator can ``await``
    it uniformly (Celery publish itself is synchronous and fast).
    """
    from pdf_chat.ingestion.tasks import process_page_task

    async def _publish() -> None:
        process_page_task.apply_async(
            args=[task_id],
            kwargs={"tenant_id": tenant_id},
            queue=get_pdf_settings().ingest_queue,
        )

    return _publish()


def build_ingest_deps(
    session: Any,
    *,
    sha256: str,
    blob_writer: Callable[..., Awaitable[str]],
):
    """Assemble :class:`IngestDeps` over a live session + a chosen blob writer.

    ``blob_writer`` is the ONLY thing that differs between the two entry points:
    ``/api/pdf/upload`` uploads the bytes; the bridge returns the existing blob
    uri without re-uploading. Everything else (hash, preflight, repos, enqueue,
    commit) is shared here so the ingest contract lives in one place.
    """
    from pdf_chat.control_plane.orchestrator import IngestDeps
    from pdf_chat.control_plane.repositories import PageManifestRepo, UploadManifestRepo
    from pdf_chat.ingestion.preflight import run_preflight

    async def _commit() -> None:
        await session.commit()

    return IngestDeps(
        upload_repo=UploadManifestRepo(session),
        page_repo=PageManifestRepo(session),
        hash_fn=lambda _b: sha256,
        preflight_fn=run_preflight,
        blob_writer=blob_writer,
        enqueue_fn=enqueue_page,
        commit=_commit,
    )


async def handle_upload(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    sha256: str,
    tenant_id: str,
    user_id: str,
    container_id: str,
    connection_string: str,
    container_name: str,
) -> dict[str, Any]:
    """Ingest a PDF whose bytes are in the request (native ``/api/pdf/upload``).

    Uploads the bytes to the tenant's shared Azure container under
    ``pdfs/<tenant>/<sha256>/<filename>`` (namespaced away from Excel/Parquet
    blobs), writes the manifest tagged with ``container_id`` (so the page worker
    can resolve the per-tenant connection string), and fans out the page tasks.
    """
    from app.core.database import async_session  # type: ignore
    from pdf_chat.control_plane.orchestrator import ingest_document

    blob_writer = _uploading_blob_writer(connection_string, container_name)

    async with async_session() as session:
        deps = build_ingest_deps(session, sha256=sha256, blob_writer=blob_writer)
        result = await ingest_document(
            file_bytes,
            tenant_id,
            user_id,
            {},
            deps=deps,
            filename=filename,
            content_type=content_type,
            container_id=container_id,
            source_file_id=None,  # native upload has no originating File row
        )

    return {
        "upload_id": result.upload_id,
        "status": result.status,
        "deduplicated": result.deduplicated,
    }


def _uploading_blob_writer(connection_string: str, container_name: str):
    """Async writer that uploads PDF bytes to the tenant container, returns the uri.

    Stored under ``pdfs/<tenant_id>/<sha256>/<filename>`` so PDF blobs are
    namespaced away from the Excel/Parquet blobs in the SAME container.
    """

    async def blob_writer(
        *, file_bytes: bytes, tenant_id: str, sha256: str, filename: str
    ) -> str:
        from azure.storage.blob.aio import BlobServiceClient  # type: ignore

        blob_path = f"pdfs/{tenant_id}/{sha256}/{filename}"
        async with BlobServiceClient.from_connection_string(connection_string) as client:
            bc = client.get_blob_client(container=container_name, blob=blob_path)
            await bc.upload_blob(file_bytes, overwrite=True)
        return f"az://{container_name}/{blob_path}"

    return blob_writer

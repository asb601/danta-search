"""upload_service — wires IngestDeps and calls the ingest orchestrator.

Imported late (inside the upload route body) so the module is importable
without infra. The caller resolves Azure credentials from the app's
ContainerConfig (same storage account and container as the Excel pipeline),
then passes them here. No PDF-specific storage account is introduced.
"""
from __future__ import annotations

from typing import Any


async def handle_upload(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    sha256: str,
    tenant_id: str,
    user_id: str,
    connection_string: str,
    container_name: str,
) -> dict[str, Any]:
    """Orchestrate a PDF upload into the tenant's shared Azure container.

    Uses the same ``connection_string`` and ``container_name`` from
    ``ContainerConfig`` that the Excel/CSV pipeline uses — PDFs land under
    ``pdfs/<tenant_id>/<sha256>/<filename>`` within that container.
    """
    from app.core.database import async_session  # type: ignore
    from pdf_chat.control_plane.orchestrator import IngestDeps, ingest_document
    from pdf_chat.control_plane.repositories import PageManifestRepo, UploadManifestRepo
    from pdf_chat.ingestion.preflight import run_preflight

    blob_writer = _build_blob_writer(connection_string, container_name)

    async def _enqueue(task_id: str, _tenant_id: str) -> None:
        # TODO: wire to Celery task for page-level extraction fan-out
        pass

    async with async_session() as db:
        upload_repo = UploadManifestRepo(db)
        page_repo = PageManifestRepo(db)

        async def _commit() -> None:
            await db.commit()

        deps = IngestDeps(
            upload_repo=upload_repo,
            page_repo=page_repo,
            hash_fn=lambda b: sha256,
            preflight_fn=run_preflight,
            blob_writer=blob_writer,
            enqueue_fn=_enqueue,
            commit=_commit,
        )

        result = await ingest_document(
            file_bytes,
            tenant_id,
            user_id,
            {},
            deps=deps,
            filename=filename,
            content_type=content_type,
        )

    return {
        "upload_id": result.upload_id,
        "status": result.status,
        "deduplicated": result.deduplicated,
    }


def _build_blob_writer(connection_string: str, container_name: str):
    """Async callable that uploads PDF bytes and returns the blob URI.

    Stores under ``pdfs/<tenant_id>/<sha256>/<filename>`` so PDF blobs are
    namespaced away from Excel/Parquet blobs within the shared container.
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

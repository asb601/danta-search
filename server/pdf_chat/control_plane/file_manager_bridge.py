"""File-manager → PDF-pipeline bridge.

The file-manager upload path (``/api/files/upload-url`` → ``/confirm-upload``)
streams bytes straight to the tenant's Azure container via SAS — the server
never holds them. CSV/XLSX are auto-ingested there; PDFs were silently left
``not_ingested`` because the CSV pipeline can't process them.

This module bridges that gap WITHOUT duplicating the ingest contract: a Celery
task (run on the dedicated PDF queue) downloads the already-uploaded blob, then
reuses :func:`upload_service.build_ingest_deps` + the orchestrator exactly like
the native ``/api/pdf/upload`` path. The only difference is the blob writer —
here it's a NO-OP that returns the existing blob uri (the bytes are already in
the container), so nothing is re-uploaded.

The originating ``files.id`` is recorded on the manifest (``source_file_id``) so
reconciliation mirrors the document status back onto the File row — that's what
flips the file-manager badge from "not ingested" to "ingested".
"""
from __future__ import annotations

import asyncio
import mimetypes
from typing import Any

try:
    from celery import shared_task  # type: ignore

    _HAS_CELERY = True
except ImportError:  # pragma: no cover - import-safe without Celery
    shared_task = None  # type: ignore
    _HAS_CELERY = False


async def _download_blob_bytes(
    connection_string: str, container_name: str, blob_path: str
) -> bytes:
    """Fetch the whole document blob (per-tenant connection string).

    One PDF held briefly in worker memory is acceptable (preflight + hashing need
    the bytes, and PyMuPDF later streams page-by-page from the stored blob). Uses
    the same client shape as ``parquet_service.py`` — no new auth path.
    """
    from azure.storage.blob.aio import BlobServiceClient  # type: ignore

    async with BlobServiceClient.from_connection_string(connection_string) as client:
        bc = client.get_blob_client(container=container_name, blob=blob_path)
        stream = await bc.download_blob()
        return await stream.readall()


async def _set_file_status(session: Any, file_id: str, status: str) -> None:
    """Best-effort update of ``files.ingest_status`` (mirrors the manifest)."""
    from app.models.file import File  # type: ignore

    f = await session.get(File, file_id)
    if f is not None:
        f.ingest_status = status


async def run_bridge_ingest(file_id: str) -> dict[str, Any]:
    """Document-level ingest for a file-manager PDF (already in blob storage).

    Returns the ingest result dict. Raises on unrecoverable wiring errors so
    Celery records the failure; the File row is marked FAILED first so the UI
    never shows a stuck "pending".
    """
    from app.core.database import async_session  # type: ignore
    from app.models.container import ContainerConfig  # type: ignore
    from app.models.file import File  # type: ignore
    from app.models.user import User  # type: ignore
    from app.services.ingestion_config import IngestStatus  # type: ignore
    from pdf_chat.control_plane.orchestrator import ingest_document
    from pdf_chat.control_plane.upload_service import build_ingest_deps
    from pdf_chat.ingestion.fingerprint import compute_sha256

    # ── 1. Resolve everything to PLAIN SCALARS inside one session, then mark the
    #       File in-flight. Nothing ORM-bound escapes the block (no detached reads).
    async with async_session() as session:
        file_row = await session.get(File, file_id)
        if file_row is None or not file_row.blob_path or not file_row.container_id:
            raise ValueError(f"bridge: file {file_id!r} missing blob_path/container_id")
        container_id = file_row.container_id
        blob_path = file_row.blob_path
        display_name = file_row.name or _name_from_blob(blob_path)
        user_id = file_row.owner_id or file_row.uploaded_by_id or ""

        config = await session.get(ContainerConfig, container_id)
        if config is None or not config.connection_string:
            raise ValueError(f"bridge: no container config for file {file_id!r}")
        connection_string = config.connection_string
        container_name = config.container_name
        # Tenant boundary == the organization PDF chat scopes retrieval by. Resolve
        # it the same way the rest of the app does (org), loading the owner so an
        # org-less container can still fall back to the owner's org — never silently
        # under container_id (which org-scoped chat could not query). See
        # :func:`_resolve_tenant_id`.
        owner = await session.get(User, user_id) if user_id else None
        tenant_id = _resolve_tenant_id(
            config, owner, file_id=file_id, container_id=container_id
        )

        file_row.ingest_status = IngestStatus.RUNNING.value
        await session.commit()

    # ── 2. Download the already-uploaded blob (per-tenant connection string).
    try:
        file_bytes = await _download_blob_bytes(
            connection_string, container_name, blob_path
        )
    except Exception:
        async with async_session() as session:
            await _set_file_status(session, file_id, IngestStatus.FAILED.value)
            await session.commit()
        raise

    sha256 = compute_sha256(file_bytes)
    content_type = mimetypes.guess_type(blob_path)[0] or "application/pdf"

    # No-op writer: the bytes already live in the tenant container, so the manifest
    # simply points at the existing blob — never a re-upload.
    async def _existing_blob_writer(**_kw: Any) -> str:
        return f"az://{container_name}/{blob_path}"

    # ── 3. Same ingest contract as /api/pdf/upload (manifest + page fan-out).
    async with async_session() as session:
        deps = build_ingest_deps(session, sha256=sha256, blob_writer=_existing_blob_writer)
        result = await ingest_document(
            file_bytes,
            tenant_id,
            user_id,
            {},
            deps=deps,
            filename=display_name,
            content_type=content_type,
            container_id=container_id,
            source_file_id=file_id,
        )

    return {
        "upload_id": result.upload_id,
        "status": result.status,
        "deduplicated": result.deduplicated,
    }


def _resolve_tenant_id(
    container_config: Any,
    owner_user: Any,
    *,
    file_id: str,
    container_id: str,
) -> str:
    """Resolve the tenant (organization) a bridged document is indexed under.

    PDF chat scopes retrieval by ``principal.tenant_id`` == the querying user's
    ``organization_id`` (see ``api.routes._principal_from_user``), and the native
    ``/api/pdf/upload`` path indexes under that same org. A bridged document must
    therefore land under the SAME organization or chat can never reach it.

    Resolution order (data-driven — reads attributes, no magic values):

      1. the container's ``organization_id`` — the native-upload tenant, then
      2. the owning user's ``organization_id`` — the chat principal's tenant
         (covers an org-less container whose owner still has an org), then
      3. **fail loudly** — never silently index under ``container_id``, which an
         org-scoped chat principal would not query (that was the prior bug).

    The two ``organization_id`` reads are the single source of truth for "tenant
    == organization"; changing that identity is a one-line change here.
    """
    tenant_id = getattr(container_config, "organization_id", None) or getattr(
        owner_user, "organization_id", None
    )
    if not tenant_id:
        raise ValueError(
            f"bridge: cannot resolve a tenant (organization) for file {file_id!r} "
            f"in container {container_id!r} — neither the container nor its owner "
            f"has an organization_id. Indexing under container_id would hide the "
            f"document from org-scoped PDF chat, so the bridge refuses rather than "
            f"silently mis-scoping it."
        )
    return str(tenant_id)


def _name_from_blob(blob_path: str) -> str:
    """Display filename from a blob path (final path segment)."""
    return blob_path.rsplit("/", 1)[-1] or "document.pdf"


if _HAS_CELERY:  # pragma: no cover - requires infra

    @shared_task(name="pdf_chat.bridge.run_pdf_document_ingest")  # type: ignore[misc]
    def run_pdf_document_ingest(file_id: str) -> dict[str, Any]:
        """Celery entry point for the file-manager bridge (runs on the PDF queue)."""
        return asyncio.run(run_bridge_ingest(file_id))

else:

    def run_pdf_document_ingest(file_id: str) -> dict[str, Any]:  # type: ignore[misc]
        """Stub when Celery is absent — call ``run_bridge_ingest`` directly in tests."""
        raise RuntimeError("Celery required for run_pdf_document_ingest")

"""Read-side services for the PDF status + documents routes.

``api/routes.py`` imports these lazily; until now the module did not exist, so
``GET /api/pdf/status/{id}`` and ``GET /api/pdf/documents`` fell through to their
503 fallback and the frontend could never see a document. Both functions are
tenant-scoped (the caller passes the JWT-derived ``tenant_id``) and read-only.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from pdf_chat.models.enums import PageStatus
from pdf_chat.models.manifests import PageManifest, UploadManifest


async def get_upload_status(upload_id: str, *, tenant_id: str) -> "dict[str, Any] | None":
    """Reconciled document status + per-page counters, or None if not found.

    Tenant-scoped: a principal can only read its own tenant's documents.
    Shape matches :class:`schemas.pdf_schemas.StatusResponse`.
    """
    from app.core.database import async_session  # type: ignore

    async with async_session() as session:
        manifest = await session.get(UploadManifest, upload_id)
        if manifest is None or manifest.tenant_id != tenant_id:
            return None

        rows = (
            await session.execute(
                select(PageManifest.status, func.count())
                .where(PageManifest.upload_id == upload_id)
                .group_by(PageManifest.status)
            )
        ).all()

    by_status = {status: int(count) for status, count in rows}
    total = sum(by_status.values())
    succeeded = by_status.get(PageStatus.SUCCEEDED.value, 0)
    failed = by_status.get(PageStatus.FAILED_TERMINAL.value, 0)
    pending = total - succeeded - failed

    return {
        "upload_id": manifest.upload_id,
        "status": manifest.status,
        "page_count": manifest.page_count or total,
        "pages_succeeded": succeeded,
        "pages_failed": failed,
        "pages_pending": max(pending, 0),
        "error_message": manifest.error_message,
    }


async def list_documents(tenant_id: str) -> list[dict[str, Any]]:
    """All of a tenant's documents (newest first).

    Shape matches :class:`schemas.pdf_schemas.DocumentSummary`.
    """
    from app.core.database import async_session  # type: ignore

    async with async_session() as session:
        manifests = (
            await session.execute(
                select(UploadManifest)
                .where(UploadManifest.tenant_id == tenant_id)
                .order_by(UploadManifest.created_at.desc())
            )
        ).scalars().all()

    return [
        {
            "upload_id": m.upload_id,
            "status": m.status,
            "page_count": m.page_count,
            "mime_type": m.mime_type,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in manifests
    ]

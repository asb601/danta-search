"""Async repositories over the Control Plane manifest tables.

Thin persistence layer — these classes ONLY read and write rows. No business
logic lives here: dedup decisions, state transitions, and reconciliation are
computed by ``dedup.py`` / ``state_machine.py`` and the *result* is persisted
through these repos. This keeps the decision logic pure/unit-testable and the
repos a faithful mirror of the database.

SQLAlchemy 2.0 async style (``select`` / ``update`` + ``AsyncSession``), matching
``server/app/services`` conventions. The repos do NOT commit — the caller owns
the transaction boundary (so a whole upload + page-fan-out can be one atomic
unit). They ``flush`` where a returned object must be populated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pdf_chat.models.enums import PageStatus
from pdf_chat.models.manifests import PageManifest, UploadManifest

__all__ = ["UploadManifestRepo", "PageManifestRepo"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UploadManifestRepo:
    """Persistence for ``pdf_upload_manifest`` — one row per document."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_upload(self, **fields: Any) -> UploadManifest:
        """Insert a new document-level manifest row and return it.

        Caller supplies at least ``blob_uri``, ``sha256``, ``tenant_id``,
        ``user_id``. Defaults (``upload_id``, ``status``, timestamps) come from
        the model. Flushed so ``upload_id`` is populated for page fan-out.
        """
        row = UploadManifest(**fields)
        self._session.add(row)
        await self._session.flush()
        return row

    async def set_status(
        self,
        upload_id: str,
        status: str,
        *,
        error_message: str | None = None,
        **fields: Any,
    ) -> None:
        """Update the document status (+ optional error and other columns)."""
        values: dict[str, Any] = {"status": status, "updated_at": _now()}
        if error_message is not None:
            values["error_message"] = error_message
        values.update(fields)
        await self._session.execute(
            update(UploadManifest)
            .where(UploadManifest.upload_id == upload_id)
            .values(**values)
        )

    async def find_by_sha256(
        self, sha256: str, tenant_id: str
    ) -> UploadManifest | None:
        """Return the most recent manifest for this fingerprint within a tenant.

        Tenant-scoped on purpose: dedup isolation is per-tenant (see
        ``decide_dedup``). Newest row wins so the caller sees the latest attempt.
        """
        result = await self._session.execute(
            select(UploadManifest)
            .where(
                UploadManifest.sha256 == sha256,
                UploadManifest.tenant_id == tenant_id,
            )
            .order_by(UploadManifest.created_at.desc())
        )
        return result.scalars().first()


class PageManifestRepo:
    """Persistence for ``pdf_page_manifest`` — one row per page task."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_pages(
        self, pages: Iterable[dict[str, Any]]
    ) -> list[PageManifest]:
        """Bulk-insert page rows.

        Each dict must carry ``task_id``, ``upload_id``, ``page_num`` (other
        columns default in the model). Returns the ORM objects added.
        """
        rows = [PageManifest(**p) for p in pages]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def set_page_status(
        self,
        task_id: str,
        status: str,
        **fields: Any,
    ) -> None:
        """Update a single page's status, stamping lifecycle timestamps.

        ``started_at`` is set when entering ``running``; ``completed_at`` is set
        when entering any settled state. Any extra columns (``parser_used``,
        ``confidence``, ``error_message``) pass straight through via ``**fields``.
        """
        values: dict[str, Any] = {"status": status}
        if status == PageStatus.RUNNING.value:
            values.setdefault("started_at", _now())
        elif status in {
            PageStatus.SUCCEEDED.value,
            PageStatus.FAILED_TERMINAL.value,
            PageStatus.FAILED_RETRYABLE.value,
            PageStatus.NEEDS_HUMAN_REVIEW.value,
        }:
            values.setdefault("completed_at", _now())
        values.update(fields)
        await self._session.execute(
            update(PageManifest)
            .where(PageManifest.task_id == task_id)
            .values(**values)
        )

    async def get_pending_pages(self, upload_id: str) -> Sequence[PageManifest]:
        """Crash recovery: pages not yet settled — pending OR retryable.

        These are the tasks a reconciler must (re)queue after a worker crash.
        Settled pages (succeeded / failed_terminal / needs_human_review) are
        intentionally excluded so completed work is never re-processed.
        """
        settled = [s.value for s in (
            PageStatus.SUCCEEDED,
            PageStatus.FAILED_TERMINAL,
            PageStatus.NEEDS_HUMAN_REVIEW,
        )]
        result = await self._session.execute(
            select(PageManifest)
            .where(
                PageManifest.upload_id == upload_id,
                PageManifest.status.notin_(settled),
            )
            .order_by(PageManifest.page_num)
        )
        return result.scalars().all()

    async def get_all_pages(self, upload_id: str) -> Sequence[PageManifest]:
        """All pages for a document (used to reconcile the document status)."""
        result = await self._session.execute(
            select(PageManifest)
            .where(PageManifest.upload_id == upload_id)
            .order_by(PageManifest.page_num)
        )
        return result.scalars().all()

    async def increment_retry(self, task_id: str) -> None:
        """Atomically bump ``retry_count`` for a page (server-side increment)."""
        await self._session.execute(
            update(PageManifest)
            .where(PageManifest.task_id == task_id)
            .values(retry_count=PageManifest.retry_count + 1)
        )

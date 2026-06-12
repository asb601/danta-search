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

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pdf_chat.models.enums import PageStatus
from pdf_chat.models.manifests import PageManifest, UploadManifest

__all__ = [
    "UploadManifestRepo",
    "PageManifestRepo",
    "set_default_blob_reader",
    "set_default_container_blob_reader",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is.

    Lets ``load_page_inputs`` drive BOTH the real async seam bodies (which hit the
    DB / blob store / PyMuPDF) AND the simple sync lambdas the pure unit tests
    inject as instance attributes, without the loader knowing which it is. Mirrors
    the identical bridge in ``ingestion/tasks.py`` (kept local to avoid a
    control_plane → ingestion import at module load).
    """
    if inspect.isawaitable(value):
        return await value
    return value


# ── Process-wide default blob reader (installed by the worker bootstrap) ──────
#
# ``_download`` needs a blob client built from the per-org Azure connection
# string, which is NOT a pdf_chat env var — it must be injected. The worker
# bootstrap installs a default reader here (parallel to
# ``model_router.set_default_budget_store``); tests inject a fake at the seam
# instead. Leaving this ``None`` keeps ``repositories.py`` import-safe with zero
# infra and lets the local-disk path (file:// / plain paths) work unchanged.
_DEFAULT_BLOB_READER: "Callable[[str], bytes] | None" = None


def set_default_blob_reader(reader: "Callable[[str], bytes] | None") -> None:
    """Install the process-wide blob reader (``blob_uri -> bytes``).

    Called by the worker bootstrap with a closure over a real Azure
    ``BlobServiceClient``. ``reader`` may be sync or async; ``_download`` bridges
    both via ``_maybe_await``.
    """
    global _DEFAULT_BLOB_READER
    _DEFAULT_BLOB_READER = reader


# ── Per-CONTAINER blob reader (installed by the worker bootstrap) ─────────────
#
# Unlike ``_DEFAULT_BLOB_READER`` (one global Azure connection string), this
# reader resolves the PER-TENANT connection string from ``ContainerConfig`` by
# ``container_id`` — the same per-org credential the CSV pipeline uses. This is
# what makes the page worker able to read a document that lives in a tenant's own
# storage container (the common production case, where there is NO single global
# connection string). Signature: ``(blob_uri, container_id) -> bytes`` (sync or
# async). ``None`` until the worker bootstrap installs it; ``_download`` then
# falls back to the global reader / local path.
_DEFAULT_CONTAINER_BLOB_READER: "Callable[[str, str], Any] | None" = None


def set_default_container_blob_reader(
    reader: "Callable[[str, str], Any] | None",
) -> None:
    """Install the process-wide per-container blob reader (``(uri, cid) -> bytes``)."""
    global _DEFAULT_CONTAINER_BLOB_READER
    _DEFAULT_CONTAINER_BLOB_READER = reader


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
        tenant_id: str | None = None,
        **fields: Any,
    ) -> int:
        """Update the document status (+ optional error and other columns).

        ``tenant_id`` is OPTIONAL and BACKWARD COMPATIBLE: the orchestrator callers
        pass none and keep the upload_id-only UPDATE. When supplied (the delete
        path), a ``tenant_id`` predicate is added so a tenant can NEVER mutate
        another tenant's row (SECURITY). Returns the affected ``rowcount`` so the
        caller can distinguish a real update from an unknown-id / cross-tenant
        no-op (0 rows → the route's 404). Existing callers ignore the return.
        """
        values: dict[str, Any] = {"status": status, "updated_at": _now()}
        if error_message is not None:
            values["error_message"] = error_message
        values.update(fields)
        stmt = update(UploadManifest).where(UploadManifest.upload_id == upload_id)
        if tenant_id is not None:
            stmt = stmt.where(UploadManifest.tenant_id == tenant_id)
        result = await self._session.execute(stmt.values(**values))
        return result.rowcount

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

    def __init__(
        self,
        session: AsyncSession,
        *,
        blob_reader: "Callable[[str], bytes] | None" = None,
    ) -> None:
        self._session = session
        # Optional per-repo blob reader override. When None, ``_download`` falls
        # back to the process-wide default (worker bootstrap) and then to the
        # local-disk path, so the repo is functional in prod AND in tests.
        self._blob_reader = blob_reader

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

    async def load_page_inputs(self, task_id: str, *, tenant_id: str | None = None):
        """Load the per-page extraction inputs for the worker.

        Returns ``(page_obj, page_image_bytes, coverage, doc_id, acl, page_num)``
        — exactly what ``ingestion/tasks.py`` unpacks and ``extract_page_elements``
        consumes. The rendered ``fitz.Page`` + its PNG come from streaming the
        document blob to the one requested page (Hard rule #1: the whole PDF is
        never resident — PyMuPDF streams page-by-page). ``coverage`` is the
        measured extractable-text ratio derived at render time; ``doc_id`` is the
        document identity (``UploadManifest.upload_id``); ``page_num`` is the real
        page number from the row (so chunks carry their true page, never 0).

        Architectural decision (vs. the original stubs): there is NO per-page blob
        column or splitter in this module — the orchestrator writes ONE doc-level
        blob (``UploadManifest.blob_uri``) and the spec streams it per page. So the
        three seams re-point at ``(doc blob + page_num)``: ``_fetch_row`` returns
        the joined ``PageManifest``/``UploadManifest`` row, ``_download`` fetches
        the doc blob, and ``_render_page`` streams to ``row.page_num``.

        The three seams are ``async`` (``_fetch_row`` needs ``self._session``); we
        bridge through ``_maybe_await`` so the pure tests' sync fakes stay green.

        ``tenant_id`` is threaded to ``_fetch_row`` so the row fetch scopes the
        lookup to the owning tenant (no cross-tenant page read).
        """
        row = await _maybe_await(self._fetch_row(task_id, tenant_id=tenant_id))
        # Stash the doc's container so the real ``_download`` can resolve the
        # per-tenant connection string. Set as an attribute (not a call arg) so
        # the 1-arg ``_download`` fakes the unit tests inject keep working.
        self._doc_container_id = getattr(row, "container_id", None)
        blob = await _maybe_await(self._download(row.blob_uri))
        rendered = await _maybe_await(
            self._render_page(blob, page_num=row.page_num)
        )
        # ``_render_page`` returns (page_obj, image_bytes, coverage). Sync test
        # fakes may still return the legacy 2-tuple — fall back to the row's
        # text_coverage_ratio when present so the pinned loader test stays green.
        if len(rendered) == 3:
            page_obj, image_bytes, coverage = rendered
        else:  # pragma: no cover - legacy 2-tuple fake compatibility
            page_obj, image_bytes = rendered
            coverage = getattr(row, "text_coverage_ratio", 0.0)
        doc_id = getattr(row, "doc_id", None) or row.upload_id
        return (
            page_obj,
            image_bytes,
            coverage,
            doc_id,
            row.acl_snapshot,
            row.page_num,
        )

    async def _fetch_row(self, task_id: str, *, tenant_id: str | None = None):
        """Fetch the page row joined to its document, scoped to the tenant.

        Tenant identity + ACL + the doc blob uri live on ``UploadManifest`` (not
        ``PageManifest``), so this is a two-table read joined on ``upload_id``. The
        ``tenant_id`` predicate (when supplied) guarantees a page can NEVER be
        loaded outside its owning tenant (multi-tenant Hard rule #3 / sentinel
        WARN). A missing/cross-tenant row is a PERMANENT failure → the worker DLQs
        rather than retrying forever.

        Returns a Row exposing ``.page_num``, ``.upload_id``, ``.blob_uri``,
        ``.acl_snapshot``.
        """
        stmt = (
            select(
                PageManifest.page_num,
                PageManifest.upload_id,
                UploadManifest.blob_uri,
                UploadManifest.acl_snapshot,
                UploadManifest.container_id,
            )
            .join(
                UploadManifest,
                PageManifest.upload_id == UploadManifest.upload_id,
            )
            .where(PageManifest.task_id == task_id)
        )
        if tenant_id is not None:
            stmt = stmt.where(UploadManifest.tenant_id == tenant_id)
        row = (await self._session.execute(stmt)).first()
        if row is None:
            # Lazy import keeps repositories.py infra-free at module load and
            # avoids a hard control_plane → ingestion edge at import time.
            from pdf_chat.ingestion.tasks import PermanentError

            raise PermanentError(
                f"page row not found or cross-tenant for task_id={task_id!r}"
            )
        return row

    async def _download(self, blob_uri: str, *, container_id: str | None = None) -> bytes:
        """Download the document blob bytes for ``blob_uri``.

        Mirrors the established blob-access path WITHOUT inventing a new auth path:
          * a local-disk / ``file://`` path is read directly (test + dev),
          * when ``container_id`` is known AND a per-container reader is installed,
            the bytes are fetched with the PER-TENANT connection string resolved
            from ``ContainerConfig`` (the production case — documents live in each
            tenant's own storage container, with no single global credential), and
          * otherwise an ``az://``/``https://``/``blob://`` uri falls back to the
            injected per-repo reader → process-wide global reader (one shared
            Azure connection string), matching ``parquet_service.py``'s client shape.

        The container + blob are parsed from the uri (data-driven; never
        hardcoded), matching ``datafusion_client``'s ``az://container/blob`` shape.
        """
        if self._is_local_path(blob_uri):
            return self._read_local(blob_uri)
        # Prefer the per-tenant reader when we know which container the doc is in.
        # The container id arrives either as an explicit kwarg (direct callers) or
        # via the attribute ``load_page_inputs`` stashed from the joined manifest.
        cid = container_id or getattr(self, "_doc_container_id", None)
        if cid and _DEFAULT_CONTAINER_BLOB_READER is not None:
            return await _maybe_await(_DEFAULT_CONTAINER_BLOB_READER(blob_uri, cid))
        reader = self._blob_reader or _DEFAULT_BLOB_READER
        if reader is None:
            from pdf_chat.ingestion.tasks import PermanentError

            raise PermanentError(
                "no blob reader wired (worker bootstrap must install one) and "
                f"blob_uri is not a local path: {blob_uri!r}"
            )
        return await _maybe_await(reader(blob_uri))

    @staticmethod
    def _is_local_path(blob_uri: str) -> bool:
        """True when ``blob_uri`` points at the local filesystem (file:// / path)."""
        scheme = urlparse(blob_uri).scheme
        return scheme in ("", "file")

    @staticmethod
    def _read_local(blob_uri: str) -> bytes:
        """Read bytes from a local path or ``file://`` uri."""
        parsed = urlparse(blob_uri)
        path = parsed.path if parsed.scheme == "file" else blob_uri
        return Path(path).read_bytes()

    async def _render_page(self, blob: bytes, *, page_num: int):
        """Stream the document blob to ``page_num`` and render it.

        REUSES ``page_reader.stream_pages`` (the canonical ``fitz.open`` site —
        Hard rule #1: only one page is resident at a time) and
        ``page_reader.render_page_png`` (the canonical page → PNG site). Returns
        ``(page_obj, image_bytes, coverage)`` where:
          * ``page_obj`` is the real ``fitz.Page`` that ``extract_digital_page``
            consumes (``.get_text("dict")`` / ``.rect``),
          * ``image_bytes`` is the rendered PNG the OCR path consumes, and
          * ``coverage`` is the MEASURED extractable-text ratio (text-span area /
            page area) computed via the existing ``page_routing.text_coverage_ratio``
            — fully data-driven, no magic numbers.

        A requested page outside the document is a PERMANENT failure (the worker
        DLQs rather than retrying a structural error).
        """
        from pdf_chat.ingestion.page_reader import render_page_png, stream_pages
        from pdf_chat.ingestion.page_routing import text_coverage_ratio

        for n, page in stream_pages(blob):
            if n != page_num:
                continue
            image_bytes = render_page_png(page)
            rect = page.rect
            page_area = float(rect.width) * float(rect.height)
            text_area = 0.0
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:  # 0 == text block in PyMuPDF
                    continue
                bx = block.get("bbox", [0, 0, 0, 0])
                text_area += max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
            coverage = text_coverage_ratio(text_area=text_area, page_area=page_area)
            return page, image_bytes, coverage

        from pdf_chat.ingestion.tasks import PermanentError

        raise PermanentError(
            f"page {page_num} out of range for document blob"
        )

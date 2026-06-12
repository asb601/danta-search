"""Stages 7 + 9 + 14 — Celery worker: per-page extraction with retry/DLQ.

One Celery task per page (Stage 7). ``acks_late=True`` returns a task to the
queue if the worker dies mid-page. Transient failures retry with exponential
backoff; once retries are exhausted the page is marked terminal and its id is
pushed to the per-tenant dead-letter queue (Stage 4.4).

The Celery + Redis imports are GUARDED so this module imports with zero infra.
The retry/DLQ control flow lives in the pure, testable ``_run_page_extraction``;
the Celery decorator is applied to ``process_page_task`` only when Celery is
present, otherwise a clear stub is exported.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..config import get_pdf_settings
from ..models.enums import PageStatus
from .chunker import chunk_elements
from .extraction_confidence import propagate_confidence
from .page_extraction import extract_page_elements
from .retrieval_embeddings_shim import embed_texts_batched

try:
    from celery import shared_task  # type: ignore

    _HAS_CELERY = True
except ImportError:  # pragma: no cover - exercised only without infra
    shared_task = None  # type: ignore
    _HAS_CELERY = False

try:
    import redis  # type: ignore

    _HAS_REDIS = True
except ImportError:  # pragma: no cover
    redis = None  # type: ignore
    _HAS_REDIS = False


class TransientError(Exception):
    """Temporary failure (network/OCR timeout) — eligible for retry."""


class PermanentError(Exception):
    """Permanent failure (corrupt page bytes) — straight to terminal + DLQ."""


def dlq_key(tenant_id: str) -> str:
    """Redis LIST key for a tenant's ingestion dead-letter queue."""
    return f"dlq:ingestion:{tenant_id}"


def retry_countdown(retries: int, base_delay: int | None = None) -> int:
    """Exponential backoff: ``base_delay * 2**retries`` (pure, testable)."""
    if base_delay is None:
        base_delay = get_pdf_settings().retry_base_delay
    return base_delay * (2 ** retries)


def push_to_dlq(redis_client: Any, tenant_id: str, task_id: str) -> None:
    """Push a terminally-failed task id onto the tenant DLQ (best-effort)."""
    if redis_client is None:
        return
    redis_client.lpush(dlq_key(tenant_id), task_id)


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is.

    Lets ``_run_page_extraction`` drive BOTH the real async control-plane repo
    (``PageManifestRepo.set_page_status`` is ``async def``) AND simple sync fakes
    in unit tests, without the repo needing to know which it is.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def _set_page_status(
    page_repo: Any, task_id: str, status: str, *, error: str | None = None
) -> None:
    """Set a page status against either repo surface (async or sync).

    Bridges the naming/signature gap (Team A's repo uses the async
    ``set_page_status(task_id, status, **fields)`` with ``error_message``; older
    fakes use ``set_status(task_id, status, error=...)``). We prefer the real
    ``set_page_status`` and pass ``error_message`` when an error is present.
    """
    set_page_status = getattr(page_repo, "set_page_status", None)
    if set_page_status is not None:
        if error is not None:
            await _maybe_await(set_page_status(task_id, status, error_message=error))
        else:
            await _maybe_await(set_page_status(task_id, status))
        return
    # Fallback to the legacy ``set_status`` surface (test fakes).
    set_status = page_repo.set_status
    if error is not None:
        await _maybe_await(set_status(task_id, status, error=error))
    else:
        await _maybe_await(set_status(task_id, status))


async def _run_page_extraction(
    task_id: str,
    *,
    tenant_id: str,
    extract_fn: Callable[[str], Any],
    page_repo: Any,
    redis_client: Any = None,
    retries: int = 0,
    max_retries: int | None = None,
    on_retry: Callable[[int], None] | None = None,
) -> str:
    """Core per-page extraction with the spec's retry/DLQ semantics.

    This is the testable heart of ``process_page_task`` — independent of Celery
    so the state transitions can be unit-tested without a broker. It is ``async``
    so it can drive the real async ``PageManifestRepo`` (Team A); sync fakes are
    supported transparently via ``_maybe_await``.

    Flow:
        running → extract → succeeded
        TransientError + retries < max → increment_retry, schedule retry
                                          (``on_retry(countdown)``), re-raise
        TransientError + retries >= max → failed_terminal + DLQ
        PermanentError                  → failed_terminal + DLQ

    Args:
        task_id: page manifest id (== page task id).
        tenant_id: owning tenant (for DLQ key + isolation).
        extract_fn: does the actual Stage 9 extraction; raises Transient/Permanent.
            May be sync or async (awaited via ``_maybe_await``).
        page_repo: page manifest repo. Either Team A's async ``PageManifestRepo``
            (``set_page_status`` / ``increment_retry``) or a sync fake
            (``set_status`` / ``increment_retry``) — both work.
        redis_client: Redis client for DLQ lpush (optional).
        retries: the current Celery retry count (``self.request.retries``).
        max_retries: cap (defaults to config ``max_retries``).
        on_retry: invoked with the backoff ``countdown`` when a retry is
            scheduled (Celery binds this to ``self.retry``). May be sync/async.

    Returns:
        The terminal :class:`PageStatus` value reached this attempt.

    Raises:
        TransientError: re-raised when a retry is scheduled (Celery requeues).
    """
    if max_retries is None:
        max_retries = get_pdf_settings().max_retries

    await _set_page_status(page_repo, task_id, PageStatus.RUNNING.value)
    try:
        await _maybe_await(extract_fn(task_id))
        await _set_page_status(page_repo, task_id, PageStatus.SUCCEEDED.value)
        return PageStatus.SUCCEEDED.value

    except TransientError as exc:
        if retries < max_retries:
            await _maybe_await(page_repo.increment_retry(task_id))
            await _set_page_status(page_repo, task_id, PageStatus.FAILED_RETRYABLE.value)
            countdown = retry_countdown(retries)
            if on_retry is not None:
                await _maybe_await(on_retry(countdown))
            raise  # Celery wrapper converts this into self.retry(...)
        # retries exhausted → terminal + DLQ
        await _set_page_status(
            page_repo, task_id, PageStatus.FAILED_TERMINAL.value, error=str(exc)
        )
        push_to_dlq(redis_client, tenant_id, task_id)
        return PageStatus.FAILED_TERMINAL.value

    except PermanentError as exc:
        await _set_page_status(
            page_repo, task_id, PageStatus.FAILED_TERMINAL.value, error=str(exc)
        )
        push_to_dlq(redis_client, tenant_id, task_id)
        return PageStatus.FAILED_TERMINAL.value


def run_page_pipeline(
    *,
    page: Any,
    page_image_bytes: bytes,
    coverage: float,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
    writer: Any,
) -> int:
    """Full per-page chain: extract → confidence → chunk → embed → write.

    Pure orchestration over injected backends (the extractors/embedder are
    module-scope and monkeypatchable; ``writer`` is a Neo4jWriter-like object).
    Returns the number of chunks written. This IS the worker's real extract step.
    """
    elements = extract_page_elements(
        page=page, page_image_bytes=page_image_bytes, coverage=coverage,
        doc_id=doc_id, page_num=page_num, tenant_id=tenant_id, acl=acl,
    )
    if not elements:
        return 0
    element_conf = {el.element_id: el.confidence for el in elements}
    chunks = chunk_elements(elements)
    chunks = propagate_confidence(chunks, element_conf, container_id=tenant_id)
    vectors = embed_texts_batched([c.text for c in chunks], container_id=tenant_id)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
    return writer.write_chunks(chunks)


def _build_redis_client():  # pragma: no cover - requires infra
    if not _HAS_REDIS:
        return None
    return redis.Redis.from_url(get_pdf_settings().redis_url)  # type: ignore[union-attr]


if _HAS_CELERY:  # pragma: no cover - requires infra

    _settings = get_pdf_settings()

    @shared_task(  # type: ignore[misc]
        bind=True,
        max_retries=_settings.max_retries,
        default_retry_delay=_settings.retry_base_delay,
        queue=_settings.ingest_queue,
        acks_late=True,
    )
    def process_page_task(self, task_id: str, *, tenant_id: str):
        """Celery entry point — delegates control flow to ``_run_page_extraction``.

        The real ``PageManifestRepo`` is async and AsyncSession-backed, so the
        page work runs inside an event loop with a freshly-opened session bound
        to the repo. ``extract_fn`` is wired by the worker bootstrap. The retry
        is bound to ``self.retry`` via ``on_retry``.
        """
        import asyncio

        from app.core.database import async_session  # type: ignore  # late import
        from ..control_plane import PageManifestRepo  # type: ignore  # Team A

        redis_client = _build_redis_client()

        def _on_retry(countdown: int):
            raise self.retry(countdown=countdown)

        async def _run() -> str:
            # One AsyncSession per task → the repo persists status transitions;
            # the session owns the transaction (commit on success path).
            async with async_session() as session:
                page_repo = PageManifestRepo(session)

                async def _extract(tid: str) -> None:
                    # Worker bootstrap supplies the rendered page + coverage from
                    # the page manifest; the Neo4jWriter is built from settings.
                    from .neo4j_writer import Neo4jWriter

                    s = get_pdf_settings()
                    writer = Neo4jWriter(s.neo4j_uri, s.neo4j_user, s.neo4j_password,
                                         database=s.neo4j_database)
                    page_obj, page_image, coverage, doc_id, acl, page_num = (
                        await page_repo.load_page_inputs(tid, tenant_id=tenant_id)
                    )
                    run_page_pipeline(
                        page=page_obj, page_image_bytes=page_image, coverage=coverage,
                        doc_id=doc_id, page_num=page_num, tenant_id=tenant_id, acl=acl,
                        writer=writer,
                    )

                try:
                    result = await _run_page_extraction(
                        task_id,
                        tenant_id=tenant_id,
                        extract_fn=_extract,
                        page_repo=page_repo,
                        redis_client=redis_client,
                        retries=self.request.retries,
                        on_retry=_on_retry,
                    )
                finally:
                    await session.commit()
                return result

        result = asyncio.run(_run())
        # Once this page committed its terminal status, reconcile the DOCUMENT if
        # every page has now settled (uses its own session so the commit above is
        # visible). Best-effort: a missed finalize is recovered by the next page.
        try:
            from pdf_chat.control_plane.finalizer import finalize_if_complete

            upload_id = task_id.split(":page:")[0]
            asyncio.run(finalize_if_complete(upload_id))
        except Exception:  # pragma: no cover - never fail the page on reconcile
            pass
        return result

    @shared_task(  # type: ignore[misc]
        bind=True,
        max_retries=_settings.max_retries,
        default_retry_delay=_settings.retry_base_delay,
        queue=_settings.ingest_queue,
        acks_late=True,
    )
    def build_document_graph_task(
        self, upload_id: str, *, tenant_id: str, container_id: str | None = None
    ):
        """Phase-2 + Phase-5 finalization for ONE settled document.

        Enqueued by ``control_plane.finalizer`` the first time a document reaches a
        terminal indexed status. Both phases are idempotent, so a retry (transient
        Neo4j/LLM failure) safely re-runs them. Errors retry with the same
        exponential backoff as page extraction; once retries are exhausted the
        document is already retrievable as vector-only chunks, so the graph build
        failing is degraded — never fatal to chat.
        """
        import asyncio

        from pdf_chat.control_plane.graph_build import build_document_graph

        try:
            return asyncio.run(
                build_document_graph(upload_id, tenant_id, container_id)
            )
        except Exception as exc:  # transient infra → retry with backoff
            raise self.retry(
                exc=exc, countdown=retry_countdown(self.request.retries)
            )

else:

    def process_page_task(task_id: str, *, tenant_id: str):  # type: ignore[misc]
        """Stub used when Celery is not installed (pure import safety)."""
        raise RuntimeError(
            "Celery is required to run process_page_task as a task but is not "
            "installed. Use _run_page_extraction directly for in-process or test "
            "execution."
        )

    def build_document_graph_task(  # type: ignore[misc]
        upload_id: str, *, tenant_id: str, container_id: str | None = None
    ):
        """Stub when Celery is absent — call ``build_document_graph`` directly."""
        raise RuntimeError(
            "Celery is required to run build_document_graph_task as a task but is "
            "not installed. Use control_plane.graph_build.build_document_graph "
            "directly for in-process or test execution."
        )

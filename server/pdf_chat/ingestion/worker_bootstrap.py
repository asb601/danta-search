"""Celery worker bootstrap — installs process-wide ingestion dependencies.

Runs ONCE per worker process (via Celery's ``worker_process_init`` signal) to
wire the two pieces of process-global state the per-page extraction path needs:

  1. the escalation **budget store** (``model_router.set_default_budget_store``)
     so the per-call escalation gate can actually reserve against Redis — without
     it ``escalation_allowed`` fails safe and denies every escalation; and
  2. the default **blob reader** (``repositories.set_default_blob_reader``) so
     ``PageManifestRepo._download`` can fetch the document blob for an ``az://``
     uri (the local-disk path needs no reader).

Everything here is import-safe with ZERO infra: the redis / azure imports are
guarded, and when a dependency is absent the corresponding default is left UNSET
so the system fails safe (escalation denied; ``_download`` raises a clear
``PermanentError`` for non-local uris) rather than crashing the worker.

The three page-input seams (``_fetch_row`` / ``_download`` / ``_render_page``)
are real method bodies on ``PageManifestRepo`` (they own ``self._session``), so
they are NOT installed here — only the process-global stores are.
"""
from __future__ import annotations

import os
from typing import Callable

from ..config import get_pdf_settings
from ..control_plane.repositories import set_default_blob_reader
from ..model_router import RedisBudgetStore, set_default_budget_store
from .tasks import _build_redis_client

__all__ = ["worker_bootstrap", "install_budget_store", "install_blob_reader"]


def install_budget_store() -> bool:
    """Install the Redis-backed escalation budget store. Returns True if wired.

    Reuses the existing ``REDIS_URL`` builder (``tasks._build_redis_client``) — no
    new Redis instance (Hard rule). When redis is unavailable the store is left
    ``None`` so escalation fails safe (deny), never crashing the worker.
    """
    client = _build_redis_client()
    if client is None:
        set_default_budget_store(None)
        return False
    set_default_budget_store(RedisBudgetStore(client))
    return True


def _build_blob_reader() -> "Callable[[str], bytes] | None":
    """Build a blob reader closure over a real Azure client, or None.

    Mirrors ``app/services/parquet_service.py``'s established access
    (``BlobServiceClient.from_connection_string`` → ``get_blob_client`` →
    ``download_blob().readall()``) — no new auth path is invented. The
    per-org-encrypted connection string is supplied to the worker via env
    (``PDF_BLOB_CONNECTION_STRING`` / ``AZURE_STORAGE_CONNECTION_STRING``); when
    it is absent OR azure-storage-blob is not installed, returns ``None`` and the
    local-disk download path remains the only one (``_download`` fails safe for
    ``az://`` uris). The container + blob are parsed from each uri (data-driven).
    """
    conn = os.getenv("PDF_BLOB_CONNECTION_STRING") or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    if not conn:
        return None
    try:  # guarded — import only when a connection string is configured
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except ImportError:  # pragma: no cover - exercised only without infra
        return None

    default_container = get_pdf_settings().blob_container

    def _read(blob_uri: str) -> bytes:
        container, blob = _parse_blob_uri(blob_uri, default_container)
        client = BlobServiceClient.from_connection_string(conn)
        bc = client.get_blob_client(container=container, blob=blob)
        return bc.download_blob().readall()

    return _read


def _parse_blob_uri(blob_uri: str, default_container: str) -> tuple[str, str]:
    """Parse ``az://container/blob`` (or ``blob://``/``https://``) → (container, blob).

    Data-driven: the container is derived from the uri, falling back to the
    configured ``blob_container`` only when the uri carries no host segment.
    """
    from urllib.parse import urlparse

    parsed = urlparse(blob_uri)
    if parsed.scheme in ("az", "blob") and parsed.netloc:
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        # https://<account>.blob.core.windows.net/<container>/<blob>
        parts = parsed.path.lstrip("/").split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return default_container, parts[0]
    # No host segment — treat the whole path as the blob in the default container.
    return default_container, blob_uri


def install_blob_reader() -> bool:
    """Install the default blob reader. Returns True if a real reader was wired."""
    reader = _build_blob_reader()
    set_default_blob_reader(reader)
    return reader is not None


def worker_bootstrap(**_kwargs) -> None:
    """Install all process-global ingestion dependencies (idempotent).

    Wired to Celery's ``worker_process_init`` below so it runs once per worker
    process. Safe to call directly in tests.
    """
    install_budget_store()
    install_blob_reader()


try:  # pragma: no cover - requires Celery infra
    from celery.signals import worker_process_init  # type: ignore

    worker_process_init.connect(worker_bootstrap)
except ImportError:  # pragma: no cover - import-safe without Celery
    pass

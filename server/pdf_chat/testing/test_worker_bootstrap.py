"""Pure tests for the worker bootstrap (no live infra).

Covers the two pieces of process-global state the bootstrap installs:
  * the escalation budget store (RedisBudgetStore from REDIS_URL), and
  * the default blob reader (built from an env connection string, or None).

All tests reset the process-global state in a finally block so they never leak
into the other suite tests (global-state leak is the #1 risk flagged in the spec).
"""
from __future__ import annotations


def test_install_budget_store_wires_redis_store(monkeypatch):
    from pdf_chat import model_router
    from pdf_chat.ingestion import worker_bootstrap

    class _FakeRedis:
        def get(self, key):
            return None

        def incr(self, key):
            return 1

    monkeypatch.setattr(worker_bootstrap, "_build_redis_client", lambda: _FakeRedis())
    prev = model_router._DEFAULT_STORE
    try:
        wired = worker_bootstrap.install_budget_store()
        assert wired is True
        assert isinstance(model_router._DEFAULT_STORE, model_router.RedisBudgetStore)
    finally:
        model_router.set_default_budget_store(prev)


def test_install_budget_store_is_redis_safe(monkeypatch):
    """When no redis client can be built, the store is left None (fail-safe)."""
    from pdf_chat import model_router
    from pdf_chat.ingestion import worker_bootstrap

    monkeypatch.setattr(worker_bootstrap, "_build_redis_client", lambda: None)
    prev = model_router._DEFAULT_STORE
    try:
        wired = worker_bootstrap.install_budget_store()
        assert wired is False
        assert model_router._DEFAULT_STORE is None
    finally:
        model_router.set_default_budget_store(prev)


def test_install_blob_reader_none_without_connection_string(monkeypatch):
    from pdf_chat.control_plane import repositories as repo_mod
    from pdf_chat.ingestion import worker_bootstrap

    monkeypatch.delenv("PDF_BLOB_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    prev = repo_mod._DEFAULT_BLOB_READER
    try:
        wired = worker_bootstrap.install_blob_reader()
        assert wired is False
        assert repo_mod._DEFAULT_BLOB_READER is None
    finally:
        repo_mod.set_default_blob_reader(prev)


def test_parse_blob_uri_is_data_driven():
    from pdf_chat.ingestion.worker_bootstrap import _parse_blob_uri

    assert _parse_blob_uri("az://cont/path/doc.pdf", "default") == ("cont", "path/doc.pdf")
    assert _parse_blob_uri("blob://t1/sha", "default") == ("t1", "sha")
    assert _parse_blob_uri(
        "https://acct.blob.core.windows.net/cont/doc.pdf", "default"
    ) == ("cont", "doc.pdf")
    # No host segment → falls back to the configured default container.
    assert _parse_blob_uri("just-a-blob-name", "default") == ("default", "just-a-blob-name")


def test_worker_bootstrap_installs_both(monkeypatch):
    from pdf_chat import model_router
    from pdf_chat.control_plane import repositories as repo_mod
    from pdf_chat.ingestion import worker_bootstrap

    class _FakeRedis:
        def get(self, key):
            return None

        def incr(self, key):
            return 1

    monkeypatch.setattr(worker_bootstrap, "_build_redis_client", lambda: _FakeRedis())
    monkeypatch.delenv("PDF_BLOB_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    prev_store = model_router._DEFAULT_STORE
    prev_reader = repo_mod._DEFAULT_BLOB_READER
    try:
        worker_bootstrap.worker_bootstrap()
        assert isinstance(model_router._DEFAULT_STORE, model_router.RedisBudgetStore)
        # No conn string → reader stays None (local-disk path remains usable).
        assert repo_mod._DEFAULT_BLOB_READER is None
    finally:
        model_router.set_default_budget_store(prev_store)
        repo_mod.set_default_blob_reader(prev_reader)

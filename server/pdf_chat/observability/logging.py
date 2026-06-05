"""structlog wiring for pdf_chat with trace-ID binding (Phase 6 hardening).

The main app's logger (server/app/core/logger.py) already configures structlog
process-wide. We reuse that configuration and only add named loggers under the
``pdf_chat.*`` namespace plus a helper to bind a trace_id/tenant_id onto every
event so a request can be correlated end-to-end across ingestion + query.

Pure + import-safe: structlog.get_logger() works even when no processors have
been configured (it returns a lazy proxy), so importing this never touches infra.
"""
from __future__ import annotations

from typing import Any

import structlog


def get_pdf_logger(name: str) -> Any:
    """Return a named structlog logger under the ``pdf_chat`` namespace."""
    return structlog.get_logger(f"pdf_chat.{name}")


def bind_trace(logger: Any, trace_id: str, tenant_id: str) -> Any:
    """Bind correlation fields so every subsequent event carries them.

    Returns a NEW bound logger; the original is unchanged (structlog loggers are
    immutable — ``.bind()`` copies). Callers keep the returned logger.
    """
    return logger.bind(trace_id=trace_id, tenant_id=tenant_id)

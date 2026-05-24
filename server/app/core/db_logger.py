"""Non-blocking ai_pipeline event writer → server_logs DB table.

All public functions are fire-and-forget: they schedule an asyncio.Task and
return immediately.  A DB error in the background task is silently discarded —
DB log failures NEVER propagate to the query pipeline.

Usage:
    from app.core.db_logger import log_pipeline_event as _db_log
    _db_log(
        event="query_received",
        trace_id=req_trace_id,
        actor_user_id=user_id,
        details={"query": query[:500]},
    )
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.database import async_session
from app.models.server_log import ServerLog

_VALID_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


async def _write(row_kwargs: dict) -> None:
    """Write one ServerLog row in an independent DB session. Never raises."""
    try:
        async with async_session() as sess:
            sess.add(ServerLog(**row_kwargs))
            await sess.commit()
    except Exception:
        pass  # silently discard — never affect the pipeline


def log_pipeline_event(
    *,
    event: str,
    level: str = "info",
    trace_id: str | None = None,
    actor_user_id: str | None = None,
    actor_email: str | None = None,
    actor_role: str | None = None,
    domain_tag: str | None = None,
    duration_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Schedule a non-blocking DB write for an ai_pipeline event.

    Safe to call from any async context.  Returns immediately — the actual DB
    write happens in a background asyncio.Task.
    """
    row_kwargs: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "log_type": "ai_pipeline",
        "event": event[:80],
        "level": level if level in _VALID_LEVELS else "info",
        "trace_id": trace_id or None,
        "actor_user_id": actor_user_id or None,
        "actor_email": actor_email or None,
        "actor_role": actor_role or None,
        "domain_tag": domain_tag or None,
        "duration_ms": duration_ms,
        "details": details if details else None,
    }
    try:
        asyncio.get_running_loop().create_task(_write(row_kwargs))
    except RuntimeError:
        pass  # no running event loop (test context) — skip silently


async def log_ingest_event(
    *,
    event: str,
    level: str = "info",
    trace_id: str | None = None,
    file_id: str | None = None,
    file_name: str | None = None,
    domain_tag: str | None = None,
    actor_user_id: str | None = None,
    actor_email: str | None = None,
    actor_role: str | None = None,
    duration_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Write an ingestion event to server_logs.

    This is an awaitable (not fire-and-forget) so it works correctly inside
    Celery worker async stages called via _run_async().  DB errors are silently
    discarded — a log failure never aborts the ingestion pipeline.
    """
    row_kwargs: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "log_type": "ingestion",
        "event": event[:80],
        "level": level if level in _VALID_LEVELS else "info",
        "trace_id": trace_id or None,
        "actor_user_id": actor_user_id or None,
        "actor_email": actor_email or None,
        "actor_role": actor_role or None,
        "file_id": file_id or None,
        "file_name": file_name or None,
        "domain_tag": domain_tag or None,
        "duration_ms": duration_ms,
        "details": details if details else None,
    }
    await _write(row_kwargs)

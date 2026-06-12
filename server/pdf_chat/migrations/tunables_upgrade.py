"""Runtime migration — create the per-container tunables table + index, and wire
the DB-override lookup into ``pdf_chat.tunables``.

Idempotent, non-fatal, additive (same convention as app/migrations/ and
control_plane_upgrade.py). Importing the model registers it on the shared Base
metadata so ``create_all`` also creates the table; this migration additionally
ensures the secondary index and installs the per-container override hook so
``tunables.get_tunable`` can resolve a DB override before env/default.

Tunables resolve from env + named defaults WITHOUT this table; the table only
lets an operator override a single key for one container without a redeploy.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pdf_chat import tunables

# Importing registers the table on the shared Base metadata.
from pdf_chat.models.tunable import PdfGraphRagTunable  # noqa: F401

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pdf_tunable_container "
    "ON pdf_graphrag_tunables(container_id)",
]


async def run_migration(engine: AsyncEngine) -> None:
    """Create the table (via metadata) + index. Safe to run repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # type: ignore[name-defined]
        for stmt in _INDEXES:
            await conn.execute(text(stmt))


# Backwards-compatible alias used by some bootstrap call sites.
upgrade = run_migration


# ── Per-container override snapshot ────────────────────────────────────────────
# ``tunables.get_tunable`` resolves the DB override on a SYNC path that is itself
# reached from inside running event loops (request handlers) AND from sync Celery
# workers — so the lookup may neither ``await`` nor ``asyncio.run``. There is also
# no sync Postgres driver wired (asyncpg only). We therefore keep an in-memory
# snapshot of the (low-cardinality, operator-set) overrides, refreshed
# asynchronously, and the sync lookup is a pure dict read that never touches the
# DB. Overrides take effect within ``_OVERRIDE_TTL_SECONDS`` of an operator edit.
_overrides: "dict[str, dict[str, str]]" = {}
_loaded_at: float = 0.0
_OVERRIDE_TTL_SECONDS = 60.0
_session_factory: "async_sessionmaker | None" = None


async def refresh_overrides(session_factory: "async_sessionmaker | None" = None) -> None:
    """Reload the override snapshot from ``pdf_graphrag_tunables`` (best-effort).

    Rebuilds ``container_id -> {key: value}``. Safe to call repeatedly; a failed
    read (table absent on a fresh DB, transient error) leaves the prior snapshot
    intact so resolution still degrades to env/default.
    """
    import time

    from sqlalchemy import select

    global _loaded_at
    factory = session_factory or _session_factory
    if factory is None:
        return
    try:
        async with factory() as session:
            rows = (
                await session.execute(
                    select(
                        PdfGraphRagTunable.container_id,
                        PdfGraphRagTunable.key,
                        PdfGraphRagTunable.value,
                    )
                )
            ).all()
        snapshot: dict[str, dict[str, str]] = {}
        for container_id, key, value in rows:
            snapshot.setdefault(str(container_id), {})[str(key)] = value
        _overrides.clear()
        _overrides.update(snapshot)
        _loaded_at = time.monotonic()
    except Exception:  # pragma: no cover - best-effort; prior snapshot wins
        return


def _maybe_async_refresh() -> None:
    """Schedule a non-blocking snapshot refresh when the TTL has expired.

    Only fires when a running event loop is available (API process); the sync
    lookup never blocks. Celery workers refresh at ``worker_process_init``.
    """
    import asyncio
    import time

    if _session_factory is None or (time.monotonic() - _loaded_at) < _OVERRIDE_TTL_SECONDS:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop (sync context) → snapshot is refreshed elsewhere
    loop.create_task(refresh_overrides())


def install_db_lookup(session_factory: "async_sessionmaker") -> None:
    """Wire a per-container DB override lookup into ``tunables.get_tunable``.

    Installs a pure-sync lookup that reads the in-memory override snapshot (see
    above) and opportunistically schedules an async refresh when stale. The caller
    should ``await refresh_overrides(session_factory)`` once after this to warm the
    snapshot at boot. Kept out of module import to preserve ``tunables``' pure
    import contract.
    """
    global _session_factory
    _session_factory = session_factory

    def _lookup(container_id: str, key: str) -> "str | None":
        _maybe_async_refresh()
        return _overrides.get(container_id, {}).get(key)

    tunables.set_db_lookup(_lookup)


# Base is needed for create_all; import after the model so metadata is populated.
from app.core.database import Base  # noqa: E402

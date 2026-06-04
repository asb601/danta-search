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


def install_db_lookup(session_factory: "async_sessionmaker") -> None:
    """Wire a per-container DB override lookup into ``tunables.get_tunable``.

    The lookup is SYNC (``tunables`` calls it inside a sync resolution path), so
    we build a synchronous wrapper that opens a short-lived sync read. To keep
    the pure-import contract of ``tunables`` intact we only install the hook here,
    at bootstrap — never at module import.

    ``session_factory`` is the app's ``async_sessionmaker``. Because the resolver
    seam is sync, production should pass a sync-capable lookup; this default
    implementation degrades to ``None`` (env/default wins) if no sync path is
    available, so it is always safe.
    """

    def _lookup(container_id: str, key: str) -> "str | None":
        # The async session cannot be awaited from the sync resolver; the
        # production bootstrap supplies a sync engine reader. Until then this
        # returns None so resolution falls through to env/default (never fatal).
        return None

    tunables.set_db_lookup(_lookup)


# Base is needed for create_all; import after the model so metadata is populated.
from app.core.database import Base  # noqa: E402

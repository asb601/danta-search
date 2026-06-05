"""Runtime migration — create the cross-domain bridge table + indexes.

Idempotent, non-fatal, additive (same convention as control_plane_upgrade.py and
app/migrations/). Importing the model registers it on the shared Base so
``create_all`` creates the table; this migration also adds the secondary index
the spec calls for.

NOT wired into ``app/main.py`` — productionization (worker/lifespan wiring) is
deferred (it needs live infra); see the plan's Deferred section.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Importing registers the table on the shared Base metadata.
from pdf_chat.models.bridge import PdfEntityBridge

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_container ON pdf_entity_bridge(container_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_tenant ON pdf_entity_bridge(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_status ON pdf_entity_bridge(status)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_container_entity "
    "ON pdf_entity_bridge(container_id, pdf_entity_id)",
]


async def run_migration(engine: AsyncEngine) -> None:
    """Create the bridge table + indexes only. Safe to run repeatedly.

    DDL is scoped to ``pdf_entity_bridge`` alone (``checkfirst=True`` is
    IF-NOT-EXISTS) — we deliberately do NOT ``Base.metadata.create_all`` on the
    shared Base, which would also try to materialize every CSV-side table.
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: PdfEntityBridge.__table__.create(sync_conn, checkfirst=True)
        )
        for stmt in _INDEXES:
            await conn.execute(text(stmt))


# Alias so callers can use either name (mirrors control_plane_upgrade.py style).
upgrade = run_migration

"""SME Phase-1 trust/quarantine — file_metadata.trust_state column.

Additive + idempotent runtime migration (NOT Alembic), matching the house
pattern used by app/migrations/semantic_layer_upgrade.py: a flat ``_STATEMENTS``
list applied inside a single ``engine.begin()`` transaction.

Adds:
  * file_metadata.trust_state VARCHAR(20) NOT NULL DEFAULT 'trusted'
      — coarse per-file trust state (trusted/shadow/quarantined). Default
        'trusted' so existing rows and the flag-off runtime are unchanged.
  * ix_file_metadata_trust_state (container_id, trust_state)
      — lets the consumer (retrieval/SQL-context gate) filter a tenant's files
        by trust state without a full scan.

Wiring: the lead invokes ``migrate()`` from the app/main.py lifespan migration
sequence (alongside the other runtime migrations). This module only needs to be
importable and idempotent.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    "ALTER TABLE file_metadata "
    "ADD COLUMN IF NOT EXISTS trust_state VARCHAR(20) NOT NULL DEFAULT 'trusted'",
    "CREATE INDEX IF NOT EXISTS ix_file_metadata_trust_state "
    "ON file_metadata (container_id, trust_state)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

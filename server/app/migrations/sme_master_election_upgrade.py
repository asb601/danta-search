"""SME master-election columns on semantic_entities.

Additive + idempotent. Adds the canonical-master designation written by
`semantic_layer_builder.apply_master_election`. With these columns defaulting
to FALSE/NULL, runtime behavior is unchanged until the SME flags are enabled
and an election pass runs (via semantic_rebuild — no re-ingest).

Wired in app/main.py lifespan alongside the other runtime migrations:
    from app.migrations import sme_master_election_upgrade
    await sme_master_election_upgrade.migrate()
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    "ALTER TABLE semantic_entities ADD COLUMN IF NOT EXISTS is_canonical_master BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE semantic_entities ADD COLUMN IF NOT EXISTS master_for_entity VARCHAR(120)",
    # Fast lookup of "the master for entity X in this container".
    "CREATE INDEX IF NOT EXISTS ix_semantic_entities_master "
    "ON semantic_entities (container_id, master_for_entity) WHERE is_canonical_master",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

"""
Org-RBAC overhaul — container_configs table upgrade (LANE A).

Adds:
  - container_configs.organization_id VARCHAR(36) FK organizations.id ON DELETE CASCADE (+index)
  - container_configs.storage_kind    VARCHAR(30) DEFAULT 'azure_blob'
  - container_configs.is_primary      BOOLEAN     DEFAULT FALSE

Backfill: for every organizations.container_id, set that container's
organization_id and mark it as the org's primary container.

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.org_multi_container_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE container_configs ADD COLUMN IF NOT EXISTS organization_id VARCHAR(36)",
    "ALTER TABLE container_configs ADD COLUMN IF NOT EXISTS storage_kind VARCHAR(30) DEFAULT 'azure_blob'",
    "ALTER TABLE container_configs ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT FALSE",
    "UPDATE container_configs SET storage_kind = 'azure_blob' WHERE storage_kind IS NULL",
    "UPDATE container_configs SET is_primary = FALSE WHERE is_primary IS NULL",
]

_GUARDED_DDL: list[str] = [
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'fk_container_configs_organization'
        ) THEN
            ALTER TABLE container_configs
                ADD CONSTRAINT fk_container_configs_organization
                FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;
        END IF;
    END $$;
    """,
    "CREATE INDEX IF NOT EXISTS ix_container_configs_organization ON container_configs (organization_id)",
]

# Backfill: bind each org's currently-referenced container to that org and
# flag it primary. Guarded by IS NULL / != so re-runs are no-ops.
_BACKFILL: list[str] = [
    """
    UPDATE container_configs c
    SET organization_id = o.id
    FROM organizations o
    WHERE o.container_id = c.id
      AND c.organization_id IS DISTINCT FROM o.id
    """,
    """
    UPDATE container_configs c
    SET is_primary = TRUE
    FROM organizations o
    WHERE o.container_id = c.id
      AND c.is_primary IS DISTINCT FROM TRUE
    """,
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))
    for stmt in _GUARDED_DDL:
        await conn.execute(text(stmt))
    for stmt in _BACKFILL:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("org_multi_container_upgrade: done")

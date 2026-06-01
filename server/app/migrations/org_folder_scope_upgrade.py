"""
Org-RBAC overhaul — folders table upgrade (LANE A).

Adds:
  - folders.organization_id VARCHAR(36) FK organizations.id ON DELETE CASCADE (+index)
  - folders.folder_kind     VARCHAR(20) DEFAULT 'generic'  ('org_root'|'domain'|'generic')

Backfill: resolve folders.organization_id by joining
folder.container_id -> container_configs.organization_id (populated by
org_multi_container_upgrade, which must run before this).

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.org_folder_scope_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE folders ADD COLUMN IF NOT EXISTS organization_id VARCHAR(36)",
    "ALTER TABLE folders ADD COLUMN IF NOT EXISTS folder_kind VARCHAR(20) DEFAULT 'generic'",
    "UPDATE folders SET folder_kind = 'generic' WHERE folder_kind IS NULL",
]

_GUARDED_DDL: list[str] = [
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'fk_folders_organization'
        ) THEN
            ALTER TABLE folders
                ADD CONSTRAINT fk_folders_organization
                FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE;
        END IF;
    END $$;
    """,
    "CREATE INDEX IF NOT EXISTS ix_folders_organization ON folders (organization_id)",
]

# Backfill org scope via the folder's container.
_BACKFILL: str = """
UPDATE folders f
SET organization_id = c.organization_id
FROM container_configs c
WHERE f.container_id = c.id
  AND c.organization_id IS NOT NULL
  AND f.organization_id IS DISTINCT FROM c.organization_id
"""


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))
    for stmt in _GUARDED_DDL:
        await conn.execute(text(stmt))
    await conn.execute(text(_BACKFILL))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("org_folder_scope_upgrade: done")

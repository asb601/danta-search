"""
Access-request / Org-AI schema upgrade.

Adds two columns:
  - access_requests.org_name   VARCHAR(255)  — org name the requester proposes;
                                               becomes the Organization name on
                                               owner approval.
  - org_ai_settings.postgres_url TEXT         — per-org PostgreSQL DSN. Stored
                                               encrypted-at-rest at the ORM layer
                                               (EncryptedText); the raw column is
                                               plain TEXT.

Idempotent — safe to run multiple times. Invoked from `app.main:lifespan`
alongside the other org/RBAC migrations.

Run standalone:
    python -m app.migrations.access_org_ai_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE access_requests ADD COLUMN IF NOT EXISTS org_name VARCHAR(255)",
    "ALTER TABLE org_ai_settings ADD COLUMN IF NOT EXISTS postgres_url TEXT",
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("access_org_ai_upgrade: done")

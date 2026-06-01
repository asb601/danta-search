"""
Org-RBAC overhaul — org_ai_settings table (LANE A).

Creates the per-organization AI credentials & deployment configuration table.
API keys are stored encrypted-at-rest by the application layer
(`app.core.crypto.EncryptedText`); at the DB level they are TEXT columns.

One row per organization (UNIQUE organization_id).

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.org_ai_settings_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS org_ai_settings (
        id VARCHAR(36) PRIMARY KEY,
        organization_id VARCHAR(36) NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
        chat_api_key TEXT,
        embeddings_api_key TEXT,
        fallback_api_key TEXT,
        chat_endpoint TEXT,
        chat_deployment TEXT,
        embeddings_deployment TEXT,
        fallback_deployment TEXT DEFAULT 'gpt-4o-mini',
        api_version TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("org_ai_settings_upgrade: done")

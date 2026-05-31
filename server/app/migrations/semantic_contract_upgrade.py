"""Runtime migration: semantic_contracts table. Idempotent, additive, non-fatal."""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS semantic_contracts (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        definition JSONB NOT NULL DEFAULT '{}'::jsonb,
        content_hash VARCHAR(64),
        version INTEGER NOT NULL DEFAULT 1,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_contract_container UNIQUE (container_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_semantic_contracts_container ON semantic_contracts (container_id, status)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

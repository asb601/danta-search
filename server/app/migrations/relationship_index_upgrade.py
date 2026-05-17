"""Column key registry for database-backed relationship discovery.

Adds column_key_registry: a tenant-scoped inverted index of hashed sample values.
This replaces in-memory hashmap thinking with a durable Postgres GIN-indexed
structure:

    normalized value fingerprint -> files/columns containing that value

All lookup queries are scoped by container_id, so tenants never mix.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    "ALTER TABLE file_relationships ADD COLUMN IF NOT EXISTS value_overlap_pct DOUBLE PRECISION",
    "ALTER TABLE file_relationships ADD COLUMN IF NOT EXISTS related_column VARCHAR(255)",
    """
    CREATE TABLE IF NOT EXISTS column_key_registry (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        file_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        blob_path VARCHAR(1000),
        column_name VARCHAR(255) NOT NULL,
        semantic_role VARCHAR(100),
        key_kind VARCHAR(20) NOT NULL DEFAULT 'candidate',
        cardinality BIGINT NOT NULL DEFAULT 0,
        sample_size BIGINT NOT NULL DEFAULT 0,
        unique_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
        null_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
        value_fingerprints VARCHAR(16)[] NOT NULL,
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_column_key_registry_file_column UNIQUE (file_id, column_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_ckr_container_kind ON column_key_registry (container_id, key_kind)",
    "CREATE INDEX IF NOT EXISTS ix_ckr_container_role ON column_key_registry (container_id, semantic_role)",
    "CREATE INDEX IF NOT EXISTS ix_ckr_file ON column_key_registry (file_id)",
    "CREATE INDEX IF NOT EXISTS ix_ckr_fingerprints ON column_key_registry USING GIN (value_fingerprints)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

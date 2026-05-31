"""Runtime migration: erp_classifications table.

Idempotent + additive + non-fatal, matching the project's runtime-migration
convention (no Alembic). Called from main.py lifespan.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS erp_classifications (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        file_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        source_system VARCHAR(120) NOT NULL DEFAULT 'Unknown',
        erp_module VARCHAR(120) NOT NULL DEFAULT 'Unknown',
        domain_polarity VARCHAR(20) NOT NULL DEFAULT 'neutral',
        process_role VARCHAR(120) NOT NULL DEFAULT 'unknown',
        grain TEXT,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
        evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
        source VARCHAR(20) NOT NULL DEFAULT 'unknown',
        model_version VARCHAR(80),
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_erp_classification_file UNIQUE (file_id)
    )
    """,
    "ALTER TABLE erp_classifications ADD COLUMN IF NOT EXISTS schema_fingerprint VARCHAR(64)",
    "CREATE INDEX IF NOT EXISTS ix_erp_classifications_container ON erp_classifications (container_id)",
    "CREATE INDEX IF NOT EXISTS ix_erp_classifications_polarity ON erp_classifications (container_id, domain_polarity)",
    "CREATE INDEX IF NOT EXISTS ix_erp_classifications_fingerprint ON erp_classifications (container_id, schema_fingerprint)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

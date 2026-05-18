"""Cleaning config and quarantine audit upgrade.

Adds two things:

1. container_configs.cleaning_config (JSONB, nullable)
   Per-container cleaning rules loaded at ingest time.
   Schema: {"extra_null_patterns": [...], "extra_garbage_patterns": [...]}
    Allows adding source-specific null patterns
   without a code change or redeployment.

2. file_analytics.quarantine_count (BIGINT, default 0)
   file_analytics.quarantine_sample (JSONB, nullable)
    file_analytics.cleaning_audit (JSONB, nullable)
   Audit trail of rows removed by the cleaning pipeline.
   quarantine_sample holds the first 20 dropped rows as
   [{"reason": "garbage_keyword", "row": {"col": "value", ...}}].
    cleaning_audit stores operational cleaning facts such as header row,
    delimiter, dedup skip status, temp disk bytes, and clean blob path.

Idempotent — safe to run on every startup.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    # Per-container cleaning configuration
    "ALTER TABLE container_configs "
    "ADD COLUMN IF NOT EXISTS cleaning_config JSONB",

    # Quarantine audit on FileAnalytics
    "ALTER TABLE file_analytics "
    "ADD COLUMN IF NOT EXISTS quarantine_count BIGINT DEFAULT 0",

    "ALTER TABLE file_analytics "
    "ADD COLUMN IF NOT EXISTS quarantine_sample JSONB",

    "ALTER TABLE file_analytics "
    "ADD COLUMN IF NOT EXISTS cleaning_audit JSONB",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

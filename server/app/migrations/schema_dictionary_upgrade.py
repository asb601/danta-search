"""
Schema-dictionary table upgrade.

Adds source_blob_path (nullable) and relaxes parquet_blob_path to nullable so
the dictionary can be looked up via the original CSV when parquet conversion
hasn't completed (or has failed).

Idempotent — safe to run on every startup.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    "ALTER TABLE schema_dictionaries "
    "ADD COLUMN IF NOT EXISTS source_blob_path TEXT",
    "ALTER TABLE schema_dictionaries "
    "ALTER COLUMN parquet_blob_path DROP NOT NULL",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

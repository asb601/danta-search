"""Container semantic configuration schema upgrade.

Adds container_configs.semantic_config JSONB so each tenant/container can extend
the base semantic role registry without a code deploy.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE container_configs ADD COLUMN IF NOT EXISTS semantic_config JSONB",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())
    print("semantic_config_upgrade: done")

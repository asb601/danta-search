"""
Local email+password auth schema upgrade.

Adds:
  - users.hashed_password VARCHAR   — bcrypt hash for local-auth users;
                                       NULL for Google-SSO users (org owners
                                       stay NULL and cannot use local login).

Idempotent — safe to run multiple times. Invoked from `app.main:lifespan`.

Run standalone:
    python -m app.migrations.local_auth_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    # users.hashed_password — bcrypt hash; NULL = no local password (Google SSO)
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password VARCHAR",
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("local_auth_upgrade: done")

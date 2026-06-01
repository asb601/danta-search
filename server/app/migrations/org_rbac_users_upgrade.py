"""
Org-RBAC overhaul — users table upgrade (LANE A).

Adds:
  - users.auth_provider      VARCHAR(20) DEFAULT 'google'
  - users.is_platform_admin  BOOLEAN     DEFAULT FALSE (+index)

Backfill (best-effort): the earliest-created `is_admin` user is promoted to
`is_platform_admin = TRUE` so an existing deployment retains a platform owner.

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.org_rbac_users_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'google'",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_platform_admin BOOLEAN DEFAULT FALSE",
    "UPDATE users SET auth_provider = 'google' WHERE auth_provider IS NULL",
    "CREATE INDEX IF NOT EXISTS ix_users_is_platform_admin ON users (is_platform_admin) WHERE is_platform_admin = TRUE",
]

# Backfill: promote the earliest-created admin to platform admin, but only if
# no platform admin exists yet (so re-runs do not flip-flop).
_BACKFILL: str = """
UPDATE users
SET is_platform_admin = TRUE
WHERE id = (
    SELECT id FROM users
    WHERE is_admin = TRUE
    ORDER BY created_at ASC
    LIMIT 1
)
AND NOT EXISTS (
    SELECT 1 FROM users WHERE is_platform_admin = TRUE
)
"""


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))
    await conn.execute(text(_BACKFILL))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("org_rbac_users_upgrade: done")

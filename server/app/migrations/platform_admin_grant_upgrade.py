"""
Org-RBAC overhaul — platform_admin_grants table (LANE A).

Creates the auditable grant table binding a platform admin (cross-org
superuser) to a specific organization. Grants are revoked via `status` rather
than deletion so history is retained.

One grant per (organization, platform_admin_user) — enforced by a UNIQUE
constraint.

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.platform_admin_grant_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS platform_admin_grants (
        id VARCHAR(36) PRIMARY KEY,
        organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        platform_admin_user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        granted_by VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        revoked_at TIMESTAMPTZ,
        CONSTRAINT uq_platform_admin_grant UNIQUE (organization_id, platform_admin_user_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_platform_admin_grants_org ON platform_admin_grants (organization_id)",
    "CREATE INDEX IF NOT EXISTS ix_platform_admin_grants_user ON platform_admin_grants (platform_admin_user_id)",
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("platform_admin_grant_upgrade: done")

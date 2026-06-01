"""
Org-RBAC overhaul — manager_domain_assignments table (LANE A).

Creates the per-(user, organization, domain) grant table that supersedes the
coarse `users.allowed_domains` array (kept for backward-compat).

Backfill: one assignment row per (user, domain) for every user that has both
an organization_id and a non-empty allowed_domains array.

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.manager_domain_assignment_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS manager_domain_assignments (
        id VARCHAR(36) PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        organization_id VARCHAR(36) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        domain_tag TEXT NOT NULL,
        is_domain_admin BOOLEAN NOT NULL DEFAULT FALSE,
        granted_by VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_manager_domain_assignment UNIQUE (user_id, organization_id, domain_tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_manager_domain_assignments_user ON manager_domain_assignments (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_manager_domain_assignments_org ON manager_domain_assignments (organization_id)",
]

# Backfill from users.allowed_domains; ON CONFLICT keeps re-runs idempotent.
# Generate the id via md5(...)::uuid so we need NO extension — Azure Postgres
# blocks `CREATE EXTENSION pgcrypto`, and gen_random_uuid() is not core before
# PG13. md5 of random+clock+keys yields a valid, unique uuid string everywhere.
_BACKFILL: list[str] = [
    """
    INSERT INTO manager_domain_assignments (id, user_id, organization_id, domain_tag)
    SELECT
        (md5(random()::text || clock_timestamp()::text || u.id || d.domain_tag))::uuid::text,
        u.id,
        u.organization_id,
        d.domain_tag
    FROM users u
    CROSS JOIN LATERAL unnest(u.allowed_domains) AS d(domain_tag)
    WHERE u.organization_id IS NOT NULL
      AND u.allowed_domains IS NOT NULL
      AND array_length(u.allowed_domains, 1) > 0
      AND d.domain_tag IS NOT NULL
      AND d.domain_tag <> ''
    ON CONFLICT (user_id, organization_id, domain_tag) DO NOTHING
    """,
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))
    for stmt in _BACKFILL:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("manager_domain_assignment_upgrade: done")

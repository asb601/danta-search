"""
Org-RBAC overhaul — organizations table upgrade (LANE A).

Adds:
  - organizations.owner_user_id          VARCHAR(36) FK users.id ON DELETE SET NULL
  - organizations.onboarding_state       VARCHAR(40) DEFAULT 'created'
  - organizations.onboarding_completed_at TIMESTAMPTZ NULL
  - organizations.slug                   VARCHAR(255) UNIQUE NULL

Backfill: derive a blob-safe lowercase `slug` from `name` for orgs missing one,
with a numeric suffix on collision so the UNIQUE constraint holds.

Idempotent — safe to run multiple times. Non-fatal. Invoked from
`app.main:lifespan`.

Run standalone:
    python -m app.migrations.org_rbac_org_upgrade
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS owner_user_id VARCHAR(36)",
    "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS onboarding_state VARCHAR(40) DEFAULT 'created'",
    "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS onboarding_completed_at TIMESTAMPTZ",
    "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS slug VARCHAR(255)",
    "UPDATE organizations SET onboarding_state = 'created' WHERE onboarding_state IS NULL",
]

# FK + UNIQUE added separately and guarded so re-runs do not error on duplicate
# constraint creation (Postgres has no ADD CONSTRAINT IF NOT EXISTS).
_GUARDED_DDL: list[str] = [
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'fk_organizations_owner_user'
        ) THEN
            ALTER TABLE organizations
                ADD CONSTRAINT fk_organizations_owner_user
                FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL;
        END IF;
    END $$;
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_organizations_slug ON organizations (slug) WHERE slug IS NOT NULL",
]

# Backfill: blob-safe slug from name. Non-alnum -> '-', collapse repeats, trim,
# lowercase; de-dupe with a row-number suffix when names collide.
_BACKFILL: str = """
WITH base AS (
    SELECT
        id,
        NULLIF(
            trim(
                BOTH '-' FROM regexp_replace(lower(name), '[^a-z0-9]+', '-', 'g')
            ),
            ''
        ) AS raw_slug
    FROM organizations
    WHERE slug IS NULL
),
ranked AS (
    SELECT
        id,
        COALESCE(raw_slug, 'org') AS raw_slug,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(raw_slug, 'org') ORDER BY id
        ) AS rn
    FROM base
)
UPDATE organizations o
SET slug = CASE
    WHEN r.rn = 1 THEN r.raw_slug
    ELSE r.raw_slug || '-' || r.rn::text
END
FROM ranked r
WHERE o.id = r.id
  AND o.slug IS NULL
"""


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))
    for stmt in _GUARDED_DDL:
        await conn.execute(text(stmt))
    await conn.execute(text(_BACKFILL))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("org_rbac_org_upgrade: done")

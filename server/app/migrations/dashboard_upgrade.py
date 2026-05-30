"""Dashboard schema.

Adds the metadata-driven dashboard layer: dashboards + dashboard_folders.
Additive and non-fatal — mirrors the existing runtime-migration pattern.
See response.txt Sections 3 and 14.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS dashboard_folders (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) REFERENCES container_configs(id) ON DELETE CASCADE,
        owner_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        parent_id VARCHAR(36) REFERENCES dashboard_folders(id) ON DELETE SET NULL,
        name VARCHAR(255) NOT NULL DEFAULT 'New folder',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dashboards (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) REFERENCES container_configs(id) ON DELETE CASCADE,
        folder_id VARCHAR(36) REFERENCES dashboard_folders(id) ON DELETE SET NULL,
        owner_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title VARCHAR(255) NOT NULL DEFAULT 'Untitled dashboard',
        description TEXT,
        config JSONB NOT NULL DEFAULT '{}'::jsonb,
        prompt_history JSONB NOT NULL DEFAULT '[]'::jsonb,
        source_file_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        is_pinned BOOLEAN NOT NULL DEFAULT FALSE,
        status VARCHAR(20) NOT NULL DEFAULT 'draft',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_dashboard_folders_owner ON dashboard_folders (owner_id)",
    "CREATE INDEX IF NOT EXISTS ix_dashboard_folders_container ON dashboard_folders (container_id)",
    "CREATE INDEX IF NOT EXISTS ix_dashboards_owner ON dashboards (owner_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS ix_dashboards_container ON dashboards (container_id)",
    "CREATE INDEX IF NOT EXISTS ix_dashboards_folder ON dashboards (folder_id)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

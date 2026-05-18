"""Audit log schema upgrade.

Creates durable, queryable audit logs for API activity. These logs are used by
/api/logs/audit so regular users see only their own activity, domain-scoped
operators see their domains, and admins see everything.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.database import engine

_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id VARCHAR(36) PRIMARY KEY,
        event_type VARCHAR(40) NOT NULL DEFAULT 'request',
        action VARCHAR(160) NOT NULL,
        actor_user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
        actor_email VARCHAR(320),
        actor_name VARCHAR(255),
        actor_role VARCHAR(20),
        actor_is_admin BOOLEAN NOT NULL DEFAULT FALSE,
        actor_allowed_domains TEXT[],
        actor_organization_id VARCHAR(36),
        method VARCHAR(10),
        path VARCHAR(500),
        route_template VARCHAR(500),
        status_code INTEGER,
        duration_ms DOUBLE PRECISION,
        ip_address VARCHAR(80),
        user_agent TEXT,
        domain_tag TEXT,
        container_id VARCHAR(36),
        file_id VARCHAR(36),
        file_name VARCHAR(500),
        folder_id VARCHAR(36),
        folder_name VARCHAR(255),
        target_user_id VARCHAR(36),
        target_user_email VARCHAR(320),
        target_user_name VARCHAR(255),
        details JSONB,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs (action)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_user_id ON audit_logs (actor_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_email ON audit_logs (actor_email)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_name ON audit_logs (actor_name)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_role ON audit_logs (actor_role)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_status_code ON audit_logs (status_code)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_path ON audit_logs (path)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_route_template ON audit_logs (route_template)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_domain_tag ON audit_logs (domain_tag)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_container_id ON audit_logs (container_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_file_id ON audit_logs (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_folder_id ON audit_logs (folder_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_target_user_id ON audit_logs (target_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_target_user_email ON audit_logs (target_user_email)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_domains ON audit_logs USING GIN (actor_allowed_domains) WHERE actor_allowed_domains IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_details ON audit_logs USING GIN (details)",
]


async def _run(conn: AsyncConnection) -> None:
    for stmt in _STATEMENTS:
        await conn.execute(text(stmt))


async def migrate() -> None:
    async with engine.begin() as conn:
        await _run(conn)


if __name__ == "__main__":
    asyncio.run(migrate())
    print("audit_log_schema_upgrade: done")

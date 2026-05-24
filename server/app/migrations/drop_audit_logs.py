"""Drop the legacy audit_logs table.

All audit events are now written to server_logs (log_type='audit').
The audit_logs table and its indexes are dropped here once server_logs
has been verified as the active write target.
"""
from __future__ import annotations

from sqlalchemy import text

from app.core.database import engine


async def migrate() -> None:
    async with engine.begin() as conn:
        # Drop all legacy audit_logs indexes first (CASCADE handles it, but being explicit)
        await conn.execute(text("DROP TABLE IF EXISTS audit_logs CASCADE"))

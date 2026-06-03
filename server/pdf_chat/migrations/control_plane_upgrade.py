"""Runtime migration — create Control Plane tables + indexes.

Idempotent, non-fatal, additive (same convention as app/migrations/). Call from
the app lifespan after Base metadata create_all, or run standalone. Importing the
models registers them on Base so create_all also creates them; this migration
adds the secondary indexes the spec calls for.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Importing registers the tables on the shared Base metadata.
from pdf_chat.models.manifests import UploadManifest, PageManifest, QueryAuditLog  # noqa: F401

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pdf_upload_sha256 ON pdf_upload_manifest(sha256)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_upload_tenant ON pdf_upload_manifest(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_upload_status ON pdf_upload_manifest(status)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_page_upload_id ON pdf_page_manifest(upload_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_page_status ON pdf_page_manifest(status)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_audit_user ON pdf_query_audit_log(user_id, created_at)",
]


async def run_migration(engine: AsyncEngine) -> None:
    """Create tables (via metadata) + indexes. Safe to run repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # type: ignore[name-defined]
        for stmt in _INDEXES:
            await conn.execute(text(stmt))


# Base is needed for create_all; import after models so metadata is populated.
from app.core.database import Base  # noqa: E402

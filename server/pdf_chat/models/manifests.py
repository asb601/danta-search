"""Control Plane ORM models — the PostgreSQL-backed management layer.

These tables do NOT store document content (that lives in Neo4j). They track the
status of every document and every page-level task, enabling crash recovery,
partial success, deduplication, and retrieval audit. Registered on the app's
shared Base so create_all/migrations pick them up in the same database.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

from pdf_chat.models.enums import DocStatus, PageStatus


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UploadManifest(Base):
    """One row per uploaded document — the document-level job ticket."""
    __tablename__ = "pdf_upload_manifest"

    upload_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    blob_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_length: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    acl_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    doc_version: Mapped[int] = mapped_column(Integer, default=1)
    parser_plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    preflight_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default=DocStatus.UPLOADED.value, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class PageManifest(Base):
    """One row per page — per-page task ticket for crash recovery + partial success."""
    __tablename__ = "pdf_page_manifest"

    task_id: Mapped[str] = mapped_column(Text, primary_key=True)  # {upload_id}:page:{n:06d}
    upload_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("pdf_upload_manifest.upload_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    page_num: Mapped[int] = mapped_column(Integer, nullable=False)
    element_type: Mapped[str] = mapped_column(Text, nullable=False, default="page")
    parser_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default=PageStatus.PENDING.value, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    @staticmethod
    def make_task_id(upload_id: str, page_num: int) -> str:
        return f"{upload_id}:page:{page_num:06d}"


class QueryAuditLog(Base):
    """One row per retrieval — records ACL decisions for compliance/audit."""
    __tablename__ = "pdf_query_audit_log"

    audit_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    returned_chunks: Mapped[list | None] = mapped_column(ARRAY(Text), nullable=True)
    denied_chunks: Mapped[list | None] = mapped_column(ARRAY(Text), nullable=True)
    acl_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

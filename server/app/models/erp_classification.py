"""ERP classification persistence.

One row per file holding the business-context classification produced by the
ERP classifier (or edited by an admin). Kept as a dedicated table — rather than
columns on FileMetadata — so it can carry provenance, confidence, evidence, and
a human-override flag without bloating the hot catalog projection, and so a
re-ingest cleanly upserts it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ErpClassification(Base):
    __tablename__ = "erp_classifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )

    source_system: Mapped[str] = mapped_column(String(120), default="Unknown")
    erp_module: Mapped[str] = mapped_column(String(120), default="Unknown")
    domain_polarity: Mapped[str] = mapped_column(String(20), default="neutral")  # customer|vendor|neutral
    process_role: Mapped[str] = mapped_column(String(120), default="unknown")
    grain: Mapped[str | None] = mapped_column(Text, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[list | None] = mapped_column(JSONB, default=list)
    # "llm" | "human_override" | "unknown"
    source: Mapped[str] = mapped_column(String(20), default="unknown")
    model_version: Mapped[str | None] = mapped_column(String(80), nullable=True)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("file_id", name="uq_erp_classification_file"),
    )

"""Phase 4 — cross-domain PDF↔CSV bridge ORM model.

One row records the reconciliation of a PDF ``Entity`` against the CSV master-key
registry. A bridge is LINKED only when the PDF entity's literal *values* overlap
a real master key (value evidence via ``fingerprint_value``) above the tunable
gates; a name match alone, or a sub-threshold overlap, is REFUSED — never a
silent top-match. Registered on the shared ``Base`` so create_all / the runtime
migration create it in the same PostgreSQL database. Tenant-isolated via
``container_id`` / ``tenant_id``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Float, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BridgeStatus(str, Enum):
    """pdf_entity_bridge.status — the reconciliation outcome."""
    LINKED = "linked"      # value overlap cleared every gate → bound to a master key
    REFUSED = "refused"    # name-only / sub-threshold → no join (never top-match)


class PdfEntityBridge(Base):
    """One PDF entity ↔ CSV semantic-entity reconciliation record."""
    __tablename__ = "pdf_entity_bridge"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    container_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    pdf_entity_id: Mapped[str] = mapped_column(Text, nullable=False)
    # Populated only on LINKED rows; the CSV-side SemanticEntity this entity maps to.
    semantic_entity_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_master_file_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_master_column: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_semantic_role: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_overlap_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    overlap_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pdf_value_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Full value-evidence trail (entity label, gate scores, thresholds, decisions).
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default=BridgeStatus.REFUSED.value, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("idx_pdf_bridge_container_entity", "container_id", "pdf_entity_id"),
    )

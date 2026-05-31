"""Semantic Contract persistence (the compiled "Danta Semantic Contract").

One governed contract per container, compiled from existing ingestion artifacts
(FileMetadata + ErpClassification + approved SemanticRelationship + SchemaDictionary)
and stored as JSONB. A content hash drives cache invalidation: when ingestion or
an approval changes the inputs, the recomputed hash differs and the cache is
rebuilt. The contract is the single surface the planner/dry-plan reason against.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SemanticContract(Base):
    __tablename__ = "semantic_contracts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )

    # The compiled contract: {source_systems, models[], relationships[],
    # metrics[], process_chains, instructions[]}.
    definition: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    # sha256 of the canonical inputs — recompute trigger / staleness check.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="active")
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("container_id", name="uq_semantic_contract_container"),
    )

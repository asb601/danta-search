import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SemanticEntity(Base):
    __tablename__ = "semantic_entities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    entity_name: Mapped[str] = mapped_column(String(120), nullable=False)
    primary_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attributes: Mapped[list | None] = mapped_column(JSONB, default=list)
    metrics: Mapped[list | None] = mapped_column(JSONB, default=list)
    dimensions: Mapped[list | None] = mapped_column(JSONB, default=list)
    grain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(50), default="ingestion")
    status: Mapped[str] = mapped_column(String(20), default="active")
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("file_id", name="uq_semantic_entities_file"),
    )


class SemanticRelationship(Base):
    __tablename__ = "semantic_relationships"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    source_relationship_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("file_relationships.id", ondelete="CASCADE"), nullable=True
    )
    file_a_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    file_b_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    from_entity: Mapped[str] = mapped_column(String(120), nullable=False)
    to_entity: Mapped[str] = mapped_column(String(120), nullable=False)
    from_column: Mapped[str] = mapped_column(String(255), nullable=False)
    to_column: Mapped[str] = mapped_column(String(255), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(30), nullable=False)
    join_rule: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    approval_status: Mapped[str] = mapped_column(String(20), default="candidate")
    risk_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="active")
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("source_relationship_id", name="uq_semantic_relationship_source"),
    )

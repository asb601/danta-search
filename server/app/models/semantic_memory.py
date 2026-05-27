import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SemanticMemoryRecord(Base):
    __tablename__ = "semantic_memory_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    memory_type: Mapped[str] = mapped_column(String(40), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_terms: Mapped[list] = mapped_column(JSONB, default=list)
    behaviors: Mapped[list] = mapped_column(JSONB, default=list)
    dimensions: Mapped[dict] = mapped_column(JSONB, default=dict)
    constraints: Mapped[dict] = mapped_column(JSONB, default=dict)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    authority_score: Mapped[float] = mapped_column(Float, default=0.0)
    governance_status: Mapped[str] = mapped_column(String(20), default="candidate")
    status: Mapped[str] = mapped_column(String(20), default="active")
    source: Mapped[str] = mapped_column(String(50), default="ingestion")
    source_file_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=True
    )
    source_entity_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("semantic_entities.id", ondelete="SET NULL"), nullable=True
    )
    source_relationship_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("semantic_relationships.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "container_id",
            "memory_type",
            "canonical_key",
            name="uq_semantic_memory_canonical",
        ),
        Index("idx_smr_container_status_type", "container_id", "status", "governance_status", "memory_type"),
        Index("idx_smr_source_file", "source_file_id"),
        Index("idx_smr_terms_gin", "normalized_terms", postgresql_using="gin"),
        Index("idx_smr_behaviors_gin", "behaviors", postgresql_using="gin"),
    )


class SemanticMemoryEvidence(Base):
    __tablename__ = "semantic_memory_evidence"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("semantic_memory_records.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    evidence_key: Mapped[str] = mapped_column(String(120), nullable=False)
    evidence_value: Mapped[dict | list | str | None] = mapped_column(JSONB, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "memory_id",
            "source_type",
            "source_id",
            "evidence_key",
            name="uq_semantic_memory_evidence_source",
        ),
        Index("idx_sme_memory", "memory_id"),
        Index("idx_sme_file", "file_id"),
    )


class SemanticMemoryLink(Base):
    __tablename__ = "semantic_memory_links"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    source_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("semantic_memory_records.id", ondelete="CASCADE"), nullable=False
    )
    target_memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("semantic_memory_records.id", ondelete="CASCADE"), nullable=False
    )
    link_type: Mapped[str] = mapped_column(String(40), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "source_memory_id",
            "target_memory_id",
            "link_type",
            name="uq_semantic_memory_link",
        ),
        Index("idx_sml_container", "container_id", "status", "link_type"),
    )


class SemanticMemoryAssetIndex(Base):
    __tablename__ = "semantic_memory_asset_index"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("semantic_memory_records.id", ondelete="CASCADE"), nullable=False
    )
    index_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    terms: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("memory_id", "file_id", "index_kind", name="uq_semantic_memory_asset"),
        Index("idx_smai_file", "file_id", "index_kind", "score"),
        Index("idx_smai_container", "container_id", "index_kind"),
    )


class SemanticMemoryTermIndex(Base):
    __tablename__ = "semantic_memory_term_index"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("semantic_memory_records.id", ondelete="CASCADE"), nullable=False
    )
    term: Mapped[str] = mapped_column(String(120), nullable=False)
    token_class: Mapped[str] = mapped_column(String(30), default="term")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("container_id", "term", "memory_id", name="uq_semantic_memory_term"),
        Index("idx_smti_container_term", "container_id", "term", "status"),
        Index("idx_smti_memory", "memory_id"),
    )


class BrainContextTrace(Base):
    __tablename__ = "brain_context_traces"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    container_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    selected_memory_ids: Mapped[list] = mapped_column(JSONB, default=list)
    ambiguity_flags: Mapped[list] = mapped_column(JSONB, default=list)
    retrieval_guidance: Mapped[dict] = mapped_column(JSONB, default=dict)
    execution_envelope: Mapped[dict] = mapped_column(JSONB, default=dict)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    caps: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("idx_bct_request", "request_id"),
        Index("idx_bct_container_created", "container_id", "created_at"),
        Index("idx_bct_query_hash", "query_hash"),
    )


from app.models.container import ContainerConfig  # noqa: E402, F401
from app.models.file import File  # noqa: E402, F401
from app.models.semantic_layer import SemanticEntity, SemanticRelationship  # noqa: E402, F401
from app.models.user import User  # noqa: E402, F401
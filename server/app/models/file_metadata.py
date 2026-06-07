import uuid
from datetime import datetime, timezone, date

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, String, BigInteger, Text, Date, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class FileMetadata(Base):
    __tablename__ = "file_metadata"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    blob_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    container_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=True
    )
    columns_info: Mapped[list | None] = mapped_column(JSONB, default=list)
    row_count: Mapped[int] = mapped_column(BigInteger, default=0)
    ai_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    good_for: Mapped[list | None] = mapped_column(JSONB, default=list)
    key_metrics: Mapped[list | None] = mapped_column(JSONB, default=list)
    key_dimensions: Mapped[list | None] = mapped_column(JSONB, default=list)
    date_range_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_range_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    delimiter: Mapped[str] = mapped_column(String(10), default=",")
    sample_rows: Mapped[list | None] = mapped_column(JSONB, default=list)
    ingest_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ontology layer — column semantic role map (ingestion-time, permanent)
    # Schema: {"source_col": "custom:<kind>:<label>", ...}
    # Populated by column_role_resolver at ingest time.
    # At query time the planner reads this — zero LLM calls for join resolution.
    # role_source: "llm" | "llm_dynamic" | future resolver source labels
    column_semantic_roles: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    role_source: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Phase 5: Role confidence evidence (ingestion-time, permanent)
    # Schema: {"source_col": {"confidence": 0.92, "signals": ["column_name", "value_pattern"], "source": "llm"}}
    # Only populated for files ingested after Phase 5 roll-out. Older files have None.
    column_role_evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    # Phase 5: Per-file ingestion confidence score (computed at end of pipeline)
    # overall: 0.0–1.0 aggregate of role quality + edge quality + metadata completeness
    # signals: breakdown dict for observability / admin UI
    ingestion_confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ingestion_confidence_signals: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    # SME Phase-1 trust/quarantine layer — coarse per-file trust state derived
    # (at ingest completion, flag-gated) from the EXISTING confidence level +
    # ingestion-audit severity. See services/trust_state.py for the contract.
    # server_default "trusted" so pre-existing rows and the flag-off path are
    # byte-identical to today.
    trust_state: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="trusted"
    )

    # Retrieval-engine columns (PHASE 1)
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    description_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    file: Mapped["File"] = relationship("File")


from app.models.file import File  # noqa: E402, F401

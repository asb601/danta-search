import uuid

from sqlalchemy import Integer, String, Float, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class FileRelationship(Base):
    __tablename__ = "file_relationships"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_a_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    file_b_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    file_a_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_b_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    shared_column: Mapped[str] = mapped_column(String(255), nullable=False)
    related_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # The resolved typed semantic role that makes this join valid.
    # Example: "custom:entity_key:record".
    # Without this, the planner cannot distinguish a real join key from a
    # coincidental column name match (two files both having a column called "id").
    semantic_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # How the role was resolved: "glossary" | "heuristic" | "llm"
    role_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    value_overlap_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    join_type: Mapped[str] = mapped_column(String(20), default="LEFT JOIN")

    # Phase 5: Edge provenance — number of overlapping fingerprinted key values
    # that confirmed this relationship. Lower values = weaker evidence.
    evidence_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Phase 5: Edge provenance context for observability / audit
    # Schema: {"card_a": int, "card_b": int, "role_a": str, "role_b": str,
    #          "key_kind_a": str, "key_kind_b": str}
    edge_provenance: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    __table_args__ = (
        UniqueConstraint("file_a_id", "file_b_id", "shared_column"),
    )

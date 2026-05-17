import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ColumnKeyRegistry(Base):
    """Tenant-scoped inverted index of join-key value fingerprints.

    This is the durable database-backed version of a hashmap:
      fingerprint -> candidate file/column owners.

    The GIN index is created in migrations/relationship_index_upgrade.py.
    """

    __tablename__ = "column_key_registry"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("container_configs.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    blob_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    semantic_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    key_kind: Mapped[str] = mapped_column(String(20), nullable=False, default="candidate")
    cardinality: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    sample_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    unique_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    null_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    value_fingerprints: Mapped[list[str]] = mapped_column(ARRAY(String(16)), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        UniqueConstraint("file_id", "column_name", name="uq_column_key_registry_file_column"),
    )

"""
SchemaDictionary — tracks uploaded data-dictionary / field-definition files.

When a file whose column names pattern-match to (field_name, description) is
ingested, a SchemaDictionary row is created pointing at the converted parquet.

At query time, the agent pre-loads all schema dicts for the container into a
flat {field_name_lower → description} dict and uses it to:
  1. Enrich inspect_column output with business meaning.
  2. Answer direct "what does X column mean?" questions via
     lookup_field_definition without touching the prompt.

One org can register multiple dictionaries (e.g. FBL3N fields, BSEG fields,
an HR glossary). All are merged at load time — the first match wins.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SchemaDictionary(Base):
    __tablename__ = "schema_dictionaries"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Which container this dictionary belongs to.
    container_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("container_configs.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The file that was uploaded.
    file_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Azure blob path of the converted parquet.
    parquet_blob_path: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Column names discovered by detection heuristics.
    field_name_col: Mapped[str] = mapped_column(String(200), nullable=False)
    description_col: Mapped[str] = mapped_column(String(200), nullable=False)
    # Optional secondary column for extended notes / long text.
    notes_col: Mapped[str | None] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        # One registration per file — re-ingest replaces the row via upsert.
        UniqueConstraint("file_id", name="uq_schema_dictionary_file_id"),
    )

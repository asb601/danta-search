import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.core.crypto import EncryptedText


class ContainerConfig(Base):
    __tablename__ = "container_configs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    container_name: Mapped[str] = mapped_column(String(255), nullable=False)
    connection_string: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Per-container cleaning rules (JSONB) ────────────────────────────────
    # Optional overrides loaded at ingest time and passed to get_cleaning_profile().
    # No code change or redeploy needed to add client-specific null patterns.
    #
    # Schema:
    #   {
    #       "extra_null_patterns":    ["k.a.", "N/V", "#VALUE!"],
    #       "extra_garbage_patterns": [".*Zwischensumme.*"]
    #   }
    cleaning_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    # Optional semantic role extensions for this container.
    # Schema:
    #   {
    #       "roles": [
    #           {"role": "claim", "kind": "entity_key", "description": "insurance claim id"},
    #           {"role": "loss_amount", "kind": "additive_measure", "default_aggregation": "SUM"}
    #       ]
    #   }
    semantic_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)

    creator: Mapped["User"] = relationship("User")


from app.models.user import User  # noqa: E402, F401

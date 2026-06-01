"""
OrgAISettings — per-organization Azure OpenAI / LLM credentials & deployments.

Org-RBAC overhaul. Lets each tenant bring its own AI keys and deployments
instead of relying on the global process-wide configuration. Secret material
(API keys) is stored encrypted-at-rest via the shared EncryptedText decorator.

One row per organization (UNIQUE organization_id).
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.crypto import EncryptedText


class OrgAISettings(Base):
    __tablename__ = "org_ai_settings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # ── Secret material (encrypted at rest) ─────────────────────────────────
    chat_api_key: Mapped[str | None] = mapped_column(EncryptedText, nullable=True)
    embeddings_api_key: Mapped[str | None] = mapped_column(
        EncryptedText, nullable=True
    )
    fallback_api_key: Mapped[str | None] = mapped_column(
        EncryptedText, nullable=True
    )
    # Per-org PostgreSQL DSN (carries a DB password → encrypted at rest).
    postgres_url: Mapped[str | None] = mapped_column(EncryptedText, nullable=True)

    # ── Endpoints / deployments (non-secret) ────────────────────────────────
    chat_endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    chat_deployment: Mapped[str | None] = mapped_column(Text, nullable=True)
    embeddings_deployment: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_deployment: Mapped[str | None] = mapped_column(
        Text, nullable=True, default="gpt-4o-mini"
    )
    api_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    organization: Mapped["Organization"] = relationship("Organization")


from app.models.organization import Organization  # noqa: E402, F401

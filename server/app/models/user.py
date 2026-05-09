import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    picture: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    # Role label exposed to the client so the UI can adapt.
    # Allowed values: "user" | "developer" | "admin".
    # `is_admin` remains the source of truth for backend permission checks;
    # `role` is mostly informational (e.g. to label external API consumers
    # as "developer" so the UI hides admin pages).
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    # Domain access control (PHASE 15):
    # NULL / empty list → unrestricted (user sees all domains, like an admin)
    # Non-empty list → user may only access files in folders tagged with these domains
    allowed_domains: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True, default=None
    )
    # Multi-tenancy (Phase 16): which organization this user belongs to.
    # NULL = global / platform admin (no organization scoping applied).
    # Non-NULL = ALL chat queries are forcibly scoped to this org's container.
    organization_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    folders: Mapped[list["Folder"]] = relationship(
        "Folder", back_populates="owner", cascade="all, delete-orphan"
    )
    organization: Mapped["Organization | None"] = relationship(
        "Organization", back_populates="users", foreign_keys=[organization_id]
    )


# Avoid circular import — import here so relationship resolves
from app.models.folder import Folder  # noqa: E402, F401
from app.models.organization import Organization  # noqa: E402, F401

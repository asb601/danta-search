"""
Organization model — multi-tenant root entity.

Each Organization represents one client company (e.g. "Acme Corp", "TechCo").
It owns:
  - one ContainerConfig (their Azure blob container — sole source of their data)
  - many Users (employees of that organization)

Hard-tenancy guarantees enforced elsewhere:
  - Non-admin users' chat queries are scoped to their org's container_id
    (the body.container_id field is ignored for non-admin users).
  - allowed_domains continues to act as a sub-filter inside the org.
  - Global / platform admins (organization_id IS NULL on user) bypass scoping.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # The Azure blob container this organization's data lives in.
    # Hard-bound 1:1 — one org owns one container.
    container_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("container_configs.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Org-RBAC overhaul:
    # The user who owns / created this organization (org-level super-admin).
    owner_user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Onboarding lifecycle state machine.
    # e.g. 'created' | 'container_provisioned' | 'ai_configured' | 'completed'.
    onboarding_state: Mapped[str] = mapped_column(
        String(40), nullable=False, default="created"
    )
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # URL- and blob-safe unique slug derived from the org name.
    slug: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    container: Mapped["ContainerConfig | None"] = relationship(
        "ContainerConfig", foreign_keys=[container_id]
    )
    users: Mapped[list["User"]] = relationship(
        "User",
        back_populates="organization",
        foreign_keys="User.organization_id",
    )
    owner: Mapped["User | None"] = relationship(
        "User", foreign_keys=[owner_user_id]
    )


# Late imports to break circular references
from app.models.container import ContainerConfig  # noqa: E402, F401
from app.models.user import User  # noqa: E402, F401

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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    container: Mapped["ContainerConfig | None"] = relationship("ContainerConfig")
    users: Mapped[list["User"]] = relationship("User", back_populates="organization")


# Late imports to break circular references
from app.models.container import ContainerConfig  # noqa: E402, F401
from app.models.user import User  # noqa: E402, F401

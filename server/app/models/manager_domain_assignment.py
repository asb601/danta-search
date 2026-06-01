"""
ManagerDomainAssignment — per-(user, organization, domain) grant.

Org-RBAC overhaul. Replaces the coarse `users.allowed_domains` array with an
explicit, organization-scoped assignment row that can additionally mark a user
as a *domain admin* (manager who can administer that domain within the org).

`users.allowed_domains` is preserved for backward-compat; this table is the
forward source of truth and is backfilled from it at migration time.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ManagerDomainAssignment(Base):
    __tablename__ = "manager_domain_assignments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    domain_tag: Mapped[str] = mapped_column(Text, nullable=False)
    # Whether this user administers (not merely accesses) the domain in this org.
    is_domain_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    granted_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    granter: Mapped["User | None"] = relationship(
        "User", foreign_keys=[granted_by]
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "organization_id",
            "domain_tag",
            name="uq_manager_domain_assignment",
        ),
        Index("ix_manager_domain_assignments_user", "user_id"),
        Index("ix_manager_domain_assignments_org", "organization_id"),
    )


from app.models.user import User  # noqa: E402, F401
from app.models.organization import Organization  # noqa: E402, F401

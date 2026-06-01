"""
PlatformAdminGrant — explicit grant of a platform admin onto an organization.

Org-RBAC overhaul. A platform admin (cross-org superuser) gains scoped access
to a specific organization through an auditable grant row that can be revoked
without deleting history.

One active grant per (organization, platform_admin_user) — enforced by a
UNIQUE constraint; `status` toggles between 'active' and 'revoked'.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PlatformAdminGrant(Base):
    __tablename__ = "platform_admin_grants"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    platform_admin_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    granted_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # 'active' | 'revoked'
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    organization: Mapped["Organization"] = relationship("Organization")
    platform_admin: Mapped["User"] = relationship(
        "User", foreign_keys=[platform_admin_user_id]
    )
    granter: Mapped["User | None"] = relationship(
        "User", foreign_keys=[granted_by]
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "platform_admin_user_id",
            name="uq_platform_admin_grant",
        ),
        Index("ix_platform_admin_grants_org", "organization_id"),
        Index("ix_platform_admin_grants_user", "platform_admin_user_id"),
    )


from app.models.user import User  # noqa: E402, F401
from app.models.organization import Organization  # noqa: E402, F401

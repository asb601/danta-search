import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, default="request", index=True)
    action: Mapped[str] = mapped_column(String(160), nullable=False, index=True)

    actor_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_email: Mapped[str | None] = mapped_column(String(320), nullable=True, index=True)
    actor_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    actor_role: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    actor_is_admin: Mapped[bool] = mapped_column(default=False, nullable=False)
    actor_allowed_domains: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    actor_organization_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    method: Mapped[str | None] = mapped_column(String(10), nullable=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    route_template: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(80), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    domain_tag: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    container_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    file_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    file_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    folder_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    folder_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    target_user_email: Mapped[str | None] = mapped_column(String(320), nullable=True, index=True)
    target_user_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    actor: Mapped["User | None"] = relationship("User", foreign_keys=[actor_user_id])


Index("idx_audit_logs_created_actor", AuditLog.created_at.desc(), AuditLog.actor_user_id)
Index("idx_audit_logs_actor_domains", AuditLog.actor_allowed_domains, postgresql_using="gin")
Index("idx_audit_logs_details", AuditLog.details, postgresql_using="gin")


from app.models.user import User  # noqa: E402, F401

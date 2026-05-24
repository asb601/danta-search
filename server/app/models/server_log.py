"""
Unified server_logs table — replaces all text log files and the audit_logs table.

Log types:
    audit       HTTP request / user action log (replaces audit_logs table + audit.log)
    ai_pipeline LLM queries, tool calls, SQL execution (replaces ai_pipeline.log)
    ingestion   Document pipeline stages and timings (subset of ai_pipeline.log)
    system      Auth, upload, blob, folder, container events (replaces system.log)
    llm         Per-call LLM request/response detail (replaces llm_calls.log)
    cost        Token usage and billing events (replaces costs.log)

RBAC enforced at query time via a single scope function:
    admin   → no filter
    manager → WHERE domain_tag IN (allowed_domains) OR actor_user_id = self
    member  → WHERE log_type = 'ai_pipeline' AND actor_user_id = self
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

# Valid log_type values — checked at write time
LOG_TYPES = {"audit", "ai_pipeline", "ingestion", "system", "llm", "cost"}

# Valid level values
LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


class ServerLog(Base):
    __tablename__ = "server_logs"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ── Classification ────────────────────────────────────────────────────────
    # log_type drives which tab shows this row and which RBAC branch applies.
    log_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # event is the structlog event name, e.g. "request", "llm_input", "chain_start"
    event: Mapped[str] = mapped_column(String(80), nullable=False)
    # level for filtering by severity
    level: Mapped[str] = mapped_column(String(10), nullable=False, default="info")

    # ── Actor — RBAC filtering columns ───────────────────────────────────────
    # These three are the hot path for every WHERE clause — all indexed.
    actor_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    actor_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    actor_role: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # domain_tag is the manager's primary scope filter.
    # Matches the SAP domain / folder domain assigned to the user.
    domain_tag: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # ── Resource context ─────────────────────────────────────────────────────
    # trace_id groups all events belonging to a single pipeline run or query session.
    trace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    file_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # ── HTTP context (audit log only) ─────────────────────────────────────────
    method: Mapped[str | None] = mapped_column(String(10), nullable=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # ── Payload ───────────────────────────────────────────────────────────────
    # Anything event-specific (prompts, token counts, step timings, error detail, etc.)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Relationship ──────────────────────────────────────────────────────────
    actor: Mapped["User | None"] = relationship("User", foreign_keys=[actor_user_id])


# ── Indexes ───────────────────────────────────────────────────────────────────
# Ordered by query priority: time desc is the default sort for every tab.
Index("idx_sl_created_at",   ServerLog.created_at.desc())

# RBAC hot path — every non-admin query filters on one or more of these.
Index("idx_sl_actor_user",   ServerLog.actor_user_id)
Index("idx_sl_domain_tag",   ServerLog.domain_tag)
Index("idx_sl_log_type",     ServerLog.log_type)

# Composite: the two most common combined filters
# manager:  log_type + domain_tag + created_at
# member:   log_type + actor_user_id + created_at
Index("idx_sl_type_domain",  ServerLog.log_type, ServerLog.domain_tag,    ServerLog.created_at.desc())
Index("idx_sl_type_actor",   ServerLog.log_type, ServerLog.actor_user_id, ServerLog.created_at.desc())

# Supporting lookups
Index("idx_sl_trace_id",     ServerLog.trace_id)
Index("idx_sl_level",        ServerLog.level)
Index("idx_sl_details_gin",  ServerLog.details, postgresql_using="gin")


from app.models.user import User  # noqa: E402, F401

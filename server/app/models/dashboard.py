"""
Dashboard and DashboardFolder models — persistent, metadata-driven dashboards.

Design (see response.txt Section 3):
  - A Dashboard is a workspace owned by one user, scoped to a tenant container.
  - Its full render contract lives in `config` (JSONB DashboardConfig) so the
    frontend renderer is a pure function of metadata — no per-dashboard code.
  - `prompt_history` records each generation prompt so a dashboard can be
    regenerated/refreshed later (and audited) without losing what produced it.
  - The embedded dataset snapshot inside `config.widgets[].data` lets users
    return later and see analytics instantly WITHOUT regenerating.
  - DashboardFolders organize dashboards like projects/workspaces and may nest.
  - Tenant isolation mirrors every other table: `container_id` scoping.

This module adds NO query logic. Generation reuses the existing agent runtime;
these models only persist the resulting metadata.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DashboardFolder(Base):
    """A workspace/folder that groups dashboards. May nest via parent_id."""

    __tablename__ = "dashboard_folders"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("container_configs.id", ondelete="CASCADE"),
        nullable=True,
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("dashboard_folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="New folder")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_dashboard_folders_owner", "owner_id"),
        Index("ix_dashboard_folders_container", "container_id"),
    )


class Dashboard(Base):
    """A persisted, metadata-driven dashboard. `config` is the render contract."""

    __tablename__ = "dashboards"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    container_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("container_configs.id", ondelete="CASCADE"),
        nullable=True,
    )
    folder_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("dashboard_folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, default="Untitled dashboard"
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    # DashboardConfig — the versioned, render-ready JSON (response.txt 8.3).
    config: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="'{}'::jsonb"
    )
    # [{prompt, created_at, widget_ids}] — enables regeneration / audit.
    prompt_history: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="'[]'::jsonb"
    )
    # File/blob ids the dashboard draws from (for provenance + invalidation).
    source_file_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="'[]'::jsonb"
    )

    is_pinned: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # draft | generating | ready | error
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft", server_default="draft"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_dashboards_owner", "owner_id", "updated_at"),
        Index("ix_dashboards_container", "container_id"),
        Index("ix_dashboards_folder", "folder_id"),
    )

"""Pydantic request/response schemas for the dashboard layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Folders --------------------------------------------------------------

class DashboardFolderCreate(BaseModel):
    name: str = Field(default="New folder", max_length=255)
    parent_id: str | None = None
    container_id: str | None = None


class DashboardFolderUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    parent_id: str | None = None


class DashboardFolderOut(BaseModel):
    id: str
    name: str
    parent_id: str | None
    container_id: str | None
    created_at: str


# ---- Dashboards -----------------------------------------------------------

class DashboardCreate(BaseModel):
    title: str = Field(default="Untitled dashboard", max_length=255)
    description: str | None = None
    folder_id: str | None = None
    container_id: str | None = None


class DashboardUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    folder_id: str | None = None
    is_pinned: bool | None = None


class DashboardGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    container_id: str | None = None
    max_widgets: int = Field(default=6, ge=1, le=8)
    append: bool = False


class DashboardSummary(BaseModel):
    id: str
    title: str
    description: str | None
    folder_id: str | None
    is_pinned: bool
    status: str
    widget_count: int
    created_at: str
    updated_at: str


class DashboardOut(BaseModel):
    id: str
    title: str
    description: str | None
    folder_id: str | None
    container_id: str | None
    is_pinned: bool
    status: str
    config: dict
    prompt_history: list
    source_file_ids: list
    created_at: str
    updated_at: str

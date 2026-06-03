"""Pydantic v2 request/response schemas for the PDF RAG API.

Pure schema module — imports only pydantic and the foundation enums. Safe to
import with zero infra installed. Every schema maps to a spec stage:

- UploadResponse / StatusResponse  → Spec §5 Stage 0/4/14 (upload + control plane)
- ChatRequest / ChatResponse       → Spec §6 (query retrieval pipeline)
- DocumentSummary / DeleteResponse → GET/DELETE /documents

These are the API surface; the internal agent state lives in ``agent/state.py``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict

from pdf_chat.models.enums import DocStatus


class UploadResponse(BaseModel):
    """Returned immediately from POST /upload (Spec §5 Stage 0 + Stage 2 dedup).

    The client polls GET /status/{upload_id} after receiving this. ``deduplicated``
    is True when the SHA-256 fingerprint matched an already-indexed document for
    the same tenant, so no processing was queued.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(..., description="Document-level job ticket id (upload_manifest PK).")
    status: DocStatus = Field(..., description="Current document lifecycle status.")
    deduplicated: bool = Field(
        False,
        description="True if SHA-256 matched an existing indexed doc for this tenant (skip).",
    )


class StatusResponse(BaseModel):
    """Returned from GET /status/{upload_id} (Spec §5 Stage 14 reconciliation).

    Page counters are derived from the page_manifest rows so the client can show
    granular progress and surface partial-success situations.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    status: DocStatus
    page_count: int = 0
    pages_succeeded: int = 0
    pages_failed: int = 0
    pages_pending: int = 0
    error_message: str | None = None


class Citation(BaseModel):
    """One inline [N] citation pointing at the source chunk's document + page."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(..., description="Citation marker matching [N] in the answer text.")
    doc_id: str
    page: int


class ChatRequest(BaseModel):
    """POST /chat body (Spec §6 Stage 1).

    ``tenant_id``/groups/user are normally taken from the JWT; tenant_id is kept
    here for explicitness and test ergonomics. ``doc_ids`` optionally scopes
    retrieval to specific documents; None means search the whole tenant.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural-language question.")
    tenant_id: str = Field(..., description="Tenant isolation key (also enforced from JWT).")
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document ids (None = all)."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=200, description="Override retrieval fan-out (defaults from config)."
    )


class ChatResponse(BaseModel):
    """POST /chat response (Spec §6 Stage 10).

    ``answer`` is grounded strictly in retrieved context with inline [N] markers
    resolved by ``citations``. ``cached`` indicates a Redis cache hit (Stage 5).
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: int = 0
    cached: bool = False


class DocumentSummary(BaseModel):
    """One row in GET /documents — a tenant's uploaded document and its status."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    status: DocStatus
    page_count: int | None = None
    mime_type: str | None = None
    created_at: str | None = Field(
        default=None, description="ISO-8601 upload timestamp."
    )


class DeleteResponse(BaseModel):
    """DELETE /documents/{id} result."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    deleted: bool
    chunks_removed: int = 0

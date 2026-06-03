"""FastAPI router for the PDF RAG system.

Mounted under ``/api/pdf``. Routes are THIN: they validate input, do
auth/tenant resolution, and delegate to the ingestion control plane (Team A/B)
or the query agent (this team). All heavy/infra dependencies are imported LATE,
inside the route bodies, so this module imports cleanly with zero infra and
WITHOUT requiring the other teams' modules to exist yet.

Route → spec stage map:
  POST   /upload              → Spec §5 Stage 0–6 (stream to blob, fingerprint,
                                 preflight, manifest writes, page-task fan-out)
  GET    /status/{upload_id}  → Spec §5 Stage 14 (reconciled doc + page counters)
  POST   /chat                → Spec §6 Stage 1–10 (the query agent pipeline)
  GET    /documents           → list tenant documents
  DELETE /documents/{id}      → remove a document + its Neo4j chunks

NOTE — how to mount in server/app/main.py (do NOT edit main.py here):
    from pdf_chat.api.routes import pdf_router, _resolve_current_user
    from app.dependencies import get_current_user
    # Wire the JWT principal: every route derives tenant/user/groups from the
    # token, never from client input. This override binds the standalone auth
    # bridge to the app's real get_current_user (which carries its own
    # HTTPBearer + db sub-dependencies) so FastAPI resolves them per request.
    app.dependency_overrides[_resolve_current_user] = get_current_user
    app.include_router(pdf_router)          # already prefixed with /api/pdf
The router is self-prefixed, so no extra prefix argument is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from pdf_chat.schemas.pdf_schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DeleteResponse,
    DocumentSummary,
    StatusResponse,
    UploadResponse,
)

pdf_router = APIRouter(prefix="/api/pdf", tags=["pdf"])


# --------------------------------------------------------------------------- #
# Auth / principal resolution.
#
# `Principal` is the security context derived SOLELY from the verified JWT — the
# client never supplies tenant/user/groups. `get_principal` is a FastAPI
# dependency that imports `app.dependencies.get_current_user` LATE (inside the
# function body) so this module imports with zero infra and without requiring
# app.* to be importable at definition time.
# --------------------------------------------------------------------------- #
@dataclass
class Principal:
    """Token-derived security context. The source of truth for tenant isolation."""

    user_id: str
    tenant_id: str
    groups: list[str]


def _principal_from_user(user: Any) -> Principal:
    """Map the app's `User` ORM object onto a `Principal`.

    Tenant is the user's ``organization_id`` (the tenant boundary in this app);
    groups come from ``allowed_domains``. Defensive ``getattr`` so a shape change
    in the User model degrades to a safe, fail-closed empty value rather than an
    AttributeError. NOTE: if the app's principal shape changes, only this mapping
    needs updating — the security surface (no client-trusted tenant) is unchanged.
    """
    user_id = str(getattr(user, "id", "") or "")
    tenant_id = str(getattr(user, "organization_id", "") or "")
    groups = list(getattr(user, "allowed_domains", None) or [])
    return Principal(user_id=user_id, tenant_id=tenant_id, groups=groups)


async def _resolve_current_user() -> Any:  # pragma: no cover - infra
    """Override seam for the app's JWT auth — wired at mount time, NOT at import.

    This is a FastAPI sub-dependency of :func:`get_principal`. It is NEVER bound
    to ``app.*`` at module-import time (so this module imports with zero infra).
    Instead the host app sets::

        app.dependency_overrides[_resolve_current_user] = get_current_user

    which makes FastAPI resolve the app's ``get_current_user`` (and ITS own
    HTTPBearer + db sub-dependencies) per request and inject the authenticated
    ``User``. Until that override is installed this fails CLOSED with 503 — there
    is no anonymous / client-trusted path to tenant data.
    """
    raise HTTPException(status_code=503, detail="Auth backend not wired.")


async def get_principal(
    request_user: Any = Depends(_resolve_current_user),
) -> Principal:  # pragma: no cover - infra
    """FastAPI dependency: resolve the current principal from the JWT.

    ``request_user`` is injected by FastAPI via :func:`_resolve_current_user`
    (overridden to the app's ``get_current_user`` when mounted — see module
    docstring). We then derive the tenant/user/groups from it. The client never
    supplies these values, so a forged ``tenant_id`` in the request body cannot
    widen access.
    """
    if request_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return _principal_from_user(request_user)


def _enforce_tenant(principal: Principal, client_tenant_id: str | None) -> str:
    """Return the trusted tenant id, rejecting any mismatching client value.

    The client MUST NOT be able to act on another tenant by passing a forged
    ``tenant_id`` in the body/query. We trust ONLY the token. A client-supplied
    value that disagrees with the token is a 403 (not silently honored).
    """
    token_tenant = principal.tenant_id
    if not token_tenant:
        raise HTTPException(status_code=403, detail="No tenant on principal.")
    if client_tenant_id and client_tenant_id != token_tenant:
        raise HTTPException(
            status_code=403, detail="tenant_id does not match the authenticated principal."
        )
    return token_tenant


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@pdf_router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    tenant_id: str | None = Form(default=None),
    principal: Principal = Depends(get_principal),
):
    """Spec §5 Stage 0–6 — accept a PDF, stream to blob, fingerprint, preflight,
    write the upload + page manifests, and fan out one Celery task per page.

    Tenant + user are taken from the JWT principal — a client-supplied
    ``tenant_id`` form field is only accepted if it MATCHES the token (else 403).
    The control plane (Team A) decides dedup via SHA-256; if the same bytes were
    already indexed for this tenant we return the existing ``upload_id`` with
    ``deduplicated=True`` and queue nothing.
    """
    from pdf_chat.ingestion.fingerprint import compute_sha256

    trusted_tenant = _enforce_tenant(principal, tenant_id)

    file_bytes = await file.read()  # NOTE: production path streams to blob; control plane reads byte-ranges per page.
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    sha256 = compute_sha256(file_bytes)

    # Late imports — control plane / ingestion may not be importable in isolation.
    try:
        from pdf_chat.control_plane.upload_service import handle_upload  # type: ignore
    except Exception as exc:  # pragma: no cover - infra not present
        raise HTTPException(
            status_code=503,
            detail="Upload service unavailable (control plane not deployed).",
        ) from exc

    result = await handle_upload(
        file_bytes=file_bytes,
        filename=file.filename or "document.pdf",
        content_type=file.content_type,
        sha256=sha256,
        tenant_id=trusted_tenant,
        user_id=principal.user_id,
    )
    return UploadResponse(
        upload_id=result["upload_id"],
        status=result["status"],
        deduplicated=result.get("deduplicated", False),
    )


@pdf_router.get("/status/{upload_id}", response_model=StatusResponse)
async def get_status(upload_id: str, principal: Principal = Depends(get_principal)):
    """Spec §5 Stage 14 — reconciled document status + per-page progress counters.

    Tenant-scoped: a principal can only read the status of its own tenant's
    documents (tenant from the JWT).
    """
    try:
        from pdf_chat.control_plane.status_service import get_upload_status  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail="Status service unavailable (control plane not deployed)."
        ) from exc

    status = await get_upload_status(upload_id, tenant_id=principal.tenant_id)
    if status is None:
        raise HTTPException(status_code=404, detail="upload_id not found.")
    return StatusResponse(**status)


@pdf_router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, principal: Principal = Depends(get_principal)):
    """Spec §6 Stage 1–10 — run the query agent: embed → cache → hybrid retrieve
    → rerank → ACL filter → lazy extract → assemble → generate → cache → audit.

    tenant/user/groups come from the JWT principal — a forged ``tenant_id`` in
    the body is rejected (403) if it disagrees with the token. The agent is pure
    orchestration; real adapters are wired lazily via ``build_default_deps`` so
    this route imports without infra.
    """
    from pdf_chat.agent.graph import build_default_deps, run_pdf_chat
    from pdf_chat.agent.state import PdfChatState

    trusted_tenant = _enforce_tenant(principal, body.tenant_id)

    state = PdfChatState(
        query=body.query,
        tenant_id=trusted_tenant,
        user_id=principal.user_id,
        groups=principal.groups,
        doc_ids=body.doc_ids,
        top_k=body.top_k,
    )
    deps = build_default_deps()
    result = await run_pdf_chat(state, deps)
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    return ChatResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        chunks_used=result.chunks_used(),
        cached=result.cached,
    )


@pdf_router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(principal: Principal = Depends(get_principal)):
    """List the principal's tenant documents (tenant from the JWT)."""
    try:
        from pdf_chat.control_plane.status_service import list_documents as _list  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail="Document service unavailable (control plane not deployed)."
        ) from exc

    rows = await _list(principal.tenant_id)
    return [DocumentSummary(**row) for row in rows]


@pdf_router.delete("/documents/{upload_id}", response_model=DeleteResponse)
async def delete_document(upload_id: str, principal: Principal = Depends(get_principal)):
    """Delete a document: remove its Neo4j chunks and mark the manifest deleted.

    Tenant-scoped to the JWT principal so one tenant cannot delete another's doc.
    """
    try:
        from pdf_chat.control_plane.delete_service import delete_document as _delete  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail="Delete service unavailable (control plane not deployed)."
        ) from exc

    result = await _delete(upload_id=upload_id, tenant_id=principal.tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail="upload_id not found.")
    return DeleteResponse(**result)

"""
Logs API — stream log files from the server for debugging.

GET  /api/logs/files                    → list available log files
GET  /api/logs/{filename}               → tail N lines from a log file
GET  /api/logs/{filename}/search?q=...  → search a log file
GET  /api/logs/file-timings             → upload + ingestion + parquet timing per file
GET  /api/logs/pipeline/tail?n=100      → last N pipeline events, pretty-formatted plain text
GET  /api/logs/pipeline/stream          → SSE live stream of pipeline events (pretty-formatted)

Auth: admin only (ADMIN_EMAIL from settings).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import LOG_DIR, format_pipeline_line
from app.dependencies import get_current_user, require_admin
from app.models.audit_log import AuditLog
from app.models.background_job import BackgroundJob
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.user import User

router = APIRouter(prefix="/logs", tags=["logs"])

# Only allow reading known log files — prevent path traversal
_ALLOWED_FILES = {"system.log", "ai_pipeline.log", "llm_calls.log", "costs.log", "pipeline.log", "audit.log"}


def _safe_log_path(filename: str) -> Path:
    """Resolve filename and ensure it's within LOG_DIR and in the allowed set."""
    # Strip any path components — only allow bare filenames
    clean = Path(filename).name
    if clean not in _ALLOWED_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown log file: {clean}")
    path = (LOG_DIR / clean).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Log file not found: {clean}")
    return path


# ── Pipeline-specific endpoints ─────────────────────────────────────────────

@router.get("/pipeline/tail")
async def pipeline_tail(
    n: int = Query(default=100, ge=1, le=2000),
    _: User = Depends(require_admin),
) -> StreamingResponse:
    """Return the last N pipeline events as pretty-formatted plain text.

    Hit this in a browser or with curl when you can't watch the server terminal:
        curl https://your-vm/api/v1/logs/pipeline/tail?n=50
    """
    path = LOG_DIR / "pipeline.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="pipeline.log not found — no queries yet?")

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail_lines = [l for l in raw_lines if l.strip()][-n:]

    formatted = []
    for line in tail_lines:
        formatted.append(format_pipeline_line(line))

    body = "\n".join(formatted) + "\n"
    return StreamingResponse(
        iter([body]),
        media_type="text/plain; charset=utf-8",
    )


@router.get("/pipeline/stream")
async def pipeline_stream(
    _: User = Depends(require_admin),
) -> StreamingResponse:
    """Live SSE stream of pipeline events, pretty-formatted.

    Equivalent to 'tail -f pipeline.log' but accessible over HTTP without SSH:
        curl -N https://your-vm/api/v1/logs/pipeline/stream

    Keeps the connection open and pushes new events as they arrive (0.3 s poll).
    Press Ctrl+C to stop.
    """
    path = LOG_DIR / "pipeline.log"

    async def _event_generator():
        # Yield a keepalive comment immediately so curl confirms connection
        yield ": connected to pipeline stream\n\n"

        # Open file and seek to end — only stream NEW events from this point
        def _open_at_end():
            f = open(path, "r", encoding="utf-8", errors="replace")  # noqa: WPS515
            f.seek(0, 2)
            return f

        if not path.is_file():
            yield "data: [pipeline.log not found — no queries yet]\n\n"
            return

        f = await asyncio.to_thread(_open_at_end)
        try:
            while True:
                line = await asyncio.to_thread(f.readline)
                if line and line.strip():
                    pretty = format_pipeline_line(line)
                    # SSE: each line of the block becomes a separate data: field
                    sse_lines = "\n".join(
                        f"data: {l}" for l in pretty.splitlines()
                    )
                    yield f"{sse_lines}\n\n"
                else:
                    # No new data — send keepalive comment every 0.3 s
                    await asyncio.sleep(0.3)
                    yield ": heartbeat\n\n"
        finally:
            await asyncio.to_thread(f.close)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer SSE
        },
    )


# ── Ingest-specific events endpoint ──────────────────────────────────────────

# Events emitted by the ingest pipeline (ingestion_service + data_preprocessor
# + analytics_service + parquet_service) — all carry pipeline="ingest" in
# structlog contextvars so we can filter them out of ai_pipeline.log.
_INGEST_EVENTS = {
    "chain_start", "chain_end", "chain_skip", "cleanup", "step",
    "preprocess", "analytics_compute", "parquet_conversion",
    "parquet_conversion_job_update_failed", "status_update_failed",
    "parquet_service",
}


@router.get("/ingest-events")
async def ingest_events(
    lines: int = Query(default=300, ge=1, le=2000),
    _: User = Depends(require_admin),
) -> dict:
    """
    Return the last N ingestion events from ai_pipeline.log.

    Filters to lines where pipeline=='ingest' (set by _ensure_trace())
    or whose event name is a known ingestion event.  Chat events are excluded.
    """
    path = LOG_DIR / "ai_pipeline.log"
    if not path.is_file():
        return {"total_lines": 0, "returned": 0, "lines": []}

    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ingest: list[dict] = []
    for line in all_lines:
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
            if ev.get("pipeline") == "ingest" or ev.get("event") in _INGEST_EVENTS:
                ingest.append(ev)
        except (json.JSONDecodeError, ValueError):
            pass

    tail = ingest[-lines:]
    return {"total_lines": len(ingest), "returned": len(tail), "lines": tail}


# ── Generic log file endpoints ────────────────────────────────────────────────

@router.get("/files")
async def list_log_files(_: User = Depends(require_admin)) -> dict:
    """List available log files with sizes."""
    files = []
    for name in sorted(_ALLOWED_FILES):
        path = LOG_DIR / name
        if path.exists():
            size_kb = round(path.stat().st_size / 1024, 1)
            files.append({"name": name, "size_kb": size_kb})
    return {"log_dir": str(LOG_DIR), "files": files}


@router.get("/file-timings")
async def file_timings(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Return upload, ingestion, and parquet conversion timing per file (most recent first)."""
    files_result = await db.execute(
        select(File).order_by(File.created_at.desc()).limit(limit)
    )
    files = files_result.scalars().all()
    if not files:
        return {"files": []}

    file_ids = [f.id for f in files]

    meta_result = await db.execute(
        select(FileMetadata).where(FileMetadata.file_id.in_(file_ids))
    )
    meta_map = {m.file_id: m for m in meta_result.scalars().all()}

    jobs_result = await db.execute(
        select(BackgroundJob).where(
            BackgroundJob.file_id.in_(file_ids),
            BackgroundJob.job_type == "parquet_conversion",
        )
    )
    jobs_map = {j.file_id: j for j in jobs_result.scalars().all()}

    rows = []
    for f in files:
        meta = meta_map.get(f.id)
        job = jobs_map.get(f.id)

        upload_secs = f.upload_duration_secs

        ingestion_secs = None
        if meta and meta.ingested_at and f.created_at:
            ingestion_secs = round((meta.ingested_at - f.created_at).total_seconds(), 1)

        parquet_secs = None
        if job and job.completed_at and job.started_at:
            parquet_secs = round((job.completed_at - job.started_at).total_seconds(), 1)

        # Processing = ingestion + parquet (complete server-side time)
        processing_secs = None
        if ingestion_secs is not None:
            processing_secs = ingestion_secs
            if parquet_secs is not None:
                processing_secs = round(processing_secs + parquet_secs, 1)

        # Total = upload + processing (end-to-end)
        total_secs = None
        if upload_secs is not None and processing_secs is not None:
            total_secs = round(upload_secs + processing_secs, 1)

        rows.append({
            "file_id": f.id,
            "name": f.name,
            "size": f.size,
            "ingest_status": f.ingest_status,
            "uploaded_at": f.created_at.isoformat() if f.created_at else None,
            "upload_secs": upload_secs,
            "ingested_at": meta.ingested_at.isoformat() if meta and meta.ingested_at else None,
            "ingestion_secs": ingestion_secs,
            "parquet_status": job.status if job else None,
            "parquet_secs": parquet_secs,
            "processing_secs": processing_secs,
            "total_secs": total_secs,
            "parquet_error": job.error_message if job else None,
        })

    return {"files": rows}


# ── Scoped audit log endpoints ───────────────────────────────────────────────

_DOMAIN_LOG_ROLES = {"developer", "manager"}


def _has_admin_log_scope(user: User) -> bool:
    return bool(user.is_admin or user.role == "admin")


def _has_domain_log_scope(user: User) -> bool:
    return user.role in _DOMAIN_LOG_ROLES


def _audit_scope_clause(user: User):
    """Return the row-level visibility predicate for the requesting user."""
    if _has_admin_log_scope(user):
        return None

    if _has_domain_log_scope(user):
        domains = list(user.allowed_domains or [])
        if not domains:
            return None
        return or_(
            AuditLog.actor_user_id == user.id,
            AuditLog.domain_tag.in_(domains),
            AuditLog.actor_allowed_domains.op("&&")(domains),
        )

    return AuditLog.actor_user_id == user.id


def _apply_audit_scope(stmt, user: User):
    scope = _audit_scope_clause(user)
    return stmt.where(scope) if scope is not None else stmt


def _audit_row(row: AuditLog) -> dict:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "event_type": row.event_type,
        "action": row.action,
        "actor": {
            "user_id": row.actor_user_id,
            "email": row.actor_email,
            "name": row.actor_name,
            "role": row.actor_role,
            "is_admin": row.actor_is_admin,
            "allowed_domains": row.actor_allowed_domains,
            "organization_id": row.actor_organization_id,
        },
        "request": {
            "method": row.method,
            "path": row.path,
            "route_template": row.route_template,
            "status_code": row.status_code,
            "duration_ms": row.duration_ms,
            "ip_address": row.ip_address,
            "user_agent": row.user_agent,
        },
        "context": {
            "domain_tag": row.domain_tag,
            "container_id": row.container_id,
            "file_id": row.file_id,
            "file_name": row.file_name,
            "folder_id": row.folder_id,
            "folder_name": row.folder_name,
            "target_user_id": row.target_user_id,
            "target_user_email": row.target_user_email,
            "target_user_name": row.target_user_name,
        },
        "details": row.details,
        "error": row.error,
    }


@router.get("/audit/users")
async def audit_users(
    q: str | None = Query(default=None, min_length=1, max_length=120),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return visible actors for the audit user/name filter."""
    stmt = select(
        AuditLog.actor_user_id,
        AuditLog.actor_email,
        AuditLog.actor_name,
        AuditLog.actor_role,
    ).where(AuditLog.actor_user_id.isnot(None)).distinct()
    stmt = _apply_audit_scope(stmt, current_user)

    if q:
        term = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(AuditLog.actor_email).like(term),
                func.lower(AuditLog.actor_name).like(term),
            )
        )

    rows = (await db.execute(stmt.order_by(AuditLog.actor_email).limit(limit))).all()
    return {
        "users": [
            {
                "user_id": row.actor_user_id,
                "email": row.actor_email,
                "name": row.actor_name,
                "role": row.actor_role,
            }
            for row in rows
        ]
    }


@router.get("/audit")
async def audit_log(
    lines: int = Query(default=100, ge=1, le=2000),
    user: str | None = Query(default=None, min_length=1, max_length=120),
    email: str | None = Query(default=None, min_length=1, max_length=320),
    name: str | None = Query(default=None, min_length=1, max_length=255),
    role: str | None = Query(default=None, min_length=1, max_length=40),
    domain: str | None = Query(default=None, min_length=1, max_length=120),
    action: str | None = Query(default=None, min_length=1, max_length=160),
    path: str | None = Query(default=None, min_length=1, max_length=240),
    status_code: int | None = Query(default=None, ge=100, le=599),
    event_type: str | None = Query(default=None, min_length=1, max_length=40),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return scoped structured audit logs.

    Visibility:
      * admin: every row
      * developer/manager with unrestricted domains: every row
      * developer/manager with allowed_domains: own rows + matching domains
      * regular user: own rows only
    """
    stmt = select(AuditLog)
    stmt = _apply_audit_scope(stmt, current_user)

    if user:
        term = f"%{user.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(AuditLog.actor_email).like(term),
                func.lower(AuditLog.actor_name).like(term),
            )
        )
    if email:
        stmt = stmt.where(func.lower(AuditLog.actor_email).like(f"%{email.lower()}%"))
    if name:
        stmt = stmt.where(func.lower(AuditLog.actor_name).like(f"%{name.lower()}%"))
    if role:
        stmt = stmt.where(func.lower(AuditLog.actor_role) == role.lower())
    if domain:
        stmt = stmt.where(func.lower(AuditLog.domain_tag) == domain.lower())
    if action:
        stmt = stmt.where(func.lower(AuditLog.action).like(f"%{action.lower()}%"))
    if path:
        stmt = stmt.where(func.lower(AuditLog.path).like(f"%{path.lower()}%"))
    if status_code is not None:
        stmt = stmt.where(AuditLog.status_code == status_code)
    if event_type:
        stmt = stmt.where(func.lower(AuditLog.event_type) == event_type.lower())

    rows = (await db.execute(stmt.order_by(AuditLog.created_at.desc()).limit(lines))).scalars().all()
    return {
        "scope": "admin" if _has_admin_log_scope(current_user) else "domain" if _has_domain_log_scope(current_user) else "self",
        "returned": len(rows),
        "lines": [_audit_row(row) for row in rows],
    }


@router.get("/{filename}")
async def tail_log(
    filename: str,
    lines: int = Query(default=100, ge=1, le=2000),
    _: User = Depends(require_admin),
) -> dict:
    """Return the last N lines of a log file (default 100, max 2000)."""
    path = _safe_log_path(filename)
    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = all_lines[-lines:]

    # Try to parse each line as JSON for structured output
    parsed = []
    for line in tail:
        try:
            parsed.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            parsed.append({"raw": line})

    return {"file": filename, "total_lines": len(all_lines), "returned": len(parsed), "lines": parsed}


@router.get("/{filename}/search")
async def search_log(
    filename: str,
    q: str = Query(..., min_length=1, max_length=200),
    lines: int = Query(default=50, ge=1, le=500),
    _: User = Depends(require_admin),
) -> dict:
    """Search a log file for lines containing query string (case-insensitive)."""
    path = _safe_log_path(filename)
    q_lower = q.lower()
    all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    matches = []
    for i, line in enumerate(all_lines):
        if q_lower in line.lower():
            try:
                matches.append({"line_num": i + 1, "data": json.loads(line)})
            except (json.JSONDecodeError, ValueError):
                matches.append({"line_num": i + 1, "data": {"raw": line}})
            if len(matches) >= lines:
                break

    return {"file": filename, "query": q, "matches": len(matches), "lines": matches}


@router.get("/{filename}/download")
async def download_log(
    filename: str,
    _: User = Depends(require_admin),
) -> FileResponse:
    """Download a raw log file as a plain-text attachment."""
    path = _safe_log_path(filename)
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{filename}", status_code=204)
async def clear_log(
    filename: str,
    _: User = Depends(require_admin),
) -> None:
    """Truncate a log file to zero bytes for fresh logging.

    Truncating (instead of deleting) preserves any open file handles held by
    structlog/the logging module so writes continue to land in the same file.
    """
    path = _safe_log_path(filename)
    # Open in write mode and immediately close — atomically truncates to 0 bytes
    with open(path, "w", encoding="utf-8"):
        pass
    return None

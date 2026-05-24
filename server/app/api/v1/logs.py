"""
Logs API — query server_logs table with RBAC-scoped access.

GET  /api/logs/audit                  → structured audit log (own/domain/all based on role)
GET  /api/logs/audit/users            → distinct actor list for the audit user filter
GET  /api/logs/ingest-events          → ingestion pipeline events
GET  /api/logs/ingest-timings         → per-file ingest stage timings
GET  /api/logs/file-timings           → upload + ingestion + parquet timing per file (DB)
GET  /api/logs/pipeline/tail?n=100    → last N pipeline events, pretty-formatted (admin only)
GET  /api/logs/pipeline/stream        → SSE live stream of pipeline events (admin only)
GET  /api/logs/{log_type}             → tail N rows for any log_type (RBAC scoped)
GET  /api/logs/{log_type}/search?q=   → full-text search inside a log_type (RBAC scoped)

RBAC:
  admin   → all rows, all log_types
  manager → all log_types, rows where domain_tag in allowed_domains OR actor is self
  member  → ai_pipeline rows where actor is self only
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import LOG_DIR, format_pipeline_line
from app.dependencies import get_current_user, require_admin
from app.models.background_job import BackgroundJob
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.server_log import LOG_TYPES, ServerLog
from app.models.user import User

router = APIRouter(prefix="/logs", tags=["logs"])

# Filename alias → log_type mapping for backward-compatible URL patterns.
# Only aliases that should remain accessible via the /{log_type} endpoint.
_FILENAME_TO_LOG_TYPE: dict[str, str] = {
    "ai_pipeline.log": "ai_pipeline",
    "llm_calls.log":   "llm",
    "costs.log":       "cost",
    "system.log":      "system",
    "audit.log":       "audit",
    "ingestion.log":   "ingestion",
}


# ── RBAC scope builder ────────────────────────────────────────────────────────

def _build_log_scope(user: User, stmt, log_types: list[str] | None = None):
    """
    Apply RBAC row-visibility rules to `stmt`, then optionally filter by log_types.

    admin   → no restriction
    manager → rows where domain_tag IN allowed_domains OR actor_user_id = self
    member  → only ai_pipeline rows where actor_user_id = self
    """
    if user.is_admin or user.role == "admin":
        if log_types:
            stmt = stmt.where(ServerLog.log_type.in_(log_types))
        return stmt

    if user.role == "manager":
        if log_types:
            stmt = stmt.where(ServerLog.log_type.in_(log_types))
        domains = list(user.allowed_domains or [])
        if domains:
            stmt = stmt.where(
                or_(
                    ServerLog.domain_tag.in_(domains),
                    ServerLog.actor_user_id == user.id,
                )
            )
        # manager with no domains = unrestricted scope (no extra WHERE)
        return stmt

    # member: ai_pipeline only, own rows only
    allowed = [t for t in (log_types or ["ai_pipeline"]) if t == "ai_pipeline"]
    if not allowed:
        # member asked for a tab outside their scope — force empty result
        stmt = stmt.where(text("false"))
        return stmt
    stmt = stmt.where(
        ServerLog.log_type.in_(allowed),
        ServerLog.actor_user_id == user.id,
    )
    return stmt


def _resolve_log_type(alias: str) -> str:
    """Map a filename alias (e.g. llm_calls.log) or bare log_type to a validated log_type."""
    resolved = _FILENAME_TO_LOG_TYPE.get(alias, alias)
    if resolved not in LOG_TYPES:
        raise HTTPException(status_code=404, detail=f"Unknown log type: {alias}")
    return resolved


# ── Row serialiser ────────────────────────────────────────────────────────────

def _sl_row(row: ServerLog) -> dict:
    """Serialise a ServerLog row, surfacing details fields at the top level."""
    details = row.details or {}
    created = row.created_at.isoformat() if row.created_at else None
    return {
        "id": row.id,
        "created_at": created,
        # `timestamp` alias so LogLine and IngestEventRow components work without changes
        "timestamp": created,
        "log_type": row.log_type,
        "event": row.event,
        "level": row.level,
        # duration_ms exposed at top level for all log types so LogLine can render it
        "duration_ms": row.duration_ms if row.duration_ms is not None else details.get("duration_ms"),
        # Actor
        "actor": {
            "user_id": row.actor_user_id,
            "email": row.actor_email,
            "role": row.actor_role,
        },
        "domain_tag": row.domain_tag,
        # Resource context
        "trace_id": row.trace_id,
        "file_id": row.file_id,
        "file_name": row.file_name,
        # HTTP context (populated for audit rows only)
        "request": {
            "method": row.method,
            "path": row.path,
            "status_code": row.status_code,
            "duration_ms": row.duration_ms,
            "ip_address": row.ip_address,
            # pulled from details JSONB (written by audit_log.py)
            "route_template": details.get("route_template"),
            "user_agent": details.get("user_agent"),
        } if row.log_type == "audit" else None,
        # Full JSONB payload — always present so callers can access any field
        "details": details,
    }


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


@router.get("/ingest-events")
async def ingest_events(
    lines: int = Query(default=300, ge=1, le=2000),
    trace_id: str | None = Query(default=None, max_length=36),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the last N ingestion pipeline events from server_logs."""
    stmt = (
        select(ServerLog)
        .where(ServerLog.log_type == "ingestion")
        .order_by(ServerLog.created_at.desc())
        .limit(lines)
    )
    if trace_id:
        stmt = stmt.where(ServerLog.trace_id == trace_id)

    rows = (await db.execute(stmt)).scalars().all()
    # Return chronological order (oldest first) for timeline readability
    rows_sorted = sorted(rows, key=lambda r: r.created_at)
    events = []
    for r in rows_sorted:
        ev = {
            "event": r.event,
            "level": r.level,
            "timestamp": r.created_at.isoformat() if r.created_at else None,
            "trace_id": r.trace_id,
            "file_id": r.file_id,
            "filename": r.file_name,
            "domain_tag": r.domain_tag,
        }
        ev.update(r.details or {})
        events.append(ev)

    return {"total": len(events), "returned": len(events), "lines": events}


@router.get("/ai-pipeline-events")
async def ai_pipeline_events(
    lines: int = Query(default=300, ge=1, le=2000),
    trace_id: str | None = Query(default=None, max_length=36),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the last N AI pipeline query events from server_logs, details flattened.

    Mirrors the shape of ingest-events so the PipelineEventRow component can
    access fields like ev.query, ev.sql, ev.answer directly at the top level.
    """
    stmt = (
        select(ServerLog)
        .where(ServerLog.log_type == "ai_pipeline")
        .order_by(ServerLog.created_at.desc())
        .limit(lines)
    )
    if trace_id:
        stmt = stmt.where(ServerLog.trace_id == trace_id)

    rows = (await db.execute(stmt)).scalars().all()
    # Chronological order for timeline readability
    rows_sorted = sorted(rows, key=lambda r: r.created_at)
    events = []
    for r in rows_sorted:
        ev = {
            "event": r.event,
            "level": r.level,
            "timestamp": r.created_at.isoformat() if r.created_at else None,
            "duration_ms": r.duration_ms,
            "trace_id": r.trace_id,
            "file_id": r.file_id,
            "file_name": r.file_name,
            "domain_tag": r.domain_tag,
            "actor_user_id": r.actor_user_id,
            "actor_email": r.actor_email,
            "actor_role": r.actor_role,
        }
        # Flatten details so PipelineEventRow can access query, sql, answer etc. directly
        ev.update(r.details or {})
        events.append(ev)

    return {"total": len(events), "returned": len(events), "lines": events}


@router.get("/ingest-timings")
async def ingest_timings(
    last_n_files: int = Query(default=20, ge=1, le=100),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Per-file, per-step timing breakdown from ingestion events in server_logs."""
    from collections import defaultdict

    stmt = (
        select(ServerLog)
        .where(ServerLog.log_type == "ingestion")
        .order_by(ServerLog.created_at.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()

    # Group by trace_id
    traces: dict[str, list[ServerLog]] = defaultdict(list)
    for row in rows:
        tid = row.trace_id or row.file_id
        if tid:
            traces[tid].append(row)

    files = []
    for tid, events in traces.items():
        details_list = [(r, r.details or {}) for r in events]

        def _ev(r, d): return d.get("event") or r.event

        start_row = next((r for r, d in details_list if _ev(r, d) == "chain_start"), None) or events[0]
        end_row = next((r for r, d in details_list if _ev(r, d) == "chain_end"), None)
        if end_row is None:
            end_row = next(
                (r for r, d in reversed(details_list)
                 if _ev(r, d) == "ingest_stage"
                 and d.get("stage") == "complete"
                 and d.get("status") == "done"),
                None,
            )
        filename_row = next((r for r in events if r.file_name), None)

        steps = []
        for r, d in details_list:
            ev_name = _ev(r, d)
            if ev_name == "step":
                entry = {
                    "step": d.get("step"),
                    "name": d.get("name"),
                    "status": d.get("status"),
                    "duration_ms": r.duration_ms or d.get("duration_ms"),
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                }
            elif ev_name == "ingest_stage":
                entry = {
                    "step": d.get("stage"),
                    "name": d.get("stage"),
                    "status": d.get("status"),
                    "duration_ms": r.duration_ms or d.get("duration_ms"),
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                }
            else:
                continue
            for k in ("encoding", "safe_for_raw_sample", "reason", "original_rows",
                      "clean_rows", "clean_blob_path", "error"):
                if k in d:
                    entry[k] = d[k]
            steps.append(entry)

        probe_row = next(
            (r for r, d in details_list if _ev(r, d) == "step" and d.get("name") == "probe"),
            None,
        )
        probe_d = probe_row.details or {} if probe_row else None

        files.append({
            "file_id": tid,
            "filename": (filename_row.file_name if filename_row else None),
            "status": "done" if end_row else "in_progress",
            "total_ms": (end_row.details or {}).get("duration_ms") if end_row else None,
            "started_at": start_row.created_at.isoformat() if start_row.created_at else None,
            "probe": {
                "safe_for_raw_sample": probe_d.get("safe_for_raw_sample"),
                "encoding": probe_d.get("encoding"),
                "reason": probe_d.get("reason"),
                "duration_ms": probe_row.duration_ms or probe_d.get("duration_ms"),
            } if probe_d is not None else None,
            "steps": steps,
        })

    files.sort(key=lambda f: f.get("started_at") or "", reverse=True)
    return {"files": files[:last_n_files]}



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
        ).order_by(BackgroundJob.started_at.desc())
    )
    jobs_map: dict[str, BackgroundJob] = {}
    for job in jobs_result.scalars().all():
        jobs_map.setdefault(job.file_id, job)

    rows = []
    for f in files:
        meta = meta_map.get(f.id)
        job = jobs_map.get(f.id)
        has_core_ingest = bool(meta and meta.ingested_at) or f.ingest_status == "ingested"
        visible_parquet_status = job.status if job and has_core_ingest else None

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
            "parquet_status": visible_parquet_status,
            "parquet_secs": parquet_secs if has_core_ingest else None,
            "processing_secs": processing_secs,
            "total_secs": total_secs,
            "parquet_error": job.error_message if job and has_core_ingest else None,
        })

    return {"files": rows}


# ── Audit log endpoints ───────────────────────────────────────────────────────

@router.get("/audit/users")
async def audit_users(
    q: str | None = Query(default=None, min_length=1, max_length=120),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return visible distinct actors from server_logs for the audit user filter."""
    stmt = (
        select(
            ServerLog.actor_user_id,
            ServerLog.actor_email,
            ServerLog.actor_role,
        )
        .where(
            ServerLog.log_type == "audit",
            ServerLog.actor_user_id.isnot(None),
        )
        .distinct()
    )
    stmt = _build_log_scope(current_user, stmt, log_types=["audit"])

    if q:
        term = f"%{q.lower()}%"
        stmt = stmt.where(func.lower(ServerLog.actor_email).like(term))

    rows = (await db.execute(stmt.order_by(ServerLog.actor_email).limit(limit))).all()
    return {
        "users": [
            {"user_id": r.actor_user_id, "email": r.actor_email, "role": r.actor_role}
            for r in rows
        ]
    }


@router.get("/audit")
async def audit_log(
    lines: int = Query(default=100, ge=1, le=2000),
    email: str | None = Query(default=None, min_length=1, max_length=320),
    role: str | None = Query(default=None, min_length=1, max_length=40),
    domain: str | None = Query(default=None, min_length=1, max_length=120),
    event: str | None = Query(default=None, min_length=1, max_length=160),
    path_filter: str | None = Query(default=None, alias="path", min_length=1, max_length=240),
    status_code: int | None = Query(default=None, ge=100, le=599),
    level: str | None = Query(default=None, min_length=1, max_length=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return RBAC-scoped audit rows from server_logs."""
    stmt = select(ServerLog).where(ServerLog.log_type == "audit")
    stmt = _build_log_scope(current_user, stmt, log_types=["audit"])

    if email:
        stmt = stmt.where(func.lower(ServerLog.actor_email).like(f"%{email.lower()}%"))
    if role:
        stmt = stmt.where(func.lower(ServerLog.actor_role) == role.lower())
    if domain:
        stmt = stmt.where(func.lower(ServerLog.domain_tag) == domain.lower())
    if event:
        stmt = stmt.where(func.lower(ServerLog.event).like(f"%{event.lower()}%"))
    if path_filter:
        stmt = stmt.where(func.lower(ServerLog.path).like(f"%{path_filter.lower()}%"))
    if status_code is not None:
        stmt = stmt.where(ServerLog.status_code == status_code)
    if level:
        stmt = stmt.where(func.lower(ServerLog.level) == level.lower())

    rows = (await db.execute(stmt.order_by(ServerLog.created_at.desc()).limit(lines))).scalars().all()

    scope = "admin" if (current_user.is_admin or current_user.role == "admin") \
        else "domain" if current_user.role == "manager" \
        else "self"

    return {"scope": scope, "returned": len(rows), "lines": [_sl_row(r) for r in rows]}


# ── Generic DB log query endpoints ────────────────────────────────────────────
# These replace the old file-based /{filename} and /{filename}/search endpoints.
# The {log_type} path param accepts both bare log types (e.g. "llm") and legacy
# filename aliases (e.g. "llm_calls.log") for frontend backward compatibility.

@router.get("/{log_type}/search")
async def search_log(
    log_type: str,
    q: str = Query(..., min_length=1, max_length=200),
    lines: int = Query(default=50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Full-text search across event name and details JSONB for a given log_type."""
    resolved = _resolve_log_type(log_type)
    q_lower = q.lower()

    stmt = select(ServerLog).where(
        ServerLog.log_type == resolved,
        or_(
            func.lower(ServerLog.event).like(f"%{q_lower}%"),
            # JSONB → text cast so the search covers any nested field value
            text(
                "lower(server_logs.details::text) like :q"
            ).bindparams(q=f"%{q_lower}%"),
        ),
    )
    stmt = _build_log_scope(current_user, stmt, log_types=[resolved])

    rows = (await db.execute(stmt.order_by(ServerLog.created_at.desc()).limit(lines))).scalars().all()
    return {
        "log_type": resolved,
        "query": q,
        "returned": len(rows),
        "matches": len(rows),
        "lines": [_sl_row(r) for r in rows],
    }


@router.get("/{log_type}")
async def tail_log(
    log_type: str,
    lines: int = Query(default=100, ge=1, le=2000),
    level: str | None = Query(default=None, max_length=20),
    domain: str | None = Query(default=None, max_length=120),
    trace_id: str | None = Query(default=None, max_length=36),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the last N rows for the given log_type, RBAC scoped."""
    resolved = _resolve_log_type(log_type)

    stmt = select(ServerLog)
    stmt = _build_log_scope(current_user, stmt, log_types=[resolved])

    if level:
        stmt = stmt.where(func.lower(ServerLog.level) == level.lower())
    if domain:
        stmt = stmt.where(func.lower(ServerLog.domain_tag) == domain.lower())
    if trace_id:
        stmt = stmt.where(ServerLog.trace_id == trace_id)

    rows = (await db.execute(stmt.order_by(ServerLog.created_at.desc()).limit(lines))).scalars().all()
    return {
        "log_type": resolved,
        "returned": len(rows),
        "lines": [_sl_row(r) for r in rows],
    }


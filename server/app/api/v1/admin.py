"""
Admin API — internal endpoints for monitoring and cost tracking.

GET  /api/admin/cost-summary
POST /api/admin/reingest-all
GET  /api/admin/domains
PATCH /api/admin/users/{user_id}/domains
PATCH /api/admin/folders/{folder_id}/domain
GET  /api/admin/files/eligible
GET  /api/admin/departments/{domain_name}/files
POST /api/admin/departments/{domain_name}/ai-assign
POST /api/admin/departments/{domain_name}/assign
DELETE /api/admin/departments/{domain_name}/files/{file_id}
GET  /api/admin/missing-parquet
POST /api/admin/retry-parquet
"""
import asyncio
import re
import uuid
from collections.abc import Sequence

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cost_tracker import get_session_summary
from app.core.config import get_settings
from app.core.database import async_session, get_db
from app.core.logger import ingest_logger
from app.dependencies import get_current_user, require_admin
from app.agent.graph.graph import invalidate_catalog_cache
from app.services.audit_log import record_audit_event_safe
from app.services.ingestion_config import IngestStatus, is_parquet_source_file, is_supported_ingest_file
from app.models.background_job import BackgroundJob
from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.models.folder import Folder
from app.models.user import User
from app.worker.celery_app import celery_app
from app.worker.ingest_tasks import run_ingest_pipeline

router = APIRouter(prefix="/admin", tags=["admin"])


def _reingest_batch_size() -> int:
    return max(1, int(get_settings().REINGEST_BATCH_SIZE))


def _reingest_batch_delay_seconds() -> int:
    return max(0, int(get_settings().REINGEST_BATCH_DELAY_SECONDS))


_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _safe_celery_error(exc: BaseException) -> str:
    message = str(exc) or exc.__class__.__name__
    message = re.sub(r"(redis|rediss)://[^@\s]+@", r"\1://***@", message)
    return message[:500]


def _assert_ingestion_queue_available() -> None:
    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1, timeout=3)

        backend_client = getattr(celery_app.backend, "client", None)
        if backend_client is not None:
            backend_client.ping()
    except Exception as exc:
        error = _safe_celery_error(exc)
        ingest_logger.error("ingestion_queue_unavailable", error=error)
        raise HTTPException(
            status_code=503,
            detail=(
                "Ingestion queue is unavailable. Redis/Celery is down or stale; "
                "restart redis-server, gchat, and gchat-celery, then retry."
            ),
        ) from exc


def _inspect_ingestion_runtime() -> dict:
    settings = get_settings()
    queue_name = settings.INGEST_NORMAL_QUEUE
    health: dict = {
        "broker_ok": False,
        "result_backend_ok": False,
        "workers": {},
        "queue": {"name": queue_name},
        "errors": {},
    }

    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1, timeout=3)
        health["broker_ok"] = True
    except Exception as exc:
        health["errors"]["broker"] = _safe_celery_error(exc)

    try:
        backend_client = getattr(celery_app.backend, "client", None)
        if backend_client is not None:
            backend_client.ping()
        health["result_backend_ok"] = True
    except Exception as exc:
        health["errors"]["result_backend"] = _safe_celery_error(exc)

    try:
        inspector = celery_app.control.inspect(timeout=3)
        health["workers"] = {
            "ping": inspector.ping() or {},
            "active": inspector.active() or {},
            "reserved": inspector.reserved() or {},
            "scheduled": inspector.scheduled() or {},
        }
    except Exception as exc:
        health["errors"]["inspect"] = _safe_celery_error(exc)

    try:
        import redis  # noqa: PLC0415

        broker = redis.Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        broker.ping()
        health["queue"]["length"] = broker.llen(queue_name)
    except Exception as exc:
        health["errors"]["queue_length"] = _safe_celery_error(exc)

    return health


def _active_ingest_file_ids() -> set[str]:
    """Best-effort extraction of file IDs from Celery active/reserved/scheduled tasks."""
    try:
        inspector = celery_app.control.inspect(timeout=3)
        task_groups = [
            inspector.active() or {},
            inspector.reserved() or {},
            inspector.scheduled() or {},
        ]
    except Exception as exc:
        ingest_logger.warning("celery_active_file_inspect_failed", error=_safe_celery_error(exc))
        return set()

    file_ids: set[str] = set()
    for tasks_by_worker in task_groups:
        for tasks in tasks_by_worker.values():
            for task in tasks or []:
                request = task.get("request", task) if isinstance(task, dict) else task
                file_ids.update(_UUID_PATTERN.findall(repr(request)))
    return file_ids


async def _mark_dispatch_failed(db: AsyncSession, file_ids: Sequence[str], error: str) -> None:
    if not file_ids:
        return

    await db.execute(
        update(File)
        .where(File.id.in_(file_ids))
        .values(ingest_status=IngestStatus.FAILED.value)
    )

    result = await db.execute(select(File).where(File.id.in_(file_ids)))
    files_by_id = {file.id: file for file in result.scalars().all()}
    result = await db.execute(select(FileMetadata).where(FileMetadata.file_id.in_(file_ids)))
    metadata_by_file_id = {metadata.file_id: metadata for metadata in result.scalars().all()}
    ingest_error = f"Ingestion dispatch failed before worker start: {error}"[:1000]

    for file_id in file_ids:
        file = files_by_id.get(file_id)
        metadata = metadata_by_file_id.get(file_id)
        if metadata:
            metadata.ingest_error = ingest_error
        elif file:
            db.add(FileMetadata(
                id=str(uuid.uuid4()),
                file_id=file_id,
                blob_path=file.blob_path,
                container_id=file.container_id,
                ingest_error=ingest_error,
            ))

    await db.commit()


@router.get("/ingestion-health")
async def ingestion_health(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    runtime = await asyncio.to_thread(_inspect_ingestion_runtime)

    status_rows = await db.execute(
        select(File.ingest_status, func.count()).group_by(File.ingest_status)
    )
    status_counts = {status or "unknown": count for status, count in status_rows.all()}

    stuck_rows = await db.execute(
        select(File.id, File.name, File.ingest_status, File.created_at)
        .where(File.ingest_status.in_([IngestStatus.PENDING.value, IngestStatus.RUNNING.value, IngestStatus.FAILED.value]))
        .order_by(File.created_at.desc())
        .limit(50)
    )
    files = [
        {
            "id": file_id,
            "name": name,
            "ingest_status": status,
            "created_at": created_at.isoformat() if created_at else None,
        }
        for file_id, name, status, created_at in stuck_rows.all()
    ]

    return {
        "runtime": runtime,
        "file_status_counts": status_counts,
        "attention_files": files,
    }


@router.get("/cost-summary")
async def cost_summary(current_user: User = Depends(get_current_user)) -> dict:
    return get_session_summary()


# ── Re-ingest all files ──────────────────────────────────────────────────────


@router.post("/reingest-all")
async def reingest_all(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Wipe all metadata / analytics / relationships, reset every
    CSV/TXT/TSV file to not_ingested, and re-run the full pipeline.

    Concurrency safety:
      * Files already in an active state ('pending' / 'running') are NOT
        reset and NOT re-queued — another reingest already owns them.
      * The remaining files are reset and queued; each one is then
        atomically claimed inside `_batch_reingest`, so even if this
        endpoint is hit twice in quick succession, no file is processed
        twice in parallel.
    """
    # Find all ingestable files
    result = await db.execute(select(File))
    all_files = list(result.scalars().all())
    ingestable = [f for f in all_files if is_supported_ingest_file(f.name)]
    if not ingestable:
        raise HTTPException(status_code=400, detail="No ingestable files found.")

    _assert_ingestion_queue_available()

    # Exclude only files that are truly active in Celery. Stale RUNNING rows
    # from a dead worker must be reset/requeued or they look stuck forever.
    active_file_ids = await asyncio.to_thread(_active_ingest_file_ids)
    in_flight = [
        f.id
        for f in ingestable
        if f.ingest_status == IngestStatus.RUNNING.value and f.id in active_file_ids
    ]
    in_flight_ids = set(in_flight)
    stale_running = [
        f.id
        for f in ingestable
        if f.ingest_status == IngestStatus.RUNNING.value and f.id not in active_file_ids
    ]
    file_ids = [f.id for f in ingestable if f.id not in in_flight_ids]
    if in_flight:
        ingest_logger.info(
            "reingest_all_skipping_in_flight",
            in_flight_count=len(in_flight),
        )
    if stale_running:
        ingest_logger.warning(
            "reingest_all_requeueing_stale_running",
            stale_running_count=len(stale_running),
        )
    if not file_ids:
        return {
            "message": "All ingestable files are already being processed.",
            "file_count": 0,
            "in_flight_count": len(in_flight),
        }

    # Delete old metadata, analytics, relationships ONLY for files we will reingest.
    # Leaving in-flight files' rows alone prevents their running pipeline from
    # crashing on a missing FK target.
    await db.execute(delete(FileRelationship).where(FileRelationship.file_a_id.in_(file_ids)))
    await db.execute(delete(FileRelationship).where(FileRelationship.file_b_id.in_(file_ids)))
    await db.execute(delete(FileAnalytics).where(FileAnalytics.file_id.in_(file_ids)))
    await db.execute(delete(FileMetadata).where(FileMetadata.file_id.in_(file_ids)))
    await db.execute(
        delete(BackgroundJob).where(
            BackgroundJob.file_id.in_(file_ids),
            BackgroundJob.job_type == "parquet_conversion",
        )
    )

    # Reset ingest status AND preprocessed flag so the preprocessor actually runs again.
    # Without resetting is_preprocessed, ingest_file skips preprocessing entirely
    # and re-reads the already-preprocessed (possibly corrupted) CSV blob as-is.
    await db.execute(
        update(File)
        .where(File.id.in_(file_ids))
        .values(ingest_status=IngestStatus.PENDING.value, is_preprocessed=False)
    )
    await db.commit()
    invalidate_catalog_cache()

    # Dispatch each file to Celery in small waves. This keeps the API responsive
    # on small VMs when someone clicks "ingest all" for many large files.
    batch_size = _reingest_batch_size()
    batch_delay_seconds = _reingest_batch_delay_seconds()
    task_ids = []
    schedule = []
    try:
        for idx, fid in enumerate(file_ids):
            countdown = (idx // batch_size) * batch_delay_seconds
            task = run_ingest_pipeline.apply_async(args=[fid], countdown=countdown, retry=False)
            task_ids.append(task.id)
            schedule.append({"file_id": fid, "task_id": task.id, "countdown": countdown})
    except Exception as exc:
        error = _safe_celery_error(exc)
        unscheduled_file_ids = file_ids[len(task_ids):]
        await _mark_dispatch_failed(db, unscheduled_file_ids, error)
        ingest_logger.error(
            "reingest_all_dispatch_failed",
            error=error,
            queued_count=len(task_ids),
            failed_count=len(unscheduled_file_ids),
        )
        raise HTTPException(
            status_code=503,
            detail="Ingestion queue unavailable while scheduling files. Restart Redis/Celery/API and retry.",
        ) from exc

    ingest_logger.info(
        "reingest_all_queued",
        admin_id=admin.id,
        admin_email=admin.email,
        admin_name=admin.name,
        file_count=len(file_ids),
        in_flight_count=len(in_flight),
        stale_running_count=len(stale_running),
        batch_size=batch_size,
        batch_delay_seconds=batch_delay_seconds,
        task_ids=task_ids,
        backend="celery",
    )

    try:
        await record_audit_event_safe(
            actor=admin,
            event_type="action",
            action="admin.reingest_all",
            status_code=200,
            path="/api/admin/reingest-all",
            route_template="/api/admin/reingest-all",
            details={
                "file_count": len(file_ids),
                "in_flight_count": len(in_flight),
                "stale_running_count": len(stale_running),
                "batch_size": batch_size,
                "batch_delay_seconds": batch_delay_seconds,
                "scheduled_tasks": schedule,
            },
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        ingest_logger.warning("reingest_all_audit_failed", error=str(exc)[:300])

    return {
        "message": "Re-ingestion queued",
        "file_count": len(file_ids),
        "in_flight_count": len(in_flight),
        "stale_running_count": len(stale_running),
        "batch_size": batch_size,
        "batch_delay_seconds": batch_delay_seconds,
        "task_ids": task_ids,
    }


# ── Domain access control ────────────────────────────────────────────────────

class _UserDomainsBody(BaseModel):
    allowed_domains: list[str] | None  # None = unrestricted


class _FolderDomainBody(BaseModel):
    domain_tag: str | None  # None = untagged (public)


class _CreateDomainBody(BaseModel):
    name: str


@router.get("/domains")
async def list_domains(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return all distinct non-null domain tags currently in use on folders."""
    rows = (await db.execute(
        select(Folder.domain_tag).where(Folder.domain_tag.isnot(None)).distinct()
    )).scalars().all()
    return {"domains": sorted(rows)}


@router.post("/domains")
async def create_domain(
    body: _CreateDomainBody,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Create a new domain by registering a top-level system folder tagged with it.
    The folder becomes the home for files belonging to that domain.
    """
    import uuid as _uuid
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")

    # Check for existing domain folder with same tag
    existing = (await db.execute(
        select(Folder).where(Folder.domain_tag == name).limit(1)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail=f"Domain '{name}' already exists")

    folder = Folder(
        id=str(_uuid.uuid4()),
        name=name,
        owner_id=admin.id,
        parent_id=None,
        container_id=None,
        domain_tag=name,
    )
    db.add(folder)
    await db.commit()
    # Invalidate catalog so this new domain folder is reflected in chat scope
    # immediately (default 5-min TTL would otherwise hide it from users).
    invalidate_catalog_cache()
    try:
        await record_audit_event_safe(
            actor=admin,
            action="admin.create_domain",
            event_type="action",
            status_code=200,
            path="/api/admin/domains",
            route_template="/api/admin/domains",
            domain_tag=name,
            folder_id=folder.id,
            folder_name=folder.name,
            details={"domain": name},
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        ingest_logger.warning("create_domain_audit_failed", domain=name, error=str(exc)[:300])
    return {"domain": name, "folder_id": folder.id}


@router.patch("/users/{user_id}/domains")
async def set_user_domains(
    user_id: str,
    body: _UserDomainsBody,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Set a user's allowed_domains list.
    Pass null to remove restrictions (unrestricted access).
    """
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Normalise: empty list → None (unrestricted)
    old_domains = user.allowed_domains
    domains = body.allowed_domains if body.allowed_domains else None
    await db.execute(
        update(User).where(User.id == user_id).values(allowed_domains=domains)
    )
    await db.commit()
    # Invalidate catalog so the next chat request rebuilds visibility from scratch.
    # Without this, the user keeps seeing the previous (broader) catalog for up
    # to the cache TTL window.
    invalidate_catalog_cache()
    try:
        await record_audit_event_safe(
            actor=admin,
            action="admin.set_user_domains",
            event_type="action",
            status_code=200,
            path=f"/api/admin/users/{user_id}/domains",
            route_template="/api/admin/users/{user_id}/domains",
            target_user_id=user.id,
            target_user_email=user.email,
            target_user_name=user.name,
            details={"old_allowed_domains": old_domains, "new_allowed_domains": domains},
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        ingest_logger.warning("set_user_domains_audit_failed", user_id=user_id, error=str(exc)[:300])
    return {"user_id": user_id, "allowed_domains": domains}


@router.patch("/folders/{folder_id}/domain")
async def set_folder_domain(
    folder_id: str,
    body: _FolderDomainBody,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Tag a folder with a domain string (e.g. "finance", "hr").
    Pass null to remove the tag (folder becomes public).
    """
    folder = await db.get(Folder, folder_id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    old_domain = folder.domain_tag
    await db.execute(
        update(Folder).where(Folder.id == folder_id).values(domain_tag=body.domain_tag or None)
    )
    await db.commit()
    # Invalidate catalog so the folder's new domain_tag is reflected in chat
    # scope immediately. Without this, files in the folder remain visible to
    # users outside the new domain for up to 5 minutes (the cache TTL).
    invalidate_catalog_cache()
    try:
        await record_audit_event_safe(
            actor=admin,
            action="admin.set_folder_domain",
            event_type="action",
            status_code=200,
            path=f"/api/admin/folders/{folder_id}/domain",
            route_template="/api/admin/folders/{folder_id}/domain",
            domain_tag=body.domain_tag or None,
            folder_id=folder.id,
            folder_name=folder.name,
            details={"old_domain_tag": old_domain, "new_domain_tag": body.domain_tag or None},
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        ingest_logger.warning("set_folder_domain_audit_failed", folder_id=folder_id, error=str(exc)[:300])
    return {"folder_id": folder_id, "domain_tag": body.domain_tag or None}


# ── Department ↔ File assignment ─────────────────────────────────────────────

class _BulkAssignBody(BaseModel):
    file_ids: list[str]


def _score_file_for_domain(good_for: list, ai_description: str | None,
                            key_metrics: list, key_dimensions: list,
                            file_name: str, domain_name: str) -> float:
    """
    Keyword-score a file's metadata against a domain name.
    Returns a float ≥ 0. Files with score ≥ 1.0 are considered a match.
    """
    domain_words = set(re.split(r"[\s\-_/,]+", domain_name.lower())) - {
        "", "and", "the", "of", "for", "a", "an", "in", "to"
    }
    if not domain_words:
        return 0.0

    score = 0.0
    desc_lower = (ai_description or "").lower()
    name_lower = file_name.lower()

    for word in domain_words:
        # good_for describes use cases → highest weight
        for item in (good_for or []):
            if word in str(item).lower():
                score += 2.0

        # ai_description
        if word in desc_lower:
            score += 1.0

        # key_metrics
        for m in (key_metrics or []):
            if word in str(m).lower():
                score += 1.5

        # key_dimensions
        for d in (key_dimensions or []):
            if word in str(d).lower():
                score += 1.0

        # file name itself
        if word in name_lower:
            score += 0.5

    return score


@router.get("/files/eligible")
async def list_eligible_files(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Return all files with their AI metadata — used by the manual file picker
    and the AI-sort preview in the Profile → Domains tab.
    """
    rows = (await db.execute(
        select(File, FileMetadata, Folder)
        .join(FileMetadata, FileMetadata.file_id == File.id, isouter=True)
        .join(Folder, Folder.id == File.folder_id, isouter=True)
        .order_by(File.name)
    )).all()

    files = []
    for file, meta, folder in rows:
        files.append({
            "file_id": file.id,
            "name": file.name,
            "folder_id": file.folder_id,
            "current_domain": folder.domain_tag if folder else None,
            "ai_description": meta.ai_description if meta else None,
            "good_for": meta.good_for if meta else [],
            "key_metrics": meta.key_metrics if meta else [],
            "key_dimensions": meta.key_dimensions if meta else [],
            "ingest_status": file.ingest_status,
        })

    return {"files": files, "count": len(files)}


@router.get("/departments/{domain_name}/files")
async def list_department_files(
    domain_name: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return files currently assigned to a department's folder."""
    domain_folder = (await db.execute(
        select(Folder).where(Folder.domain_tag == domain_name).limit(1)
    )).scalar_one_or_none()
    if not domain_folder:
        raise HTTPException(status_code=404, detail=f"Department '{domain_name}' not found")

    files = (await db.execute(
        select(File).where(File.folder_id == domain_folder.id).order_by(File.name)
    )).scalars().all()

    return {
        "domain": domain_name,
        "folder_id": domain_folder.id,
        "files": [
            {"file_id": f.id, "name": f.name, "ingest_status": f.ingest_status}
            for f in files
        ],
        "count": len(files),
    }


@router.post("/departments/{domain_name}/ai-assign")
async def ai_assign_files_to_department(
    domain_name: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    AI keyword-match: reads each file's good_for, ai_description, key_metrics
    and assigns files where score ≥ 1.0 to this department's folder.
    Only touches files that are not already in a domain-tagged folder.
    """
    domain_folder = (await db.execute(
        select(Folder).where(Folder.domain_tag == domain_name).limit(1)
    )).scalar_one_or_none()
    if not domain_folder:
        raise HTTPException(status_code=404, detail=f"Department '{domain_name}' not found")

    rows = (await db.execute(
        select(File, FileMetadata, Folder)
        .join(FileMetadata, FileMetadata.file_id == File.id, isouter=True)
        .join(Folder, Folder.id == File.folder_id, isouter=True)
    )).all()

    assigned_ids: list[str] = []
    assigned_names: list[str] = []

    for file, meta, existing_folder in rows:
        # Skip files already pinned to a domain (respect existing assignments)
        if existing_folder and existing_folder.domain_tag:
            continue
        # Skip files without metadata (not yet ingested)
        if not meta:
            continue

        score = _score_file_for_domain(
            good_for=meta.good_for or [],
            ai_description=meta.ai_description,
            key_metrics=meta.key_metrics or [],
            key_dimensions=meta.key_dimensions or [],
            file_name=file.name,
            domain_name=domain_name,
        )

        if score >= 1.0:
            assigned_ids.append(file.id)
            assigned_names.append(file.name)

    if assigned_ids:
        await db.execute(
            update(File)
            .where(File.id.in_(assigned_ids))
            .values(folder_id=domain_folder.id)
        )
        await db.commit()
        invalidate_catalog_cache()

    return {
        "domain": domain_name,
        "assigned_count": len(assigned_ids),
        "assigned_files": assigned_names,
    }


@router.post("/departments/{domain_name}/assign")
async def manually_assign_files_to_department(
    domain_name: str,
    body: _BulkAssignBody,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Manually assign a list of file IDs to a department's folder."""
    if not body.file_ids:
        raise HTTPException(status_code=400, detail="No file_ids provided")

    domain_folder = (await db.execute(
        select(Folder).where(Folder.domain_tag == domain_name).limit(1)
    )).scalar_one_or_none()
    if not domain_folder:
        raise HTTPException(status_code=404, detail=f"Department '{domain_name}' not found")

    await db.execute(
        update(File)
        .where(File.id.in_(body.file_ids))
        .values(folder_id=domain_folder.id)
    )
    await db.commit()
    invalidate_catalog_cache()

    return {"domain": domain_name, "assigned_count": len(body.file_ids)}


@router.delete("/departments/{domain_name}/files/{file_id}")
async def remove_file_from_department(
    domain_name: str,
    file_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Remove a file from a department by unsetting its folder assignment."""
    domain_folder = (await db.execute(
        select(Folder).where(Folder.domain_tag == domain_name).limit(1)
    )).scalar_one_or_none()
    if not domain_folder:
        raise HTTPException(status_code=404, detail=f"Department '{domain_name}' not found")

    file = await db.get(File, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    if file.folder_id != domain_folder.id:
        raise HTTPException(status_code=400, detail="File is not in this department")

    await db.execute(
        update(File).where(File.id == file_id).values(folder_id=None)
    )
    await db.commit()
    invalidate_catalog_cache()

    return {"file_id": file_id, "domain": domain_name, "unassigned": True}


# ── Parquet status / retry ────────────────────────────────────────────────────

@router.get("/missing-parquet")
async def list_missing_parquet(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Return all CSV/TSV files that have been ingested but whose Parquet
    conversion record is missing or incomplete.  Includes the latest
    BackgroundJob status and error message so the UI can show why it failed.
    """
    rows = (await db.execute(
        select(File, FileAnalytics)
        .join(FileAnalytics, FileAnalytics.file_id == File.id, isouter=True)
        .where(File.ingest_status == IngestStatus.INGESTED.value)
    )).all()

    # Collect file_ids that are missing parquet
    candidates = []
    for file, analytics in rows:
        if not is_parquet_source_file(file.blob_path or file.name):
            continue
        if analytics is None or not analytics.parquet_blob_path:
            candidates.append((file, analytics))

    if not candidates:
        return {"files": [], "count": 0}

    # Fetch latest BackgroundJob per file in one query
    candidate_ids = [f.id for f, _ in candidates]
    job_rows = (await db.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.file_id.in_(candidate_ids),
            BackgroundJob.job_type == "parquet_conversion",
        )
        .order_by(BackgroundJob.started_at.desc())
    )).scalars().all()
    # Keep only the most recent job per file
    latest_job: dict[str, BackgroundJob] = {}
    for job in job_rows:
        if job.file_id not in latest_job:
            latest_job[job.file_id] = job

    missing = []
    for file, analytics in candidates:
        job = latest_job.get(file.id)
        missing.append({
            "file_id": file.id,
            "name": file.name,
            "blob_path": file.blob_path,
            "has_analytics": analytics is not None,
            "job_status": job.status if job else None,
            "job_error": job.error_message if job else None,
            "last_attempt": job.started_at.isoformat() if job and job.started_at else None,
        })

    return {"files": missing, "count": len(missing)}


@router.post("/retry-parquet")
async def retry_missing_parquet(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Retry parquet conversion for any ingested CSV/text file that is missing its
    parquet_blob_path. Calls trigger_parquet_conversion() directly — does NOT
    re-run AI description, embedding, or metadata.

    With the in-place overwrite design, the blob_path always points to the single
    source-of-truth clean CSV — there are no orphaned blob paths to recover.
    """
    from app.services.analytics_service import trigger_parquet_conversion
    from app.models.container import ContainerConfig

    # ── Skip files that already have a parquet job currently running ────────
    # Prevents double-clicking Retry All from launching duplicate work.
    running_jobs = (await db.execute(
        select(BackgroundJob.file_id).where(
            BackgroundJob.job_type == "parquet_conversion",
            BackgroundJob.status == "running",
        )
    )).scalars().all()
    in_flight: set[str] = set(running_jobs)

    # ── Ingested CSVs missing parquet → parquet only ─────────────────────────
    no_parquet_rows = (await db.execute(
        select(File, FileAnalytics, ContainerConfig)
        .join(FileAnalytics, FileAnalytics.file_id == File.id, isouter=True)
        .join(ContainerConfig, ContainerConfig.id == File.container_id, isouter=True)
        .where(
            File.ingest_status == IngestStatus.INGESTED.value,
            File.blob_path.isnot(None),
        )
    )).all()

    parquet_tasks = []
    no_parquet_count = 0
    for f, fa, container in no_parquet_rows:
        if not is_parquet_source_file(f.blob_path or f.name):
            continue
        if fa is not None and fa.parquet_blob_path:
            continue
        if f.id in in_flight:  # already converting — skip
            continue
        if not container or not container.connection_string:
            continue
        no_parquet_count += 1
        parquet_tasks.append((f.id, f.blob_path, container.connection_string, container.container_name))

    skipped_in_flight = len(in_flight)
    ingest_logger.info(
        "retry_parquet_started",
        admin_id=admin.id,
        missing_parquet=no_parquet_count,
        skipped_in_flight=skipped_in_flight,
        total=no_parquet_count,
    )

    # Fire parquet conversions sequentially — one per file to avoid OOM on big CSVs
    async def _run_parquet_tasks() -> None:
        for file_id, blob_path, conn_str, cont_name in parquet_tasks:
            try:
                await trigger_parquet_conversion(
                    file_id=file_id,
                    blob_path=blob_path,
                    connection_string=conn_str,
                    container_name=cont_name,
                )
            except Exception as exc:
                # trigger_parquet_conversion already logs to BackgroundJob;
                # this guard keeps the loop running for the remaining files.
                ingest_logger.warning(
                    "retry_parquet_task_failed",
                    file_id=file_id,
                    blob_path=blob_path,
                    error=str(exc)[:500],
                )

    if parquet_tasks:
        asyncio.create_task(_run_parquet_tasks())

    msg = "Parquet retry started"
    if skipped_in_flight:
        msg += f" — {skipped_in_flight} already running, skipped"
    return {
        "message": msg,
        "missing_parquet": no_parquet_count,
        "skipped_in_flight": skipped_in_flight,
        "total": no_parquet_count,
    }

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from app.api.v1.chat_common import IngestRequest
from app.core.database import get_db
from app.core.logger import ingest_logger
from app.dependencies import get_current_user, require_developer  # noqa: F401 (get_current_user used in ingest_status)
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder
from app.models.user import User
from app.services.ingestion_config import (
    REPROCESS_SCOPES,
    IngestStatus,
    is_supported_ingest_file,
    scope_forces_preprocess,
    supported_ingest_extensions,
)
from app.worker.ingest_tasks import run_ingest_pipeline, run_scoped_ingest

router = APIRouter()


class ReprocessRequest(BaseModel):
    container_id: str
    scope: str  # one of: refresh_rules | re_analyze | full_rebuild


@router.post("/reprocess")
async def reprocess_container(
    body: ReprocessRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_developer),
):
    """Scoped, parallel re-ingestion of every file in a container.

    Maps the 3 UI actions to stage subsets and fans out ONE scoped chain per
    file (so Celery runs them in parallel). Returns immediately with the count
    queued. Faster wall-clock comes from worker concurrency — raise
    CELERY_WORKER_CONCURRENCY (default is capped low) to widen the fan-out.
    """
    if body.scope not in REPROCESS_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scope '{body.scope}'. Use one of: {', '.join(REPROCESS_SCOPES)}.",
        )

    rows = (await db.execute(select(File).where(File.container_id == body.container_id))).scalars().all()
    ingestable = [f for f in rows if is_supported_ingest_file(f.name)]

    # Domain scope: a developer with allowed_domains only touches their folders.
    if admin.allowed_domains:
        scoped: list[File] = []
        for f in ingestable:
            folder = await db.get(Folder, f.folder_id) if f.folder_id else None
            fdomain = folder.domain_tag if folder else None
            if fdomain is None or fdomain in admin.allowed_domains:
                scoped.append(f)
        ingestable = scoped

    if not ingestable:
        raise HTTPException(status_code=400, detail="No ingestable files found in this container.")

    file_ids = [f.id for f in ingestable]

    # full_rebuild is the only scope that re-runs preprocessing → clear the flag
    # so clean_file_stage actually re-preprocesses.
    if scope_forces_preprocess(body.scope):
        await db.execute(update(File).where(File.id.in_(file_ids)).values(is_preprocessed=False))
        await db.commit()

    task_ids: list[str] = []
    for fid in file_ids:
        result = run_scoped_ingest.delay(fid, body.scope)
        task_ids.append(result.id)

    ingest_logger.info(
        "reprocess_queued",
        admin_id=admin.id,
        container_id=body.container_id,
        scope=body.scope,
        file_count=len(file_ids),
        backend="celery",
    )
    return {"queued": len(file_ids), "scope": body.scope, "container_id": body.container_id, "task_ids": task_ids}


@router.get("/reprocess-status")
async def reprocess_status(
    container_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Lightweight progress for the 're-ingestion in progress, please wait' banner.

    Counts files by ingest_status for the container. `in_progress` is true while
    any file is running/pending — the UI shows a wait notice and chat-may-be-
    slower hint during that window (ingestion is using cores reserved by the
    x-2 rule, so chat stays usable but can be slower)."""
    rows = (
        await db.execute(
            select(File.ingest_status, func.count())
            .where(File.container_id == container_id)
            .group_by(File.ingest_status)
        )
    ).all()
    counts = {str(status): int(n) for status, n in rows}
    running = counts.get(IngestStatus.RUNNING.value, 0)
    pending = counts.get(IngestStatus.PENDING.value, 0)
    total = sum(counts.values())
    done = counts.get(IngestStatus.INGESTED.value, 0)
    return {
        "container_id": container_id,
        "in_progress": (running + pending) > 0,
        "running": running,
        "pending": pending,
        "done": done,
        "total": total,
    }


@router.post("/ingest")
async def ingest_files(
    body: IngestRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_developer),
):
    if not body.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided.")

    valid_ids: list[str] = []
    for fid in body.file_ids:
        file = await db.get(File, fid)
        if not file:
            continue
        if not is_supported_ingest_file(file.name):
            continue
        # Domain scope: if developer has domain restrictions, skip files outside their domains
        if admin.allowed_domains:
            folder = await db.get(Folder, file.folder_id) if file.folder_id else None
            folder_domain = folder.domain_tag if folder else None
            if folder_domain and folder_domain not in admin.allowed_domains:
                continue  # silently skip — outside this developer's scope
        valid_ids.append(fid)

    if not valid_ids:
        allowed = ", ".join(sorted(supported_ingest_extensions()))
        raise HTTPException(status_code=400, detail=f"No valid ingestable files found. Supported: {allowed}")

    # If the caller wants to force re-preprocessing, clear the flag so clean_file_stage
    # re-runs preprocess_file() for each file regardless of its current state.
    if body.force_preprocess:
        await db.execute(
            update(File)
            .where(File.id.in_(valid_ids))
            .values(is_preprocessed=False)
        )
        await db.commit()

    # Dispatch each file to a Celery worker — isolated from the API event loop.
    # The worker runs preprocessing + DuckDB sample + AI description + parquet
    # conversion in a separate process. API returns immediately.
    task_ids: list[str] = []
    for fid in valid_ids:
        result = run_ingest_pipeline.delay(fid)
        task_ids.append(result.id)

    ingest_logger.info(
        "ingest_queued",
        admin_id=admin.id,
        file_count=len(valid_ids),
        file_ids=valid_ids,
        task_ids=task_ids,
        backend="celery",
    )
    return {"queued": len(valid_ids), "file_ids": valid_ids, "task_ids": task_ids}


@router.get("/ingest-status/{file_id}")
async def ingest_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    file = await db.get(File, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found.")

    result = await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
    metadata = result.scalar_one_or_none()

    return {
        "file_id": file_id,
        "ingest_status": file.ingest_status,
        "ai_description": metadata.ai_description if metadata else None,
        "columns": [c["name"] for c in metadata.columns_info] if metadata and metadata.columns_info else [],
        "row_count": metadata.row_count if metadata else None,
        "error": metadata.ingest_error if metadata else None,
    }

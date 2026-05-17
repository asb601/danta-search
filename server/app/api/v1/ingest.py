from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat_common import IngestRequest
from app.core.database import get_db
from app.core.logger import ingest_logger
from app.dependencies import get_current_user, require_developer  # noqa: F401 (get_current_user used in ingest_status)
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder
from app.models.user import User
from app.worker.ingest_tasks import run_ingest_pipeline

router = APIRouter()


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
        ext = (file.name or "").rsplit(".", 1)[-1].lower()
        if ext not in ("csv", "txt", "tsv"):
            continue
        # Domain scope: if developer has domain restrictions, skip files outside their domains
        if admin.allowed_domains:
            folder = await db.get(Folder, file.folder_id) if file.folder_id else None
            folder_domain = folder.domain_tag if folder else None
            if folder_domain and folder_domain not in admin.allowed_domains:
                continue  # silently skip — outside this developer's scope
        valid_ids.append(fid)

    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid CSV/TXT files found.")

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

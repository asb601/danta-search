import asyncio
import mimetypes
import time
import uuid
from datetime import datetime, timezone

import structlog
from azure.storage.blob import BlobServiceClient
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session, get_db
from app.core.logger import container_logger, blob_logger, db_logger, ingest_logger
from app.dependencies import require_admin, require_developer, get_current_user
from app.worker.ingest_tasks import run_ingest_pipeline, run_semantic_rebuild_container
from app.models.background_job import BackgroundJob
from app.models.container import ContainerConfig
from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder
from app.models.user import User
from app.schemas.container import (
    ContainerCreate,
    ContainerOut,
    ContainerSemanticConfigUpdate,
    ContainerSemanticRebuildRequest,
    ContainerSyncResponse,
)
from app.services.semantic_rebuild import evaluate_container_semantics
from app.services.semantic_roles import ROLE_KINDS, is_dynamic_role, role_catalog

router = APIRouter(prefix="/containers", tags=["containers"])


# ── helpers ──────────────────────────────────────────────────────────────────


async def _get_or_create_folder_path(
    path_parts: list[str],
    container_id: str,
    owner_id: str,
    db: AsyncSession,
) -> str | None:
    """Recursively create folder hierarchy from blob path segments. Return deepest folder id."""
    if not path_parts:
        return None
    parent_id: str | None = None
    for part in path_parts:
        result = await db.execute(
            select(Folder).where(
                Folder.name == part,
                Folder.parent_id == parent_id if parent_id else Folder.parent_id.is_(None),
                Folder.container_id == container_id,
            )
        )
        folder = result.scalar_one_or_none()
        if not folder:
            folder = Folder(
                id=str(uuid.uuid4()),
                name=part,
                parent_id=parent_id,
                owner_id=owner_id,
                container_id=container_id,
            )
            db.add(folder)
            await db.flush()
        parent_id = folder.id
    return parent_id


def _guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _validate_semantic_config(config: dict | None) -> dict | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise HTTPException(status_code=422, detail="semantic_config must be a JSON object")
    roles = config.get("roles", [])
    if roles is None:
        roles = []
    if not isinstance(roles, list):
        raise HTTPException(status_code=422, detail="semantic_config.roles must be a list")
    for idx, role in enumerate(roles):
        if not isinstance(role, dict):
            raise HTTPException(status_code=422, detail=f"semantic_config.roles[{idx}] must be an object")
        if not role.get("role") or not role.get("kind"):
            raise HTTPException(status_code=422, detail=f"semantic_config.roles[{idx}] requires role and kind")
        if role.get("kind") not in ROLE_KINDS:
            raise HTTPException(
                status_code=422,
                detail=f"semantic_config.roles[{idx}].kind must be one of {list(ROLE_KINDS)}",
            )
        role_id = str(role.get("role") or "")
        if role_id.startswith("custom:") and not is_dynamic_role(role_id):
            raise HTTPException(
                status_code=422,
                detail=f"semantic_config.roles[{idx}].role is not a valid custom role id",
            )
    role_catalog(config)
    return config


async def sync_container(container_id: str, user_id: str) -> None:
    """Background task: list all blobs in an Azure container and upsert file records."""
    async with async_session() as db:
        try:
            start = time.perf_counter()
            container_logger.info("sync_started", container_id=container_id)

            result = await db.execute(
                select(ContainerConfig).where(ContainerConfig.id == container_id)
            )
            config = result.scalar_one_or_none()
            if not config:
                container_logger.error("sync_failed", container_id=container_id, reason="not_found")
                return

            # All Azure SDK calls are synchronous — run in thread
            blob_start = time.perf_counter()
            blob_logger.info("blob_list_started", container_name=config.container_name)
            blob_service = await asyncio.to_thread(
                BlobServiceClient.from_connection_string, config.connection_string
            )
            container_client = await asyncio.to_thread(
                blob_service.get_container_client, config.container_name
            )
            blobs = await asyncio.to_thread(lambda: list(container_client.list_blobs()))
            blob_logger.info("blob_list_complete", blob_count=len(blobs), duration_ms=round((time.perf_counter() - blob_start) * 1000, 2))

            # Build set of blob paths currently in Azure
            remote_paths = {blob.name for blob in blobs if not blob.name.endswith("/")}

            # ── Remove files deleted from Azure ──
            db_files_result = await db.execute(
                select(File).where(File.container_id == container_id)
            )
            db_files = list(db_files_result.scalars().all())
            removed = 0
            for db_file in db_files:
                if db_file.blob_path not in remote_paths:
                    await db.delete(db_file)
                    removed += 1
            if removed:
                container_logger.info("sync_removed_stale", container_id=container_id, removed=removed)

            # ── Add new files from Azure ──
            synced = 0
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue

                # Check if already synced
                existing = await db.execute(
                    select(File.id).where(File.blob_path == blob.name, File.container_id == container_id)
                )
                if existing.scalar_one_or_none():
                    continue

                parts = blob.name.split("/")
                filename = parts[-1]
                folder_parts = parts[:-1]

                parent_folder_id = await _get_or_create_folder_path(
                    folder_parts, container_id, user_id, db
                )

                file = File(
                    id=str(uuid.uuid4()),
                    name=filename,
                    content_type=_guess_mime(filename),
                    size=blob.size or 0,
                    folder_id=parent_folder_id,
                    owner_id=user_id,
                    container_id=container_id,
                    blob_path=blob.name,
                    ingest_status="not_ingested",
                )
                db.add(file)
                synced += 1

            # ── Clean up empty folders for this container ──
            orphan_result = await db.execute(
                select(Folder).where(Folder.container_id == container_id)
            )
            orphan_folders = list(orphan_result.scalars().all())
            cleaned_folders = 0
            for folder in orphan_folders:
                child_files = await db.execute(
                    select(File.id).where(File.folder_id == folder.id).limit(1)
                )
                child_folders = await db.execute(
                    select(Folder.id).where(Folder.parent_id == folder.id).limit(1)
                )
                if not child_files.scalar_one_or_none() and not child_folders.scalar_one_or_none():
                    await db.delete(folder)
                    cleaned_folders += 1
            if cleaned_folders:
                container_logger.info("sync_cleaned_folders", container_id=container_id, removed=cleaned_folders)

            config.last_synced_at = datetime.now(timezone.utc)
            await db.commit()
            container_logger.info("sync_complete", container_id=container_id, added=synced, removed=removed, duration_ms=round((time.perf_counter() - start) * 1000, 2))

            # ── Auto-ingest un-ingested CSV/TXT files ──
            # Dispatch each file to a Celery worker — completely isolated from
            # this event loop. The sync endpoint returns in milliseconds;
            # workers process files concurrently across all worker processes.
            ingestable_result = await db.execute(
                select(File).where(
                    File.container_id == container_id,
                    File.ingest_status == "not_ingested",
                )
            )
            ingestable = [
                f for f in ingestable_result.scalars().all()
                if (f.name or "").rsplit(".", 1)[-1].lower() in ("csv", "txt", "tsv")
            ]
            if ingestable:
                task_ids = [run_ingest_pipeline.delay(f.id).id for f in ingestable]
                ingest_logger.info(
                    "sync_auto_ingest_queued",
                    container_id=container_id,
                    file_count=len(ingestable),
                    task_ids=task_ids,
                    backend="celery",
                )

        except Exception as exc:
            container_logger.exception("sync_failed", container_id=container_id, error=str(exc))
            await db.rollback()


# ── endpoints ────────────────────────────────────────────────────────────────


@router.get("", response_model=list[ContainerOut])
async def list_containers(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    container_logger.info("list_requested")

    result = await db.execute(
        select(
            ContainerConfig,
            func.count(File.id).label("file_count"),
        )
        .outerjoin(File, File.container_id == ContainerConfig.id)
        .group_by(ContainerConfig.id)
        .order_by(ContainerConfig.created_at.desc())
    )
    rows = result.all()
    containers = [
        ContainerOut(
            id=config.id,
            name=config.name,
            container_name=config.container_name,
            last_synced_at=config.last_synced_at,
            file_count=file_count,
            created_at=config.created_at,
            semantic_config=config.semantic_config,
        )
        for config, file_count in rows
    ]
    container_logger.info("list_complete", count=len(containers), duration_ms=round((time.perf_counter() - start) * 1000, 2))
    return containers


@router.get("/{container_id}", response_model=ContainerOut)
async def get_container(
    container_id: str,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(
            ContainerConfig,
            func.count(File.id).label("file_count"),
        )
        .outerjoin(File, File.container_id == ContainerConfig.id)
        .where(ContainerConfig.id == container_id)
        .group_by(ContainerConfig.id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Container not found")
    config, file_count = row
    return ContainerOut(
        id=config.id,
        name=config.name,
        container_name=config.container_name,
        last_synced_at=config.last_synced_at,
        file_count=file_count,
        created_at=config.created_at,
        semantic_config=config.semantic_config,
    )


@router.post("", response_model=ContainerOut, status_code=status.HTTP_201_CREATED)
async def create_container(
    body: ContainerCreate,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    container_logger.info("create_started", name=body.name, container_name=body.container_name)

    # Validate connection by trying to access the container
    try:
        blob_start = time.perf_counter()
        blob_logger.info("container_validate_started", container_name=body.container_name)
        blob_service = await asyncio.to_thread(
            BlobServiceClient.from_connection_string, body.connection_string
        )
        container_client = await asyncio.to_thread(
            blob_service.get_container_client, body.container_name
        )
        await asyncio.to_thread(container_client.get_container_properties)
        blob_logger.info("container_validate_complete", duration_ms=round((time.perf_counter() - blob_start) * 1000, 2))
    except Exception as e:
        container_logger.error("create_failed", reason="connection_error", error=str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to Azure container '{body.container_name}': {e}",
        )

    config = ContainerConfig(
        name=body.name,
        container_name=body.container_name,
        connection_string=body.connection_string,
        created_by=admin.id,
        semantic_config=_validate_semantic_config(body.semantic_config),
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)

    # Kick off background sync
    asyncio.create_task(sync_container(config.id, admin.id))

    container_logger.info("create_complete", container_id=config.id, duration_ms=round((time.perf_counter() - start) * 1000, 2))

    return ContainerOut(
        id=config.id,
        name=config.name,
        container_name=config.container_name,
        last_synced_at=config.last_synced_at,
        file_count=0,
        created_at=config.created_at,
        semantic_config=config.semantic_config,
    )


@router.patch("/{container_id}/semantic-config", response_model=ContainerOut)
async def update_semantic_config(
    container_id: str,
    body: ContainerSemanticConfigUpdate,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ContainerConfig).where(ContainerConfig.id == container_id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Container not found")

    config.semantic_config = _validate_semantic_config(body.semantic_config)
    await db.commit()
    await db.refresh(config)

    file_count = (
        await db.execute(select(func.count(File.id)).where(File.container_id == container_id))
    ).scalar_one()
    return ContainerOut(
        id=config.id,
        name=config.name,
        container_name=config.container_name,
        last_synced_at=config.last_synced_at,
        file_count=file_count,
        created_at=config.created_at,
        semantic_config=config.semantic_config,
    )


@router.post("/{container_id}/semantic-rebuild")
async def trigger_semantic_rebuild(
    container_id: str,
    body: ContainerSemanticRebuildRequest | None = None,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ContainerConfig.id).where(ContainerConfig.id == container_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Container not found")

    options = body or ContainerSemanticRebuildRequest()
    task = run_semantic_rebuild_container.delay(
        container_id,
        options.re_resolve_roles,
        options.batch_size,
    )
    ingest_logger.info(
        "semantic_rebuild_queued",
        admin_id=admin.id,
        container_id=container_id,
        task_id=task.id,
        re_resolve_roles=options.re_resolve_roles,
        batch_size=options.batch_size,
    )
    return {
        "message": "Semantic rebuild queued",
        "container_id": container_id,
        "task_id": task.id,
        "re_resolve_roles": options.re_resolve_roles,
        "batch_size": options.batch_size,
    }


@router.get("/{container_id}/semantic-evaluation")
async def get_semantic_evaluation(
    container_id: str,
    _admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await evaluate_container_semantics(container_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{container_id}/sync", response_model=ContainerSyncResponse)
async def trigger_sync(
    container_id: str,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ContainerConfig).where(ContainerConfig.id == container_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Container not found")

    asyncio.create_task(sync_container(container_id, admin.id))
    return ContainerSyncResponse(message="Sync started", container_id=container_id)


@router.delete("/{container_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_container(
    container_id: str,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    container_logger.info("delete_started", container_id=container_id)

    result = await db.execute(
        select(ContainerConfig).where(ContainerConfig.id == container_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Container not found")

    # Delete all files and folders belonging to this container
    # First get all file IDs so we can clean up dependent tables
    file_id_rows = (await db.execute(
        select(File.id).where(File.container_id == container_id)
    )).scalars().all()

    if file_id_rows:
        await db.execute(
            FileMetadata.__table__.delete().where(FileMetadata.file_id.in_(file_id_rows))
        )
        await db.execute(
            FileAnalytics.__table__.delete().where(FileAnalytics.file_id.in_(file_id_rows))
        )
        await db.execute(
            BackgroundJob.__table__.delete().where(BackgroundJob.file_id.in_(file_id_rows))
        )

    await db.execute(
        File.__table__.delete().where(File.container_id == container_id)
    )
    await db.execute(
        Folder.__table__.delete().where(Folder.container_id == container_id)
    )
    await db.delete(config)
    await db.commit()

    # Evict DuckDB thread-local connection so the connection string is no longer held in RAM
    from app.core.duckdb_client import _clear_connection
    _clear_connection(config.connection_string)

    container_logger.info("delete_complete", container_id=container_id, duration_ms=round((time.perf_counter() - start) * 1000, 2))

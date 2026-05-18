"""Celery task graph for staged ingestion.

Public entrypoint:
    run_ingest_pipeline.delay(file_id)

The public task is now only a workflow starter. The real work runs as separate
Celery tasks so each stage has its own retry/failure boundary:
    clean -> parquet -> metadata -> AI description -> ontology -> embedding
    -> OpenSearch -> analytics -> relationships -> complete
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import structlog
from celery import chain
from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import get_settings
from app.services.ingestion_config import (
    INGEST_PIPELINE_TASK_NAME,
    INGEST_STAGE_SPECS,
    SEMANTIC_REBUILD_TASK_NAME,
    PayloadStatus,
    StageName,
    celery_ingest_task_options,
    celery_semantic_rebuild_task_options,
    stage_names,
    stage_task_name,
)
from app.worker.celery_app import celery_app

Payload = dict[str, Any]
_INGEST_TASK_OPTIONS = celery_ingest_task_options()
_SEMANTIC_REBUILD_TASK_OPTIONS = celery_semantic_rebuild_task_options()


def _file_id_from_payload(payload_or_file_id: Payload | str) -> str:
    if isinstance(payload_or_file_id, dict):
        return str(payload_or_file_id["file_id"])
    return str(payload_or_file_id)


def _failed_payload(file_id: str, stage: str, exc: BaseException, retries: int) -> Payload:
    return {
        "file_id": file_id,
        "status": PayloadStatus.FAILED.value,
        "failed_stage": stage,
        "error": str(exc)[:500],
        "retries": retries,
    }


def _run_async(coro: Awaitable[Payload]) -> Payload:
    return asyncio.run(coro)


def _run_stage(
    task: Any,
    stage: StageName | str,
    payload: Payload | str,
    func: Callable[[Payload], Awaitable[Payload]],
) -> Payload:
    """Run one async stage with consistent retry and final failure handling."""
    stage_value = stage.value if isinstance(stage, StageName) else str(stage)
    if isinstance(payload, dict) and payload.get("status") == PayloadStatus.FAILED.value:
        return payload

    file_id = _file_id_from_payload(payload)
    stage_payload: Payload = payload if isinstance(payload, dict) else {"file_id": file_id}

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        pipeline="ingest",
        file_id=file_id,
        stage=stage_value,
        worker="celery",
    )

    try:
        return _run_async(func(stage_payload))
    except (SoftTimeLimitExceeded, Exception) as exc:
        if task.request.retries >= task.max_retries:
            from app.services.ingestion_stages import mark_ingestion_failed

            asyncio.run(mark_ingestion_failed(file_id, stage_value, exc))
            return _failed_payload(file_id, stage_value, exc, task.request.retries)
        raise task.retry(exc=exc)
    finally:
        structlog.contextvars.clear_contextvars()


@celery_app.task(
    bind=True,
    name=INGEST_PIPELINE_TASK_NAME,
    **_INGEST_TASK_OPTIONS,
)
def run_ingest_pipeline(self, file_id: str) -> Payload:
    """Start the staged ingestion graph for one file.

    Kept under the original task name so API routes do not need to know about
    the internal stage graph.
    """
    try:
        from app.services.ingestion_stages import prepare_pipeline

        prepared = asyncio.run(prepare_pipeline(file_id))
    except (SoftTimeLimitExceeded, Exception) as exc:
        if self.request.retries >= self.max_retries:
            from app.services.ingestion_stages import mark_ingestion_failed

            asyncio.run(mark_ingestion_failed(file_id, "prepare", exc))
            return _failed_payload(file_id, "prepare", exc, self.request.retries)
        raise self.retry(exc=exc)

    if prepared.get("status") != PayloadStatus.QUEUED.value:
        return prepared

    ordered_tasks = [_TASK_BY_STAGE[spec.stage] for spec in INGEST_STAGE_SPECS]
    workflow = chain(
        ordered_tasks[0].s({"file_id": file_id}),
        *(task.s() for task in ordered_tasks[1:]),
    ).apply_async()

    return {
        "file_id": file_id,
        "status": PayloadStatus.QUEUED.value,
        "workflow_id": workflow.id,
        "stages": stage_names(),
    }


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.CLEAN),
    **_INGEST_TASK_OPTIONS,
)
def clean_file_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import clean_file_stage

    return _run_stage(self, StageName.CLEAN, payload, clean_file_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.PARQUET),
    **_INGEST_TASK_OPTIONS,
)
def parquet_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import parquet_stage

    return _run_stage(self, StageName.PARQUET, payload, parquet_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.METADATA),
    **_INGEST_TASK_OPTIONS,
)
def metadata_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import metadata_stage

    return _run_stage(self, StageName.METADATA, payload, metadata_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.AI_DESCRIPTION),
    **_INGEST_TASK_OPTIONS,
)
def ai_description_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import ai_description_stage

    return _run_stage(self, StageName.AI_DESCRIPTION, payload, ai_description_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.ONTOLOGY),
    **_INGEST_TASK_OPTIONS,
)
def ontology_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import ontology_stage

    return _run_stage(self, StageName.ONTOLOGY, payload, ontology_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.EMBEDDING),
    **_INGEST_TASK_OPTIONS,
)
def embedding_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import embedding_stage

    return _run_stage(self, StageName.EMBEDDING, payload, embedding_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.OPENSEARCH),
    **_INGEST_TASK_OPTIONS,
)
def opensearch_index_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import opensearch_stage

    return _run_stage(self, StageName.OPENSEARCH, payload, opensearch_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.ANALYTICS),
    **_INGEST_TASK_OPTIONS,
)
def analytics_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import analytics_stage

    return _run_stage(self, StageName.ANALYTICS, payload, analytics_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.RELATIONSHIPS),
    **_INGEST_TASK_OPTIONS,
)
def relationship_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import relationship_stage

    return _run_stage(self, StageName.RELATIONSHIPS, payload, relationship_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.SEMANTIC_LAYER),
    **_INGEST_TASK_OPTIONS,
)
def semantic_layer_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import semantic_layer_stage

    return _run_stage(self, StageName.SEMANTIC_LAYER, payload, semantic_layer_stage)


@celery_app.task(
    bind=True,
    name=stage_task_name(StageName.COMPLETE),
    **_INGEST_TASK_OPTIONS,
)
def complete_ingestion_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import complete_ingestion_stage

    return _run_stage(self, StageName.COMPLETE, payload, complete_ingestion_stage)


_TASK_BY_STAGE = {
    StageName.CLEAN: clean_file_task,
    StageName.PARQUET: parquet_task,
    StageName.METADATA: metadata_task,
    StageName.AI_DESCRIPTION: ai_description_task,
    StageName.ONTOLOGY: ontology_task,
    StageName.EMBEDDING: embedding_task,
    StageName.OPENSEARCH: opensearch_index_task,
    StageName.ANALYTICS: analytics_task,
    StageName.RELATIONSHIPS: relationship_task,
    StageName.SEMANTIC_LAYER: semantic_layer_task,
    StageName.COMPLETE: complete_ingestion_task,
}


@celery_app.task(
    bind=True,
    name=SEMANTIC_REBUILD_TASK_NAME,
    **_SEMANTIC_REBUILD_TASK_OPTIONS,
)
def run_semantic_rebuild_container(
    self,
    container_id: str,
    re_resolve_roles: bool = True,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run a semantic-only rebuild for one container."""
    try:
        from app.services.semantic_rebuild import rebuild_container_semantics

        return asyncio.run(
            rebuild_container_semantics(
                container_id,
                re_resolve_roles=re_resolve_roles,
                batch_size=batch_size or get_settings().INGEST_SEMANTIC_REBUILD_BATCH_SIZE,
            )
        )
    except (SoftTimeLimitExceeded, Exception) as exc:
        if self.request.retries >= self.max_retries:
            return {
                "container_id": container_id,
                "status": PayloadStatus.FAILED.value,
                "error": str(exc)[:500],
                "retries": self.request.retries,
            }
        raise self.retry(exc=exc)

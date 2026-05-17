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

from app.worker.celery_app import celery_app

Payload = dict[str, Any]


def _file_id_from_payload(payload_or_file_id: Payload | str) -> str:
    if isinstance(payload_or_file_id, dict):
        return str(payload_or_file_id["file_id"])
    return str(payload_or_file_id)


def _failed_payload(file_id: str, stage: str, exc: BaseException, retries: int) -> Payload:
    return {
        "file_id": file_id,
        "status": "failed",
        "failed_stage": stage,
        "error": str(exc)[:500],
        "retries": retries,
    }


def _run_async(coro: Awaitable[Payload]) -> Payload:
    return asyncio.run(coro)


def _run_stage(
    task: Any,
    stage: str,
    payload: Payload | str,
    func: Callable[[Payload], Awaitable[Payload]],
) -> Payload:
    """Run one async stage with consistent retry and final failure handling."""
    if isinstance(payload, dict) and payload.get("status") == "failed":
        return payload

    file_id = _file_id_from_payload(payload)
    stage_payload: Payload = payload if isinstance(payload, dict) else {"file_id": file_id}

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        pipeline="ingest",
        file_id=file_id,
        stage=stage,
        worker="celery",
    )

    try:
        return _run_async(func(stage_payload))
    except (SoftTimeLimitExceeded, Exception) as exc:
        if task.request.retries >= task.max_retries:
            from app.services.ingestion_stages import mark_ingestion_failed

            asyncio.run(mark_ingestion_failed(file_id, stage, exc))
            return _failed_payload(file_id, stage, exc, task.request.retries)
        raise task.retry(exc=exc)
    finally:
        structlog.contextvars.clear_contextvars()


@celery_app.task(
    bind=True,
    name="gchat.ingest_pipeline",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
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

    if prepared.get("status") != "queued":
        return prepared

    workflow = chain(
        clean_file_task.s({"file_id": file_id}),
        parquet_task.s(),
        metadata_task.s(),
        ai_description_task.s(),
        ontology_task.s(),
        embedding_task.s(),
        opensearch_index_task.s(),
        analytics_task.s(),
        relationship_task.s(),
        semantic_layer_task.s(),
        complete_ingestion_task.s(),
    ).apply_async()

    return {
        "file_id": file_id,
        "status": "queued",
        "workflow_id": workflow.id,
        "stages": [
            "clean",
            "parquet",
            "metadata",
            "ai_description",
            "ontology",
            "embedding",
            "opensearch",
            "analytics",
            "relationships",
            "semantic_layer",
            "complete",
        ],
    }


@celery_app.task(
    bind=True,
    name="gchat.ingest.clean",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def clean_file_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import clean_file_stage

    return _run_stage(self, "clean", payload, clean_file_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.parquet",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def parquet_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import parquet_stage

    return _run_stage(self, "parquet", payload, parquet_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.metadata",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def metadata_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import metadata_stage

    return _run_stage(self, "metadata", payload, metadata_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.ai_description",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def ai_description_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import ai_description_stage

    return _run_stage(self, "ai_description", payload, ai_description_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.ontology",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def ontology_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import ontology_stage

    return _run_stage(self, "ontology", payload, ontology_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.embedding",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def embedding_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import embedding_stage

    return _run_stage(self, "embedding", payload, embedding_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.opensearch",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def opensearch_index_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import opensearch_stage

    return _run_stage(self, "opensearch", payload, opensearch_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.analytics",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def analytics_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import analytics_stage

    return _run_stage(self, "analytics", payload, analytics_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.relationships",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def relationship_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import relationship_stage

    return _run_stage(self, "relationships", payload, relationship_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.semantic_layer",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def semantic_layer_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import semantic_layer_stage

    return _run_stage(self, "semantic_layer", payload, semantic_layer_stage)


@celery_app.task(
    bind=True,
    name="gchat.ingest.complete",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def complete_ingestion_task(self, payload: Payload) -> Payload:
    from app.services.ingestion_stages import complete_ingestion_stage

    return _run_stage(self, "complete", payload, complete_ingestion_stage)


@celery_app.task(
    bind=True,
    name="gchat.semantic.rebuild_container",
    max_retries=1,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=300,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="ingest_normal",
)
def run_semantic_rebuild_container(
    self,
    container_id: str,
    re_resolve_roles: bool = True,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Run a semantic-only rebuild for one container."""
    try:
        from app.services.semantic_rebuild import rebuild_container_semantics

        return asyncio.run(
            rebuild_container_semantics(
                container_id,
                re_resolve_roles=re_resolve_roles,
                batch_size=batch_size,
            )
        )
    except (SoftTimeLimitExceeded, Exception) as exc:
        if self.request.retries >= self.max_retries:
            return {
                "container_id": container_id,
                "status": "failed",
                "error": str(exc)[:500],
                "retries": self.request.retries,
            }
        raise self.retry(exc=exc)

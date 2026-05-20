from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from app.core.config import Settings, get_settings


class IngestStatus(StrEnum):
    NOT_INGESTED = "not_ingested"
    PENDING = "pending"
    RUNNING = "running"
    INGESTED = "ingested"
    FAILED = "failed"


class PayloadStatus(StrEnum):
    QUEUED = "queued"
    SKIPPED = "skipped"
    ALREADY_RUNNING = "already_running"
    FAILED = "failed"
    DONE = "done"


class StageName(StrEnum):
    CLEAN = "clean"
    PARQUET = "parquet"
    METADATA = "metadata"
    AI_DESCRIPTION = "ai_description"
    ONTOLOGY = "ontology"
    EMBEDDING = "embedding"
    OPENSEARCH = "opensearch"
    ANALYTICS = "analytics"
    RELATIONSHIPS = "relationships"
    SEMANTIC_LAYER = "semantic_layer"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StageSpec:
    stage: StageName
    task_name: str


INGEST_PIPELINE_TASK_NAME = "gchat.ingest_pipeline"
SEMANTIC_REBUILD_TASK_NAME = "gchat.semantic.rebuild_container"

INGEST_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(StageName.CLEAN, "gchat.ingest.clean"),
    StageSpec(StageName.METADATA, "gchat.ingest.metadata"),
    StageSpec(StageName.AI_DESCRIPTION, "gchat.ingest.ai_description"),
    StageSpec(StageName.ONTOLOGY, "gchat.ingest.ontology"),
    StageSpec(StageName.EMBEDDING, "gchat.ingest.embedding"),
    StageSpec(StageName.OPENSEARCH, "gchat.ingest.opensearch"),
    StageSpec(StageName.PARQUET, "gchat.ingest.parquet"),
    StageSpec(StageName.ANALYTICS, "gchat.ingest.analytics"),
    StageSpec(StageName.RELATIONSHIPS, "gchat.ingest.relationships"),
    StageSpec(StageName.SEMANTIC_LAYER, "gchat.ingest.semantic_layer"),
    StageSpec(StageName.COMPLETE, "gchat.ingest.complete"),
)


def _split_csv(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = [str(item) for item in value]

    items: list[str] = []
    for raw in raw_items:
        item = raw.strip()
        if item.lower() in {"<empty>", "<blank>", "<null-token-empty>"}:
            items.append("")
        elif item:
            items.append(item)
    return tuple(dict.fromkeys(items))


def configured_tokens(value: str | Iterable[str], *, lower: bool = True) -> tuple[str, ...]:
    items = _split_csv(value)
    if lower:
        return tuple(item.lower() for item in items)
    return items


def _exts(value: str | Iterable[str], *, dotted: bool = False) -> frozenset[str]:
    normalized: list[str] = []
    for item in _split_csv(value):
        ext = item.lower().lstrip(".")
        if not ext:
            continue
        normalized.append(f".{ext}" if dotted else ext)
    return frozenset(normalized)


def file_extension(filename: str | None, *, dotted: bool = False) -> str:
    ext = Path(filename or "").suffix.lower().lstrip(".")
    return f".{ext}" if dotted and ext else ext


def supported_ingest_extensions(*, dotted: bool = False) -> frozenset[str]:
    return _exts(get_settings().INGEST_SUPPORTED_EXTENSIONS, dotted=dotted)


def auto_ingest_extensions(*, dotted: bool = False) -> frozenset[str]:
    return _exts(get_settings().INGEST_AUTO_EXTENSIONS, dotted=dotted)


def text_ingest_extensions(*, dotted: bool = False) -> frozenset[str]:
    return _exts(get_settings().INGEST_TEXT_EXTENSIONS, dotted=dotted)


def excel_ingest_extensions(*, dotted: bool = False) -> frozenset[str]:
    return _exts(get_settings().INGEST_EXCEL_EXTENSIONS, dotted=dotted)


def parquet_source_extensions(*, dotted: bool = False) -> frozenset[str]:
    return _exts(get_settings().INGEST_PARQUET_EXTENSIONS, dotted=dotted)


def preprocess_extensions(*, dotted: bool = False) -> frozenset[str]:
    return text_ingest_extensions(dotted=dotted) | excel_ingest_extensions(dotted=dotted)


def is_supported_ingest_file(filename: str | None) -> bool:
    return file_extension(filename) in supported_ingest_extensions()


def is_auto_ingest_file(filename: str | None) -> bool:
    return file_extension(filename) in auto_ingest_extensions()


def is_text_ingest_file(filename: str | None) -> bool:
    return file_extension(filename) in text_ingest_extensions()


def is_excel_ingest_file(filename: str | None) -> bool:
    return file_extension(filename) in excel_ingest_extensions()


def is_parquet_source_file(filename: str | None) -> bool:
    return file_extension(filename) in parquet_source_extensions()


def parquet_blob_path_for(blob_path: str) -> str:
    ext = get_settings().INGEST_PARQUET_EXTENSION.strip().lstrip(".") or "parquet"
    return f"{blob_path.rsplit('.', 1)[0]}.{ext}"


def parquet_extension(*, dotted: bool = False) -> str:
    ext = get_settings().INGEST_PARQUET_EXTENSION.strip().lower().lstrip(".") or "parquet"
    return f".{ext}" if dotted else ext


def null_tokens() -> tuple[str, ...]:
    return _split_csv(get_settings().INGEST_NULL_TOKENS)


def null_tokens_lower() -> frozenset[str]:
    return frozenset(token.strip().lower() for token in null_tokens())


def schema_filename_tokens() -> frozenset[str]:
    return frozenset(token.lower() for token in _split_csv(get_settings().INGEST_SCHEMA_FILENAME_TOKENS))


def glossary_filename_tokens() -> frozenset[str]:
    return frozenset(token.lower() for token in _split_csv(get_settings().INGEST_GLOSSARY_FILENAME_TOKENS))


def schema_field_name_tokens() -> frozenset[str]:
    return frozenset(token.lower() for token in _split_csv(get_settings().INGEST_SCHEMA_FIELD_NAME_TOKENS))


def schema_description_tokens() -> frozenset[str]:
    return frozenset(token.lower() for token in _split_csv(get_settings().INGEST_SCHEMA_DESCRIPTION_TOKENS))


def schema_notes_tokens() -> frozenset[str]:
    return frozenset(token.lower() for token in _split_csv(get_settings().INGEST_SCHEMA_NOTES_TOKENS))


def stage_names() -> list[str]:
    return [spec.stage.value for spec in INGEST_STAGE_SPECS]


def stage_task_name(stage: StageName) -> str:
    for spec in INGEST_STAGE_SPECS:
        if spec.stage == stage:
            return spec.task_name
    raise KeyError(stage)


def ingest_task_names() -> tuple[str, ...]:
    return (INGEST_PIPELINE_TASK_NAME, *(spec.task_name for spec in INGEST_STAGE_SPECS), SEMANTIC_REBUILD_TASK_NAME)


def celery_ingest_task_options(settings: Settings | None = None) -> dict[str, object]:
    active = settings or get_settings()
    return {
        "max_retries": max(0, int(active.INGEST_TASK_MAX_RETRIES)),
        "default_retry_delay": max(0, int(active.INGEST_TASK_DEFAULT_RETRY_DELAY_SECONDS)),
        "retry_backoff": bool(active.INGEST_TASK_RETRY_BACKOFF),
        "retry_backoff_max": max(0, int(active.INGEST_TASK_RETRY_BACKOFF_MAX_SECONDS)),
        "acks_late": bool(active.INGEST_TASK_ACKS_LATE),
        "reject_on_worker_lost": bool(active.INGEST_TASK_REJECT_ON_WORKER_LOST),
        "queue": active.INGEST_NORMAL_QUEUE,
    }


def celery_semantic_rebuild_task_options(settings: Settings | None = None) -> dict[str, object]:
    options = celery_ingest_task_options(settings)
    active = settings or get_settings()
    options["max_retries"] = max(0, int(active.INGEST_SEMANTIC_REBUILD_MAX_RETRIES))
    options["default_retry_delay"] = max(0, int(active.INGEST_SEMANTIC_REBUILD_DEFAULT_RETRY_DELAY_SECONDS))
    return options


@lru_cache
def celery_task_routes() -> dict[str, dict[str, str]]:
    queue = get_settings().INGEST_NORMAL_QUEUE
    return {task_name: {"queue": queue} for task_name in ingest_task_names()}
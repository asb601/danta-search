from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SERVER_ROOT = Path(__file__).resolve().parents[2]


LEGACY_POLICY_PATHS: dict[str, tuple[str, ...]] = {
    "REINGEST_BATCH_SIZE": ("runtime", "reingest_batch_size"),
    "REINGEST_BATCH_DELAY_SECONDS": ("runtime", "reingest_batch_delay_seconds"),
    "PARQUET_CONVERSION_CONCURRENCY": ("runtime", "parquet_conversion_concurrency"),
    "INGEST_PREPROCESS_CONCURRENCY": ("runtime", "preprocess_concurrency"),
    "INGEST_EXCEL_PREPROCESS_CONCURRENCY": ("runtime", "excel_preprocess_concurrency"),
    "CELERY_WORKER_CONCURRENCY": ("runtime", "celery_worker_concurrency"),
    "CELERY_WORKER_PREFETCH_MULTIPLIER": ("runtime", "celery_worker_prefetch_multiplier"),
    "CELERY_RESULT_EXPIRES_SECONDS": ("runtime", "celery_result_expires_seconds"),
    "INGEST_HIGH_QUEUE": ("queues", "high"),
    "INGEST_NORMAL_QUEUE": ("queues", "normal"),
    "INGEST_LOW_QUEUE": ("queues", "low"),
    "INGEST_TASK_MAX_RETRIES": ("task", "max_retries"),
    "INGEST_TASK_DEFAULT_RETRY_DELAY_SECONDS": ("task", "default_retry_delay_seconds"),
    "INGEST_TASK_RETRY_BACKOFF": ("task", "retry_backoff"),
    "INGEST_TASK_RETRY_BACKOFF_MAX_SECONDS": ("task", "retry_backoff_max_seconds"),
    "INGEST_TASK_ACKS_LATE": ("task", "acks_late"),
    "INGEST_TASK_REJECT_ON_WORKER_LOST": ("task", "reject_on_worker_lost"),
    "INGEST_TASK_SOFT_TIME_LIMIT_SECONDS": ("task", "soft_time_limit_seconds"),
    "INGEST_TASK_TIME_LIMIT_SECONDS": ("task", "time_limit_seconds"),
    "INGEST_SEMANTIC_REBUILD_MAX_RETRIES": ("semantic_rebuild", "max_retries"),
    "INGEST_SEMANTIC_REBUILD_DEFAULT_RETRY_DELAY_SECONDS": ("semantic_rebuild", "default_retry_delay_seconds"),
    "INGEST_SEMANTIC_REBUILD_BATCH_SIZE": ("semantic_rebuild", "batch_size"),
    "INGEST_SUPPORTED_EXTENSIONS": ("extensions", "supported"),
    "INGEST_AUTO_EXTENSIONS": ("extensions", "auto_ingest"),
    "INGEST_TEXT_EXTENSIONS": ("extensions", "text"),
    "INGEST_EXCEL_EXTENSIONS": ("extensions", "excel"),
    "INGEST_PARQUET_EXTENSIONS": ("extensions", "parquet_sources"),
    "INGEST_PARQUET_EXTENSION": ("extensions", "parquet_output"),
    "INGEST_DUCKDB_SAMPLE_ROWS": ("duckdb", "sample_rows"),
    "INGEST_DUCKDB_QUERY_TIMEOUT_SECONDS": ("duckdb", "query_timeout_seconds"),
    "INGEST_DUCKDB_QUERY_MAX_ROWS": ("duckdb", "query_max_rows"),
    "INGEST_COLUMN_UNIQUE_VALUES_LIMIT": ("columns", "unique_values_limit"),
    "INGEST_COLUMN_SAMPLE_VALUES_LIMIT": ("columns", "sample_values_limit"),
    "INGEST_ROLE_RESOLVER_MAX_COLUMNS": ("role_resolver", "max_columns"),
    "INGEST_ROLE_RESOLVER_SAMPLE_VALUES": ("role_resolver", "sample_values"),
    "INGEST_ROLE_RESOLVER_GLOSSARY_ITEMS": ("role_resolver", "glossary_items"),
    "INGEST_ROLE_RESOLVER_PREVIEW_ITEMS": ("role_resolver", "preview_items"),
    "INGEST_ROLE_RESOLVER_MAX_COMPLETION_TOKENS": ("role_resolver", "max_completion_tokens"),
    "INGEST_FILE_DESCRIPTION_SAMPLE_ROWS": ("file_description", "sample_rows"),
    "INGEST_FILE_DESCRIPTION_MAX_COMPLETION_TOKENS": ("file_description", "max_completion_tokens"),
    "INGEST_LLM_RETRY_DELAYS_SECONDS": ("llm", "retry_delays_seconds"),
    "INGEST_IDENTIFIER_EXACT_NAMES": ("identifier_detection", "exact_names"),
    "INGEST_IDENTIFIER_SUFFIXES": ("identifier_detection", "suffixes"),
    "INGEST_IDENTIFIER_PREFIXES": ("identifier_detection", "prefixes"),
    "INGEST_PREPROCESS_CHUNK_ROWS": ("preprocess", "chunk_rows"),
    "INGEST_SMALL_FILE_THRESHOLD_MB": ("preprocess", "small_file_threshold_mb"),
    "INGEST_HEADER_SCAN_ROWS": ("preprocess", "header_scan_rows"),
    "INGEST_TYPE_DETECT_SAMPLE_ROWS": ("preprocess", "type_detect_sample_rows"),
    "INGEST_PROBE_BYTES": ("preprocess", "probe_bytes"),
    "INGEST_AZURE_READ_BUFFER_BYTES": ("preprocess", "azure_read_buffer_bytes"),
    "INGEST_UPLOAD_BLOCK_SIZE_BYTES": ("preprocess", "upload_block_size_bytes"),
    "INGEST_MIN_EXCEL_TMP_FREE_BYTES": ("preprocess", "min_excel_tmp_free_bytes"),
    "INGEST_EXCEL_TMP_FREE_MULTIPLIER": ("preprocess", "excel_tmp_free_multiplier"),
    "INGEST_QUARANTINE_SAMPLE_ROWS": ("preprocess", "quarantine_sample_rows"),
    "INGEST_MALFORMED_SAMPLE_ROWS": ("preprocess", "malformed_sample_rows"),
    "INGEST_LOG_SAMPLE_ITEMS": ("preprocess", "log_sample_items"),
    "INGEST_DELIMITER_DETECT_BYTES": ("preprocess", "delimiter_detect_bytes"),
    "INGEST_DELIMITER_CONSISTENCY_THRESHOLD": ("preprocess", "delimiter_consistency_threshold"),
    "INGEST_HEADER_NUMERIC_PENALTY_WEIGHT": ("preprocess", "header_numeric_penalty_weight"),
    "INGEST_HEADER_AVG_LEN_TARGET": ("preprocess", "header_avg_len_target"),
    "INGEST_HEADER_LEN_PENALTY_SPAN": ("preprocess", "header_len_penalty_span"),
    "INGEST_HEADER_EARLY_ROW_LIMIT": ("preprocess", "header_early_row_limit"),
    "INGEST_HEADER_EARLY_SCORE_THRESHOLD": ("preprocess", "header_early_score_threshold"),
    "INGEST_HEADER_SCORE_EPSILON": ("preprocess", "header_score_epsilon"),
    "INGEST_PARQUET_AZURE_BUFFER_BYTES": ("parquet", "azure_buffer_bytes"),
    "INGEST_PARQUET_BLOCK_BYTES": ("parquet", "block_bytes"),
    "INGEST_PARQUET_READ_BLOCK_BYTES": ("parquet", "read_block_bytes"),
    "INGEST_PARQUET_COMPRESSION": ("parquet", "compression"),
    "INGEST_PARQUET_COMPRESSION_LEVEL": ("parquet", "compression_level"),
    "INGEST_PARQUET_AUTO_DICT_MAX_CARDINALITY": ("parquet", "auto_dict_max_cardinality"),
    "INGEST_PARQUET_CATEGORY_MAX_RATIO": ("parquet", "category_max_ratio"),
    "INGEST_PARQUET_CATEGORY_MAX_DISTINCT": ("parquet", "category_max_distinct"),
    "INGEST_PARQUET_TOP_VALUES": ("parquet", "top_values"),
    "INGEST_PARQUET_PROGRESS_MAX_PCT": ("parquet", "progress_max_pct"),
    "INGEST_PARQUET_PROGRESS_BATCH_PCT": ("parquet", "progress_batch_pct"),
    "INGEST_ANALYTICS_VALUE_COUNT_COLUMNS": ("analytics", "value_count_columns"),
    "INGEST_ANALYTICS_VALUE_COUNT_TOP_VALUES": ("analytics", "value_count_top_values"),
    "INGEST_ANALYTICS_CROSSTAB_DIMENSIONS": ("analytics", "crosstab_dimensions"),
    "INGEST_ANALYTICS_CROSSTAB_METRICS": ("analytics", "crosstab_metrics"),
    "INGEST_ANALYTICS_CROSSTAB_TOP_ROWS": ("analytics", "crosstab_top_rows"),
    "INGEST_SEMANTIC_COMPONENT_LIMIT": ("semantic_layer", "component_limit"),
    "INGEST_FAILURE_SAMPLE_LIMIT": ("semantic_layer", "failure_sample_limit"),
    "INGEST_NULL_TOKENS": ("cleaning", "null_tokens"),
    "INGEST_SCHEMA_FILENAME_TOKENS": ("dictionary_detection", "schema_filename_tokens"),
    "INGEST_GLOSSARY_FILENAME_TOKENS": ("dictionary_detection", "glossary_filename_tokens"),
    "INGEST_SCHEMA_FIELD_NAME_TOKENS": ("dictionary_detection", "field_name_tokens"),
    "INGEST_SCHEMA_DESCRIPTION_TOKENS": ("dictionary_detection", "description_tokens"),
    "INGEST_SCHEMA_NOTES_TOKENS": ("dictionary_detection", "notes_tokens"),
    "INGEST_SEMANTIC_ENRICHMENT_MAX_COMPLETION_TOKENS": ("semantic_enrichment", "max_completion_tokens"),
    "INGEST_SEMANTIC_ENRICHMENT_MAX_ADDITIONS": ("semantic_enrichment", "max_additions"),
}

_DYNAMIC_DEFAULTS = {
    "REINGEST_BATCH_SIZE",
    "REINGEST_BATCH_DELAY_SECONDS",
    "PARQUET_CONVERSION_CONCURRENCY",
    "INGEST_PREPROCESS_CONCURRENCY",
    "INGEST_EXCEL_PREPROCESS_CONCURRENCY",
    "CELERY_WORKER_CONCURRENCY",
}


def _available_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _available_memory_gib() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    return page_size * page_count / (1024 ** 3)


def _default_ingest_parallelism() -> int:
    """Cores to give ingestion, leaving headroom so chat stays responsive.

    Rule (hardware-aware, not a hardcoded number — works on any VM):
        ingestion_cores = max(1, x - 2)        where x = detected CPU cores
    i.e. RESERVE 2 cores for the chat/API process. Worked examples:
        x = 2  → max(1, 0) = 1   (ingestion 1, chat keeps 1 — small-box rule)
        x = 4  → 2               (ingestion 2, chat keeps 2)
        x = 8  → 6               (ingestion 6, chat keeps 2)
    Celery workers only occupy cores while there is work, so these cores are
    used DURING ingestion and freed when idle — "x-2 when triggered" for free.

    A memory bound still applies (each worker process needs RAM), and an upper
    sanity cap guards huge boxes (real ceiling there is the Azure TPM quota, not
    CPU). Override anytime via CELERY_WORKER_CONCURRENCY in the policy/env.
    """
    cpu = _available_cpu_count()
    cpu_bound = max(1, cpu - 2)  # reserve 2 cores for chat (1 on a 2-core box)
    memory_gib = _available_memory_gib()
    memory_bound = max(1, int(memory_gib // 2)) if memory_gib else cpu_bound
    _SANITY_CAP = 32  # backstop for very large hosts; tune via quota, not here
    return max(1, min(cpu_bound, memory_bound, _SANITY_CAP))


def _dynamic_default(name: str) -> int:
    if name == "INGEST_EXCEL_PREPROCESS_CONCURRENCY":
        return max(1, _default_ingest_parallelism() // 2)
    if name == "REINGEST_BATCH_DELAY_SECONDS":
        return max(5, 60 // _available_cpu_count())
    return _default_ingest_parallelism()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"ingestion policy must be a JSON object: {path}")
    return data


def _policy_file_path(raw_path: str) -> Path:
    path = Path(raw_path or "config/ingestion_policy.json")
    if not path.is_absolute():
        path = SERVER_ROOT / path
    return path


def _coerce_env_value(raw: str, fallback: Any) -> Any:
    if isinstance(fallback, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(fallback, int) and not isinstance(fallback, bool):
        return int(raw)
    if isinstance(fallback, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class IngestionPolicy:
    data: Mapping[str, Any]

    def lookup(self, path: tuple[str, ...]) -> Any:
        cursor: Any = self.data
        for part in path:
            if not isinstance(cursor, Mapping) or part not in cursor:
                return None
            cursor = cursor[part]
        return cursor

    def legacy_value(self, name: str) -> Any:
        if name not in LEGACY_POLICY_PATHS:
            raise AttributeError(name)

        value = self.lookup(LEGACY_POLICY_PATHS[name])
        fallback = _dynamic_default(name) if value is None and name in _DYNAMIC_DEFAULTS else value
        env_value = os.getenv(name)
        if env_value is not None:
            return _coerce_env_value(env_value, fallback)
        if fallback is None:
            raise AttributeError(f"missing ingestion policy value: {name}")
        return fallback

    def as_legacy_mapping(self) -> dict[str, Any]:
        return {name: self.legacy_value(name) for name in LEGACY_POLICY_PATHS}


@lru_cache
def get_ingestion_policy() -> IngestionPolicy:
    from app.core.config import get_settings

    settings = get_settings()
    policy_path = _policy_file_path(getattr(settings, "INGESTION_POLICY_FILE", ""))
    data = _read_json_file(policy_path)

    inline_json = os.getenv("INGESTION_POLICY_JSON") or getattr(settings, "INGESTION_POLICY_JSON", "")
    if inline_json.strip():
        inline = json.loads(inline_json)
        if not isinstance(inline, dict):
            raise ValueError("INGESTION_POLICY_JSON must be a JSON object")
        data = _deep_merge(data, inline)

    return IngestionPolicy(data)
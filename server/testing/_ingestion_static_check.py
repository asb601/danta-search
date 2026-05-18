"""
Static guard for ingestion pipeline configuration drift.
Usage: cd server && python3 -m testing._ingestion_static_check
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from app.services.ingestion_policy import LEGACY_POLICY_PATHS, get_ingestion_policy


SERVER_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = SERVER_ROOT / "app"

ALLOW_CONFIG = {
    APP_ROOT / "core" / "config.py",
    APP_ROOT / "services" / "ingestion_config.py",
}

PIPELINE_FILES = sorted(
    path for path in APP_ROOT.rglob("*.py")
    if "__pycache__" not in path.parts
)

FORBIDDEN_PATTERNS = {
    "duplicated csv auto-ingest tuple": re.compile(r'\("csv",\s*"txt",\s*"tsv"\)'),
    "duplicated Excel extension set": re.compile(r'\{\s*"\.xlsx",\s*"\.xls",\s*"\.xlsm",\s*"\.xlsb"\s*\}'),
    "hardcoded DuckDB sample size": re.compile(r"sample_size\s*=\s*500"),
    "hardcoded DuckDB sample limit": re.compile(r"LIMIT\s+500\b"),
    "hardcoded DuckDB null list": re.compile(r"nullstr\s*=\s*\["),
    "hardcoded ingest queue decorator": re.compile(r'queue\s*=\s*"ingest_normal"'),
    "hardcoded ingest retry count": re.compile(r"max_retries\s*=\s*3"),
    "hardcoded ingest retry delay": re.compile(r"default_retry_delay\s*=\s*30"),
    "hardcoded ingest retry backoff max": re.compile(r"retry_backoff_max\s*=\s*300"),
    "raw ingest status assignment": re.compile(r'ingest_status\s*=\s*"(?:not_ingested|pending|running|ingested|failed)"'),
    "raw pending status comparison": re.compile(r'ingest_status\s*==\s*"pending"'),
    "business-specific role example": re.compile(
        r"business_entity|custom:[a-z_]+:(?:claim|policy|premium|coverage|vendor|customer|invoice|sales|purchase)\b"
    ),
    "source-specific schema example": re.compile(r"fbl3n|bseg|vendor master", re.IGNORECASE),
    "source-specific ingestion term": re.compile(
        r"\bSAP\b|\bOracle\b|\bLEDGER\b|\bINVOICE\b|ledger_|invoice_num|set_of_books|code_combination",
        re.IGNORECASE,
    ),
}

REQUIRED_CONFIG_NAMES = [
    "INGEST_SUPPORTED_EXTENSIONS",
    "INGEST_AUTO_EXTENSIONS",
    "INGEST_TEXT_EXTENSIONS",
    "INGEST_EXCEL_EXTENSIONS",
    "INGEST_PARQUET_EXTENSIONS",
    "INGEST_DUCKDB_SAMPLE_ROWS",
    "INGEST_NULL_TOKENS",
    "INGEST_PREPROCESS_CHUNK_ROWS",
    "INGEST_PARQUET_READ_BLOCK_BYTES",
    "INGEST_TASK_MAX_RETRIES",
    "INGEST_TASK_DEFAULT_RETRY_DELAY_SECONDS",
    "INGEST_TASK_RETRY_BACKOFF_MAX_SECONDS",
    "INGEST_NORMAL_QUEUE",
]

LEGACY_CONFIG_FIELD_RE = re.compile(
    r"^\s+(?:INGEST_[A-Z0-9_]+|REINGEST_BATCH_SIZE|REINGEST_BATCH_DELAY_SECONDS|"
    r"PARQUET_CONVERSION_CONCURRENCY|CELERY_WORKER_CONCURRENCY|"
    r"CELERY_WORKER_PREFETCH_MULTIPLIER|CELERY_RESULT_EXPIRES_SECONDS)\s*:\s*[^\n=]+=",
    re.MULTILINE,
)

REQUIRED_USAGES = {
    APP_ROOT / "api" / "v1" / "files.py": ["is_auto_ingest_file", "IngestStatus"],
    APP_ROOT / "api" / "v1" / "ingest.py": ["is_supported_ingest_file"],
    APP_ROOT / "api" / "v1" / "admin.py": ["is_supported_ingest_file", "is_parquet_source_file"],
    APP_ROOT / "api" / "v1" / "containers.py": ["is_auto_ingest_file", "IngestStatus"],
    APP_ROOT / "core" / "duckdb_client.py": ["INGEST_DUCKDB_SAMPLE_ROWS", "null_tokens"],
    APP_ROOT / "services" / "data_preprocessor.py": ["INGEST_PREPROCESS_CHUNK_ROWS", "text_ingest_extensions"],
    APP_ROOT / "services" / "parquet_service.py": ["INGEST_PARQUET_READ_BLOCK_BYTES", "schema_field_name_tokens", "null_tokens"],
    APP_ROOT / "worker" / "ingest_tasks.py": ["celery_ingest_task_options", "INGEST_STAGE_SPECS"],
    APP_ROOT / "worker" / "celery_app.py": ["celery_task_routes", "INGEST_NORMAL_QUEUE"],
}


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def main() -> int:
    failures: list[str] = []

    config_text = (APP_ROOT / "core" / "config.py").read_text(encoding="utf-8")
    for match in LEGACY_CONFIG_FIELD_RE.finditer(config_text):
        failures.append(
            f"app/core/config.py:{_line_number(config_text, match.start())}: "
            f"ingestion policy value belongs outside core Settings: {match.group(0).strip()}"
        )

    policy = get_ingestion_policy()
    for name in REQUIRED_CONFIG_NAMES:
        if name not in LEGACY_POLICY_PATHS:
            failures.append(f"missing ingestion policy mapping: {name}")
            continue
        try:
            policy.legacy_value(name)
        except Exception as exc:
            failures.append(f"missing ingestion policy value: {name}: {exc}")

    for path in PIPELINE_FILES:
        text = path.read_text(encoding="utf-8")
        if path not in ALLOW_CONFIG:
            for label, pattern in FORBIDDEN_PATTERNS.items():
                for match in pattern.finditer(text):
                    rel = path.relative_to(SERVER_ROOT)
                    failures.append(f"{rel}:{_line_number(text, match.start())}: {label}: {match.group(0)!r}")

    for path, tokens in REQUIRED_USAGES.items():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(SERVER_ROOT)
        for token in tokens:
            if token not in text:
                failures.append(f"{rel}: expected config-driven usage token missing: {token}")

    if failures:
        print("INGESTION STATIC CHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("INGESTION STATIC CHECK PASSED")
    print(f"  scanned_files={len(PIPELINE_FILES)}")
    print(f"  required_policy_values={len(REQUIRED_CONFIG_NAMES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
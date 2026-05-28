"""inspect_data_format tool — preview a few rows to understand structure before writing SQL.

Bound to the FULL catalog (not just the retrieval shortlist) so the LLM can
preview any file the user has access to — including files surfaced via
search_catalog. Files with cached sample_rows return instantly; files
without cached samples are probed with a bounded SQL LIMIT against the
parquet/CSV path.
"""
from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.duckdb_client import execute_query_sync as _duckdb_execute
from app.core.datafusion_client import execute_query_sync as _datafusion_execute
from app.core.logger import pipeline_logger
from app.services.file_identity import FileIdentityMap
from app.services.promotion_state import mark_data_inspected, mark_schema_inspected


def _execute(sql: str, connection_string: str, container_name: str | None, max_rows: int) -> tuple:
    if get_settings().QUERY_ENGINE == "datafusion":
        return _datafusion_execute(sql, connection_string, max_rows=max_rows, container_name=container_name)
    return _duckdb_execute(sql, connection_string, max_rows=max_rows)


def _sql_path(blob_path: str, parquet_paths: dict[str, str], container: str) -> str:
    if parquet_paths and blob_path in parquet_paths:
        return f"read_parquet('az://{container}/{parquet_paths[blob_path]}')"
    sample_rows = max(1, int(get_settings().INGEST_DUCKDB_SAMPLE_ROWS))
    return (
        f"read_csv_auto('az://{container}/{blob_path}', "
        f"sample_size={sample_rows}, null_padding=true, ignore_errors=true)"
    )


def build_sample_tool(
    catalog: list[dict],
    parquet_paths: dict[str, str] | None = None,
    container_name: str | None = None,
    connection_string: str | None = None,
    file_identities: FileIdentityMap | None = None,
    state_store: dict | None = None,
) -> list:
    """Return the inspect_data_format tool bound to the full request catalog.

    Args:
        catalog:           every catalog entry visible to the user (FULL list,
                           not the retrieval shortlist). Each entry must
                           carry blob_path; sample_rows is optional.
        parquet_paths:     {csv_blob_path -> parquet_blob_path} for SQL probe
                           fallback. If empty, no SQL probe is attempted.
        container_name:    Azure container for SQL probe.
        connection_string: Azure connection string for SQL probe.
    """
    catalog_by_blob: dict[str, dict] = {
        e["blob_path"]: e for e in catalog if e.get("blob_path")
    }
    parquet_paths = parquet_paths or {}

    def _resolve_blob_path(file_ref: str) -> str | None:
        if not file_ref:
            return None

        identity = file_identities.resolve_reference(file_ref) if file_identities else None
        if identity and identity.blob_path in catalog_by_blob:
            return identity.blob_path

        query = file_ref.lower().strip()
        if query.startswith("az://"):
            query = query.split("/", 3)[-1]

        for candidate in catalog_by_blob:
            if candidate.lower() == query:
                return candidate

        query_stem = os.path.splitext(query)[0]
        if not query_stem:
            return None
        for candidate in catalog_by_blob:
            cand_stem = os.path.splitext(candidate.lower())[0]
            if query_stem == cand_stem or query_stem in cand_stem:
                return candidate
        return None

    def _probe_sample_rows(resolved_blob: str, n: int) -> list[dict]:
        """Bounded SQL probe for files without cached sample_rows.

        Returns [] silently on any error so the tool can degrade gracefully.
        """
        if not (container_name and connection_string):
            return []
        try:
            sql = (
                f"SELECT * FROM {_sql_path(resolved_blob, parquet_paths, container_name)} "
                f"LIMIT {max(1, min(n, 20))}"
            )
            rows, _total = _execute(sql, connection_string, container_name, max_rows=n)
            return rows or []
        except Exception as exc:
            pipeline_logger.warning(
                "inspect_data_format_probe_failed",
                blob_path=resolved_blob,
                error=str(exc)[:200],
            )
            return []

    @tool
    def inspect_data_format(file_ref: str, n: int = 5) -> str:
        """Preview a few example rows from a specific file to understand data format,
        column names, value patterns, and date formats before writing SQL.
        Use this when you need to know what the data looks like — e.g. whether a region is
        stored as 'us-east' or 'US East', or what date format is used.
        These rows are from the beginning of the file only — do NOT use them as the answer
        to the user's question. Always run SQL on the logical table for actual results."""
        resolved_blob_path = _resolve_blob_path(file_ref)
        if not resolved_blob_path:
            available_tables = []
            if file_identities:
                available_tables = [
                    identity.sql_name
                    for identity in file_identities.prompt_identities_for_catalog(catalog[:15])
                ]
            else:
                available_tables = list(catalog_by_blob.keys())[:15]
            pipeline_logger.info(
                "inspect_data_format",
                file_ref=file_ref,
                resolved_blob_path=None,
                n=n,
                available=False,
            )
            return json.dumps({
                "error": f"Logical table '{file_ref}' not found.",
                "available_logical_tables": available_tables,
                "hint": "Pass a logical_table from search_catalog/get_file_schema before calling inspect_data_format.",
            })

        n = max(1, min(n, 20))
        entry = catalog_by_blob[resolved_blob_path]
        sample_rows = list(entry.get("sample_rows") or [])

        # File outside hydrated shortlist → probe directly.
        if not sample_rows:
            sample_rows = _probe_sample_rows(resolved_blob_path, n)

        if not sample_rows:
            pipeline_logger.info(
                "inspect_data_format",
                file_ref=file_ref,
                resolved_blob_path=resolved_blob_path,
                n=n,
                available=False,
            )
            return json.dumps({
                "error": "No sample rows available.",
                "hint": "Use inspect_column for per-column dtype/samples, or run_sql with LIMIT 5.",
            })

        rows = sample_rows[:n]
        pipeline_logger.info(
            "inspect_data_format",
            file_ref=file_ref,
            resolved_blob_path=resolved_blob_path,
            n=n,
            available=True,
            total_sample_rows=len(sample_rows),
            columns=list(rows[0].keys()) if rows else [],
            rows=rows,
        )
        identity = file_identities.identity_for_blob(resolved_blob_path) if file_identities else None
        mark_data_inspected(
            state_store,
            file_id=identity.canonical_id if identity else entry.get("file_id"),
            logical_table=identity.sql_name if identity else resolved_blob_path,
            tool="inspect_data_format",
        )
        mark_schema_inspected(
            state_store,
            file_id=identity.canonical_id if identity else entry.get("file_id"),
            logical_table=identity.sql_name if identity else resolved_blob_path,
            tool="inspect_data_format",
        )
        return json.dumps({
            "logical_table": identity.sql_name if identity else resolved_blob_path,
            "canonical_id": identity.canonical_id if identity else entry.get("file_id"),
            "format_preview": rows,
            "note": "These are example rows for understanding data format only. Use run_sql with the logical table for real answers.",
        }, default=str)

    return [inspect_data_format]

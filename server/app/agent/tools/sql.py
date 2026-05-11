"""SQL execution tool — run_sql.

Key properties:
  - Uses Parquet path when available (10-50x faster than CSV)
  - Synchronous — runs inside LangGraph's thread pool, no event loop needed
  - Results capped at 1000 rows server-side with truncation warning
"""
from __future__ import annotations

import json
import time

from langchain_core.tools import tool

from app.core.duckdb_client import execute_query_sync as _duckdb_execute
from app.core.datafusion_client import execute_query_sync as _datafusion_execute
from app.core.config import get_settings
from app.core.logger import chat_logger, pipeline_logger
from app.core import metrics
from app.agent.tools.sql_safety import validate_and_normalise


def _execute(sql: str, connection_string: str, container_name: str, max_rows: int) -> tuple:
    """Route to DataFusion or DuckDB based on QUERY_ENGINE config flag."""
    if get_settings().QUERY_ENGINE == "datafusion":
        return _datafusion_execute(sql, connection_string, max_rows=max_rows, container_name=container_name)
    return _duckdb_execute(sql, connection_string, max_rows=max_rows)


def build_sql_tools(
    connection_string: str,
    container_name: str,
    parquet_blob_path: str | None,
    state_store: dict,
    allowed_blob_paths: set[str] | None = None,
) -> list:
    """Return SQL tools bound to connection context.

    allowed_blob_paths: every az:// path the user is authorised to query for
    this request.  Derived from the catalog shortlist in graph.py and used
    by validate_and_normalise to close the prompt-injection gap.
    """

    @tool
    def run_sql(sql: str) -> str:
        """Execute a DuckDB SQL query against Azure Blob Storage files.
        The file paths and column names are in the system prompt — use them directly.
        Parquet syntax: read_parquet('az://CONTAINER/filename.parquet')
        CSV syntax:     read_csv_auto('az://CONTAINER/filename.csv')
        Use TRY_CAST for date columns. Results are capped at 20 rows server-side.
        Returns row_count, total_rows, column names, and all result rows (up to 20).
        If total_rows > 20 the query returned more data — refine with WHERE/LIMIT/GROUP BY."""
        try:
            sql = validate_and_normalise(sql, allowed_blob_paths=allowed_blob_paths)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

        sql_upper = sql.upper()

        # ── Log the complete SQL before execution ──────────────────────────────
        pipeline_logger.info("sql_execute_start", sql=sql)

        t_exec = time.perf_counter()
        try:
            rows, total = _execute(sql, connection_string, container_name, max_rows=20)
            duration_ms = round((time.perf_counter() - t_exec) * 1000, 2)

            # ── Log full result: columns + first 20 rows + timing ──────────────
            pipeline_logger.info(
                "sql_execute_done",
                sql=sql,
                duration_ms=duration_ms,
                rows_returned=len(rows),
                total_rows=total,
                columns=list(rows[0].keys()) if rows else [],
                preview_rows=rows[:20],  # first 20 rows in the log
            )

            chat_logger.info("run_sql_result",
                             sql_preview=sql[:300],
                             rows_returned=len(rows),
                             total_rows=total,
                             duration_ms=duration_ms)

            state_store["sql_results"] = rows
            state_store["sql_total_rows"] = total
            resp: dict = {
                "row_count": len(rows),
                "total_rows": total,
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
            }
            if total > len(rows):
                resp["warning"] = (
                    f"Results truncated: showing {len(rows)} of {total} total rows. "
                    "Add a LIMIT, WHERE, or GROUP BY to get complete results."
                )
            # Detect failed join: SQL has JOIN but joined columns came back entirely null
            if rows and "JOIN" in sql_upper:
                all_null_cols = [
                    col for col in rows[0].keys()
                    if all(
                        row.get(col) is None or row.get(col) == ""
                        for row in rows
                    )
                ]
                if all_null_cols:
                    resp["join_warning"] = (
                        f"JOIN produced 0 matches: columns {all_null_cols} are entirely null. "
                        "The two files use incompatible ID systems — do NOT retry or recast the join. "
                        "STOP. Query the primary file alone using its own IDs, return that data, "
                        "and tell the user which columns could not be enriched and why."
                    )
            return json.dumps(resp, default=str)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t_exec) * 1000, 2)
            metrics.inc("llm_sql_failure_count")
            pipeline_logger.error(
                "sql_execute_error",
                sql=sql,
                duration_ms=duration_ms,
                error=str(exc),  # full error, no truncation
            )
            error_msg = str(exc)[:500]
            # Parquet Int64 dtype errors mean the file has empty strings in nullable
            # integer columns — a Parquet conversion issue, not a SQL logic error.
            # Give the LLM an explicit recovery hint so it tries specific columns
            # instead of retrying SELECT *.
            if any(tok in error_msg for tok in ("dtype", "Int64", "Invalid value ''")):
                return json.dumps({
                    "error": error_msg,
                    "hint": (
                        "This file has a Parquet data type issue: some nullable integer "
                        "columns (typed Int64 in the schema) contain empty strings that "
                        "cannot be read. Do NOT retry SELECT *. Instead, call "
                        "get_file_schema to identify which columns are typed Int64, "
                        "then SELECT only the non-Int64 columns you need. "
                        "The file's string, float64, datetime, and lowercase int64 "
                        "columns will work fine."
                    ),
                })
            return json.dumps({"error": error_msg})

    return [run_sql]

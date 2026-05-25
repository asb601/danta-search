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
import app.services.sql_repair as _repair
from app.services.execution_guards import ExecutionGuard, ExecutionGuardError
from app.services.sql_plan_signature import compute_plan_signature as _compute_plan_sig


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
    sql_ctx=None,  # SQLContext | None — passed to repair layer
) -> list:
    """Return SQL tools bound to connection context.

    allowed_blob_paths: every az:// path the user is authorised to query for
    this request.  Derived from the catalog shortlist in graph.py and used
    by validate_and_normalise to close the prompt-injection gap.
    """
    # Build the execution guard once per tool instantiation (per request), not once
    # per SQL call. ExecutionGuard.__init__ constructs AstValidationConfig and reads
    # ExecutionPolicy — hoisting avoids that cost on every run_sql invocation.
    _guard = ExecutionGuard()

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

        # ── Execution safety guards (pre-execution, deterministic) ─────────────
        # Checks: SQL length, JOIN count, Cartesian joins, file scan count.
        # Raises ExecutionGuardError when a structural safety limit is breached.
        # These are structural checks only — no cost model, no query planning.
        try:
            _guard.check_pre_execution(sql)
        except ExecutionGuardError as ge:
            pipeline_logger.error("execution_guard_rejected", reason=str(ge)[:200], sql_preview=sql[:200])
            return json.dumps({"error": str(ge)})

        # ── Retry governance: structural plan-signature deduplication ──────────
        # Compute a logical-plan fingerprint from the normalized SQL.  If this
        # exact plan was already executed and returned 0 rows in this request,
        # skip execution and return a terminal response.  None fingerprints
        # (sqlglot unavailable or parse failure) are fail-open — never block.
        _plan_sig = _compute_plan_sig(sql)
        if _plan_sig is not None:
            _zero_sigs: set[str] = state_store.get("_zero_row_plan_sigs", set())
            if _plan_sig in _zero_sigs:
                pipeline_logger.info(
                    "sql_plan_duplicate_zero_row",
                    plan_signature=_plan_sig,
                    sql_preview=sql[:200],
                )
                return json.dumps({
                    "row_count": 0,
                    "total_rows": 0,
                    "columns": [],
                    "rows": [],
                    "terminal_note": (
                        "This query has the same logical structure as a prior query "
                        "that returned 0 rows (same source files, same join graph, "
                        "same aggregation shape, same filter predicates). "
                        "Executing it again will return the same empty result. "
                        "CONCLUDE: no matching records exist for this analytical plan. "
                        "Do NOT retry with cosmetic SQL changes (aliases, LIMIT, "
                        "column order, TRY_CAST). "
                        "If a genuinely different angle is needed, use search_catalog "
                        "to find alternative files or reformulate with different "
                        "filter logic, aggregation, or source files."
                    ),
                })

        sql_upper = sql.upper()

        # ── Log the complete SQL before execution ──────────────────────────────
        pipeline_logger.info("sql_execute_start", sql=sql)

        # ── Bounded repair loop (max 2 repair attempts after the first failure) ──
        # Tier 1: deterministic pattern rewrites (zero cost).
        # Tier 2: focused LLM call with approved joins/columns as constraints.
        # Re-validates repaired SQL through validate_and_normalise each time.
        # Never retries on data-shape errors (dtype/Int64) — those need schema
        # inspection, not SQL repair.
        _MAX_REPAIR = 2
        current_sql = sql
        last_exc: Exception | None = None
        final_rows: list | None = None
        final_total: int | None = None

        t_exec = time.perf_counter()
        for _repair_attempt in range(_MAX_REPAIR + 1):
            try:
                final_rows, final_total = _execute(
                    current_sql, connection_string, container_name, max_rows=20
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                _exc_ms = round((time.perf_counter() - t_exec) * 1000, 2)
                metrics.inc("llm_sql_failure_count")
                pipeline_logger.error(
                    "sql_execute_error",
                    sql=current_sql,
                    duration_ms=_exc_ms,
                    error=str(exc),
                    repair_attempt=_repair_attempt,
                )
                # Parquet dtype errors require schema inspection, not SQL repair;
                # skip the repair loop and surface the hint directly.
                if any(tok in str(exc) for tok in ("dtype", "Int64", "Invalid value ''")):
                    break
                if _repair_attempt < _MAX_REPAIR:
                    repaired = _repair.attempt_repair(
                        current_sql, exc, sql_ctx, attempt_number=_repair_attempt
                    )
                    if repaired and repaired != current_sql:
                        try:
                            repaired = validate_and_normalise(
                                repaired, allowed_blob_paths=allowed_blob_paths
                            )
                            pipeline_logger.info(
                                "sql_repair_retry",
                                repair_attempt=_repair_attempt + 1,
                                sql=repaired,
                            )
                            current_sql = repaired
                            continue
                        except ValueError:
                            pass  # repair introduced forbidden keywords — stop
                break  # no repair possible

        if last_exc is not None:
            error_msg = str(last_exc)[:500]
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

        rows, total = final_rows, final_total  # type: ignore[assignment]
        duration_ms = round((time.perf_counter() - t_exec) * 1000, 2)
        # Refresh sql_upper in case the SQL was repaired during the loop
        sql_upper = current_sql.upper()

        # ── Record 0-row plan signature for retry governance ───────────────────
        # Use the final (possibly repaired) SQL for the recorded signature so
        # that the exact plan that ran — not the original pre-repair plan — is
        # what gets deduplicated on future attempts.
        if total == 0:
            _final_sig = _compute_plan_sig(current_sql)
            if _final_sig is not None:
                state_store.setdefault("_zero_row_plan_sigs", set()).add(_final_sig)
                pipeline_logger.debug("sql_zero_row_plan_recorded", plan_signature=_final_sig)

        # ── Post-execution guard: large result set warning ─────────────────────
        _exec_warning = _guard.check_post_execution(rows, total)

        # ── Log full result: columns + first 20 rows + timing ─────────────────
        pipeline_logger.info(
            "sql_execute_done",
            sql=current_sql,
            duration_ms=duration_ms,
            rows_returned=len(rows),
            total_rows=total,
            columns=list(rows[0].keys()) if rows else [],
            preview_rows=rows[:20],
        )

        chat_logger.info("run_sql_result",
                         sql_preview=current_sql[:300],
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
        if _exec_warning:
            resp["execution_warning"] = _exec_warning
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

    return [run_sql]

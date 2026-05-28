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
from app.services.file_identity import FileIdentityMap
from app.services.logical_sql import SQLCanonicalizationError, canonicalize_logical_sql
from app.services.promotion_state import (
    PromotionRequiredError,
    promoted_physical_uris,
    require_sql_promotion,
)
from app.services.schema_sql_validator import (
    TableSchema,
    overlay_inspected_schemas,
    validate_logical_sql_schema,
)
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
    file_identities: FileIdentityMap | None = None,
    allowed_file_ids: set[str] | None = None,
    sql_ctx=None,  # SQLContext | None — passed to repair layer
    schema_index: dict[str, TableSchema] | None = None,
) -> list:
    """Return SQL tools bound to connection context.

    file_identities / allowed_file_ids: canonical request-local identity map.
    The LLM submits logical SQL; this tool resolves logical table names to
    canonical file IDs and only then emits physical executor SQL.

    allowed_blob_paths is retained as a secondary invariant after canonical
    resolution so a repaired/internal SQL string still cannot scan outside the
    current request's authorized physical URIs.
    """
    # Build the execution guard once per tool instantiation (per request), not once
    # per SQL call. ExecutionGuard.__init__ constructs AstValidationConfig and reads
    # ExecutionPolicy — hoisting avoids that cost on every run_sql invocation.
    _guard = ExecutionGuard()

    def _active_allowed_blob_paths() -> set[str] | None:
        promoted = promoted_physical_uris(state_store, file_identities)
        if promoted is not None:
            return promoted
        return allowed_blob_paths

    @tool
    def run_sql(sql: str) -> str:
        """Execute read-only SQL against logical tables from the current catalog.
        Use logical table names in FROM/JOIN clauses, for example:
        SELECT * FROM ORDERS
        Do not use physical storage functions, storage URIs, blob paths, or parquet filenames.
        Runtime resolves logical tables to canonical file IDs and physical storage internally.
        Use TRY_CAST for date columns. Results are capped at 20 rows server-side.
        Returns row_count, total_rows, column names, and all result rows (up to 20).
        If total_rows > 20 the query returned more data — refine with WHERE/LIMIT/GROUP BY."""
        _attempt: dict = {
            "logical_sql": sql,
            "status": "started",
            "referenced_file_ids": [],
        }
        state_store.setdefault("sql_attempts", []).append(_attempt)
        schema_warnings: list[dict] = []

        if file_identities is not None:
            try:
                logical_sql = sql
                canonical = canonicalize_logical_sql(
                    logical_sql,
                    file_identities,
                    allowed_file_ids=allowed_file_ids,
                )
                _attempt.update({
                    "status": "canonicalized",
                    "referenced_file_ids": canonical.referenced_file_ids,
                    "referenced_tables": canonical.referenced_tables,
                    "physical_uris": canonical.physical_uris,
                })
                try:
                    require_sql_promotion(state_store, canonical.referenced_file_ids)
                except PromotionRequiredError as pe:
                    error_msg = str(pe)
                    _attempt.update({"status": "promotion_required", "error": error_msg})
                    state_store["execution_failure"] = {
                        "status": "promotion_required",
                        "error": error_msg,
                    }
                    return json.dumps({
                        "error": error_msg,
                        "fatal_execution_error": True,
                        "promotion_required": True,
                    })

                active_schema_index = overlay_inspected_schemas(
                    schema_index,
                    state_store.get("schema_columns_by_file_id"),
                )
                schema_report = validate_logical_sql_schema(
                    logical_sql,
                    file_identities,
                    active_schema_index,
                    allowed_file_ids=allowed_file_ids,
                    sql_ctx=sql_ctx,
                )
                if schema_report.warnings:
                    schema_warnings = [issue.to_dict() for issue in schema_report.warnings[:6]]
                    _attempt["schema_warnings"] = schema_warnings
                if not schema_report.ok:
                    error_payload = schema_report.to_error_payload()
                    _attempt.update({
                        "status": "schema_validation_error",
                        "error": error_payload["error"],
                        "schema_issues": error_payload.get("issues", []),
                    })
                    state_store["execution_failure"] = {
                        "status": "schema_validation_error",
                        "error": error_payload["error"],
                        "issues": error_payload.get("issues", []),
                    }
                    return json.dumps(error_payload)

                sql = canonical.executable_sql
            except SQLCanonicalizationError as ve:
                error_msg = str(ve)
                _attempt.update({"status": "authorization_error", "error": error_msg})
                state_store["execution_failure"] = {
                    "status": "authorization_error",
                    "error": error_msg,
                }
                metrics.inc("sql_blob_acl_denied")
                return json.dumps({"error": error_msg, "fatal_execution_error": True})

        try:
            sql = validate_and_normalise(sql, allowed_blob_paths=_active_allowed_blob_paths())
        except ValueError as ve:
            error_msg = str(ve)
            _attempt.update({"status": "validation_error", "error": error_msg})
            state_store["execution_failure"] = {
                "status": "validation_error",
                "error": error_msg,
            }
            return json.dumps({"error": error_msg, "fatal_execution_error": True})

        # ── Execution safety guards (pre-execution, deterministic) ─────────────
        # Checks: SQL length, JOIN count, Cartesian joins, file scan count.
        # Raises ExecutionGuardError when a structural safety limit is breached.
        # These are structural checks only — no cost model, no query planning.
        try:
            _guard.check_pre_execution(sql)
        except ExecutionGuardError as ge:
            pipeline_logger.error("execution_guard_rejected", reason=str(ge)[:200], sql_preview=sql[:200])
            error_msg = str(ge)
            _attempt.update({"status": "guard_error", "error": error_msg})
            state_store["execution_failure"] = {
                "status": "guard_error",
                "error": error_msg,
            }
            return json.dumps({"error": error_msg, "fatal_execution_error": True})

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
        _attempt["executable_sql"] = sql

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
                                repaired, allowed_blob_paths=_active_allowed_blob_paths()
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
            _attempt.update({"status": "engine_error", "error": error_msg})
            state_store["execution_failure"] = {
                "status": "engine_error",
                "error": error_msg,
            }
            if any(tok in error_msg for tok in ("dtype", "Int64", "Invalid value ''")):
                return json.dumps({
                    "error": error_msg,
                    "fatal_execution_error": True,
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
            return json.dumps({"error": error_msg, "fatal_execution_error": True})

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
        state_store.pop("execution_failure", None)
        _attempt.update({
            "status": "success",
            "rows": len(rows),
            "total_rows": total,
        })
        if _attempt.get("referenced_file_ids"):
            used = state_store.setdefault("files_used", [])
            for file_id in _attempt["referenced_file_ids"]:
                if file_id not in used:
                    used.append(file_id)
        resp: dict = {
            "row_count": len(rows),
            "total_rows": total,
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        }
        if _exec_warning:
            resp["execution_warning"] = _exec_warning
        if schema_warnings:
            resp["schema_warnings"] = schema_warnings
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

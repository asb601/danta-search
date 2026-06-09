"""SQL execution tool — run_sql.

Key properties:
  - Uses Parquet path when available (10-50x faster than CSV)
  - Synchronous — runs inside LangGraph's thread pool, no event loop needed
  - Results capped at 1000 rows server-side with truncation warning
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.tools import tool

from app.core.duckdb_client import execute_query_sync as _duckdb_execute
from app.core.datafusion_client import execute_query_sync as _datafusion_execute
from app.core.config import get_settings
from app.core.logger import chat_logger, pipeline_logger
from app.core import metrics
from app.agent.tools.sql_safety import validate_and_normalise
import app.services.sql_repair as _repair
from app.services.execution_guards import (
    ExecutionGuard,
    ExecutionGuardError,
    check_joins_approved,
    check_fanout_risk,
    distinct_projection_note,
)
from app.services.file_identity import FileIdentityMap
from app.services.logical_sql import (
    SQLCanonicalizationError,
    SQLAuthorizationError,
    SQLParseError,
    canonicalize_logical_sql,
)
from app.services.sql_plan_signature import compute_plan_signature as _compute_plan_sig

try:
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp as sg_exp
    _SQLGLOT_AVAILABLE = True
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    sg_exp = None  # type: ignore[assignment]
    _SQLGLOT_AVAILABLE = False


def _execute(
    sql: str,
    connection_string: str,
    container_name: str,
    max_rows: int,
    engine: str | None = None,
) -> tuple:
    """Route to DataFusion or DuckDB.

    Defaults to the QUERY_ENGINE config flag. Callers may force an engine: the
    LLM's free-form run_sql forces DuckDB because DataFusion is identifier-CASE-
    SENSITIVE (a bare `AMOUNT_APPLIED` resolves to lowercase and fails on the
    uppercase parquet schema), whereas DuckDB is case-insensitive and tolerates
    the model's natural SQL. The deterministic seam emits fully-quoted SQL and
    keeps the fast default engine."""
    eng = engine or get_settings().QUERY_ENGINE
    if eng == "datafusion":
        return _datafusion_execute(sql, connection_string, max_rows=max_rows, container_name=container_name)
    return _duckdb_execute(sql, connection_string, max_rows=max_rows)


def _norm(value: str) -> str:
    return (value or "").strip().strip('`"[]').lower()


def _schema_gate_payload(
    logical_sql: str,
    referenced_file_ids: list[str],
    referenced_tables: list[str],
    state_store: dict,
    file_identities: FileIdentityMap,
) -> dict[str, Any] | None:
    inspected: dict = state_store.get("_inspected_schemas", {}) or {}
    missing = [
        (fid, table)
        for fid, table in zip(referenced_file_ids, referenced_tables)
        if fid not in inspected
    ]
    if missing:
        return {
            "error": "Schema inspection required before run_sql.",
            "schema_required": True,
            "required_schema_tools": [
                {"tool": "get_file_schema", "file_ref": table}
                for _, table in missing
            ],
            "hint": "Call get_file_schema for every table in the SQL, then retry using only columns shown by the schema tool.",
        }

    if not _SQLGLOT_AVAILABLE:
        return _business_semantic_guard(logical_sql)

    try:
        tree = sqlglot.parse_one(
            logical_sql,
            dialect="duckdb",
            error_level=sqlglot.errors.ErrorLevel.RAISE,
        )
    except Exception:
        return _business_semantic_guard(logical_sql)

    cte_names = {
        _norm(getattr(cte, "alias_or_name", "") or "")
        for cte in tree.find_all(sg_exp.CTE)
    }
    qualifier_to_fid: dict[str, str] = {}
    for table in tree.find_all(sg_exp.Table):
        table_name = table.name
        if not table_name or _norm(table_name) in cte_names:
            continue
        try:
            identity = file_identities.resolve_table(table_name)
        except Exception:
            continue
        qualifier_to_fid[_norm(table_name)] = identity.canonical_id
        alias = table.alias_or_name
        if alias:
            qualifier_to_fid[_norm(alias)] = identity.canonical_id

    output_aliases = {
        _norm(alias.alias)
        for alias in tree.find_all(sg_exp.Alias)
        if getattr(alias, "alias", None)
    }
    schema_cols = {
        fid: set((inspected.get(fid) or {}).get("columns") or set())
        for fid in referenced_file_ids
    }
    unknown: list[dict[str, str]] = []
    for column in tree.find_all(sg_exp.Column):
        col_name = column.name
        if not col_name or col_name == "*":
            continue
        col_key = _norm(col_name)
        if col_key in output_aliases:
            continue
        qualifier = _norm(column.table or "")
        if qualifier:
            fid = qualifier_to_fid.get(qualifier)
            if fid and col_key not in schema_cols.get(fid, set()):
                unknown.append({
                    "table": (inspected.get(fid) or {}).get("logical_table") or qualifier,
                    "column": col_name,
                })
        elif len(referenced_file_ids) == 1:
            fid = referenced_file_ids[0]
            if col_key not in schema_cols.get(fid, set()):
                unknown.append({
                    "table": (inspected.get(fid) or {}).get("logical_table") or referenced_tables[0],
                    "column": col_name,
                })
        elif not any(col_key in cols for cols in schema_cols.values()):
            unknown.append({"table": "<unqualified>", "column": col_name})

    if unknown:
        return {
            "error": "SQL references columns that were not present in inspected schemas.",
            "schema_validation_error": True,
            "unknown_columns": unknown[:12],
            "hint": "Rewrite the SQL using only exact column names returned by get_file_schema. Do not use semantic role labels as SQL column names.",
        }

    return _business_semantic_guard(logical_sql)


def _business_semantic_guard(logical_sql: str) -> dict[str, Any] | None:
    sql = " ".join((logical_sql or "").split())
    lowered = sql.lower()
    aliases = {
        re.sub(r"[^a-z0-9]", "", alias.lower())
        for alias in re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)
    }
    checks = (
        (r"approved\w*\s*=\s*'?(?:y|yes|true|approved)'?", ("pending", "open", "waiting"), "approved status is positive evidence, not pending/open evidence"),
        (r"delivered\w*\s*=\s*'?(?:y|yes|true|delivered)'?", ("notdelivered", "pending", "issue", "problem"), "delivered status is positive evidence, not a delivery issue"),
        (r"matched\w*\s*=\s*'?(?:y|yes|true|matched)'?", ("unmatched", "mismatch", "issue", "problem"), "matched status is positive evidence, not an invoice issue"),
        (r"paid\w*\s*=\s*'?(?:y|yes|true|paid)'?", ("unpaid", "pending", "open", "issue"), "paid status is positive evidence, not an unpaid/open issue"),
        (r"cleared\w*\s*=\s*'?(?:y|yes|true|cleared)'?", ("uncleared", "open", "pending"), "cleared status is positive evidence, not open/uncleared evidence"),
    )
    for condition_pattern, bad_alias_tokens, message in checks:
        if not re.search(condition_pattern, lowered, flags=re.IGNORECASE):
            continue
        if any(any(token in alias for token in bad_alias_tokens) for alias in aliases):
            return {
                "error": "SQL alias conflicts with status semantics.",
                "semantic_validation_error": True,
                "hint": f"{message}. Use the opposite/null/not-equal condition for that facet, or report that the facet cannot be verified from inspected columns.",
            }
    return None


def build_sql_tools(
    connection_string: str,
    container_name: str,
    parquet_blob_path: str | None,
    state_store: dict,
    allowed_blob_paths: set[str] | None = None,
    file_identities: FileIdentityMap | None = None,
    allowed_file_ids: set[str] | None = None,
    sql_ctx=None,  # SQLContext | None — passed to repair layer
    contract: dict | None = None,  # Danta Semantic Contract — GATE B dry-plan
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

        # ── GATE B: dry-plan against the contract (logical SQL, pre-rewrite) ──
        # Validates declared joins / exposed columns BEFORE execution. Default
        # OFF; enable with SEMANTIC_CONTRACT_DRY_PLAN_ENABLED=true. Never raises.
        if contract:
            try:
                from app.core.config import get_settings as _gs  # noqa: PLC0415
                if bool(getattr(_gs(), "SEMANTIC_CONTRACT_DRY_PLAN_ENABLED", False)):
                    from app.services.contract.dry_plan import dry_plan_sql  # noqa: PLC0415

                    def _resolve_fid(name: str):
                        if file_identities is None:
                            return None
                        try:
                            return file_identities.resolve_table(name).canonical_id
                        except Exception:
                            return None

                    _verdict = dry_plan_sql(sql, contract, resolve_file_id=_resolve_fid)
                    if not _verdict.ok:
                        _payload = _verdict.to_error_payload()
                        _attempt.update({"status": "dry_plan_rejected", "error": _payload["error"]})
                        pipeline_logger.info("gate_b_dry_plan_reject", violations=_verdict.violations)
                        return json.dumps(_payload)
            except Exception as _dp_exc:
                pipeline_logger.warning("gate_b_dry_plan_error", error=str(_dp_exc)[:200])

        canonical = None
        if file_identities is not None:
            try:
                canonical = canonicalize_logical_sql(
                    sql,
                    file_identities,
                    allowed_file_ids=allowed_file_ids,
                )
                _attempt.update({
                    "status": "canonicalized",
                    "referenced_file_ids": canonical.referenced_file_ids,
                    "referenced_tables": canonical.referenced_tables,
                    "physical_uris": canonical.physical_uris,
                })
                gate_payload = _schema_gate_payload(
                    canonical.logical_sql,
                    canonical.referenced_file_ids,
                    canonical.referenced_tables,
                    state_store,
                    file_identities,
                )
                if gate_payload:
                    _attempt.update({
                        "status": "schema_gate_rejected",
                        "error": gate_payload.get("error", "schema gate rejected"),
                    })
                    return json.dumps(gate_payload)

                # ── Relationship-graph join enforcement (SME, flag-gated) ──────
                # CLAUDE.md rule #3: joins must be relationship-validated. The
                # structural ExecutionGuard below only checks JOIN shape (count,
                # cartesian, scan cap); it cannot tell an approved pair from a
                # fabricated one. When SME join-enforce is on, reject any JOIN
                # whose table pair is not in the approved relationship set the
                # context already carries. Default OFF → byte-identical behavior.
                # Checked on LOGICAL SQL (table names intact, pre-physical rewrite).
                # Fail-OPEN: an unexpected error here must never crash run_sql on
                # the hot path — on any exception we log and fall through to the
                # existing structural guards + execution (mirrors GATE-B).
                try:
                    _settings = get_settings()
                    if (
                        getattr(_settings, "SME_MODE_ENABLED", False)
                        and getattr(_settings, "SME_JOIN_ENFORCE_ENABLED", False)
                    ):
                        _approved = getattr(sql_ctx, "approved_joins", None) or []
                        _join_report = check_joins_approved(
                            canonical.logical_sql, _approved, file_identities
                        )
                        if not _join_report.ok:
                            _pair = _join_report.unapproved_pair or ("?", "?")
                            _pair_str = f"{_pair[0]} ⋈ {_pair[1]}"
                            pipeline_logger.info(
                                "sme_join_enforce_reject",
                                unapproved_pair=list(_pair),
                                checked_joins=_join_report.checked_joins,
                                approved_pairs=_join_report.approved_pair_count,
                                sql_preview=canonical.logical_sql[:200],
                            )
                            metrics.inc("sme_join_enforce_rejected")
                            _attempt.update({
                                "status": "join_not_approved",
                                "error": f"join_not_approved: {_pair_str}",
                            })
                            return json.dumps({
                                "error": "join_not_approved",
                                "detail": (
                                    f"{_pair_str} is not a validated relationship; "
                                    "analyze the tables independently or use an approved "
                                    "join path. Call extract_relations to see the approved "
                                    "join paths for these tables."
                                ),
                            })
                except Exception as _je:
                    pipeline_logger.warning("sme_join_enforce_error", error=str(_je)[:200])

                # ── Fan-out guard (always on, structural) ──────────────────────
                # Summing additive measures from BOTH sides of a join on a
                # non-unique key inflates totals via cartesian fan-out (the I11
                # $86M→$2.5B class). Block and instruct the LLM to pre-aggregate
                # each table to the join grain before joining. Checked on LOGICAL
                # SQL. Fail-OPEN: any error here falls through to execution.
                try:
                    _fanout = check_fanout_risk(canonical.logical_sql)
                    if not _fanout.ok:
                        _ft = _fanout.offending_tables or ("?", "?")
                        pipeline_logger.info(
                            "fanout_guard_reject",
                            offending_tables=list(_ft),
                            sql_preview=canonical.logical_sql[:200],
                        )
                        metrics.inc("fanout_guard_rejected")
                        _attempt.update({
                            "status": "fanout_risk",
                            "error": f"fanout_risk: {_ft[0]} ⋈ {_ft[1]}",
                        })
                        return json.dumps({
                            "error": "fanout_risk",
                            "detail": (
                                f"Summing additive measures from BOTH {_ft[0]} and "
                                f"{_ft[1]} across their join inflates the totals — each "
                                "matching row is multiplied by the other table's matching "
                                "rows. Pre-aggregate EACH table to the join key in its own "
                                f"CTE first, e.g. WITH a AS (SELECT <key>, SUM(<measure>) "
                                f"FROM {_ft[0]} GROUP BY <key>), b AS (SELECT <key>, "
                                f"SUM(<measure>) FROM {_ft[1]} GROUP BY <key>) SELECT ... "
                                "FROM a JOIN b USING(<key>)."
                            ),
                        })
                except Exception as _fe:
                    pipeline_logger.warning("fanout_guard_error", error=str(_fe)[:200])

                sql = canonical.executable_sql
            except SQLCanonicalizationError as ve:
                error_msg = str(ve)
                # Type the failure truthfully. A parse/dialect error is the model's
                # fault, NOT an access-control denial — surfacing it as an auth
                # error misleads the user (and the ACL metric). Only a genuine
                # authorization failure increments the blob-ACL metric.
                if isinstance(ve, SQLAuthorizationError):
                    status = "authorization_error"
                    metrics.inc("sql_blob_acl_denied")
                elif isinstance(ve, SQLParseError):
                    status = "sql_parse_error"
                else:
                    status = "sql_resolution_error"
                _attempt.update({"status": status, "error": error_msg})
                state_store["execution_failure"] = {
                    "status": status,
                    "error": error_msg,
                }
                return json.dumps({"error": error_msg, "fatal_execution_error": True})

        try:
            sql = validate_and_normalise(sql, allowed_blob_paths=allowed_blob_paths)
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
        # Count distinct LOGICAL tables (not partition paths) so a multi-month
        # logical table's partition fan-out isn't mistaken for cross-domain scan.
        _logical_table_count = (
            len(set(canonical.referenced_tables)) if canonical is not None else None
        )
        try:
            _guard.check_pre_execution(sql, logical_table_count=_logical_table_count)
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
                    current_sql, connection_string, container_name, max_rows=20,
                    # Force DuckDB for the LLM's free-form SQL: case-insensitive
                    # identifiers, so the model's natural `AMOUNT_APPLIED` works
                    # against the uppercase parquet schema (DataFusion would fail
                    # it). The deterministic seam keeps the configured engine.
                    engine="duckdb",
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
        if total > len(rows):
            resp["warning"] = (
                f"Results truncated: showing {len(rows)} of {total} total rows. "
                "Add a LIMIT, WHERE, or GROUP BY to get complete results."
            )
        # Distinct-vs-row clarification: a multi-column SELECT DISTINCT's row count
        # is distinct TUPLES, not distinct entities (the I02 '2,422 vendors' trap).
        try:
            _logical = getattr(canonical, "logical_sql", None) if canonical is not None else None
            _dnote = distinct_projection_note(_logical) if _logical else None
            if _dnote:
                resp["distinct_semantics"] = _dnote
        except Exception:
            pass
        # 0-row diagnostic: when an aggregation/filter returns nothing, point the
        # model back at its OWN filters on the SAME tables before it abandons them
        # for a different (often wrong-domain) table. The most common cause is an
        # out-of-range date window or a guessed status/category literal that does
        # not exist in the data — both fixable here, not by switching tables.
        if total == 0 and not rows:
            _has_date_filter = "BETWEEN" in sql_upper or bool(
                re.search(r"(>=|<=|>|<)\s*'?\d{4}-\d{2}-\d{2}", current_sql)
            )
            _has_literal_eq = bool(re.search(r"=\s*'[^']+'", current_sql))
            if _has_date_filter or _has_literal_eq:
                resp["zero_row_diagnostic"] = (
                    "0 rows. Before switching to a different table or domain, diagnose "
                    "your OWN query on the SAME table(s): "
                    "(1) if you filtered a date range, run SELECT MIN(<datecol>), MAX(<datecol>) "
                    "to confirm the window overlaps the data — the data may simply not cover "
                    "that period; "
                    "(2) if you equated a column to a text literal, SELECT DISTINCT that column "
                    "to confirm the value exists (never invent values like 'Shipped'); "
                    "(3) drop your most specific filter and re-run. "
                    "A 0-row aggregation is NOT permission to join to an unrelated business domain."
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

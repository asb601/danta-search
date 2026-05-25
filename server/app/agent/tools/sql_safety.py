"""SQL safety layer — validate and normalise SQL before it reaches any query engine.

Called by both run_sql (sql.py) and inspect_column (column.py).

Rules:
  1. Reject any SQL containing DML / DDL keywords that could mutate state.
  2. Auto-inject LIMIT 10000 when no LIMIT is present (prevents runaway scans).
  3. (Optional) Reject any az:// blob path not in the caller-supplied allowlist —
     closes the prompt-injection gap where a malicious instruction could direct
     the LLM to query files outside the user's authorized catalog.
"""
from __future__ import annotations

import re

from app.core import metrics
from app.core.logger import chat_logger
from app.services.sql_ast_validator import (
    AstValidationConfig,
    AstValidatorMode,
    SQLGLOT_AVAILABLE,
    validate_sql_ast,
)

# AstValidationConfig for the sql_safety layer.
# Only the statement_type check matters here — thresholds are not enforced
# at this layer (execution_guards.py handles them).  Mode is SHADOW so the
# AST result never blocks on its own; the regex _FORBIDDEN_RE remains
# authoritative at this layer.
_SQL_SAFETY_AST_CFG = AstValidationConfig(
    max_joins        = 999,   # not enforced here
    max_scan_files   = 999,   # not enforced here
    allow_cross_join = True,  # not enforced here
    mode             = AstValidatorMode.SHADOW,
)

# ── Forbidden token patterns ──────────────────────────────────────────────────
# Each entry is a word that, when found as a standalone SQL token (word boundary),
# causes the query to be rejected outright. Upper-case — we compare against
# sql.upper() before matching.
_FORBIDDEN: tuple[str, ...] = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "COPY",
    "ATTACH",
    "DETACH",
    "EXEC",
    "EXECUTE",
    "PRAGMA",
    "VACUUM",
    "CHECKPOINT",
    "LOAD",
    "INSTALL",
    "CALL",
)

# Compiled once — match as whole word so "CREATED_AT" doesn't trip "CREATE"
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN) + r")\b"
)


# ── az:// path extractor ──────────────────────────────────────────────────────
# Extracts every az://container/... path that appears inside single-quoted
# string literals in the SQL.  We normalise by stripping trailing whitespace /
# closing quote so the comparison is robust.
_AZ_PATH_RE = re.compile(r"az://[^\s'\"]+", re.IGNORECASE)


def _extract_az_paths(sql: str) -> list[str]:
    return _AZ_PATH_RE.findall(sql)


def _ast_shadow_statement_check(sql: str) -> None:
    """
    Run the AST validator in shadow mode for statement-type detection.

    Emits structured telemetry and logs disagreements between the regex
    forbidden-keyword check and the AST structural parse.  Never raises —
    shadow mode only.  Called only when SQLGLOT_AVAILABLE is True.

    Disagreement cases of interest:
      regex_deny=True,  ast_deny=False  — regex false positive (e.g. a column
          named "EXECUTE_DATE" tripping the EXECUTE keyword).
      regex_deny=False, ast_deny=True   — AST caught a non-SELECT that the
          regex list doesn't include (unlikely but worth knowing).
    """
    report = validate_sql_ast(sql, _SQL_SAFETY_AST_CFG)
    if not report.parse_ok:
        return  # Can't compare — skip

    ast_is_select = report.statement_type == "Select"
    regex_passed  = not bool(_FORBIDDEN_RE.search(sql.upper()))

    if ast_is_select != regex_passed:
        chat_logger.warning(
            "validator_disagreement",
            layer           = "sql_safety",
            ast_is_select   = ast_is_select,
            regex_passed    = regex_passed,
            statement_type  = report.statement_type,
            sql_fingerprint = report.sql_fingerprint,
        )


def validate_and_normalise(
    sql: str,
    allowed_blob_paths: set[str] | None = None,
) -> str:
    """Validate SQL and return a (possibly modified) safe version.

    Args:
        sql:                Raw SQL from the LLM tool call.
        allowed_blob_paths: If supplied, every az:// path in the SQL must appear
                            in this set.  Pass the catalog's authorised paths to
                            close the prompt-injection gap.  None = skip check.

    Raises ValueError with a human-readable message on any violation.
    Returns the SQL with LIMIT injected if missing.
    """
    sql = sql.strip()
    if not sql:
        raise ValueError("Empty SQL query.")

    match = _FORBIDDEN_RE.search(sql.upper())
    if match:
        metrics.inc("sql_forbidden_count")
        raise ValueError(
            f"SQL contains forbidden keyword '{match.group(1)}'. "
            "Only SELECT queries are allowed."
        )

    # ── AST statement-type shadow check ──────────────────────────────────────
    # Run in SHADOW mode — regex above is still authoritative at this layer.
    # Emits "validator_ast_result" telemetry and logs "validator_disagreement"
    # if the AST sees a non-SELECT that the regex keyword list missed (or the
    # opposite: regex fires on a benign column name, AST knows it's a SELECT).
    if SQLGLOT_AVAILABLE:
        _ast_shadow_statement_check(sql)

    # ── Blob path allowlist check ─────────────────────────────────────────────
    if allowed_blob_paths is not None:
        for path in _extract_az_paths(sql):
            if path not in allowed_blob_paths:
                metrics.inc("sql_blob_acl_denied")
                raise ValueError(
                    f"Blob path '{path}' is not in the authorised file list for "
                    "this request. Only paths from the current catalog may be queried."
                )

    # Auto-inject LIMIT — prevent unbounded full-table scans
    sql_upper = sql.upper()
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + " LIMIT 10000"

    return sql

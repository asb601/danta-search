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

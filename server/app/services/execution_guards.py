"""Deterministic execution safety guards.

PURPOSE:
  Prevent runaway SQL execution before a query reaches the engine.
  All checks are structural/syntactic — no cost-model AI, no query planning,
  no I/O.  Fast (<1ms), deterministic, and never raises on the happy path.

GUARDS:
  1. MAX_SQL_LENGTH      — rejects queries that are implausibly long.
     Long SQL is usually a sign of runaway prompt construction or LLM drift.

  2. MAX_JOINS           — rejects queries with too many explicit JOINs.
     Each JOIN multiplies potential scan size. More than 5 in a single query
     is almost always an error in an enterprise analytics context.

  3. CARTESIAN_JOIN      — rejects explicit CROSS JOINs and implicit Cartesian
     products (comma-separated FROM list with no WHERE/ON filter).
     Cartesian joins are O(N²) and will OOM small VMs.

  4. MAX_ESTIMATED_SCAN  — rejects FROM clauses referencing more unique files
     than the configured limit (does NOT count via stats; counts az:// mentions).

  5. MAX_RESULT_ROWS     — called post-execution: emits a warning when the
     engine returned more rows than expected (data shape guard, not hard reject).

DESIGN CONSTRAINTS:
  - Hard-reject guards raise ExecutionGuardError (caller returns error to LLM).
  - Post-execution guards return a warning string or None — no exceptions.
  - All limits are config-driven (ExecutionGuardConfig) so they can be overridden
    per deployment via environment without touching code.
  - No SQL parsing: regex + counting only. sqlglot is the migration target if
    more precise structural analysis is ever needed.

INTEGRATION (sql.py):
  Before _execute():
      guard = get_default_guard()
      guard.check_pre_execution(sql)   ← raises ExecutionGuardError if unsafe
  After _execute():
      warn = guard.check_post_execution(rows, total)
      if warn: resp["execution_warning"] = warn
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logger import chat_logger
from app.policies.execution_policy import get_execution_policy as _get_execution_policy


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Explicit JOIN keywords (INNER, LEFT, RIGHT, FULL, CROSS, JOIN)
_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)

# Explicit CROSS JOIN
_CROSS_JOIN_RE = re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE)

# az:// file references — each unique path counts as one scanned file
_AZ_PATH_RE = re.compile(r"az://[^\s'\"]+", re.IGNORECASE)

# FROM clause with comma (possible implicit Cartesian)
# Matches: FROM a, b  /  FROM a AS x, b AS y  — i.e. comma inside FROM before WHERE/JOIN
_FROM_COMMA_RE = re.compile(
    r"\bFROM\b[^;]*?,",
    re.IGNORECASE,
)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ExecutionGuardConfig:
    """
    All guard thresholds in one place.

    Defaults are appropriate for a single-VM deployment with limited RAM.
    Override per-deployment by constructing with different values.
    """
    max_sql_length: int     = 8_000    # characters
    max_joins: int          = 5        # explicit JOIN keywords
    max_scan_files: int     = 8        # unique az:// paths in FROM
    max_result_rows: int    = 2_000    # post-execution soft warning threshold
    allow_cross_join: bool  = False    # CROSS JOIN always forbidden by default


# Singleton default — built from ExecutionPolicy so all guard limits share
# one source of truth with the executor and agent bounds.
# Direct instantiation of ExecutionGuardConfig() still works for custom configs.
_ep = _get_execution_policy()
_DEFAULT_CONFIG = ExecutionGuardConfig(
    max_sql_length  = _ep.max_sql_length,
    max_joins       = _ep.max_joins,
    max_scan_files  = _ep.max_scan_files,
    max_result_rows = _ep.max_result_rows,
    allow_cross_join= _ep.allow_cross_join,
)


def get_default_guard() -> "ExecutionGuard":
    return ExecutionGuard(_DEFAULT_CONFIG)


# ── Exception ──────────────────────────────────────────────────────────────────

class ExecutionGuardError(ValueError):
    """
    Raised by ExecutionGuard.check_pre_execution() when a safety limit is breached.

    The error message is safe to surface to the LLM as a tool error response —
    it explains the problem and suggests corrective action without exposing
    internal infrastructure details.
    """


# ── Guard ──────────────────────────────────────────────────────────────────────

class ExecutionGuard:
    """
    Applies all pre-execution and post-execution safety checks to a SQL query.

    Usage:
        guard = ExecutionGuard(config)
        guard.check_pre_execution(sql)       # raises ExecutionGuardError if unsafe
        rows, total = engine.execute(sql)
        warning = guard.check_post_execution(rows, total)
    """

    def __init__(self, config: ExecutionGuardConfig | None = None) -> None:
        self._cfg = config or _DEFAULT_CONFIG

    # ── Pre-execution (structural, raises on violation) ────────────────────────

    def check_pre_execution(self, sql: str) -> None:
        """
        Run all structural safety checks before execution.

        Raises ExecutionGuardError on the first violation found.
        All checks are O(N) regex + counting — no query planning.
        """
        self._check_sql_length(sql)
        self._check_cross_join(sql)
        self._check_from_comma(sql)
        self._check_join_count(sql)
        self._check_scan_files(sql)

    def _check_sql_length(self, sql: str) -> None:
        if len(sql) > self._cfg.max_sql_length:
            chat_logger.warning(
                "execution_guard_sql_too_long",
                length=len(sql),
                limit=self._cfg.max_sql_length,
            )
            raise ExecutionGuardError(
                f"SQL is too long ({len(sql)} chars, limit {self._cfg.max_sql_length}). "
                "Simplify the query — break it into smaller parts, reduce inline literals, "
                "or use WITH clauses to organise complexity."
            )

    def _check_cross_join(self, sql: str) -> None:
        if not self._cfg.allow_cross_join and _CROSS_JOIN_RE.search(sql):
            chat_logger.warning("execution_guard_cross_join_detected", sql_preview=sql[:200])
            raise ExecutionGuardError(
                "CROSS JOIN detected. Cartesian products are not permitted — they produce "
                "O(N²) result sets that will exhaust memory. Replace with an explicit "
                "JOIN ... ON condition."
            )

    def _check_from_comma(self, sql: str) -> None:
        """
        Detect implicit Cartesian products: FROM a, b with no WHERE join condition.

        We only reject if the FROM comma pattern is present WITHOUT an ON or WHERE
        that bridges the tables — a rough heuristic, but safe. False positives
        (comma in subquery) are unlikely to matter: those queries should use CTEs.
        """
        if _FROM_COMMA_RE.search(sql):
            sql_upper = sql.upper()
            # If there's an ON clause or a WHERE with an equality predicate, assume
            # the comma form is intentional and filtered. Otherwise reject.
            has_join_cond = " ON " in sql_upper or (" WHERE " in sql_upper and "=" in sql_upper)
            if not has_join_cond:
                chat_logger.warning("execution_guard_implicit_cartesian", sql_preview=sql[:200])
                raise ExecutionGuardError(
                    "Implicit Cartesian join detected (comma-separated FROM without a "
                    "WHERE/ON join condition). This will produce a full cross-product. "
                    "Use explicit JOIN ... ON syntax instead."
                )

    def _check_join_count(self, sql: str) -> None:
        join_count = len(_JOIN_RE.findall(sql))
        if join_count > self._cfg.max_joins:
            chat_logger.warning(
                "execution_guard_too_many_joins",
                join_count=join_count,
                limit=self._cfg.max_joins,
            )
            raise ExecutionGuardError(
                f"Query contains {join_count} JOIN operations (limit {self._cfg.max_joins}). "
                "Reduce the number of joined tables. If multiple analyses are needed, "
                "run them as separate queries."
            )

    def _check_scan_files(self, sql: str) -> None:
        paths = set(_AZ_PATH_RE.findall(sql))
        if len(paths) > self._cfg.max_scan_files:
            chat_logger.warning(
                "execution_guard_too_many_scan_files",
                file_count=len(paths),
                limit=self._cfg.max_scan_files,
                paths=list(paths)[:10],
            )
            raise ExecutionGuardError(
                f"Query references {len(paths)} files (limit {self._cfg.max_scan_files}). "
                "Split into multiple focused queries, one per analytical domain."
            )

    # ── Post-execution (soft warning, never raises) ────────────────────────────

    def check_post_execution(self, rows: list, total: int) -> str | None:
        """
        Check result shape after execution. Returns a warning string or None.

        Never raises — post-execution guards are informational.
        """
        try:
            if total > self._cfg.max_result_rows:
                chat_logger.warning(
                    "execution_guard_large_result",
                    total_rows=total,
                    limit=self._cfg.max_result_rows,
                )
                return (
                    f"Query returned {total:,} rows — this is a very large result set. "
                    "Consider adding WHERE, GROUP BY, or LIMIT to narrow the scope. "
                    "Large result sets slow down response generation."
                )
        except Exception:
            pass
        return None

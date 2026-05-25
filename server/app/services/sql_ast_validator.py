"""AST-based SQL validator using sqlglot.

PURPOSE
-------
Replace heuristic regex-based SQL validation with parse-tree inspection.
Every check walks the sqlglot AST instead of matching text patterns, so
commas in GROUP BY / SELECT lists / function arguments and multiline ON
clauses never cause false positives.

MIGRATION STATUS — Phase 1 (dual-run)
--------------------------------------
The AST validator runs alongside the existing regex ExecutionGuard.

  Mode = PRIMARY  (default):
    AST result is authoritative when sqlglot parses successfully.
    Regex checks run as shadow validators for comparison telemetry only.
    If sqlglot fails to parse (unsupported syntax, malformed SQL), the
    regex guard takes over as the safety fallback — no query is left
    unchecked.

  Mode = SHADOW:
    AST runs and emits telemetry but never blocks execution.
    Regex remains authoritative.  Use to observe before promoting.

  Mode = DISABLED:
    AST completely bypassed.  Regex only.

  Mode is controlled by ExecutionGuardConfig.ast_mode ("primary" |
  "shadow" | "disabled").

STRUCTURAL CHECKS (in order)
-----------------------------
  1. statement_type  — Non-SELECT (INSERT/UPDATE/DROP/CREATE/etc.) → deny
  2. cartesian_join  — Comma join or any JOIN without ON/USING → deny
  3. cross_join      — Explicit CROSS JOIN (when allow_cross_join=False) → deny
  4. join_count      — JOIN count exceeds policy limit → deny
  5. scan_files      — Unique az:// blob paths exceed policy limit → deny

WHY AST BEATS REGEX FOR THESE CHECKS
--------------------------------------
  • Multiline ON:  `\nON col = col`  →  regex " ON " misses it; AST sees it.
  • GROUP BY commas: regex scans past GROUP BY and matches commas there;
    AST stops at the correct clause boundary.
  • read_parquet('az://...'): regex sees the comma inside the function
    argument; AST walks string literal nodes directly.
  • CTEs: regex can't track paren depth; AST gives a proper tree.
  • Subqueries: regex matches across scope boundaries; find_all() stays scoped.

TELEMETRY EVENT — "validator_ast_result"
-----------------------------------------
  {
    "validator":        "sqlglot_ast",
    "decision":         "allow" | "deny",
    "parse_ok":         true | false,
    "statement_type":   "Select" | "Insert" | ...,
    "table_count":      2,
    "join_count":       1,
    "has_cartesian":    false,
    "has_cross_join":   false,
    "has_on_clause":    true,
    "scan_file_count":  2,
    "cte_count":        0,
    "sql_fingerprint":  "a1b2c3d4e5f6",
    "deny_check":       "cartesian_join",   # only when denied
    "deny_reason":      "...",              # only when denied
    "parse_error":      "...",              # only when parse failed
  }

DISAGREEMENT TELEMETRY — "validator_disagreement"
--------------------------------------------------
  Emitted whenever the AST decision differs from the regex decision on the
  same structural check.  These events are the primary signal for tracking
  false positive / false negative candidates during the migration window.

  {
    "ast_deny":         ["cartesian_join"],
    "regex_deny":       [],
    "validator_used":   "ast",
    "sql_fingerprint":  "...",
  }
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp as sg_exp

    SQLGLOT_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    SQLGLOT_AVAILABLE = False  # type: ignore[assignment]

from app.core.logger import chat_logger


# ── Operation mode ─────────────────────────────────────────────────────────────

class AstValidatorMode(str, Enum):
    PRIMARY  = "primary"   # AST is authoritative; regex is comparison shadow
    SHADOW   = "shadow"    # AST runs but never blocks; regex is authoritative
    DISABLED = "disabled"  # AST completely bypassed


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class AstValidationConfig:
    """
    Validation thresholds for the AST validator.

    Mirror ExecutionGuardConfig so both validators enforce the same limits
    from the same source of truth (ExecutionPolicy).
    """
    max_joins:        int              = 5
    max_scan_files:   int              = 8
    allow_cross_join: bool             = False
    mode:             AstValidatorMode = AstValidatorMode.PRIMARY


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class AstFinding:
    """Result of a single AST check."""
    check:    str
    decision: str                     # "allow" | "deny"
    reason:   str | None    = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AstValidationReport:
    """
    Full structured report from one AST validation pass.

    Callers use .decision and .deny_finding to decide whether to block;
    .to_telemetry() produces the event dict for structlog emission.
    """
    sql_fingerprint: str
    parse_ok:        bool
    parse_error:     str | None        = None
    dialect:         str               = "duckdb"
    findings:        list[AstFinding]  = field(default_factory=list)

    # Structural summary (all fields emitted in telemetry)
    statement_type:     str | None = None
    table_count:        int        = 0
    join_count:         int        = 0
    has_cartesian_join: bool       = False
    has_cross_join:     bool       = False
    has_on_clause:      bool       = False
    scan_file_count:    int        = 0
    cte_count:          int        = 0

    @property
    def decision(self) -> str:
        """Overall decision — deny if any finding is a denial."""
        return "deny" if any(f.decision == "deny" for f in self.findings) else "allow"

    @property
    def deny_finding(self) -> AstFinding | None:
        """First denial finding, or None if all checks passed."""
        return next((f for f in self.findings if f.decision == "deny"), None)

    def to_telemetry(self) -> dict[str, Any]:
        """Flat dict suitable for structlog keyword arguments."""
        d: dict[str, Any] = {
            "validator":        "sqlglot_ast",
            "decision":         self.decision,
            "parse_ok":         self.parse_ok,
            "dialect":          self.dialect,
            "statement_type":   self.statement_type,
            "table_count":      self.table_count,
            "join_count":       self.join_count,
            "has_cartesian":    self.has_cartesian_join,
            "has_cross_join":   self.has_cross_join,
            "has_on_clause":    self.has_on_clause,
            "scan_file_count":  self.scan_file_count,
            "cte_count":        self.cte_count,
            "sql_fingerprint":  self.sql_fingerprint,
        }
        if self.parse_error:
            d["parse_error"] = self.parse_error
        if df := self.deny_finding:
            d["deny_check"]  = df.check
            d["deny_reason"] = df.reason
        return d


# ── Public entry point ─────────────────────────────────────────────────────────

def validate_sql_ast(sql: str, config: AstValidationConfig) -> AstValidationReport:
    """
    Parse and validate SQL using the sqlglot AST.

    Always returns an AstValidationReport — never raises.  Callers inspect
    .parse_ok and .decision and act based on the configured AstValidatorMode.

    Args:
        sql:    Raw SQL string (may be multiline, DuckDB dialect, with CTEs).
        config: Validation thresholds and operating mode.
    """
    fingerprint = _fingerprint(sql)
    report = AstValidationReport(
        sql_fingerprint=fingerprint,
        parse_ok=False,
        dialect="duckdb",
    )

    if not SQLGLOT_AVAILABLE:
        report.parse_error = "sqlglot not installed — regex fallback active"
        return report

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        stmt = sqlglot.parse_one(
            sql,
            dialect="duckdb",
            error_level=sqlglot.errors.ErrorLevel.RAISE,
        )
    except Exception as exc:
        report.parse_error = _truncate(str(exc), 300)
        chat_logger.warning(
            "ast_validator_parse_error",
            sql_fingerprint=fingerprint,
            error=report.parse_error,
        )
        return report

    report.parse_ok = True

    # ── Ordered checks ────────────────────────────────────────────────────────
    # 1. Statement type — short-circuit on non-SELECT before other checks.
    _check_statement_type(stmt, report)
    if report.decision == "deny":
        return report

    # 2 + 3. Join safety (cartesian and cross join — a single AST walk).
    _check_join_safety(stmt, report, config)

    # 4. Join count.
    _check_join_count(stmt, report, config)

    # 5. Scan file count.
    _check_scan_files(stmt, report, config)

    return report


# ── Internal checks ────────────────────────────────────────────────────────────

def _fingerprint(sql: str) -> str:
    return hashlib.md5(sql.encode(), usedforsecurity=False).hexdigest()[:12]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _check_statement_type(stmt: Any, report: AstValidationReport) -> None:
    """
    Block non-SELECT statements (INSERT, UPDATE, DELETE, DROP, CREATE, etc.).

    sqlglot parses `WITH cte AS (...) SELECT ...` as exp.Select with a
    `with_` arg — CTEs are correctly allowed.  Any non-Select node type is
    a DDL/DML statement that must be rejected.
    """
    t = type(stmt).__name__
    report.statement_type = t

    if isinstance(stmt, sg_exp.Select):
        report.findings.append(AstFinding(check="statement_type", decision="allow"))
        return

    report.findings.append(AstFinding(
        check    = "statement_type",
        decision = "deny",
        reason   = (
            f"Non-SELECT statement '{t}' is not permitted. "
            "Only SELECT queries are allowed."
        ),
        metadata = {"statement_type": t},
    ))


def _check_join_safety(
    stmt: Any,
    report: AstValidationReport,
    config: AstValidationConfig,
) -> None:
    """
    Walk every JOIN node in the full AST (including inside subqueries and CTEs)
    and classify each as safe or Cartesian.

    sqlglot AST join taxonomy for this check:
      kind=''     on=True/using=True  → explicit join  (SAFE)
      kind=''     on=False            → comma/implicit  (CARTESIAN)
      kind='LEFT' on=True             → explicit join  (SAFE)
      kind='CROSS'                    → CROSS JOIN     (blocked unless allowed)
      kind=''     natural=True        → NATURAL JOIN   (safe — implicit key match)

    The WHERE-equality heuristic is preserved from the regex validator:
    if there are comma joins but the query has a WHERE clause with equality
    predicates, we allow it — `FROM a, b WHERE a.id = b.id` is semantically
    equivalent to an equijoin and is common in legacy SQL.
    """
    all_joins = list(stmt.find_all(sg_exp.Join))
    report.join_count = len(all_joins)

    cartesian_findings: list[AstFinding] = []

    for join in all_joins:
        kind       = (join.args.get("kind") or "").upper()
        has_on     = join.args.get("on")      is not None
        has_using  = join.args.get("using")   is not None
        is_natural = join.args.get("natural") is not None

        if has_on or has_using or is_natural:
            report.has_on_clause = True
            continue  # Explicit condition — safe

        if kind == "CROSS":
            report.has_cross_join = True
            if not config.allow_cross_join:
                report.has_cartesian_join = True
                cartesian_findings.append(AstFinding(
                    check    = "cross_join",
                    decision = "deny",
                    reason   = (
                        "CROSS JOIN detected. Cartesian products are not permitted — "
                        "they produce O(N²) result sets that will exhaust memory. "
                        "Replace with an explicit JOIN ... ON condition."
                    ),
                    metadata = {"join_kind": "CROSS"},
                ))
        else:
            # JOIN with no ON, no USING, not NATURAL — implicit Cartesian
            # (includes comma joins: SELECT * FROM a, b)
            report.has_cartesian_join = True
            cartesian_findings.append(AstFinding(
                check    = "cartesian_join",
                decision = "deny",
                reason   = (
                    f"Implicit Cartesian join ('{kind or 'comma'}' join without "
                    "ON or USING condition) will produce a full cross-product that "
                    "exhausts memory. Use explicit JOIN ... ON syntax instead."
                ),
                metadata = {"join_kind": kind or "comma"},
            ))

    if cartesian_findings:
        # WHERE-equality fallback: FROM a, b WHERE a.id = b.id is conventional
        # equijoin syntax — allow it, matching the original regex validator.
        where_node = stmt.args.get("where")
        has_eq_filter = bool(where_node and where_node.find(sg_exp.EQ))
        if has_eq_filter:
            report.has_cartesian_join = False  # corrected — WHERE filters it
            report.findings.append(AstFinding(
                check    = "cartesian_join",
                decision = "allow",
                reason   = "Comma join allowed — WHERE clause contains equality predicate.",
                metadata = {"where_filtered": True},
            ))
        else:
            report.findings.extend(cartesian_findings)
    else:
        report.findings.append(AstFinding(
            check    = "cartesian_join",
            decision = "allow",
            metadata = {"join_count": len(all_joins)},
        ))

    # ── Structural summary (telemetry only) ────────────────────────────────
    report.table_count = len({
        t.name for t in stmt.find_all(sg_exp.Table) if t.name
    })
    cte_node = stmt.args.get("with_")
    report.cte_count = len(cte_node.expressions) if cte_node else 0


def _check_join_count(
    stmt: Any,
    report: AstValidationReport,
    config: AstValidationConfig,
) -> None:
    """Reject queries exceeding the policy JOIN limit."""
    # report.join_count already populated by _check_join_safety
    if report.join_count > config.max_joins:
        report.findings.append(AstFinding(
            check    = "join_count",
            decision = "deny",
            reason   = (
                f"Query contains {report.join_count} JOIN operations "
                f"(limit {config.max_joins}). Reduce the number of joined tables. "
                "If multiple analyses are needed, run them as separate queries."
            ),
            metadata = {"join_count": report.join_count, "limit": config.max_joins},
        ))
    else:
        report.findings.append(AstFinding(
            check    = "join_count",
            decision = "allow",
            metadata = {"join_count": report.join_count},
        ))


def _check_scan_files(
    stmt: Any,
    report: AstValidationReport,
    config: AstValidationConfig,
) -> None:
    """
    Count distinct az:// blob paths referenced in the query.

    Walks all string literal nodes in the entire AST — correctly captures
    read_parquet('az://...'), read_csv('az://...'), and any future DuckDB
    table-valued function arguments without regex boundary issues.
    """
    az_paths: set[str] = {
        lit.this
        for lit in stmt.find_all(sg_exp.Literal)
        if lit.is_string and lit.this.lower().startswith("az://")
    }
    report.scan_file_count = len(az_paths)

    if len(az_paths) > config.max_scan_files:
        report.findings.append(AstFinding(
            check    = "scan_files",
            decision = "deny",
            reason   = (
                f"Query references {len(az_paths)} files "
                f"(limit {config.max_scan_files}). "
                "Split into multiple focused queries, one per analytical domain."
            ),
            metadata = {"file_count": len(az_paths), "limit": config.max_scan_files},
        ))
    else:
        report.findings.append(AstFinding(
            check    = "scan_files",
            decision = "allow",
            metadata = {"file_count": len(az_paths)},
        ))

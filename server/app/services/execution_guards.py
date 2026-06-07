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
  - Structural checks (JOIN safety) use AST-based validation via sqlglot
    (sql_ast_validator.py).  Regex checks remain as fallback when sqlglot
    cannot parse the SQL.  See sql_ast_validator.py for the migration status
    and dual-run telemetry schema.

MIGRATION STATUS:
  Phase 1 (current): AST validator is PRIMARY for structural checks when
  sqlglot parses successfully.  Regex checks run as shadow validators;
  disagreements are logged as "validator_disagreement" events.
  If sqlglot fails to parse, regex guards remain the safety fallback.

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
from app.core.config import get_settings as _get_settings
from app.policies.execution_policy import get_execution_policy as _get_execution_policy
from app.services.sql_ast_validator import (
    AstValidationConfig,
    AstValidationReport,
    AstValidatorMode,
    SQLGLOT_AVAILABLE,
    validate_sql_ast,
)

try:
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp as _sg_exp
    _SQLGLOT_IMPORT_OK = True
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    _sg_exp = None  # type: ignore[assignment]
    _SQLGLOT_IMPORT_OK = False


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Explicit JOIN keywords (INNER, LEFT, RIGHT, FULL, CROSS, JOIN)
_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)

# Explicit CROSS JOIN
_CROSS_JOIN_RE = re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE)

# az:// file references — each unique path counts as one scanned file
_AZ_PATH_RE = re.compile(r"az://[^\s'\"]+", re.IGNORECASE)

# FROM clause with comma (possible implicit Cartesian).
# Matches: FROM a, b  /  FROM a AS x, b AS y
#
# IMPORTANT — negative lookahead stops the scan at any SQL clause keyword:
#   WHERE / JOIN (any variant) / GROUP BY / HAVING / ORDER BY / LIMIT
# Without this, [^;]*? would lazily scan past GROUP BY and match commas there,
# producing false positives on valid queries like:
#   FROM read_parquet(...) AS t LEFT JOIN ... ON t.id = u.id GROUP BY t.id, t.name
#
# MIGRATION TARGET: replace with sqlglot AST inspection (now complete —
# see sql_ast_validator.py).  Regex retained as shadow/fallback only.
_FROM_COMMA_RE = re.compile(
    r"\bFROM\b(?:(?!\b(?:WHERE|JOIN|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b)[^;])*?,",
    re.IGNORECASE,
)

# ON keyword — word-boundary safe for use in the Cartesian-join condition check.
# The space-padded " ON " string check fails when ON starts a new line
# (the LLM writes "\nON col = col", not " ON col = col").
_ON_WORD_RE = re.compile(r"\bON\b", re.IGNORECASE)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ExecutionGuardConfig:
    """
    All guard thresholds in one place.

    Defaults are appropriate for a single-VM deployment with limited RAM.
    Override per-deployment by constructing with different values.

    ast_mode controls the sqlglot AST validator:
      "primary"  — AST is authoritative for structural checks when parse
                   succeeds; regex runs as comparison shadow (default).
      "shadow"   — AST runs and emits telemetry but never blocks; regex
                   remains authoritative.  Use during initial roll-out.
      "disabled" — AST completely bypassed; regex only.
    """
    max_sql_length:  int   = 8_000    # characters
    max_joins:       int   = 5        # explicit JOIN keywords
    max_scan_files:  int   = 8        # unique az:// paths in FROM
    max_result_rows: int   = 2_000    # post-execution soft warning threshold
    allow_cross_join: bool = False    # CROSS JOIN always forbidden by default
    ast_mode:        str   = "primary"  # "primary" | "shadow" | "disabled"


# Singleton default — built from ExecutionPolicy so all guard limits share
# one source of truth with the executor and agent bounds.
# ast_mode reads from Settings.SQL_VALIDATOR_AST_MODE so it can be overridden
# at the deployment level without a code change (set env var to "shadow" or
# "disabled" for a hot rollback if AST causes unexpected issues in production).
# Direct instantiation of ExecutionGuardConfig() still works for custom configs.
_ep = _get_execution_policy()
_DEFAULT_CONFIG = ExecutionGuardConfig(
    max_sql_length  = _ep.max_sql_length,
    max_joins       = _ep.max_joins,
    max_scan_files  = _ep.max_scan_files,
    max_result_rows = _ep.max_result_rows,
    allow_cross_join= _ep.allow_cross_join,
    ast_mode        = _get_settings().SQL_VALIDATOR_AST_MODE,
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
        self._ast_cfg = AstValidationConfig(
            max_joins        = self._cfg.max_joins,
            max_scan_files   = self._cfg.max_scan_files,
            allow_cross_join = self._cfg.allow_cross_join,
            mode             = AstValidatorMode(self._cfg.ast_mode),
        )

    # ── Pre-execution (structural, raises on violation) ────────────────────────

    def check_pre_execution(self, sql: str, logical_table_count: int | None = None) -> None:
        """
        Run all structural safety checks before execution.

        Raises ExecutionGuardError on the first violation found.

        Execution order:
          1. SQL length  — character count, always regex (not SQL-structural).
          2. Structural  — AST-primary when sqlglot parses; regex fallback.
          3. Scan files  — distinct logical-table count when known, else az:// path count.

        logical_table_count, when provided by the canonicalizer, is the number of
        distinct LOGICAL tables referenced. A single logical table legitimately
        fans out to many physical partition paths (monthly/format files); the scan
        limit guards cross-domain fan-out, so it must count logical tables, not
        partitions, or it would reject every multi-period query.
        """
        self._check_sql_length(sql)
        self._run_structural_checks(sql)
        self._check_scan_files(sql, logical_table_count)

    def _run_structural_checks(self, sql: str) -> None:
        """
        Dispatch structural checks (JOIN safety, JOIN count) to the AST
        validator or the regex fallback depending on AstValidatorMode.

        PRIMARY mode (default):
          sqlglot parse succeeds → AST result is authoritative; regex runs
          in shadow mode and any disagreement is logged as
          "validator_disagreement".

          sqlglot parse fails    → regex guards run as the safety fallback;
          "ast_parse_failure_regex_fallback" is logged so the failure is
          visible in the admin log viewer.

        SHADOW mode:
          Both run; regex is authoritative; AST telemetry is emitted.

        DISABLED mode:
          AST skipped entirely; regex only.
        """
        mode = self._ast_cfg.mode

        if mode == AstValidatorMode.DISABLED:
            self._check_cross_join(sql)
            self._check_from_comma(sql)
            self._check_join_count(sql)
            return

        # Run AST validator
        ast_report: AstValidationReport = validate_sql_ast(sql, self._ast_cfg)
        # Log at DEBUG on the allow path to avoid flooding INFO in production;
        # elevate to WARNING when the validator denies or fails to parse.
        if ast_report.decision == "deny" or ast_report.parse_error:
            chat_logger.warning("validator_ast_result", **ast_report.to_telemetry())
        else:
            chat_logger.debug("validator_ast_result", **ast_report.to_telemetry())

        if ast_report.parse_ok and mode == AstValidatorMode.PRIMARY:
            # ── AST is authoritative ─────────────────────────────────────────
            # Run regex in shadow mode for comparison telemetry only.
            regex_deny: set[str] = set()
            for check_name, check_fn in [
                ("cross_join",   self._check_cross_join),
                ("from_comma",   self._check_from_comma),
                ("join_count",   self._check_join_count),
            ]:
                try:
                    check_fn(sql)
                except ExecutionGuardError:
                    regex_deny.add(check_name)

            ast_deny = {
                f.check for f in ast_report.findings if f.decision == "deny"
            }
            # Normalise check names for comparison (regex uses different names)
            ast_structural = ast_deny & {"cross_join", "cartesian_join", "join_count"}
            # "cartesian_join" from AST maps to "from_comma" in regex
            regex_normalised = {
                "cartesian_join" if n == "from_comma" else n for n in regex_deny
            }

            if ast_structural != regex_normalised:
                chat_logger.warning(
                    "validator_disagreement",
                    ast_deny    = sorted(ast_structural),
                    regex_deny  = sorted(regex_deny),
                    validator_used = "ast",
                    sql_fingerprint = ast_report.sql_fingerprint,
                )

            if ast_report.decision == "deny":
                df = ast_report.deny_finding
                raise ExecutionGuardError(df.reason)  # type: ignore[union-attr]

        elif ast_report.parse_ok and mode == AstValidatorMode.SHADOW:
            # ── Shadow mode — regex is authoritative, AST only logs ──────────
            self._check_cross_join(sql)
            self._check_from_comma(sql)
            self._check_join_count(sql)

        else:
            # ── Regex fallback — AST parse failed or DISABLED ────────────────
            if ast_report.parse_error and SQLGLOT_AVAILABLE:
                chat_logger.warning(
                    "ast_parse_failure_regex_fallback",
                    sql_fingerprint = ast_report.sql_fingerprint,
                    parse_error     = ast_report.parse_error,
                )
            self._check_cross_join(sql)
            self._check_from_comma(sql)
            self._check_join_count(sql)

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
        Detect implicit Cartesian products: FROM a, b with no explicit join condition.

        Two-stage check:
          1. _FROM_COMMA_RE must match a top-level comma in the FROM clause.
             The regex uses a negative lookahead to stop scanning at SQL clause
             keywords (WHERE, JOIN, GROUP BY, HAVING, ORDER BY, LIMIT), so commas
             in GROUP BY / HAVING / ORDER BY do NOT trigger this check.
          2. If a FROM comma is found, the query is allowed through only when an
             explicit join condition exists — either an ON clause (detected via
             \bON\b word-boundary regex, not the broken " ON " space-padded check
             which misses multiline SQL where ON starts on a new line) or a WHERE
             clause with an equality predicate.

        Known limitation: commas inside CTE definitions (WITH t1 AS (...), t2 AS (...))
        can still match _FROM_COMMA_RE because the inner FROM is at paren depth > 0 but
        the regex has no paren tracking. CTEs are handled by the has_join_cond fallback
        (they virtually always appear alongside JOIN/ON). For a fully correct solution,
        migrate to sqlglot AST inspection (see module-level migration comment).
        """
        if not _FROM_COMMA_RE.search(sql):
            return
        # A top-level comma was found before any clause keyword in the FROM section.
        # Allow if an explicit join condition exists.
        #
        # Bug fix: use \bON\b (word-boundary regex) instead of " ON " (space-padded).
        # LLM-generated multiline SQL writes the ON keyword at the start of a new line
        # ("\nON t1.id = t2.id"), producing "\nON" in sql_upper — " ON " never matches.
        has_explicit_on    = bool(_ON_WORD_RE.search(sql))
        has_filtered_where = " WHERE " in sql.upper() and "=" in sql
        if not (has_explicit_on or has_filtered_where):
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

    def _check_scan_files(self, sql: str, logical_table_count: int | None = None) -> None:
        # Count logical tables when the canonicalizer told us how many; a single
        # logical table's partition fan-out (many az:// paths) is not a violation.
        if logical_table_count is not None:
            count = logical_table_count
            unit = "logical tables"
        else:
            count = len(set(_AZ_PATH_RE.findall(sql)))
            unit = "files"
        if count > self._cfg.max_scan_files:
            chat_logger.warning(
                "execution_guard_too_many_scan_files",
                file_count=count,
                limit=self._cfg.max_scan_files,
                unit=unit,
            )
            raise ExecutionGuardError(
                f"Query references {count} {unit} (limit {self._cfg.max_scan_files}). "
                "Split into multiple focused queries, one per analytical domain."
            )

    # ── Relationship-graph join approval (additive, flag-gated) ────────────────

    def check_joins_approved(
        self,
        logical_sql: str,
        approved_joins,
        file_identities,
    ) -> "JoinApprovalReport":
        """Verify every JOIN's table pair is an approved relationship.

        Thin instance wrapper over the module-level check_joins_approved so the
        relationship guard sits alongside the structural guards on the same
        object. Stateless — no config dependency. Never raises.
        """
        return check_joins_approved(logical_sql, approved_joins, file_identities)

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


# ── Relationship-graph join approval (no fabricated joins) ──────────────────────
#
# CLAUDE.md rule #3: "Never generate arbitrary LLM joins — joins must be
# relationship-validated and ontology-backed." The structural ExecutionGuard
# above only checks JOIN *shape* (count, cartesian, scan cap); it does not know
# WHICH table pairs are a validated relationship. This guard closes that gap:
# it rejects any JOIN whose two tables are not present, as a pair, in the
# request's approved relationship set (SemanticRelationship edges surfaced via
# SQLContext.approved_joins).
#
# Fully data-driven: the approved set is the per-request graph the context
# already carries — there are NO hardcoded table names, key lists, or magic
# literals here. Identity resolution is delegated to the request-local
# FileIdentityMap so a logical-table name maps to the SAME canonical id on both
# the SQL side and the approved-edge side regardless of which partition/alias
# the model wrote.

@dataclass
class JoinApprovalReport:
    """Result of the join-approval check. Never raised — inspected by caller.

    ok                 — True when every JOIN's table pair is approved (or the
                         query has no resolvable cross-table joins to check).
    parse_ok           — False when sqlglot could not parse the SQL (fail-open:
                         the structural ExecutionGuard already ran; this guard
                         declines rather than blocking on an unparseable string).
    unapproved_pair    — ("LEFT_TABLE", "RIGHT_TABLE") display names of the first
                         join whose pair was not in the approved set (None on ok).
    checked_joins      — count of cross-table joins inspected (telemetry).
    approved_pair_count— size of the approved relationship set (telemetry).
    """
    ok: bool = True
    parse_ok: bool = True
    unapproved_pair: tuple[str, str] | None = None
    checked_joins: int = 0
    approved_pair_count: int = 0


def _approved_canonical_pairs(approved_joins, file_identities) -> set[frozenset[str]]:
    """Project the approved-join set to a set of unordered canonical-id pairs.

    Each ApprovedJoin carries left_file_id / right_file_id — these may be any
    member partition id of a logical table. We collapse both endpoints to the
    logical table's representative canonical_id via the identity map so the
    comparison is partition-agnostic. Endpoints that don't resolve are kept as
    their raw id (still a valid equality key) so a missing identity never
    silently widens approval.
    """
    pairs: set[frozenset[str]] = set()
    by_id = getattr(file_identities, "by_id", {}) or {}
    for j in approved_joins or []:
        left = getattr(j, "left_file_id", None)
        right = getattr(j, "right_file_id", None)
        if not left or not right:
            continue
        left_canon = getattr(by_id.get(left), "canonical_id", None) or left
        right_canon = getattr(by_id.get(right), "canonical_id", None) or right
        pairs.add(frozenset((left_canon, right_canon)))
    return pairs


def _resolve_canonical(table_name: str, file_identities) -> str | None:
    """Resolve a logical table name to its representative canonical id, or None."""
    try:
        return file_identities.resolve_table(table_name).canonical_id
    except Exception:
        return None


def check_joins_approved(
    logical_sql: str,
    approved_joins,
    file_identities,
) -> JoinApprovalReport:
    """Reject any JOIN whose table pair is not a validated relationship.

    Operates on LOGICAL SQL (table names still present, pre-physical-rewrite).
    For every JOIN, the joined table is paired with each table already in scope
    from the same FROM (the left side may carry multiple base tables across a
    chain of joins). Each (left, joined) canonical-id pair must be present in the
    approved set. The first pair that is absent yields ok=False.

    Data-safety contract:
      • Single-table queries (no JOIN) → ok (nothing to check).
      • Unparseable SQL → parse_ok=False, ok=True (decline; structural guard
        already ran — never hard-fail an unparseable string here).
      • A join table that cannot be resolved to a canonical id is skipped (it is
        not provably unapproved; canonicalization/auth handles unknown tables).
      • Empty approved set + a real cross-table join → ok=False with the offending
        pair, so the agent degrades to independent analysis (never crashes).

    Never raises. Returns a JoinApprovalReport for the caller to act on.
    """
    report = JoinApprovalReport()
    if not logical_sql or not _SQLGLOT_IMPORT_OK or file_identities is None:
        report.parse_ok = _SQLGLOT_IMPORT_OK
        return report

    try:
        tree = sqlglot.parse_one(
            logical_sql,
            dialect="duckdb",
            error_level=sqlglot.errors.ErrorLevel.RAISE,
        )
    except Exception:
        report.parse_ok = False
        return report

    approved_pairs = _approved_canonical_pairs(approved_joins, file_identities)
    report.approved_pair_count = len(approved_pairs)

    # CTE names are query-local aliases, not catalog tables — never treat them as
    # a logical table for join approval.
    cte_names = {
        (getattr(cte, "alias_or_name", "") or "").strip().strip('`"[]').lower()
        for cte in tree.find_all(_sg_exp.CTE)
    }

    def _is_real_table(node) -> bool:
        name = node.name
        return bool(name) and name.strip().strip('`"[]').lower() not in cte_names

    def _qual_key(value: str) -> str:
        return (value or "").strip().strip('`"[]').lower()

    # Inspect each SELECT scope independently so a join in a subquery is paired
    # only with the tables of its OWN scope, not tables from an outer scope.
    for select in tree.find_all(_sg_exp.Select):
        # sqlglot stores the FROM clause under "from" in some versions and
        # "from_" in others; accept either so the guard is version-robust.
        from_node = select.args.get("from") or select.args.get("from_")
        joins = select.args.get("joins") or []
        if not joins:
            continue

        # Scope alias/name → canonical id. A table is addressable in the ON
        # clause by its alias if it has one, else by its bare name; we register
        # both so the ON-qualifier lookup resolves either form.
        alias_to_canon: dict[str, str] = {}

        def _register(tbl) -> None:
            if not _is_real_table(tbl):
                return
            canon = _resolve_canonical(tbl.name, file_identities)
            if canon is None:
                return
            alias = tbl.alias_or_name
            alias_to_canon[_qual_key(alias)] = canon
            alias_to_canon[_qual_key(tbl.name)] = canon

        # FROM base tables resolved to canonical ids — the fallback pairing target
        # for join conditions we cannot attribute to specific tables (USING /
        # unqualified ON), so a fabricated USING-join cannot slip through.
        from_canons: list[str] = []
        if from_node is not None:
            for tbl in from_node.find_all(_sg_exp.Table):
                _register(tbl)
                if _is_real_table(tbl):
                    canon = _resolve_canonical(tbl.name, file_identities)
                    if canon:
                        from_canons.append(canon)
        for join in joins:
            for tbl in join.find_all(_sg_exp.Table):
                _register(tbl)

        for join in joins:
            join_tables = [
                t for t in join.find_all(_sg_exp.Table) if _is_real_table(t)
            ]
            if not join_tables:
                # Subquery / derived-table join with no direct catalog table —
                # its own inner SELECT scope is inspected separately by the loop.
                continue

            # The relationship a join ASSERTS is exactly the table pair its
            # ON/USING condition connects. Read the distinct table qualifiers
            # referenced in the ON condition and resolve them to canonical ids.
            on_node = join.args.get("on")
            using_node = join.args.get("using")
            cond_canons: set[str] = set()
            if on_node is not None:
                for col in on_node.find_all(_sg_exp.Column):
                    qual = _qual_key(col.table)
                    canon = alias_to_canon.get(qual) if qual else None
                    if canon:
                        cond_canons.add(canon)

            for jt in join_tables:
                right_canon = _resolve_canonical(jt.name, file_identities)
                if right_canon is None:
                    # Unknown table — not provably unapproved; auth handles it.
                    continue
                # Pair the joined table with the OTHER side(s) of its own ON
                # condition. Self-pairs (same canonical id — two partitions of one
                # logical table) are not cross-table joins and are skipped.
                partners = {c for c in cond_canons if c != right_canon}
                # Fallback: a USING clause, or an ON whose qualifiers we could not
                # attribute, still asserts a real relationship. Pair against the
                # FROM base table(s) so a fabricated join cannot bypass the gate by
                # using USING(col) or an unqualified ON.
                if not partners and (using_node is not None or on_node is not None):
                    partners = {c for c in from_canons if c != right_canon}
                for partner in sorted(partners):
                    report.checked_joins += 1
                    if frozenset((partner, right_canon)) not in approved_pairs:
                        report.ok = False
                        report.unapproved_pair = (
                            _display_name(partner, file_identities),
                            _display_name(right_canon, file_identities),
                        )
                        return report

    return report


def _display_name(canonical_id: str, file_identities) -> str:
    """Best-effort human-readable table name for an error message."""
    by_id = getattr(file_identities, "by_id", {}) or {}
    identity = by_id.get(canonical_id)
    return getattr(identity, "sql_name", None) or canonical_id

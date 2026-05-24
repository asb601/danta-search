"""AST-safe SQL transformation shim — migration path from regex to structural rewrites.

PROBLEM STATEMENT:
  sql_repair.py currently uses regex surgery for two SQL rewrites:
    1. CAST → TRY_CAST (for type conversion errors)
    2. GROUP BY column append (for binder errors)

  Regex rewrites work for simple cases but are structurally unsafe for:
    - Nested CAST expressions
    - CAST inside CASE or subquery
    - String literals containing "CAST("
    - GROUP BY modificationwithin CTEs or window functions

  The correct fix is AST-level transformations — parse the SQL, walk the
  expression tree, apply targeted mutations, regenerate.

CURRENT STATUS:
  sqlglot is NOT yet a project dependency (pyproject.toml, May 2026).
  This module provides a thin shim that:
    - Tries sqlglot if available (try-import)
    - Falls back to the existing regex approach if sqlglot is absent
    - Exposes a stable interface so sql_repair.py can import from here
      and automatically get the AST path once sqlglot is added

HOW TO ACTIVATE AST MODE:
  Add sqlglot to pyproject.toml:
    [tool.poetry.dependencies]
    sqlglot = "^23"
  Then: pip install sqlglot
  The shim detects it at import time — no code changes required.

INTERFACE:
  cast_to_try_cast(sql) -> str | None
    Return rewritten SQL (CAST→TRY_CAST) or None if no change / unsafe.

  append_groupby_column(sql, column) -> str | None
    Return rewritten SQL with column appended to GROUP BY, or None if unsafe.

  Both functions are guarded:
    - Return None rather than corrupt SQL
    - Never raise on invalid input
    - AST path validates that the transformation is structurally clean

IMPORTANT — SCOPE CONSTRAINT:
  This module ONLY handles Tier 1 deterministic rewrites that have a well-defined
  AST transformation. It does NOT:
    - Interpret query semantics
    - Generate or suggest SQL
    - Fix anything requiring business vocabulary understanding
  Those remain with LLM Tier 2 in sql_repair.py.
"""
from __future__ import annotations

import re
from app.core.logger import chat_logger

# ── Try to import sqlglot (optional dependency) ────────────────────────────────
try:
    import sqlglot
    import sqlglot.expressions as exp
    _SQLGLOT_AVAILABLE = True
except ImportError:
    sqlglot = None  # type: ignore[assignment]
    exp = None      # type: ignore[assignment]
    _SQLGLOT_AVAILABLE = False


# ── Regex fallbacks (same patterns as sql_repair.py — kept in sync) ────────────

_CAST_RE_FALLBACK = re.compile(
    r"\bCAST\s*\((.+?)\s+AS\s+(DATE|TIMESTAMP|INTEGER|INT|INT32|INT64|BIGINT|FLOAT|DOUBLE|DECIMAL)\b",
    re.IGNORECASE,
)

_GROUPBY_CLAUSE_RE_FALLBACK = re.compile(
    r"\bGROUP\s+BY\s+([\s\S]+?)(?=\s+(?:HAVING|ORDER|LIMIT|UNION|EXCEPT|INTERSECT)\b|;|\Z)",
    re.IGNORECASE,
)


# ── Public interface ───────────────────────────────────────────────────────────

def is_ast_available() -> bool:
    """Return True if sqlglot is installed and AST transforms are active."""
    return _SQLGLOT_AVAILABLE


def cast_to_try_cast(sql: str) -> str | None:
    """
    Rewrite all CAST(expr AS TYPE) to TRY_CAST(expr AS TYPE) in the SQL.

    Returns:
      str  — rewritten SQL if any CAST expressions were changed.
      None — if no CAST found, if the SQL is structurally unsafe to rewrite,
             or if any error occurs during transformation.

    AST path (sqlglot available):
      Parses the SQL, walks all Cast nodes, replaces with TryCast while
      preserving nesting and context exactly. Dialect: DuckDB.

    Regex fallback (sqlglot absent):
      Uses the flat _CAST_RE pattern. Returns None if nested CASTs are
      detected (same guard as sql_repair._has_nested_cast).
    """
    if not sql or not sql.strip():
        return None

    if _SQLGLOT_AVAILABLE:
        return _ast_cast_to_try_cast(sql)
    else:
        return _regex_cast_to_try_cast(sql)


def append_groupby_column(sql: str, column: str) -> str | None:
    """
    Append `column` to the GROUP BY clause of the SQL.

    Returns:
      str  — rewritten SQL with column appended to GROUP BY.
      None — if no GROUP BY clause found, if the column is already present,
             if the SQL cannot be safely rewritten, or on any error.

    AST path (sqlglot available):
      Parses SQL, locates Group node, appends column as a bare identifier,
      regenerates. Validates the resulting SQL parses cleanly.

    Regex fallback (sqlglot absent):
      Appends via string match on the GROUP BY clause.
    """
    if not sql or not column:
        return None

    if _SQLGLOT_AVAILABLE:
        return _ast_append_groupby(sql, column)
    else:
        return _regex_append_groupby(sql, column)


# ── AST implementations ────────────────────────────────────────────────────────

def _ast_cast_to_try_cast(sql: str) -> str | None:
    """sqlglot-based CAST → TRY_CAST rewrite."""
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
        original_sql = tree.sql(dialect="duckdb")
        changed = False

        def _transform(node: "exp.Expression") -> "exp.Expression":
            nonlocal changed
            if isinstance(node, exp.Cast):
                changed = True
                return exp.TryCast(
                    this=node.this,
                    to=node.to,
                )
            return node

        transformed = tree.transform(_transform)
        if not changed:
            return None

        result = transformed.sql(dialect="duckdb")
        # Sanity: result must still parse cleanly
        sqlglot.parse_one(result, dialect="duckdb")
        chat_logger.info("sql_ast_cast_rewrite", mode="sqlglot")
        return result

    except Exception as exc:
        chat_logger.warning("sql_ast_cast_rewrite_failed", error=str(exc)[:200])
        return None


def _ast_append_groupby(sql: str, column: str) -> str | None:
    """sqlglot-based GROUP BY append."""
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")

        select_node = tree if isinstance(tree, exp.Select) else next(
            (n for n in tree.walk() if isinstance(n, exp.Select)), None
        )
        if select_node is None:
            return None

        group_node = select_node.args.get("group")
        if group_node is None:
            return None

        existing_cols = {
            col.name.upper()
            for expr in group_node.expressions
            for col in expr.find_all(exp.Column)
        }
        if column.upper() in existing_cols:
            return None  # already present

        group_node.append("expressions", exp.Column(this=exp.Identifier(this=column)))
        result = tree.sql(dialect="duckdb")
        # Sanity: result must still parse cleanly
        sqlglot.parse_one(result, dialect="duckdb")
        chat_logger.info("sql_ast_groupby_append", mode="sqlglot", column=column)
        return result

    except Exception as exc:
        chat_logger.warning("sql_ast_groupby_append_failed", error=str(exc)[:200])
        return None


# ── Regex fallback implementations ────────────────────────────────────────────

def _has_nested_cast(sql: str) -> bool:
    """Bracket-counting nested CAST detector (same logic as sql_repair.py)."""
    cast_positions = [m.end() for m in re.finditer(r"\bCAST\s*\(", sql, re.IGNORECASE)]
    if len(cast_positions) < 2:
        return False
    for start in cast_positions:
        depth, pos = 1, start
        while pos < len(sql) and depth > 0:
            if sql[pos] == "(":
                depth += 1
            elif sql[pos] == ")":
                depth -= 1
            pos += 1
        inner = sql[start : pos - 1]
        if re.search(r"\bCAST\b", inner, re.IGNORECASE):
            return True
    return False


def _regex_cast_to_try_cast(sql: str) -> str | None:
    """Flat regex CAST→TRY_CAST with nested-CAST guard."""
    if _has_nested_cast(sql):
        chat_logger.info("sql_ast_cast_nested_declined", mode="regex_fallback")
        return None
    repaired = _CAST_RE_FALLBACK.sub(r"TRY_CAST(\1 AS \2)", sql)
    if repaired == sql:
        return None
    chat_logger.info("sql_ast_cast_rewrite", mode="regex_fallback")
    return repaired


def _regex_append_groupby(sql: str, column: str) -> str | None:
    """Flat regex GROUP BY append."""
    m = _GROUPBY_CLAUSE_RE_FALLBACK.search(sql)
    if not m:
        return None
    existing = m.group(1).rstrip()
    if column.upper() in existing.upper():
        return None  # already present
    new_gb = existing + f", {column}"
    result = sql[: m.start(1)] + new_gb + sql[m.end(1):]
    chat_logger.info("sql_ast_groupby_append", mode="regex_fallback", column=column)
    return result

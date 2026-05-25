"""Logical-plan signature for SQL retry governance.

Computes a structural fingerprint of a SQL query using sqlglot AST nodes.
Two queries with the same fingerprint are considered logically equivalent —
they reference the same source files, have the same join structure, the
same aggregation shape, the same GROUP BY columns, and the same filter
predicates (including actual predicate values — values are analytically
significant and are NOT normalized away).

Cosmetic differences that produce the SAME signature (deduplication targets):
  - Column aliases in SELECT
  - Table aliases (az:// paths stripped to file basename)
  - LIMIT / OFFSET values  (execution caps, not analytical logic)
  - ORDER BY clauses       (sorting is cosmetic for empty-result detection)
  - Column order in SELECT projection
  - Whitespace / formatting
  - TRY_CAST wrappers around column references (same column, different casting)

Meaningful differences that produce DIFFERENT signatures (allow through):
  - Different source files
  - Different join tables or join conditions
  - Different aggregate functions (COUNT vs SUM vs none)
  - Different GROUP BY columns
  - Different WHERE predicates or predicate values
  - Different HAVING conditions
  - Absence vs presence of a WHERE clause

DESIGN CONSTRAINTS:
  - Never raises — returns None on parse failure (fail-open, never blocks).
  - Uses sha1 truncated to 16 hex chars — collision-resistant for
    intra-session deduplication, not a security hash.
  - Handles both classic table references (sg_exp.Table) and DuckDB
    table-valued functions (read_parquet / read_csv_auto with az:// args).
  - CTE source files are extracted from within CTE bodies, not from CTE names.
"""
from __future__ import annotations

import hashlib
import json

try:
    import sqlglot
    import sqlglot.expressions as sg_exp
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def compute_plan_signature(sql: str) -> str | None:
    """Return a 16-char hex structural plan signature, or None on failure.

    None is returned when sqlglot is unavailable or the SQL cannot be parsed.
    None signatures are fail-open — they never block execution.
    """
    if not _AVAILABLE:
        return None
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
        if not statements or statements[0] is None:
            return None
        stmt = statements[0]
        if not isinstance(stmt, sg_exp.Select):
            return None

        plan = {
            "files":       _extract_source_files(stmt),
            "joins":       _extract_join_graph(stmt),
            "aggregates":  _extract_aggregate_functions(stmt),
            "group_by":    _extract_group_by(stmt),
            "filter":      _extract_filter(stmt),
            "having":      _extract_having(stmt),
        }
        canonical = json.dumps(plan, sort_keys=True)
        # sha1 truncated — intra-session dedup, not a security primitive
        return hashlib.sha1(canonical.encode(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324
    except Exception:  # noqa: BLE001 — signature is best-effort, never raise
        return None


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _basename(path: str) -> str:
    """Strip az:// container prefix to filename, lowercase."""
    path = path.strip("'\"").rstrip("'\"")
    if "/" in path:
        path = path.rsplit("/", 1)[-1]
    return path.lower()


def _extract_source_files(stmt: sg_exp.Select) -> list[str]:
    """All az:// source file references, normalized to basename.

    DuckDB queries use table-valued functions (read_parquet, read_csv_auto)
    rather than bare table names.  Walk all Literal nodes and collect any
    that contain 'az://' — this is robust regardless of whether the file
    reference is inside a Table node, an Anonymous function, a CTE, or a
    subquery.
    """
    files: set[str] = set()
    for node in stmt.walk():
        if isinstance(node, sg_exp.Literal) and "az://" in (node.this or ""):
            files.add(_basename(node.this))
    return sorted(files)


def _extract_join_graph(stmt: sg_exp.Select) -> list[dict]:
    """Join edges: file basename + join type + ON condition structure.

    Captures the structure of each explicit JOIN so that queries with the
    same tables but different join types (INNER vs LEFT) or different ON
    conditions produce different signatures.
    """
    joins = []
    for join in stmt.find_all(sg_exp.Join):
        join_type = (join.args.get("kind") or "INNER").upper()

        # Extract the file reference from within this JOIN's table expression.
        # The file is usually inside a Literal (read_parquet/read_csv_auto call).
        join_file = ""
        for lit in join.find_all(sg_exp.Literal):
            if "az://" in (lit.this or ""):
                join_file = _basename(lit.this)
                break

        on_expr = join.args.get("on")
        joins.append({
            "file": join_file,
            "type": join_type,
            "on":   _expr_repr(on_expr) if on_expr else None,
        })
    return sorted(joins, key=lambda j: (j["file"], j["type"]))


def _extract_aggregate_functions(stmt: sg_exp.Select) -> list[str]:
    """Names of aggregate functions present anywhere in the SELECT."""
    found: set[str] = set()
    _KNOWN_AGGS = (
        sg_exp.Count, sg_exp.Sum, sg_exp.Avg, sg_exp.Max, sg_exp.Min,
        sg_exp.ArrayAgg, sg_exp.GroupConcat,
    )
    for agg_cls in _KNOWN_AGGS:
        if stmt.find(agg_cls):
            found.add(agg_cls.__name__.upper())
    # Pick up less-common aggregates by name (PERCENTILE_CONT, MEDIAN, etc.)
    for anon in stmt.find_all(sg_exp.Anonymous):
        name = (anon.name or "").upper()
        if any(kw in name for kw in (
            "AGG", "PERCENTILE", "STDDEV", "VARIANCE", "MEDIAN", "CORR",
            "COVAR", "REGR", "APPROX",
        )):
            found.add(name)
    return sorted(found)


def _extract_group_by(stmt: sg_exp.Select) -> list[str]:
    """Sorted list of GROUP BY expression representations."""
    group = stmt.args.get("group")
    if not group:
        return []
    return sorted(_expr_repr(expr) for expr in group.expressions)


def _extract_filter(stmt: sg_exp.Select) -> str | None:
    """WHERE clause as a canonical expression string."""
    where = stmt.args.get("where")
    if not where:
        return None
    return _expr_repr(where.this)


def _extract_having(stmt: sg_exp.Select) -> str | None:
    """HAVING clause as a canonical expression string."""
    having = stmt.args.get("having")
    if not having:
        return None
    return _expr_repr(having.this)


# ── Expression canonical representation ───────────────────────────────────────

def _expr_repr(expr: "sg_exp.Expression | None") -> str:
    """Produce a canonical string representation of an expression.

    Rules:
      - Literal values are kept (predicate values are analytically significant).
      - Column names: table alias prefix stripped (t1.amount → amount).
      - Operators: preserved by type.
      - CAST / TRY_CAST: stripped — the inner expression is what matters.
      - LIMIT / OFFSET: excluded (callers do not pass these nodes).
      - Recursion is capped at depth via child truncation in the generic fallback.
    """
    if expr is None:
        return ""
    if isinstance(expr, sg_exp.Literal):
        return repr(expr.this)
    if isinstance(expr, sg_exp.Column):
        return (expr.name or "?").lower()
    if isinstance(expr, sg_exp.Null):
        return "NULL"
    if isinstance(expr, sg_exp.Boolean):
        return str(expr.this).upper()
    if isinstance(expr, sg_exp.Star):
        return "*"
    # CAST / TRY_CAST — transparent to plan structure
    if isinstance(expr, (sg_exp.Cast, sg_exp.TryCast)):
        return _expr_repr(expr.this)
    # Comparison operators
    if isinstance(expr, sg_exp.EQ):
        return f"({_expr_repr(expr.left)}={_expr_repr(expr.right)})"
    if isinstance(expr, sg_exp.NEQ):
        return f"({_expr_repr(expr.left)}!={_expr_repr(expr.right)})"
    if isinstance(expr, sg_exp.GT):
        return f"({_expr_repr(expr.left)}>{_expr_repr(expr.right)})"
    if isinstance(expr, sg_exp.LT):
        return f"({_expr_repr(expr.left)}<{_expr_repr(expr.right)})"
    if isinstance(expr, sg_exp.GTE):
        return f"({_expr_repr(expr.left)}>={_expr_repr(expr.right)})"
    if isinstance(expr, sg_exp.LTE):
        return f"({_expr_repr(expr.left)}<={_expr_repr(expr.right)})"
    # Logical operators
    if isinstance(expr, sg_exp.And):
        # Sort children so AND(a, b) == AND(b, a) — commutative
        children = sorted([_expr_repr(expr.left), _expr_repr(expr.right)])
        return f"(AND {children[0]} {children[1]})"
    if isinstance(expr, sg_exp.Or):
        children = sorted([_expr_repr(expr.left), _expr_repr(expr.right)])
        return f"(OR {children[0]} {children[1]})"
    if isinstance(expr, sg_exp.Not):
        return f"(NOT {_expr_repr(expr.this)})"
    # Set membership / range
    if isinstance(expr, sg_exp.In):
        vals = sorted(_expr_repr(v) for v in expr.expressions)
        return f"({_expr_repr(expr.this)} IN [{','.join(vals)}])"
    if isinstance(expr, sg_exp.Between):
        return (
            f"({_expr_repr(expr.this)} BETWEEN "
            f"{_expr_repr(expr.args.get('low'))} AND "
            f"{_expr_repr(expr.args.get('high'))})"
        )
    if isinstance(expr, sg_exp.Is):
        return f"({_expr_repr(expr.this)} IS {_expr_repr(expr.expression)})"
    if isinstance(expr, sg_exp.Like):
        return f"({_expr_repr(expr.this)} LIKE {_expr_repr(expr.expression)})"
    # Known aggregate functions — represent structurally
    if isinstance(expr, sg_exp.Count):
        return f"COUNT({_expr_repr(expr.this)})"
    if isinstance(expr, sg_exp.Sum):
        return f"SUM({_expr_repr(expr.this)})"
    if isinstance(expr, sg_exp.Avg):
        return f"AVG({_expr_repr(expr.this)})"
    if isinstance(expr, sg_exp.Max):
        return f"MAX({_expr_repr(expr.this)})"
    if isinstance(expr, sg_exp.Min):
        return f"MIN({_expr_repr(expr.this)})"
    # Anonymous function calls (read_parquet, custom UDFs, etc.)
    if isinstance(expr, sg_exp.Anonymous):
        name = (expr.name or type(expr).__name__).upper()
        args = [_expr_repr(a) for a in (expr.expressions or [])[:4]]
        return f"{name}({','.join(args)})"
    # Generic fallback: type name + up to 4 direct children
    children: list[str] = []
    for val in expr.args.values():
        if isinstance(val, sg_exp.Expression):
            children.append(_expr_repr(val))
        elif isinstance(val, list):
            for item in val[:3]:
                if isinstance(item, sg_exp.Expression):
                    children.append(_expr_repr(item))
        if len(children) >= 4:
            break
    return f"{type(expr).__name__}({','.join(children)})"

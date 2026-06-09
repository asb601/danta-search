"""v2 RESOLVE emitter — pure, deterministic Contract → SQL string builder.

This is the EMITTER half of the deterministic query brain. Given a fully-BOUND
``Contract`` (built by ``resolver.resolve_metric_query`` from precomputed
ingestion artifacts), it renders the canonical analytical SQL for that contract.
There is NO LLM here and no DB: ``emit_sql`` is a deterministic function of the
contract's already-resolved slots.

The single most important property this module enforces is GRAIN. The query is
GROUP-BY the grain primary key (``contract.grain_pk``) and ONLY the grain key —
never by display columns. That is what turns "open receivables over 500k" into a
per-customer aggregate (one row per CUSTOMER_ID) instead of a row-level invoice
filter. Display columns are wrapped in an aggregate (``MAX(col)``) so they ride
along without splitting the grain; the measure is the contract's measure
expression; the optional HAVING threshold is applied to the aggregate so the
"over 500k" cut is per-customer, post-aggregation — not per-row.

Design properties (enforced, not aspirational):
  * Pure Python only. No LLM, no DB, no I/O — a deterministic function of inputs.
  * No hardcoded business terms or column names. Every identifier and predicate
    comes from the contract (``grain_pk`` / ``measure`` / ``measure_expr`` /
    ``filter_preds`` / ``facts``). There are no dataset-fitted literals; the only
    constants are SQL syntax and the HAVING comparison operator the caller supplies.
  * Additive. Nothing here is wired into ``graph.py``; activation is gated by the
    ``RESOLVE_CONTRACT_ENABLED`` flag at the calling site, default False.
"""
from __future__ import annotations

from app.services.resolve.contract import Contract

# Allowed HAVING comparison operators. This is SQL syntax, not a business knob:
# the caller picks one when expressing a post-aggregation threshold ("over X").
_HAVING_OPS: frozenset[str] = frozenset({">", ">=", "<", "<=", "=", "!=", "<>"})


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier for DuckDB/DataFusion (ANSI double-quote).

    Embedded double-quotes are doubled per the SQL standard so an attacker- or
    ingestion-supplied identifier cannot break out of the quoted region. Pure
    string transform; no name lists, no business logic.
    """
    return '"' + str(name).replace('"', '""') + '"'


def emit_sql(contract: Contract) -> str:
    """Render the canonical aggregated SQL for a BOUND ``Contract``.

    Shape::

        SELECT <grain_pk...>,
               MAX(<display>) AS <display>,   -- grain-preserving, optional
               <measure_expr> AS <measure>
        FROM <source_table>
        WHERE <filter_preds AND-joined>        -- optional
        GROUP BY <grain_pk...>                  -- EXACTLY the grain key
        HAVING <measure_expr> <op> <value>      -- optional, per-group threshold
        ORDER BY <measure> DESC

    The GROUP BY is exactly ``contract.grain_pk`` so aggregation is per grain
    entity. Display columns are wrapped in ``MAX(...)`` so they never split the
    grain. Raises ``ValueError`` if the grain key is empty — a grainless
    aggregate is never emitted.
    """
    grain_pk = tuple(contract.grain_pk or ())
    if not grain_pk:
        # A grainless aggregate would collapse the whole table to one row and
        # silently destroy the per-entity answer. Refuse rather than emit it.
        raise ValueError("emit_sql: contract.grain_pk is empty; refusing to emit a grainless aggregate")

    if not contract.source_table:
        raise ValueError("emit_sql: contract.source_table is unset")
    if not contract.measure or not contract.measure_expr:
        raise ValueError("emit_sql: contract.measure / measure_expr is unset")

    measure_alias = _quote_ident(contract.measure)
    measure_expr = contract.measure_expr

    # SELECT list: grain key column(s) first, then grain-preserving display
    # columns, then the measure. Display columns come from facts (caller-supplied),
    # never inferred — and are excluded from GROUP BY so the grain stays intact.
    select_parts: list[str] = [_quote_ident(col) for col in grain_pk]

    display_cols = tuple(contract.facts.get("display_cols", ()) or ())
    grain_set = set(grain_pk)
    for col in display_cols:
        # A column that is part of the grain is already selected raw; do not also
        # emit it as an aggregate (that would duplicate the column).
        if col in grain_set:
            continue
        quoted = _quote_ident(col)
        select_parts.append(f"MAX({quoted}) AS {quoted}")

    select_parts.append(f"{measure_expr} AS {measure_alias}")

    clauses: list[str] = [
        "SELECT " + ", ".join(select_parts),
        "FROM " + _quote_ident(contract.source_table),
    ]

    # WHERE: AND-join the already-SQL filter predicate strings, if any. These are
    # opaque predicate strings supplied by the resolved metric; the emitter does
    # not parse or rewrite them.
    filter_preds = tuple(p for p in (contract.filter_preds or ()) if p)
    if filter_preds:
        clauses.append("WHERE " + " AND ".join(filter_preds))

    # GROUP BY: EXACTLY the grain key. This is the line that forces per-entity
    # aggregation (the 304-vs-404 fix). Never group by display columns.
    clauses.append("GROUP BY " + ", ".join(_quote_ident(col) for col in grain_pk))

    # HAVING: optional post-aggregation threshold on the measure expression, so
    # the "over X" cut is per grain entity, not per row. Operator + value come
    # from the caller (facts["having"]); the operator is whitelisted to SQL
    # comparators so an arbitrary string cannot be injected here.
    having = contract.facts.get("having")
    if having:
        op = str(having.get("op", "")).strip()
        if op not in _HAVING_OPS:
            raise ValueError(f"emit_sql: unsupported HAVING operator {op!r}")
        value = having.get("value")
        if value is None:
            raise ValueError("emit_sql: HAVING value is missing")
        clauses.append(f"HAVING {measure_expr} {op} {_sql_literal(value)}")

    # ORDER BY the measure descending — largest first, the conventional ranking
    # for a "top / over threshold" metric question.
    clauses.append(f"ORDER BY {measure_alias} DESC")

    return "\n".join(clauses)


def _sql_literal(value: object) -> str:
    """Render a HAVING comparison value as a safe SQL literal.

    Numerics pass through as-is; anything else is rendered as a single-quoted
    string with quotes escaped. Pure transform; no business logic.
    """
    if isinstance(value, bool):
        # bool is a subclass of int — render as TRUE/FALSE, not 1/0.
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        # Guard non-finite floats: repr(inf)/repr(nan) -> bare 'inf'/'nan',
        # which are not valid SQL literals. Reject rather than emit broken SQL.
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("emit_sql: non-finite HAVING value")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"

"""[3e] RENDER — deterministic, fully-quoted SQL from a VERIFIED contract (I10).

PURE Python. The LLM never touches this stage. Every identifier is emitted
DOUBLE-QUOTED and UPPERCASE (canonicalizer-safe — DataFusion / DuckDB are
case-sensitive and the parquet schema is uppercase), matching the ``_quote``
discipline of the legacy brain emitter so the rendered SQL is byte-identical to
what the verified path produced before.

  * ``render``       — single-table aggregate. Handles entity grain (GROUP BY id)
                       and time grain (GROUP BY year[,period]). The runtime owns
                       every identifier and all arithmetic. LIFTED from
                       ``resolve.brain.render_sql``.
  * ``render_join``  — the relationship-validated, value-verified joined row-count.
                       The PK side (from the verdict) drives the GROUP BY grain.
                       LIFTED from the joined-render block in
                       ``resolve.coordinator._resolve_join``.

The navigator is self-contained — this module imports nothing from
``app.services.resolve.*``.
"""
from __future__ import annotations

from app.services.navigator.verifier import JoinVerdict
from app.services.navigator.types import ResolvedTable, VerifiedContract

# Known SQL ops for a HAVING clause guard. SQL syntax, not a business knob.
_OPS: frozenset[str] = frozenset({"=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "ILIKE"})


def _quote(name: str) -> str:
    """ANSI double-quote an identifier (DataFusion/DuckDB are case-sensitive; the
    parquet schema is uppercase). Embedded quotes doubled. Pure."""
    return '"' + str(name).replace('"', '""') + '"'


def _sql_value(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


# A measure column may itself BE a grouping column (e.g. COUNT(VENDOR_ID) at
# VENDOR_ID grain, or COUNT(YEAR) when YEAR lowercases onto the time-bucket select
# name "year"). Aliasing such a measure to its own lowercase would emit two
# case-only-variant output identifiers ("VENDOR_ID" + "vendor_id"). This reserved
# alias avoids that collision; it is a SQL-output identifier, not a business label.
_RESERVED_MEASURE_ALIAS = "measure_value"


def _measure_alias(measure_col: str, group_names: list[str]) -> str:
    """The output alias for the measure expression. Returns ``measure_col.lower()``
    normally (byte-identical to the legacy emitter for the non-colliding case) but
    a distinct reserved alias when that lowercase case-insensitively collides with
    any GROUP BY output identifier (the entity grain column, or a time-bucket select
    name like year/month/quarter). PURE — no IO, no dataset knowledge."""
    lowered = str(measure_col).lower()
    collisions = {str(n).lower() for n in group_names}
    return _RESERVED_MEASURE_ALIAS if lowered in collisions else lowered


def render(vc: VerifiedContract, time_window: tuple | None = None) -> str:
    """Deterministic, fully-quoted SQL from a verified contract. Handles entity
    grain (GROUP BY id) and time grain (GROUP BY year[,period]). PURE — the runtime
    owns every identifier and the arithmetic; the LLM touched none of this."""
    measure_expr = (f'COUNT(DISTINCT {_quote(vc.measure_col)})' if vc.agg == "COUNT_DISTINCT"
                    else f'{vc.agg}({_quote(vc.measure_col)})')

    where: list[str] = [f'{_quote(c)} {op} {_sql_value(val)}' for c, op, val in vc.filters]
    if vc.time_col and time_window:
        where.append(f'{_quote(vc.time_col)} >= {_sql_value(str(time_window[0]))}')
        where.append(f'{_quote(vc.time_col)} <= {_sql_value(str(time_window[1]))}')

    if vc.grain_kind == "time":
        dcol = _quote(vc.grain_col)
        group_sel = [f'EXTRACT(YEAR FROM {dcol}) AS "year"']
        group_by = ['"year"']
        group_names = ["year"]
        if vc.bucket == "month":
            group_sel.append(f'EXTRACT(MONTH FROM {dcol}) AS "month"')
            group_by.append('"month"'); group_names.append("month")
        elif vc.bucket == "quarter":
            group_sel.append(f'EXTRACT(QUARTER FROM {dcol}) AS "quarter"')
            group_by.append('"quarter"'); group_names.append("quarter")
        alias = _quote(_measure_alias(vc.measure_col, group_names))
        select_parts = group_sel + [f'{measure_expr} AS {alias}']
        order_by = ", ".join(group_by)
    else:
        gcol = _quote(vc.grain_col)
        alias = _quote(_measure_alias(vc.measure_col, [vc.grain_col]))
        select_parts = [gcol, f'{measure_expr} AS {alias}']
        group_by = [gcol]
        order_by = f'{alias} {vc.order}'

    sql = [f'SELECT {", ".join(select_parts)}', f'FROM {_quote(vc.table)}']
    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append("GROUP BY " + ", ".join(group_by))
    if vc.having:
        op = str(vc.having.get("op", "")).strip()
        if op in _OPS and vc.having.get("value") is not None:
            sql.append(f'HAVING {measure_expr} {op} {_sql_value(vc.having["value"])}')
    sql.append("ORDER BY " + order_by)
    if vc.top_n:
        sql.append(f'LIMIT {int(vc.top_n)}')
    return "\n".join(sql)


def render_join(
    verdict: JoinVerdict, a: ResolvedTable, b: ResolvedTable, col_a: str, col_b: str,
) -> str:
    """Render the relationship-validated, value-verified joined row-count. The PK
    side (from ``verdict.pk_side``) drives the GROUP BY grain. LIFTED from
    ``coordinator._resolve_join``. PURE — the join came from value evidence, never
    the LLM (I7). ``col_a`` / ``col_b`` are the verified key columns on ``a`` / ``b``."""
    if verdict.pk_side == a.blob:
        pk_table, pk_col, fk_table, fk_col = a.table, col_a, b.table, col_b
    else:
        pk_table, pk_col, fk_table, fk_col = b.table, col_b, a.table, col_a
    return (
        f'SELECT {_quote(pk_table)}.{_quote(pk_col)} AS {_quote(pk_col.lower())}, '
        f'COUNT(*) AS "match_count"\n'
        f'FROM {_quote(pk_table)}\n'
        f'JOIN {_quote(fk_table)} '
        f'ON {_quote(pk_table)}.{_quote(pk_col)} = {_quote(fk_table)}.{_quote(fk_col)}\n'
        f'GROUP BY {_quote(pk_table)}.{_quote(pk_col)}\n'
        f'ORDER BY "match_count" DESC'
    )

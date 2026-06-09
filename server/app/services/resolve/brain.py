"""v2 RESOLVE brain — query-time, evidence-grounded contract proposer.

This is the missing piece between file-selection and SQL: the LLM brain placed at
the RIGHT point. It does NOT write SQL and does NOT compute numbers. Given a small,
already-retrieved slice of candidate tables and their PER-FILE EVIDENCE (schema +
semantic roles + a few real sample rows + description — all already stored at
ingestion, no cross-file conclusions), the brain (gpt-4o-mini, temp 0) makes the
SEMANTIC judgments it is good at — which table is canonical for the question, what
one output row means (grain), which column is the measure, what to filter — and
emits a TYPED contract. Every slot is then VALUE-VERIFIED against the same
evidence, and the runtime renders deterministic, fully-quoted SQL from it.

Design properties (enforced):
  * Evidence in, contract out. The brain reads only per-file evidence; it never
    sees or relies on materialized cross-file conclusions (no governed-metric
    table, no master election, no precomputed join graph).
  * The brain never emits SQL, identifiers, or arithmetic. It fills typed slots;
    ``render_sql`` quotes every identifier (DataFusion-safe) and owns the math.
  * No hardcoded business terms. Every table/column/value comes from the question
    or the evidence. The only literals are SQL syntax and the time-bucket grammar.
  * Date math is deterministic. The brain only names the date column; the relative
    window ("this year") is resolved by the caller against the data's as-of date.
  * Abstain over guess. Not answerable from the slice, or a slot that fails
    verification → return None → the caller falls through to the agent.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.openai_client import get_client
from app.core.llm_tasks import safe_parse_json
from app.models.column_key_registry import ColumnKeyRegistry
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata

logger = structlog.get_logger("resolve.brain")

_AGGS: frozenset[str] = frozenset({"SUM", "COUNT", "AVG", "MAX", "MIN", "COUNT_DISTINCT"})
_OPS: frozenset[str] = frozenset({"=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "ILIKE"})
_BUCKETS: frozenset[str] = frozenset({"month", "quarter", "year"})
_SAMPLE_ROWS = 3          # real rows per candidate — enough to read the data, cheap to send
_MAX_SLICE = 9            # match the per-entity search top_k: the brain must SEE every
                         # genuine hit (the right table can rank 6th), not a truncated 5


def _quote(name: str) -> str:
    """ANSI double-quote an identifier (DataFusion is case-sensitive; the parquet
    schema is uppercase). Embedded quotes doubled. Pure."""
    return '"' + str(name).replace('"', '""') + '"'


def _sql_value(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


async def assemble_evidence(
    db: AsyncSession, candidates: list[dict],
) -> list[dict]:
    """Build the clean evidence packet for the candidate slice.

    ``candidates`` are catalog entries (each carrying ``file_id`` and the logical
    ``table`` name the SQL must use). For each we pull the per-file evidence the
    brain reasons over: typed columns + semantic roles + a few real sample rows +
    description. No conclusions — only what ingestion stored per file.
    """
    by_file: dict[str, dict] = {}
    for c in candidates[:_MAX_SLICE]:
        fid = c.get("file_id")
        if fid:
            by_file[fid] = c
    if not by_file:
        return []
    file_ids = list(by_file.keys())
    rows = (
        await db.execute(
            select(
                FileMetadata.file_id,
                FileMetadata.columns_info,
                FileMetadata.column_semantic_roles,
                FileMetadata.sample_rows,
                FileMetadata.ai_description,
                FileMetadata.good_for,
                FileMetadata.date_range_start,
                FileMetadata.date_range_end,
                FileMetadata.row_count,
            ).where(FileMetadata.file_id.in_(file_ids))
        )
    ).all()

    # Stored value-sets (categorical top-values, sampled at ingest) + key-registry
    # unique-rates per file, used by verify() to value-check filters and sanity-check
    # the grain. Both reads are defensive: a missing/None entry means "can't
    # disprove" → verify passes (abstain-biased), never blocks.
    value_counts_by_file: dict[str, dict] = {}
    for fid, vc in (
        await db.execute(
            select(FileAnalytics.file_id, FileAnalytics.value_counts)
            .where(FileAnalytics.file_id.in_(file_ids))
        )
    ).all():
        if isinstance(vc, dict):
            value_counts_by_file[fid] = vc

    unique_rate_by_file: dict[str, dict[str, float]] = {}
    for fid, cname, urate in (
        await db.execute(
            select(
                ColumnKeyRegistry.file_id,
                ColumnKeyRegistry.column_name,
                ColumnKeyRegistry.unique_rate,
            ).where(ColumnKeyRegistry.file_id.in_(file_ids))
        )
    ).all():
        if cname is not None:
            unique_rate_by_file.setdefault(fid, {})[str(cname).upper()] = float(urate or 0.0)

    evidence: list[dict] = []
    for fid, columns_info, roles, sample_rows, descr, good_for, d0, d1, row_count in rows:
        cat = by_file.get(fid, {})
        table = cat.get("table") or cat.get("logical_name") or cat.get("display_name")
        if not table:
            continue
        roles = roles or {}
        cols: list[dict] = []
        valid: dict[str, str] = {}          # upper -> exact-case, for verification
        for ci in (columns_info or []):
            cname = ci.get("name") if isinstance(ci, dict) else str(ci)
            if not cname:
                continue
            valid[str(cname).upper()] = str(cname)
            cols.append({
                "name": cname,
                "type": (ci.get("dtype") or ci.get("type") or "") if isinstance(ci, dict) else "",
                "role": roles.get(cname, ""),
            })
        # value_counts keyed by UPPER(column) for case-insensitive lookup in verify().
        raw_vc = value_counts_by_file.get(fid) or {}
        value_set: dict[str, dict] = {
            str(k).upper(): v for k, v in raw_vc.items() if isinstance(v, dict)
        }
        evidence.append({
            "table": table,
            "valid_cols": valid,
            "columns": cols,
            "sample_rows": (sample_rows or [])[:_SAMPLE_ROWS],
            "description": (descr or "")[:400],
            "good_for": (good_for or [])[:6],
            "coverage": f"{d0 or '?'}..{d1 or '?'}",
            "row_count": int(row_count or 0),
            "value_set": value_set,                       # UPPER(col) -> {value: count}
            "unique_rates": unique_rate_by_file.get(fid, {}),  # UPPER(col) -> unique_rate
        })
    return evidence


def _evidence_prompt(question: str, evidence: list[dict], time_window: tuple | None) -> str:
    lines: list[str] = []
    for e in evidence:
        lines.append(f"TABLE {e['table']}  (rows={e['row_count']:,}, coverage {e['coverage']})")
        lines.append(f"  purpose(good_for): {e['good_for']}")
        lines.append(f"  desc: {e['description']}")
        col_strs = [f"{c['name']}:{c['type'] or '?'}" + (f"[{c['role']}]" if c['role'] else "")
                    for c in e["columns"]]
        lines.append("  columns: " + ", ".join(col_strs))
        if e["sample_rows"]:
            lines.append(f"  sample rows: {e['sample_rows']}")
        lines.append("")
    tw = ""
    if time_window:
        tw = (f"\nA relative time window was detected for this question: "
              f"{time_window[0]} to {time_window[1]} (already resolved against the "
              f"data's latest date — use this if the question is time-scoped; pick the "
              f"date column to apply it to).")
    return f"""You are the analytical BRAIN of a data agent. You DO NOT write SQL. You read the
EVIDENCE for a few candidate tables and decide how to answer the question by filling
a typed contract. Pick the SINGLE best table from the evidence (read the sample rows
and column roles — these are ERP tables and several look alike). Choose the grain
(what one output row represents), the measure column + aggregation, and any filters.
Use ONLY table names and column names that appear verbatim in the evidence.{tw}

QUESTION: {question}

EVIDENCE:
{chr(10).join(lines)}
Return ONLY JSON:
{{
  "answerable": true|false,            // false if no table here fits the question
  "table": "EXACT_TABLE_NAME",
  "table_reason": "one line: why this table over the others",
  "grain": "entity" | "time",          // entity = one row per <id>; time = per period
  "grain_column": "EXACT_COLUMN",      // the id column (entity) or the date column (time)
  "time_bucket": "month"|"quarter"|"year"|null,   // only for grain=time
  "measure_column": "EXACT_COLUMN",
  "measure_agg": "SUM"|"COUNT"|"AVG"|"MAX"|"MIN"|"COUNT_DISTINCT",
  "filters": [{{"column":"EXACT_COLUMN","op":"=","value":"..."}}],  // e.g. open status; [] if none
  "time_filter_column": "EXACT_DATE_COLUMN"|null,  // the date column the time window applies to
  "having": {{"op":">","value":500000}}|null,      // per-group threshold, else null
  "top_n": 20|null,
  "order": "desc"|"asc"
}}
Rules: choose the measure that matches the business term (e.g. cash received → the
applied/received amount, not the original invoice amount — use roles + sample values
to decide). If the question asks "by month/quarter/year", grain=time. If it asks for
customers/vendors/etc, grain=entity with that id as grain_column. Abstain
(answerable=false) rather than force a bad table.
TWINS: several tables here may share nearly identical schemas (lookalikes). When the
question wants the GENERAL fact (e.g. all open receivables, all cash received), prefer
the CANONICAL MASTER — the most complete/granular table — which is normally the one
with the largest row count and broadest coverage, NOT a narrower view (a
history/archive/delinquency/interim subset). Only choose a subset table when the
question explicitly asks for that subset."""


async def propose_contract(
    question: str, evidence: list[dict], time_window: tuple | None,
) -> dict | None:
    """One gpt-4o-mini call → typed slots, or None on abstain/empty slice."""
    if not evidence:
        return None

    def _run() -> dict:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": _evidence_prompt(question, evidence, time_window)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=600,
        )
        return safe_parse_json((resp.choices[0].message.content or "{}").strip())

    try:
        out = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain_llm_error", error=str(exc)[:200])
        return None
    if not isinstance(out, dict) or not out.get("answerable"):
        logger.info("brain_abstain", reason=str(out.get("table_reason", ""))[:160] if isinstance(out, dict) else "bad_json")
        return None
    return out


def verify(out: dict, evidence: list[dict]) -> tuple[dict | None, str]:
    """Value-verify the brain's slots against the evidence. Returns a NORMALISED
    contract (exact-case identifiers) or (None, reason). Every identifier must
    exist in the chosen table's schema; the aggregation/op/bucket must be known."""
    by_table = {e["table"].upper(): e for e in evidence}
    ev = by_table.get(str(out.get("table", "")).upper())
    if not ev:
        return None, "table_not_in_slice"
    valid = ev["valid_cols"]

    def col(name: str) -> str | None:
        return valid.get(str(name).upper()) if name else None

    value_set: dict[str, dict] = ev.get("value_set") or {}
    unique_rates: dict[str, float] = ev.get("unique_rates") or {}

    grain_kind = out.get("grain")
    grain_col = col(out.get("grain_column"))
    if grain_kind not in ("entity", "time") or not grain_col:
        return None, "bad_grain"
    bucket = out.get("time_bucket")
    if grain_kind == "time" and bucket not in _BUCKETS:
        return None, "bad_time_bucket"
    # Grain-uniqueness sanity: an entity grain column should distinguish rows. If
    # the key registry recorded a unique_rate for it, it must be > 0; absent → pass
    # (can't disprove — abstain-biased, never block a working contract).
    if grain_kind == "entity":
        grate = unique_rates.get(grain_col.upper())
        if grate is not None and grate <= 0.0:
            return None, "grain_not_distinguishing"

    measure_col = col(out.get("measure_column"))
    agg = str(out.get("measure_agg", "")).upper()
    if not measure_col or agg not in _AGGS:
        return None, "bad_measure"

    preds: list[tuple[str, str, Any]] = []
    for f in (out.get("filters") or []):
        fc = col(f.get("column")) if isinstance(f, dict) else None
        op = str(f.get("op", "=")).upper() if isinstance(f, dict) else ""
        if not fc or op not in _OPS:
            return None, "bad_filter"
        # Value-set check: only equality-style predicates against a column that HAS
        # a stored categorical value-set are checkable. If the column has a value_set
        # AND the filter value is not among its keys (case-insensitive) → block. No
        # value_set for the column → pass (can't disprove). Range/LIKE ops are not
        # membership checks, so they pass. Abstain-biased throughout.
        fval = f.get("value")
        col_values = value_set.get(fc.upper())
        if col_values and op in ("=", "==") and fval is not None:
            keys_lower = {str(k).lower() for k in col_values.keys()}
            if str(fval).lower() not in keys_lower:
                return None, "filter_value_not_in_value_set"
        preds.append((fc, op, fval))

    tcol = col(out.get("time_filter_column")) if out.get("time_filter_column") else None
    having = out.get("having") if isinstance(out.get("having"), dict) else None
    return ({
        "table": ev["table"], "grain_kind": grain_kind, "grain_col": grain_col,
        "bucket": bucket, "measure_col": measure_col, "agg": agg,
        "filters": preds, "time_col": tcol, "having": having,
        "top_n": out.get("top_n") if isinstance(out.get("top_n"), int) else None,
        "order": "ASC" if str(out.get("order", "desc")).lower() == "asc" else "DESC",
        "reason": str(out.get("table_reason", ""))[:200],
    }, "ok")


def render_sql(c: dict, time_window: tuple | None) -> str:
    """Deterministic, fully-quoted SQL from a verified contract. Handles entity
    grain (GROUP BY id) and time grain (GROUP BY year[,period]). The runtime owns
    every identifier and the arithmetic; the brain touched none of this."""
    measure_expr = (f'COUNT(DISTINCT {_quote(c["measure_col"])})' if c["agg"] == "COUNT_DISTINCT"
                    else f'{c["agg"]}({_quote(c["measure_col"])})')
    alias = _quote(c["measure_col"].lower())

    where: list[str] = [f'{_quote(col)} {op} {_sql_value(val)}' for col, op, val in c["filters"]]
    if c["time_col"] and time_window:
        where.append(f'{_quote(c["time_col"])} >= {_sql_value(str(time_window[0]))}')
        where.append(f'{_quote(c["time_col"])} <= {_sql_value(str(time_window[1]))}')

    if c["grain_kind"] == "time":
        dcol = _quote(c["grain_col"])
        group_sel = [f'EXTRACT(YEAR FROM {dcol}) AS "year"']
        group_by = ['"year"']
        if c["bucket"] == "month":
            group_sel.append(f'EXTRACT(MONTH FROM {dcol}) AS "month"'); group_by.append('"month"')
        elif c["bucket"] == "quarter":
            group_sel.append(f'EXTRACT(QUARTER FROM {dcol}) AS "quarter"'); group_by.append('"quarter"')
        select_parts = group_sel + [f'{measure_expr} AS {alias}']
        order_by = ", ".join(group_by)
    else:
        gcol = _quote(c["grain_col"])
        select_parts = [gcol, f'{measure_expr} AS {alias}']
        group_by = [gcol]
        order_by = f'{alias} {c["order"]}'

    sql = [f'SELECT {", ".join(select_parts)}', f'FROM {_quote(c["table"])}']
    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append("GROUP BY " + ", ".join(group_by))
    if c["having"]:
        op = str(c["having"].get("op", "")).strip()
        if op in _OPS and c["having"].get("value") is not None:
            sql.append(f'HAVING {measure_expr} {op} {_sql_value(c["having"]["value"])}')
    sql.append("ORDER BY " + order_by)
    if c["top_n"]:
        sql.append(f'LIMIT {int(c["top_n"])}')
    return "\n".join(sql)


async def brain_resolve(
    db: AsyncSession, question: str, candidates: list[dict], time_window: tuple | None,
) -> dict | None:
    """Full brain path: evidence → contract → verify → SQL. Returns
    ``{sql, table, grain, measure, reason}`` or None (abstain/unverifiable)."""
    evidence = await assemble_evidence(db, candidates)
    if not evidence:
        return None
    out = await propose_contract(question, evidence, time_window)
    if not out:
        return None
    contract, reason = verify(out, evidence)
    if not contract:
        logger.info("brain_verify_failed", reason=reason)
        return None
    sql = render_sql(contract, time_window)
    logger.info("brain_resolved", table=contract["table"], grain=contract["grain_kind"],
                measure=f'{contract["agg"]}({contract["measure_col"]})', reason=contract["reason"])
    return {"sql": sql, "table": contract["table"], "grain": contract["grain_kind"],
            "measure": f'{contract["agg"]}({contract["measure_col"]})', "reason": contract["reason"]}

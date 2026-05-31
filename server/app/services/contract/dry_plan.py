"""GATE B — dry-plan: validate LLM SQL against the contract BEFORE execution.

This is the Data-Analyst reflex: resolve the plan against the governed contract
first, and fail loudly with a repairable error instead of executing a query
that will silently return garbage (a coincidental cross-system join) or 0 rows.

Hard DENY (high confidence, low false-positive risk):
  • a JOIN equating two DIFFERENT contract models for which NO relationship is
    DECLARED — the exact coincidental-join failure (SAP↔Oracle key overlap).

Advisory (reported, not denied — avoids false negatives on aliasing/calc cols):
  • a table that does not resolve to any contract model
  • a column that is not in its model's exposed columns

The function is pure and never raises; if sqlglot can't parse or the contract
is empty, it returns OK (degrade to today's behaviour). Resolution of a SQL
table token to a contract model uses the SAME identity resolver the executor
uses, so logical names (T_03_SALES_ORDER / F_<hash>) line up exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import sqlglot
    from sqlglot import exp as _exp
    _SQLGLOT = True
except Exception:  # pragma: no cover - sqlglot is a hard dep, but be safe
    _SQLGLOT = False


@dataclass
class DryPlanVerdict:
    ok: bool = True
    violations: list[dict[str, Any]] = field(default_factory=list)
    advisories: list[dict[str, Any]] = field(default_factory=list)

    def to_error_payload(self) -> dict[str, Any]:
        """Structured, repairable error for the LLM (mirrors sql.py error shape)."""
        return {
            "error": "; ".join(v["message"] for v in self.violations) or "contract violation",
            "dry_plan_violation": True,
            "violations": self.violations,
            "hint": _repair_hint(self.violations),
            "fatal_execution_error": False,  # repairable — let the agent retry
        }


def dry_plan_sql(
    sql: str,
    contract: dict[str, Any] | None,
    *,
    resolve_file_id: Callable[[str], str | None] | None = None,
    dialect: str = "duckdb",
) -> DryPlanVerdict:
    """Validate `sql` against `contract`. Returns OK on any uncertainty."""
    verdict = DryPlanVerdict()
    if not _SQLGLOT or not contract:
        return verdict
    models = contract.get("models") or []
    if not models:
        return verdict

    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return verdict  # unparseable here → existing validators handle it
    if tree is None:
        return verdict

    # ── Index the contract ────────────────────────────────────────────────────
    model_by_fid: dict[str, dict] = {m["file_id"]: m for m in models if m.get("file_id")}
    name_to_fid: dict[str, str] = {}
    for m in models:
        nm = str(m.get("name") or "").upper()
        if nm and m.get("file_id"):
            name_to_fid[nm] = m["file_id"]

    # Declared join pairs as a set of frozenset{ "FID:col", "FID:col" } (col lowercased)
    declared: set[frozenset[str]] = set()
    for r in (contract.get("relationships") or []):
        fa, fb = r.get("from_file_id"), r.get("to_file_id")
        ca, cb = str(r.get("from_column") or "").lower(), str(r.get("to_column") or "").lower()
        if fa and fb and ca and cb:
            declared.add(frozenset({f"{fa}:{ca}", f"{fb}:{cb}"}))

    # ── Resolve table tokens (alias → file_id) ────────────────────────────────
    alias_to_fid: dict[str, str] = {}
    table_tokens: list[str] = []
    for tbl in tree.find_all(_exp.Table):
        token = tbl.name
        if not token:
            continue
        table_tokens.append(token)
        fid = _resolve(token, resolve_file_id, name_to_fid)
        # bind both the real name and any alias to the file_id
        if fid:
            alias_to_fid[token.upper()] = fid
            alias = tbl.alias
            if alias:
                alias_to_fid[alias.upper()] = fid
        else:
            verdict.advisories.append({
                "check": "unknown_table",
                "table": token,
                "message": f"table '{token}' does not resolve to a contract model",
            })

    # ── Hard check: every cross-model equi-join must be DECLARED ──────────────
    # We scan ALL equality predicates in the statement — JOIN ... ON clauses AND
    # comma-join / WHERE-clause equi-joins (FROM a, b WHERE a.k = b.k) — because a
    # coincidental cross-system join can be written either way. Equalities are
    # grouped by the (unordered) pair of models they connect; a pair is allowed
    # if ANY of its equalities is a declared relationship (this admits composite
    # keys, where only one column-pair need be declared to validate the join).
    # Both sides must resolve to a known model, else we skip (no false denials).
    observed: dict[frozenset, dict] = {}
    for eq in tree.find_all(_exp.EQ):
        lt = _col_parts(eq.this)
        rt = _col_parts(eq.expression)
        if not lt or not rt:
            continue  # literal / expression on a side — not an equi-join
        (lalias, lcol), (ralias, rcol) = lt, rt
        lfid = alias_to_fid.get((lalias or "").upper())
        rfid = alias_to_fid.get((ralias or "").upper())
        if not lfid or not rfid or lfid == rfid:
            continue  # unresolved side or self-join → never deny
        pairkey = frozenset({lfid, rfid})
        colkey = frozenset({f"{lfid}:{lcol.lower()}", f"{rfid}:{rcol.lower()}"})
        bucket = observed.setdefault(pairkey, {"declared": False, "first": None})
        if colkey in declared:
            bucket["declared"] = True
        if bucket["first"] is None:
            bucket["first"] = (lalias, lcol, ralias, rcol, lfid, rfid)

    for pairkey, bucket in observed.items():
        if bucket["declared"]:
            continue  # at least one declared equality between these models → OK
        lalias, lcol, ralias, rcol, lfid, rfid = bucket["first"]
        lm = model_by_fid.get(lfid, {})
        rm = model_by_fid.get(rfid, {})
        cross_system = (
            lm.get("source_system") and rm.get("source_system")
            and lm.get("source_system") != rm.get("source_system")
        )
        verdict.ok = False
        verdict.violations.append({
            "check": "undeclared_join",
            "join": f"{lalias}.{lcol} = {ralias}.{rcol}",
            "left_model": lm.get("name"),
            "right_model": rm.get("name"),
            "left_system": lm.get("source_system"),
            "right_system": rm.get("source_system"),
            "message": (
                f"join '{lalias}.{lcol} = {ralias}.{rcol}' is NOT a declared "
                f"relationship between {lm.get('name')} and {rm.get('name')}"
                + (
                    f" (different systems: {lm.get('source_system')} vs {rm.get('source_system')})"
                    if cross_system else ""
                )
            ),
        })

    # ── Advisory: column existence (lenient — never denies) ───────────────────
    _collect_column_advisories(tree, alias_to_fid, model_by_fid, verdict)

    return verdict


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve(token: str, resolver: Callable[[str], str | None] | None, name_to_fid: dict[str, str]) -> str | None:
    if resolver is not None:
        try:
            fid = resolver(token)
            if fid:
                return fid
        except Exception:
            pass
    return name_to_fid.get(token.upper())


def _col_parts(node) -> tuple[str | None, str] | None:
    """Return (table_or_alias, column) for a Column node, else None."""
    if not isinstance(node, _exp.Column):
        return None
    col = node.name
    if not col:
        return None
    tbl = node.table or None
    return (tbl, col)


def _collect_column_advisories(tree, alias_to_fid, model_by_fid, verdict) -> None:
    exposed_by_fid: dict[str, set[str]] = {}
    for fid, m in model_by_fid.items():
        exposed_by_fid[fid] = {str(c.get("name", "")).lower() for c in (m.get("columns") or [])}

    for colnode in tree.find_all(_exp.Column):
        parts = _col_parts(colnode)
        if not parts:
            continue
        alias, col = parts
        if not alias:
            continue  # unqualified — ambiguous, skip to avoid false positives
        fid = alias_to_fid.get(alias.upper())
        if not fid:
            continue
        exposed = exposed_by_fid.get(fid)
        if not exposed:
            continue  # no column metadata for this model → can't judge
        if col.lower() not in exposed:
            verdict.advisories.append({
                "check": "unknown_column",
                "column": f"{alias}.{col}",
                "model": model_by_fid.get(fid, {}).get("name"),
                "message": f"column '{col}' is not an exposed column of {model_by_fid.get(fid, {}).get('name')}",
            })


def _repair_hint(violations: list[dict[str, Any]]) -> str:
    for v in violations:
        if v.get("check") == "undeclared_join":
            return (
                "Remove this join or use a declared relationship. If these two "
                "datasets are from different source systems, they cannot be joined "
                "— answer each separately and attribute the source."
            )
    return "Adjust the query to use only declared relationships and exposed columns."

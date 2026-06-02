"""Enforced negative-claim gate (Fix 4).

A confident-but-wrong "there is no data / 2023 is missing / no records" answer is
the most damaging failure mode in enterprise analytics. This gate inspects the
FINAL answer and the per-request SQL state and blocks any negative claim that is
not PROVEN by (1) a full-logical-table scan AND (2) a diagnosis of the empty
result. Pure, no I/O, never raises — mirrors the discipline of feasibility_gate.

The existing feasibility gate is temporal-only and pre-SQL; this runs post-SQL on
the answer text, catching the case where SQL succeeded with 0 rows and the model
concluded "none".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Deterministic negative-claim phrases (lowercased substring match).
_NEGATIVE_PHRASES = (
    "no data", "no records", "no result", "no matching", "no rows",
    "none found", "not found", "there are no", "there is no", "0 rows",
    "zero rows", "could not find", "couldn't find", "no such", "nothing",
    "is missing", "are missing", "missing entirely", "no information",
)

_DATE_FILTER_RE = re.compile(r"\b(?:between|>=|<=|>|<|date|year|month|quarter|period|fiscal)\b", re.IGNORECASE)
_LITERAL_EQ_RE = re.compile(r"=\s*'", re.IGNORECASE)
_MINMAX_RE = re.compile(r"\b(?:min|max)\s*\(", re.IGNORECASE)
_DISTINCT_RE = re.compile(r"\bdistinct\b", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)


@dataclass
class NegativeClaimVerdict:
    is_negative_claim: bool = False
    proven: bool = False
    coverage_complete: bool = False
    diagnosed: bool = False
    missing_diagnostics: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def _is_negative(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in _NEGATIVE_PHRASES)


def _successful_attempts(store: dict) -> list[dict]:
    out = []
    for a in (store or {}).get("sql_attempts") or []:
        if not isinstance(a, dict):
            continue
        status = str(a.get("status") or "")
        if status in {"executed", "success", "canonicalized", "ok"} or a.get("referenced_file_ids"):
            out.append(a)
    return out


def _coverage_complete(attempts: list[dict], file_identities) -> bool:
    """Every authorized partition of every referenced logical table was scanned."""
    if not attempts:
        return False
    scanned: set[str] = set()
    tables: set[str] = set()
    for a in attempts:
        scanned.update(str(x) for x in (a.get("referenced_file_ids") or []))
        tables.update(str(t) for t in (a.get("referenced_tables") or []))
    if not tables or file_identities is None:
        return False
    for tname in tables:
        try:
            ident = file_identities.resolve_table(tname)
        except Exception:
            return False
        members = set(ident.member_file_ids or (ident.canonical_id,))
        if not members <= scanned:
            return False
    return True


def _diagnosed(attempts: list[dict]) -> tuple[bool, list[str]]:
    """A negative claim must be backed by the diagnostic relevant to WHY it was empty."""
    sql_texts = " \n ".join(str(a.get("logical_sql") or a.get("sql") or "") for a in attempts)
    had_date_filter = bool(_DATE_FILTER_RE.search(sql_texts))
    had_literal_eq = bool(_LITERAL_EQ_RE.search(sql_texts))
    had_join = bool(_JOIN_RE.search(sql_texts))
    ran_minmax = bool(_MINMAX_RE.search(sql_texts))
    ran_distinct = bool(_DISTINCT_RE.search(sql_texts))

    missing: list[str] = []
    if had_date_filter and not ran_minmax:
        missing.append("date_window_overlap")
    if had_literal_eq and not ran_distinct:
        missing.append("distinct_value_existence")
    if had_join and not (ran_distinct or ran_minmax):
        missing.append("join_key_overlap")
    # If the query had no narrowing predicate at all, a bare empty result is a
    # diagnosis in itself (nothing to probe).
    diagnosed = not missing
    return diagnosed, missing


def evaluate_negative_claim(
    *,
    answer: str,
    store: dict,
    file_identities=None,
) -> NegativeClaimVerdict:
    """Return a verdict; never raises. proven == coverage_complete AND diagnosed."""
    try:
        if not _is_negative(answer):
            return NegativeClaimVerdict(is_negative_claim=False, proven=True)
        attempts = _successful_attempts(store or {})
        coverage = _coverage_complete(attempts, file_identities)
        diagnosed, missing = _diagnosed(attempts)
        proven = coverage and diagnosed
        return NegativeClaimVerdict(
            is_negative_claim=True,
            proven=proven,
            coverage_complete=coverage,
            diagnosed=diagnosed,
            missing_diagnostics=missing,
            signals={
                "attempts": len(attempts),
                "tables": sorted({str(t) for a in attempts for t in (a.get("referenced_tables") or [])}),
            },
        )
    except Exception:
        # Never block the pipeline on a gate error.
        return NegativeClaimVerdict(is_negative_claim=False, proven=True)


def honest_rewrite(verdict: NegativeClaimVerdict, scanned_tables: list[str] | None = None) -> str:
    """A scoped, honest replacement for an UNPROVEN negative claim."""
    tables = ", ".join(scanned_tables or verdict.signals.get("tables") or []) or "the selected tables"
    if not verdict.coverage_complete:
        return (
            f"I could not confirm this is absent. I did not scan the full extent of {tables}, "
            "so I cannot state the data is missing. Please retry — the query needs to cover every "
            "partition of the table before drawing a 'no data' conclusion."
        )
    probes = ", ".join(verdict.missing_diagnostics) or "an existence check"
    return (
        f"The query returned no rows, but I have not verified this is a true absence rather than a "
        f"filter/join mismatch. Required diagnostic(s) not yet run: {probes}. I should probe these on "
        f"{tables} before concluding there is no data."
    )

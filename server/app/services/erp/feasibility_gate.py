"""GATE A — pre-SQL business feasibility (the Business-Analyst reflex).

A senior analyst decides WHETHER a question can be answered before writing any
SQL. This gate encodes that, using only data already in memory at context-build
time. It produces:

  • a HARD short-circuit ONLY for the deterministic temporal check (exact date
    math over file date ranges) — "you asked for a period the data does not
    cover"; this is never a guess, so it is safe to answer without the LLM.
  • ADVISORY notes (polarity mismatch, unavailable dimensions, fragmented
    domains) that are injected into the prompt but NEVER block — these lean on
    classification/inference which can be wrong, so they inform rather than gate.

Shadow mode (default ON until validated): the gate logs what it WOULD short-
circuit but does not actually short-circuit, so we can compare against real
outcomes before trusting it. Flip ERP_FEASIBILITY_GATE_SHADOW=false to enforce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from app.core.config import get_settings
from app.core.logger import pipeline_logger


@dataclass
class FeasibilityVerdict:
    feasible: bool = True
    short_circuit_answer: str | None = None          # set only when feasible is False (enforced)
    would_short_circuit: bool = False                # True even in shadow mode
    advisory_notes: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    @property
    def has_advisories(self) -> bool:
        return bool(self.advisory_notes)


def _enabled() -> bool:
    return bool(getattr(get_settings(), "ERP_FEASIBILITY_GATE_ENABLED", True))


def _shadow() -> bool:
    # Enforce by default now that the check is subject-scoped (only the query's
    # primary-subject files) and sentinel-safe. Set ERP_FEASIBILITY_GATE_SHADOW
    # =true to revert to log-only.
    return bool(getattr(get_settings(), "ERP_FEASIBILITY_GATE_SHADOW", False))


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ── Temporal parsing ──────────────────────────────────────────────────────────

def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    # ISO date / datetime prefix
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def parse_requested_window(constraints: dict | None, today: date | None = None) -> tuple[date, date] | None:
    """Parse the planner's date_range constraint into an explicit [start, end].

    Handles the shapes the business_intent_planner emits:
      "2025"            → 2025-01-01 .. 2025-12-31
      "last_30_days"    → today-30 .. today
      "last_3_months"   → today-~90 .. today
      "last_month"      → previous calendar month
      "this_month" / "this_year" / "last_year"
    Returns None when there is no parseable temporal constraint.
    """
    if not constraints:
        return None
    today = today or _today()
    raw = constraints.get("date_range")
    if raw is None:
        return None

    # Already an explicit pair?
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, end = _as_date(raw[0]), _as_date(raw[1])
        if start and end:
            return (start, end)
        return None
    if isinstance(raw, dict):
        start, end = _as_date(raw.get("start")), _as_date(raw.get("end"))
        if start and end:
            return (start, end)
        return None

    token = str(raw).strip().lower()

    # Bare year, e.g. "2025"
    m = re.fullmatch(r"(\d{4})", token)
    if m:
        y = int(m.group(1))
        return (date(y, 1, 1), date(y, 12, 31))

    # "YYYY-QN" quarter, e.g. "2025-q2"
    m = re.fullmatch(r"(\d{4})[-_ ]?q([1-4])", token)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        start_month = (q - 1) * 3 + 1
        start = date(y, start_month, 1)
        end_month = start_month + 2
        end = _month_end(date(y, end_month, 1))
        return (start, end)

    m = re.fullmatch(r"last_(\d+)_days?", token)
    if m:
        n = int(m.group(1))
        return (_shift_days(today, -n), today)

    m = re.fullmatch(r"last_(\d+)_months?", token)
    if m:
        n = int(m.group(1))
        return (_shift_months(today, -n), today)

    if token == "last_month":
        first_this = today.replace(day=1)
        last_prev = _shift_days(first_this, -1)
        return (last_prev.replace(day=1), last_prev)
    if token == "this_month":
        return (today.replace(day=1), today)
    if token == "this_year":
        return (date(today.year, 1, 1), today)
    if token == "last_year":
        return (date(today.year - 1, 1, 1), date(today.year - 1, 12, 31))
    if token == "this_quarter":
        q_start_month = ((today.month - 1) // 3) * 3 + 1
        return (date(today.year, q_start_month, 1), today)
    if token == "last_quarter":
        cur_q_start_month = ((today.month - 1) // 3) * 3 + 1
        cur_q_start = date(today.year, cur_q_start_month, 1)
        last_q_end = _shift_days(cur_q_start, -1)
        last_q_start_month = ((last_q_end.month - 1) // 3) * 3 + 1
        return (date(last_q_end.year, last_q_start_month, 1), last_q_end)

    return None


def _shift_days(d: date, days: int) -> date:
    from datetime import timedelta
    return d + timedelta(days=days)


def _shift_months(d: date, months: int) -> date:
    """Calendar-correct month shift, clamping the day to the target month end."""
    total = (d.year * 12 + (d.month - 1)) + months
    year, month = divmod(total, 12)
    month += 1
    last_day = _month_end(date(year, month, 1)).day
    return date(year, month, min(d.day, last_day))


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    from datetime import timedelta
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


# ERP "valid-to infinity" sentinels (SAP 9999-12-31, Oracle 4712-12-31). A
# master/effective-dated table whose validity runs to one of these is NOT
# evidence of transactional coverage for a past period — counting it would make
# the gate think every period is "covered". Treat such windows as unusable.
_SENTINEL_YEAR = 2400


def _file_window(entry: dict) -> tuple[date, date] | None:
    """Extract a file's date coverage from a catalog entry, defensively.

    Returns None (file contributes no usable coverage) when dates are missing
    OR when the end is an open-ended ERP sentinel (year ≥ 2400)."""
    start = _as_date(entry.get("date_range_start"))
    end = _as_date(entry.get("date_range_end"))
    if not (start or end):
        dr = entry.get("date_range")
        if isinstance(dr, dict):
            start = _as_date(dr.get("start"))
            end = _as_date(dr.get("end"))
    # Drop open-ended sentinel windows — they are not transactional coverage.
    if end and end.year >= _SENTINEL_YEAR:
        return None
    if start and start.year >= _SENTINEL_YEAR:
        return None
    if start and not end:
        end = start
    if end and not start:
        start = end
    if start and end:
        return (start, end) if start <= end else (end, start)
    return None


def _overlaps(a: tuple[date, date], b: tuple[date, date]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _label(entry: dict) -> str:
    name = entry.get("display_name") or entry.get("blob_path") or entry.get("file_id") or "file"
    return str(name).rsplit("/", 1)[-1]


# The relative-time anchor is the HIGH-PERCENTILE coverage end, not the raw max.
# A handful of future-dated tables (forecasts, schedules, quote expirations) must
# not drag the dataset's effective "now" past where the transactional data ends —
# the raw max is dominated by a single 2036 forecast row. 90 = "the point by which
# the bulk of coverage has ended". Data-driven (computed from the observed date
# distribution), not a per-dataset constant.
_AS_OF_PERCENTILE = 90


def _percentile_date(dates: list[date], percentile: int) -> date | None:
    """Nearest-rank percentile of a date list (no interpolation). Pure."""
    if not dates:
        return None
    ordered = sorted(dates)
    k = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def data_as_of(ends: list[date] | None, today: date | None = None) -> date | None:
    """Robust data 'now' from coverage end-dates: the high-percentile end (where
    the BULK of the data ends), capped at the wall clock. Sentinel-safe. This is
    the SINGLE source of truth shared by the prompt anchor and the retrieval
    temporal filter, so the two never disagree. Returns None when no usable
    (non-sentinel) end-date exists → callers fall back to the wall clock."""
    today = today or _today()
    clean = [d for d in (ends or []) if d and d.year < _SENTINEL_YEAR]
    anchor = _percentile_date(clean, _AS_OF_PERCENTILE)
    return min(anchor, today) if anchor else None


def resolve_as_of_date(catalog: list[dict] | None, today: date | None = None) -> date | None:
    """The reference 'now' for relative-time resolution = the data's effective
    latest coverage (high-percentile end), capped at the wall clock. Data-driven
    (reads precomputed date_range_* metadata), sentinel-safe, outlier-robust.
    Returns None when no file carries usable date coverage.

    Anchoring relative windows ('this year', 'last month', 'YTD') to this date —
    rather than date.today() — prevents a correct parser from resolving a period
    the data does not cover (e.g. data ending 2025-05 queried under a 2026 clock).
    """
    ends = [w[1] for e in (catalog or []) if (w := _file_window(e))]
    return data_as_of(ends, today)


# ── The gate ──────────────────────────────────────────────────────────────────

def evaluate_feasibility(
    *,
    query: str,
    constraints: dict | None,
    catalog: list[dict],
    today: date | None = None,
) -> FeasibilityVerdict:
    """Run the deterministic feasibility checks. Pure, no I/O, never raises."""
    verdict = FeasibilityVerdict()
    if not _enabled():
        return verdict

    try:
        window = parse_requested_window(constraints, today=today)
    except Exception as exc:  # pure-function safety net
        pipeline_logger.warning("feasibility_parse_error", error=str(exc)[:160])
        return verdict

    if not window:
        return verdict  # no temporal claim → nothing deterministic to check

    dated = [(e, w) for e in (catalog or []) if (w := _file_window(e))]
    if not dated:
        # No file carries date metadata → we cannot prove infeasibility. Degrade.
        return verdict

    overlapping = [(e, w) for (e, w) in dated if _overlaps(w, window)]
    verdict.signals = {
        "requested_window": [window[0].isoformat(), window[1].isoformat()],
        "dated_files": len(dated),
        "overlapping_files": len(overlapping),
    }

    if overlapping:
        return verdict  # feasible — at least one file covers the period

    # ── Deterministic infeasibility: a period the data does not cover ─────────
    verdict.would_short_circuit = True
    answer = _build_no_coverage_answer(window, dated)

    if _shadow():
        pipeline_logger.info(
            "feasibility_gate_shadow",
            decision="would_short_circuit",
            query=query[:160],
            **verdict.signals,
        )
        # Surface as advisory so the LLM at least knows, but do not block.
        verdict.advisory_notes.append(
            "TEMPORAL NOTE (advisory): the requested period appears to fall "
            "outside every dated file's coverage. Confirm with MIN/MAX before "
            "concluding, and if truly absent, say so and suggest the available range."
        )
        return verdict

    verdict.feasible = False
    verdict.short_circuit_answer = answer
    pipeline_logger.info(
        "feasibility_gate_enforced",
        decision="short_circuit",
        query=query[:160],
        **verdict.signals,
    )
    return verdict


def _build_no_coverage_answer(window: tuple[date, date], dated: list[tuple[dict, tuple[date, date]]]) -> str:
    req = f"{window[0].isoformat()} to {window[1].isoformat()}"
    # Summarise available coverage, grouped by source_system when present.
    by_system: dict[str, list[tuple[date, date]]] = {}
    for entry, w in dated:
        sysname = str(entry.get("source_system") or "data").strip() or "data"
        by_system.setdefault(sysname, []).append(w)
    parts: list[str] = []
    for sysname, windows in by_system.items():
        lo = min(w[0] for w in windows)
        hi = max(w[1] for w in windows)
        parts.append(f"{sysname}: {lo.isoformat()} → {hi.isoformat()}")
    coverage = "; ".join(parts)
    return (
        f"There is no data for the requested period ({req}). "
        f"The available data covers — {coverage}. "
        f"Please confirm the period you meant (for example, the nearest covered "
        f"range), and I'll run it."
    )

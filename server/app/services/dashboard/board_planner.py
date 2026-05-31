"""
Board Planner — the dashboard "brain" (DASHBOARD_INTELLIGENCE_PLAN.txt §3-§4).

A dashboard request ("give me a dashboard for Q2 analysis") is an INTENT, not a
query. A single question→agent fan-out cannot interpret it. This module inserts a
BOARD-LEVEL orchestration stage that reasons the way a business analyst does,
BEFORE any agent call:

  S0  derive the SHARED TIME WINDOW from the catalog (real data coverage)
  S1  resolve which tables/measures/dimensions actually EXIST (catalog = the
      ingestion-time projection of the semantic layer) — resolve, never invent
  S2  ask the LLM to DESIGN a coherent board as a metric lattice
      (metric × dimension × time window × comparison), as a narrative
  S3  FEASIBILITY DRY-RUN — a CHEAP, deterministic check of every proposed widget
      against catalog metadata (does the measure exist? is the dimension small
      enough? is there a date column?) — repair or drop BEFORE spending a full
      run_agent_query. This is the "dry run" that kills dead tiles for free.
  S4  convert surviving specs → WidgetIntent (the existing run_widget contract)

Hard rules preserved:
  - NO new query brain: one widget still == one run_agent_query.
  - NEVER raises: any failure degrades to query_engine.decompose_prompt so the
    route can always return a dashboard.
  - Chat is untouched (this lives entirely in services/dashboard/).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logger import chat_logger
from app.core.openai_client import get_client
from app.services.dashboard.query_engine import (
    MAX_WIDGETS,
    WidgetIntent,
    decompose_prompt,
)

# Component-type values that recommend()/run_widget understand.
_VALID_VIZ = {
    "kpi_card", "metric_tile", "table", "line_chart", "bar_chart",
    "pie_chart", "area_chart", "heatmap", "funnel",
}

# question_type → default visualization when the planner omits one.
_DEFAULT_VIZ = {
    "kpi": "kpi_card",
    "trend": "line_chart",
    "breakdown": "bar_chart",
    "share": "pie_chart",
    "matrix": "heatmap",
    "detail": "table",
}

# Chart-readability ceilings used by the feasibility gate (mirror the
# component_catalog visualization_rules so we don't propose unreadable charts).
_SHARE_MAX_CARD = 8     # pie
_BREAKDOWN_MAX_CARD = 50  # bar
_MATRIX_MAX_CARD = 40   # heatmap second axis


@dataclass
class WidgetSpec:
    """One planned widget as a metric-lattice tuple (pre-execution)."""
    title: str
    question_type: str            # kpi|trend|breakdown|share|matrix|detail
    table: str | None = None
    measure: str | None = None
    dimension: str | None = None
    dimension2: str | None = None
    comparison: str | None = None
    viz: str | None = None


# --------------------------------------------------------------------------
# S0 — shared time window from real data coverage
# --------------------------------------------------------------------------

def _table_window(table) -> tuple[str, str] | None:
    """Widest (min, max) across a table's temporal columns, or None."""
    temporal = set(getattr(table, "temporal", []) or [])
    lo = hi = None
    for c in getattr(table, "columns", []):
        if c.name in temporal and c.min_value is not None and c.max_value is not None:
            cmin, cmax = str(c.min_value), str(c.max_value)
            lo = cmin if lo is None or cmin < lo else lo
            hi = cmax if hi is None or cmax > hi else hi
    return (lo, hi) if lo is not None and hi is not None else None


def _board_window(catalog) -> tuple[str, str] | None:
    """
    The board's shared date window: the temporal coverage of the highest-row-count
    table that has one. Deterministic and always in-range, so time-based widgets
    cannot ask for a period the data lacks (the April-2026 class of empty tile).
    """
    best = None  # (row_count, (min, max))
    for t in catalog:
        win = _table_window(t)
        if win is None:
            continue
        rc = t.row_count or 0
        if best is None or rc > best[0]:
            best = (rc, win)
    return best[1] if best else None


def _window_phrase(win: tuple[str, str] | None) -> str:
    if not win:
        return ""
    lo, hi = win
    return f" between {lo} and {hi}"


# --------------------------------------------------------------------------
# S1/S2 — LLM board design (grounded), never raises
# --------------------------------------------------------------------------

def _safe_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


async def _design_board(prompt: str, grounding_text: str, window: tuple[str, str] | None,
                        max_widgets: int) -> list[dict]:
    """One LLM call that DESIGNS the board as a metric lattice. Returns [] on any
    failure (caller falls back to decompose_prompt)."""

    def _run() -> list[dict]:
        client, _ = get_client()
        settings = get_settings()
        # The board brain is ONE call per dashboard — use the strong model when
        # available, fall back to the mini deployment.
        deployment = getattr(settings, "AZURE_OPENAI_DEPLOYMENT", None) or \
            settings.AZURE_OPENAI_DEPLOYMENT_MINI

        win_phrase = (f"{window[0]} .. {window[1]}" if window
                      else "none — no reliable date column; omit time filters")

        sys = (
            "You are a senior business-intelligence analyst. You DESIGN one coherent "
            "dashboard from a single request — you do NOT answer questions.\n\n"
            "Think like an analyst:\n"
            "1) Infer the business DOMAIN from the catalog (finance, sales, operations...).\n"
            "2) Reason as a metric lattice: each widget = (metric x dimension x time "
            "window x comparison).\n"
            "3) Build a NARRATIVE, not a bag of charts: lead with 1-2 headline KPIs, then "
            "ONE trend, then 1-2 breakdowns, an optional share or matrix, and ONE detail "
            "table.\n"
            "4) Be DATA-ADAPTIVE: only propose a widget the catalog can ACTUALLY answer. "
            "If the data supports 3 good widgets, return 3 — never pad.\n\n"
            "GROUNDING (the catalog is authoritative):\n"
            "- Use ONLY tables, columns and observed values shown in the catalog. Never invent.\n"
            "- 'measure' MUST be a column whose role is a measure/metric for that table.\n"
            "- 'dimension' MUST be a categorical or temporal column shown for that table.\n"
            "- SHARED TIME WINDOW: every time-based widget uses the shared window provided. "
            "Never request a period outside the data coverage.\n"
            "- Prefer single-table widgets; only span tables via the KNOWN JOINS listed.\n\n"
            "Return ONLY JSON."
        )
        user = f"""CATALOG (real tables, columns, roles, date coverage, observed values, known joins):
{grounding_text or '(catalog unavailable)'}

SHARED TIME WINDOW (use for every time-based widget): {win_phrase}

DASHBOARD REQUEST:
\"\"\"{prompt}\"\"\"

Return JSON (max {max_widgets} widgets):
{{"domain":"inferred domain",
 "widgets":[
  {{"title":"short title",
    "question_type":"kpi|trend|breakdown|share|matrix|detail",
    "table":"table name from the catalog",
    "measure":"measure column (or null for a detail table)",
    "dimension":"dimension/temporal column (or null for kpi/detail)",
    "dimension2":"second dimension for a matrix (or null)",
    "comparison":"e.g. 'vs previous period' (or null)",
    "viz":"kpi_card|metric_tile|line_chart|area_chart|bar_chart|pie_chart|heatmap|funnel|table|null"}}
 ]}}"""

        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=1200,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = _safe_json(raw) or {}
        widgets = parsed.get("widgets") if isinstance(parsed, dict) else None
        if isinstance(parsed, dict) and parsed.get("domain"):
            chat_logger.info("dashboard_board_domain", domain=str(parsed.get("domain"))[:80])
        return widgets if isinstance(widgets, list) else []

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        chat_logger.warning("dashboard_board_design_error", error=str(exc)[:200])
        return []


def _coerce_specs(raw_widgets: list[dict], max_widgets: int) -> list[WidgetSpec]:
    specs: list[WidgetSpec] = []
    for w in raw_widgets[:max_widgets]:
        if not isinstance(w, dict):
            continue
        qt = str(w.get("question_type") or "").strip().lower()
        if qt not in _DEFAULT_VIZ:
            qt = "detail"
        title = str(w.get("title") or "").strip()[:120]
        if not title:
            continue

        def _clean(key: str) -> str | None:
            v = w.get(key)
            v = str(v).strip() if v is not None else ""
            return v or None if v.lower() != "null" else None

        viz = _clean("viz")
        if viz and viz not in _VALID_VIZ:
            viz = None
        specs.append(WidgetSpec(
            title=title,
            question_type=qt,
            table=_clean("table"),
            measure=_clean("measure"),
            dimension=_clean("dimension"),
            dimension2=_clean("dimension2"),
            comparison=_clean("comparison"),
            viz=viz,
        ))
    return specs


# --------------------------------------------------------------------------
# S3 — feasibility dry-run (deterministic, NO agent / NO SQL / NO LLM)
# --------------------------------------------------------------------------

def _card(table, name: str | None) -> int | None:
    if not name:
        return None
    for c in getattr(table, "columns", []):
        if c.name == name:
            return c.cardinality
    return None


def _pick_dimension(table, max_card: int, exclude: set[str] | None = None) -> str | None:
    """Lowest-cardinality categorical dimension within `max_card`, else any dim."""
    exclude = exclude or set()
    best: tuple[int, str] | None = None
    fallback: str | None = None
    for name in getattr(table, "dimensions", []) or []:
        if name in exclude:
            continue
        fallback = fallback or name
        card = _card(table, name)
        if card is None:
            continue
        if card <= max_card and card >= 1:
            if best is None or card < best[0]:
                best = (card, name)
    if best:
        return best[1]
    return fallback


def _find_table_for(catalog, measure: str | None):
    """Best table for a measure: one that contains it, else the most measure-rich."""
    if measure:
        for t in catalog:
            if measure in set(t.measures):
                return t
    return max(catalog, key=lambda t: len(t.measures), default=None)


def feasibility_filter(specs: list[WidgetSpec], catalog: list) -> tuple[list[WidgetSpec], list[str]]:
    """Repair-or-drop each spec against catalog metadata. Returns (kept, reasons)."""
    by_name = {t.table_name: t for t in catalog}
    kept: list[WidgetSpec] = []
    dropped: list[tuple[WidgetSpec, str]] = []

    for s in specs:
        t = by_name.get(s.table) if s.table else None
        if t is None:
            t = _find_table_for(catalog, s.measure)
            if t is None:
                dropped.append((s, "no matching table in the catalog"))
                continue
            s.table = t.table_name

        measures = set(t.measures)
        dims = set(t.dimensions)
        temporal = set(t.temporal)
        dims_all = dims | temporal

        # --- measure (required for everything except a detail table) ---------
        if s.question_type != "detail":
            if not s.measure or s.measure not in measures:
                if measures:
                    s.measure = sorted(measures)[0]
                else:
                    dropped.append((s, f"no measure available in '{t.table_name}'"))
                    continue

        # --- trend needs a temporal column -----------------------------------
        if s.question_type == "trend" and not temporal:
            picked = _pick_dimension(t, _BREAKDOWN_MAX_CARD)
            if picked:
                s.question_type, s.viz, s.dimension = "breakdown", "bar_chart", picked
            else:
                s.question_type, s.viz, s.dimension = "kpi", "kpi_card", None

        # --- breakdown/share/matrix need a valid dimension -------------------
        if s.question_type in ("breakdown", "share", "matrix"):
            if not s.dimension or s.dimension not in dims_all:
                cap = _SHARE_MAX_CARD if s.question_type == "share" else _BREAKDOWN_MAX_CARD
                picked = _pick_dimension(t, cap)
                if picked:
                    s.dimension = picked
                else:
                    s.question_type, s.viz, s.dimension = "kpi", "kpi_card", None

        # share must be low-cardinality, else degrade to a bar breakdown
        if s.question_type == "share":
            card = _card(t, s.dimension)
            if card is not None and card > _SHARE_MAX_CARD:
                s.question_type, s.viz = "breakdown", "bar_chart"

        # matrix needs a distinct second dimension, else degrade to breakdown
        if s.question_type == "matrix":
            if (not s.dimension2 or s.dimension2 not in dims_all
                    or s.dimension2 == s.dimension):
                pick2 = _pick_dimension(t, _MATRIX_MAX_CARD, exclude={s.dimension or ""})
                if pick2:
                    s.dimension2 = pick2
                else:
                    s.question_type, s.viz = "breakdown", "bar_chart"

        kept.append(s)

    # Dedupe identical lattice tuples (the planner sometimes repeats a widget).
    seen: set[tuple] = set()
    uniq: list[WidgetSpec] = []
    for s in kept:
        key = (s.question_type, s.table, s.measure, s.dimension, s.dimension2)
        if key in seen:
            dropped.append((s, "duplicate of another widget"))
            continue
        seen.add(key)
        uniq.append(s)

    reasons = [f"Skipped '{s.title}' — {why}." for s, why in dropped]
    return uniq, reasons


# --------------------------------------------------------------------------
# S4 — specs → WidgetIntent (the existing run_widget contract)
# --------------------------------------------------------------------------

def _nl_query(spec: WidgetSpec, win_phrase: str) -> str:
    m = spec.measure or "records"
    d = spec.dimension
    comp = f" Compare {spec.comparison}." if spec.comparison else ""
    qt = spec.question_type
    if qt == "kpi":
        return f"What is the total {m}{win_phrase}?{comp}"
    if qt == "trend":
        return f"Show how total {m} changes over time{win_phrase}.{comp}"
    if qt == "breakdown":
        return f"Show total {m} broken down by {d}{win_phrase}, top 10 by {m}.{comp}"
    if qt == "share":
        return f"Show the percentage share of total {m} across each {d}{win_phrase}."
    if qt == "matrix":
        return f"Show total {m} by {d} and {spec.dimension2}{win_phrase}."
    # detail
    order = f", ordered by {spec.measure}" if spec.measure else ""
    return f"List the top 20 records from {spec.table}{win_phrase}{order}."


def specs_to_intents(specs: list[WidgetSpec], catalog: list) -> list[WidgetIntent]:
    by_name = {t.table_name: t for t in catalog}
    intents: list[WidgetIntent] = []
    for s in specs:
        table = by_name.get(s.table) if s.table else None
        # Share the in-range window for any time-based widget on a table that has
        # a date column; tables without temporal data get no time filter.
        win = _table_window(table) if table else None
        win_phrase = _window_phrase(win) if table and table.temporal else ""

        viz = s.viz if s.viz in _VALID_VIZ else _DEFAULT_VIZ.get(s.question_type)
        cols = [c for c in (s.measure, s.dimension, s.dimension2) if c]
        hints: dict = {"table": s.table} if s.table else {}
        if cols:
            hints["columns"] = cols
        if win_phrase:
            hints["time_window"] = win_phrase.strip()

        intents.append(WidgetIntent(
            title=s.title,
            nl_query=_nl_query(s, win_phrase),
            requested_viz=viz,
            hints=hints,
        ))
    return intents


# --------------------------------------------------------------------------
# Public entry point — replaces the bare decompose_prompt call in the route
# --------------------------------------------------------------------------

async def plan_widgets(
    prompt: str,
    catalog: list,
    *,
    grounding_text: str,
    max_widgets: int = MAX_WIDGETS,
) -> tuple[list[WidgetIntent], list[str]]:
    """
    Board-level planning: design → feasibility dry-run → intents. Never raises.
    Falls back to query_engine.decompose_prompt if planning yields nothing, so the
    route always gets a usable set of widget intents.

    Returns (intents, warnings) where warnings explain any dropped/repaired widgets.
    """
    max_widgets = max(1, min(int(max_widgets or MAX_WIDGETS), MAX_WIDGETS))
    warnings: list[str] = []

    try:
        window = _board_window(catalog) if catalog else None
        raw = await _design_board(prompt, grounding_text, window, max_widgets)
        specs = _coerce_specs(raw, max_widgets)
    except Exception as exc:
        chat_logger.warning("dashboard_board_plan_error", error=str(exc)[:200])
        specs = []

    if specs:
        kept, reasons = feasibility_filter(specs, catalog or [])
        warnings.extend(reasons)
        intents = specs_to_intents(kept[:max_widgets], catalog or [])
        if intents:
            chat_logger.info(
                "dashboard_board_planned",
                proposed=len(specs), feasible=len(intents), dropped=len(reasons),
            )
            return intents, warnings

    # Fallback: the original grounded single-pass decomposition.
    chat_logger.info("dashboard_board_fallback_decompose")
    intents = await decompose_prompt(prompt, grounding_text, max_widgets=max_widgets)
    return intents, warnings

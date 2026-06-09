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

# Component-type values that recommend()/run_widget understand. `ranked_bar` is
# the top-N "driver view" (who's driving a measure) — a first-class viz the brain
# may request and the densifier may emit for a high-cardinality entity dimension.
_VALID_VIZ = {
    "kpi_card", "metric_tile", "table", "line_chart", "bar_chart",
    "pie_chart", "area_chart", "heatmap", "funnel", "ranked_bar",
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
_RANK_MAX_CARD = 1000   # ranked_bar (top-N driver view; mirrors its catalog rule)


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
    # Metric DIRECTION proposed by the planner LLM: 'inverse' when a rising value
    # is BAD (cost, aging, DSO, returns, outstanding/overdue balance), else
    # 'positive' (growth). Carried into the widget config so the DeltaBadge frames
    # the delta correctly. Fail-safe to 'positive' downstream — never fabricated.
    polarity: str | None = None
    # The brain's DETAILED analytical instruction for the SQL agent (preferred
    # over the deterministic template) and the one-line chart rationale.
    query: str | None = None
    chart_rationale: str | None = None
    # STRUCTURAL ceiling for an INTRINSIC 0-100% ratio metric (collection rate,
    # fill rate). The planner sets metric_max=100 ONLY for a metric whose query
    # produces a bounded 0-100 percentage. It is the ONLY honest target a gauge/
    # progress/bullet may use without a real target column — every other
    # target-requiring tile fails closed to a plain KPI (D.1). None otherwise.
    metric_max: float | None = None


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

    win_phrase = (f"{window[0]} .. {window[1]}" if window
                  else "none — no reliable date column; omit time filters")

    sys = (
        "You are an elite business-intelligence analyst at a top-tier global "
        "consulting firm. Fortune-500 executives rely on you to turn a single "
        "request into ONE board-ready dashboard they can present in a leadership "
        "meeting. You do NOT answer questions — you DESIGN the dashboard. A great "
        "board is an ANALYST NARRATIVE, not a value dump: it tells the executive "
        "WHAT moved, BY HOW MUCH vs last period, WHO is driving it, and what the "
        "MIX looks like — using DERIVED business ratios, not just raw totals.\n\n"
        "HOW YOU THINK:\n"
        "1) Infer the business DOMAIN from the catalog and select ONLY the most "
        "relevant tables (catalogs). Ignore tables that do not serve the request.\n"
        "2) Decompose the request into a METRIC LATTICE — every widget is a "
        "(metric x dimension x time window x comparison). A vague request such as "
        "'Q2 analysis' becomes a deliberate set of executive widgets; a dense "
        "request with several measures becomes one widget per measure plus the "
        "trends and breakdowns that make them meaningful.\n"
        "3) DERIVED METRICS (the differentiator) — a real analyst leads with RATIOS "
        "and RATES computed over real columns, not bare sums. From columns PRESENT "
        "in the catalog, derive metrics such as: a COLLECTION RATE (1 - SUM(remaining"
        ")/SUM(original)); a PAYMENT RATIO (SUM(paid)/SUM(invoiced)); a FILL RATE "
        "(SUM(shipped)/SUM(ordered)); a CANCELLATION RATE; a CONTRIBUTION/SHARE % "
        "(a category's measure / the grand total, SUM() OVER()); growth (MoM/YoY % "
        "via comparison); an AVERAGE such as AOV (SUM(total)/COUNT(DISTINCT order)). "
        "Put the FORMULA in the 'query'. Set 'measure' to a REAL numerator column "
        "from the catalog (it is the binding handle, not the formula). For an "
        "INTRINSIC 0-100% ratio (collection rate, fill rate, payment ratio) set "
        "'metric_max':100. NEVER derive a metric from a column that is not in the "
        "catalog, and never invent the denominator.\n"
        "4) Pick the RIGHT chart for each insight, the way an analyst would for a "
        "meeting: KPI/metric tile for one headline number; line or area for a "
        "trend over time; bar to compare a measure across a HANDFUL of categories; "
        "ranked_bar (top-N driver view) to rank a measure across MANY entities "
        "(vendors, customers, products) — 'who is driving it'; pie ONLY for a "
        "few-category share; heatmap for two dimensions; a table for detail/top-N.\n"
        "5) For EACH widget, WRITE A PRECISE ANALYTICAL INSTRUCTION ('query') for a "
        "downstream SQL agent. Be explicit: which measure to AGGREGATE (SUM/AVG/"
        "COUNT), how to GROUP it, how to ORDER it, how many rows to return, and the "
        "exact output columns. NEVER write a vague one-liner — vague instructions "
        "produce wrong tiles. A trend MUST say: return one row per period with the "
        "period column and the summed measure, ordered chronologically. A breakdown "
        "MUST say: group by the dimension, sum the measure, return the dimension and "
        "the total, top-N descending. A KPI MUST say: return the single aggregated "
        "number. A DERIVED ratio MUST state the exact numerator/denominator formula.\n"
        "6) THE ANALYST ARC — order the board as a narrative like a PowerBI page: "
        "(a) 3-4 HEADLINE KPIs WITH a period delta (set 'comparison':'vs previous "
        "period' and the right 'polarity'); (b) a TREND over time; (c) a RANKED "
        "top-N DRIVER ('who's driving it'); (d) a COMPOSITION/breakdown (category bar "
        "or few-category share); (e) at least ONE DERIVED-RATIO tile; (f) optionally "
        "ONE cross-source view; (g) a detail TABLE last. Aim for >= 5 VISUAL widgets "
        "when the data supports them, but NEVER pad with fabricated or duplicate "
        "widgets — if the data only supports 3 strong widgets, return 3 (honest "
        "beats dense).\n"
        "7) For EACH measure, set 'polarity': 'inverse' when a RISING value is BAD "
        "(cost, spend, aging, returns, overdue/outstanding balance, defects, churn) "
        "and 'positive' otherwise (revenue, profit, volume, collection rate, fill "
        "rate). This colors the period delta; when unsure, use 'positive'.\n"
        "8) CROSS-SOURCE views — you MAY emit ONE widget that spans two tables ONLY "
        "on a VALIDATED master key shown under KNOWN JOINS (e.g. CUSTOMER_ID to join "
        "an orders table with a receivables table; VENDOR_ID to join purchasing with "
        "payables). Emit it as a 'detail' table (or a grouped bar) whose 'query' "
        "tells the agent to JOIN on that master key and return BOTH aggregated "
        "measures side by side. The agent validates the join via the relationship "
        "graph. If both columns live in ONE table (e.g. invoiced vs paid in the same "
        "payables table) NO join is needed — derive the ratio in place. NEVER join on "
        "a document key, an audit column, or a constant org id.\n\n"
        "HONESTY RULES (these are HARD — a wrong-but-confident tile is worse than no "
        "tile):\n"
        "- NO FABRICATED TARGETS: there are NO target/budget/quota columns in this "
        "data. Do NOT design gauge/progress/bullet 'vs target' tiles UNLESS the "
        "metric is an intrinsic 0-100% ratio with 'metric_max':100. Otherwise use a "
        "plain KPI tile.\n"
        "- NO DATE-DIFFERENCE METRICS: do NOT design DSO, days-to-pay, on-time %, "
        "lead time, cycle time, or any metric that subtracts two dates — the dates "
        "in this data are independent random draws, so such metrics are meaningless. "
        "Use the data's OWN date columns for trends; NEVER reference today/current_"
        "date.\n"
        "- NO VANITY TILES: do NOT headline a measure the catalog shows is all-zero "
        "or a single constant value, and do NOT break a measure down by a constant "
        "dimension (one distinct value). Prefer $ amounts over bare row counts.\n"
        "- GRAIN: on a LINE/transaction table, count entities with COUNT(DISTINCT "
        "entity_id) — never mix line-level rows with order/document counts.\n\n"
        "GROUNDING (the catalog is authoritative):\n"
        "- Use ONLY tables, columns and observed values shown. Never invent.\n"
        "- 'measure' MUST be a measure column; 'dimension' MUST be a shown "
        "categorical/temporal column for that table.\n"
        "- SHARED TIME WINDOW: every time-based widget uses the provided window. "
        "Never request a period outside the data coverage.\n"
        "- Prefer single-table widgets; only span tables via the KNOWN JOINS listed.\n\n"
        "Return ONLY JSON."
    )
    user = f"""CATALOG (real tables, columns, roles, date coverage, observed values, known joins):
{grounding_text or '(catalog unavailable)'}

SHARED TIME WINDOW (use for every time-based widget): {win_phrase}

DASHBOARD REQUEST:
\"\"\"{prompt}\"\"\"

Return JSON (max {max_widgets} widgets), ordered as the analyst arc (headline KPIs with deltas first, a trend, a ranked driver, a composition, at least one derived-ratio tile, an optional cross-source view, detail table last):
{{"domain":"inferred business domain",
 "widgets":[
  {{"title":"concise executive title",
    "question_type":"kpi|trend|breakdown|share|matrix|detail",
    "table":"table name from the catalog",
    "measure":"the REAL numerator/measure column (or null for a detail table); for a derived ratio this is the numerator column, the formula goes in query",
    "dimension":"dimension/temporal column (or null for kpi/detail)",
    "dimension2":"second dimension for a matrix (or null)",
    "comparison":"e.g. 'vs previous period' for a headline KPI delta (or null)",
    "polarity":"inverse if a RISING value is bad (cost/aging/returns/overdue), else positive",
    "metric_max":"100 ONLY for an intrinsic 0-100% ratio (collection/fill/payment rate); otherwise null",
    "viz":"kpi_card|metric_tile|line_chart|area_chart|bar_chart|ranked_bar|pie_chart|heatmap|funnel|table",
    "query":"a PRECISE analytical instruction for the SQL agent: the exact aggregation/formula (for a derived ratio state numerator/denominator explicitly), how to group, how to order, how many rows, and the exact output columns. For a cross-source view name the master key to join on and both measures to return.",
    "chart_rationale":"one line: why this chart suits an executive audience"}}
 ]}}"""

    def _run() -> list[dict]:
        client, _ = get_client()
        settings = get_settings()
        # The board brain is ONE call per dashboard. Use the SAME deployment the
        # rest of the dashboard layer uses (AZURE_OPENAI_DEPLOYMENT_MINI) — that is
        # the deployment actually wired in this environment. The richer persona
        # prompt above is what raises quality, not a larger model. Using the proven
        # deployment is also what prevents a silent drop to the terse decomposer.
        deployment = settings.AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=1800,
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
        polarity = (_clean("polarity") or "").lower() or None
        if polarity not in ("inverse", "positive"):
            polarity = None  # fail-safe: unknown direction -> default downstream
        # STRUCTURAL ratio ceiling — accept ONLY an exact 100 (the intrinsic
        # 0-100% bound). Any other value is not a real target and is dropped so a
        # gauge/progress/bullet cannot smuggle in a fabricated target (D.1).
        metric_max = None
        raw_max = w.get("metric_max")
        try:
            if raw_max is not None and float(raw_max) == 100.0:
                metric_max = 100.0
        except (TypeError, ValueError):
            metric_max = None
        specs.append(WidgetSpec(
            title=title,
            question_type=qt,
            table=_clean("table"),
            measure=_clean("measure"),
            dimension=_clean("dimension"),
            dimension2=_clean("dimension2"),
            comparison=_clean("comparison"),
            viz=viz,
            polarity=polarity,
            query=_clean("query"),
            chart_rationale=_clean("chart_rationale"),
            metric_max=metric_max,
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


def _column(table, name: str | None):
    if not name:
        return None
    for c in getattr(table, "columns", []):
        if c.name == name:
            return c
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vanity_measure_reason(table, name: str | None) -> str | None:
    """STRUCTURAL vanity check for a MEASURE column, from ingestion column_stats
    (min/max). Returns a human reason when the measure is meaningless to headline:
      - all-zero (min == max == 0)               → the SHIPPED_QUANTITY-all-0 class
      - single constant (min == max, non-null)   → no spread; an aggregate is noise
    Fail-OPEN: when min/max are absent (None) the guard cannot PROVE vanity and
    returns None (keep the widget) — never drop on missing evidence. No dataset-
    fitted literals; purely the column's own profiled spread."""
    col = _column(table, name)
    if col is None:
        return None
    lo, hi = getattr(col, "min_value", None), getattr(col, "max_value", None)
    if not (_is_number(lo) and _is_number(hi)):
        return None  # no numeric spread evidence → cannot prove vanity
    if lo == 0 and hi == 0:
        return f"the '{name}' column is all zero"
    if lo == hi:
        return f"the '{name}' column is a single constant value"
    return None


def _pick_dimension(table, max_card: int, exclude: set[str] | None = None) -> str | None:
    """Lowest-cardinality categorical dimension within `max_card`, else any dim.

    Skips SINGLE-CONSTANT dimensions (cardinality == 1, e.g. ORG_ID = 204): a
    breakdown across one value is a one-bar non-chart (D.3 vanity guard). A dim
    with NO cardinality info still serves as the fallback (fail-open on missing
    evidence)."""
    exclude = exclude or set()
    best: tuple[int, str] | None = None
    fallback: str | None = None
    for name in getattr(table, "dimensions", []) or []:
        if name in exclude:
            continue
        card = _card(table, name)
        # A KNOWN-constant dimension is never a usable breakdown axis.
        if card == 1:
            continue
        fallback = fallback or name
        if card is None:
            continue
        if card <= max_card and card >= 2:
            if best is None or card < best[0]:
                best = (card, name)
    if best:
        return best[1]
    return fallback


def _highest_card_dimension(
    table, max_card: int, *, min_card: int, exclude: set[str] | None = None
) -> str | None:
    """Highest-cardinality categorical dimension in [min_card, max_card] — the
    'who's driving it' ENTITY dimension (many vendors/customers/products). Returns
    None when no dimension in that band exists (so the caller falls back to the
    plain lowest-card pick). Used by the densifier to seat a top-N driver view."""
    exclude = exclude or set()
    best: tuple[int, str] | None = None
    for name in getattr(table, "dimensions", []) or []:
        if name in exclude:
            continue
        card = _card(table, name)
        if card is None:
            continue
        if min_card <= card <= max_card:
            if best is None or card > best[0]:
                best = (card, name)
    return best[1] if best else None


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

            # --- VANITY GUARD (D.3): drop a tile on an all-zero / single-constant
            # measure (e.g. the SHIPPED_QUANTITY-all-0 tile). Structural — driven by
            # the column's own profiled min/max, no fitted literals; fails OPEN when
            # stats are absent. A re-pick to a healthy measure could change the
            # metric the LLM intended, so we DROP (honest) rather than substitute.
            vanity = _vanity_measure_reason(t, s.measure)
            if vanity:
                dropped.append((s, vanity))
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

            # --- VANITY GUARD (D.3): a breakdown by a SINGLE-CONSTANT dimension
            # (cardinality == 1, e.g. ORG_ID = 204) is a one-bar non-chart. Re-pick
            # a non-constant readable dimension; if none exists, degrade to a plain
            # KPI on the measure rather than render a meaningless one-category chart.
            if s.question_type in ("breakdown", "share", "matrix") and _card(t, s.dimension) == 1:
                cap = _SHARE_MAX_CARD if s.question_type == "share" else _BREAKDOWN_MAX_CARD
                alt = _pick_dimension(t, cap, exclude={s.dimension or ""})
                if alt and _card(t, alt) != 1:
                    s.dimension = alt
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
        # A metric's identity is its TITLE: "Total Revenue" (SUM) and "Average
        # Revenue" (AVG) share table+measure+dims but are DIFFERENT KPIs. WidgetSpec
        # has no aggregation field, so without the title these collapse and real
        # KPIs get wrongly dropped as duplicates. Only an EXACT repeat (same title
        # + lattice) is a true duplicate.
        key = (s.question_type, s.table, s.measure, s.dimension, s.dimension2,
               (s.title or "").strip().lower())
        if key in seen:
            dropped.append((s, "duplicate of another widget"))
            continue
        seen.add(key)
        uniq.append(s)

    reasons = [f"Skipped '{s.title}' — {why}." for s, why in dropped]
    return uniq, reasons


# --------------------------------------------------------------------------
# S3.5 — DENSE COMPOSITION (deterministic, NO agent / NO SQL / NO LLM)
#
# A generated board must read like a PowerBI collage: a KPI ribbon of headline
# numbers, a trend, and a few breakdowns — at least ~5 VISUAL (non-table) widgets
# WHEN the data can honestly support them. The LLM still PROPOSES the metrics; this
# is a STRUCTURAL shaper that, from the SAME catalog columns the planner already
# resolved, fills out the composition shape the board is missing. It never
# fabricates a column and never pads thin data (honest > padded): a table with one
# measure / no temporal column yields fewer widgets, and that is correct.
# --------------------------------------------------------------------------

# Composition targets (STRUCTURAL shape, not dataset-fitted literals): aim for a
# headline ribbon plus a handful of breakdowns so the board is dense but not noisy.
_MIN_BOARD_VISUALS = 5      # the PowerBI-density floor (only when data supports it)
_MAX_RIBBON_KPIS = 4        # KPI ribbon width
_MAX_BREAKDOWNS = 3         # category breakdowns beyond the headline trend

# Structural cardinality boundary for the breakdown shaper (NOT dataset-fitted):
# a dimension with MORE distinct values than this reads as a "who's driving it"
# ENTITY dimension (many vendors/customers/products) — a full bar would be
# cluttered, so it becomes a top-N RANKED driver view. At/under it a plain bar is
# readable. Below the share ceiling (_SHARE_MAX_CARD) a part-of-whole donut reads
# best. These mirror the catalog visualization_rules (bar caps at 50, ranked_bar
# allows up to 1000), so the densifier and the recommender agree on the shape.
_RANK_MIN_CARD = _SHARE_MAX_CARD + 1   # above the share ceiling = an entity dim


def _polarity_map(specs: list[WidgetSpec]) -> dict[str, str]:
    """Build a measure -> LLM-authored polarity map from the feasible specs.

    The board-design LLM proposes per-measure direction ('inverse' when a rising
    value is bad). The densifier injects KPI/trend/breakdown widgets on those same
    measures, so it must CARRY that authored polarity (an inverse measure's rise
    must color red). Only specs that actually declared a polarity contribute — a
    measure the LLM never tagged is left ABSENT so the densifier keeps it None
    (honest: no Python inference of business direction). On a measure the LLM
    tagged inconsistently we keep the first 'inverse' seen (an inverse claim is the
    riskier/safer-to-preserve framing once the analyst asserted it)."""
    out: dict[str, str] = {}
    for s in specs:
        if not s.measure or not s.polarity:
            continue
        prior = out.get(s.measure)
        if prior is None or (prior != "inverse" and s.polarity == "inverse"):
            out[s.measure] = s.polarity
    return out


def _lattice_key(s: WidgetSpec) -> tuple:
    return (s.question_type, s.table, s.measure, s.dimension, s.dimension2)


def _measures(table) -> list[str]:
    return list(getattr(table, "measures", []) or [])


def _primary_table(specs: list[WidgetSpec], catalog: list):
    """The table the dense composition is built on: the one the most VISUAL seed
    specs reference, else the most measure-rich table in the catalog."""
    by_name = {t.table_name: t for t in catalog}
    counts: dict[str, int] = {}
    for s in specs:
        if s.question_type != "detail" and s.table in by_name:
            counts[s.table] = counts.get(s.table, 0) + 1
    if counts:
        best = max(counts, key=lambda k: counts[k])
        return by_name[best]
    return max(catalog, key=lambda t: len(_measures(t)), default=None)


def ensure_composition(
    specs: list[WidgetSpec], catalog: list, *, max_widgets: int
) -> list[WidgetSpec]:
    """Shape a feasible spec set into a dense PowerBI-style composition.

    Adds — ONLY from real catalog columns and ONLY when absent — a KPI ribbon, a
    trend (if a temporal column exists), and category breakdowns, until the board
    has >= _MIN_BOARD_VISUALS visual widgets OR the data is exhausted (whichever
    comes first). Existing specs are preserved in order and never duplicated. Never
    raises; on any structural gap it simply returns what it has.
    """
    specs = list(specs or [])
    catalog = catalog or []
    cap = max(1, int(max_widgets or MAX_WIDGETS))
    table = _primary_table(specs, catalog)
    if table is None:
        return specs[:cap]

    # The LLM authored the metric semantics (per-measure polarity). Carry that map
    # onto every densifier widget it injects so an inverse measure's KPI/trend/
    # delta colors correctly. A measure the LLM never tagged is ABSENT from the map
    # and stays None (no Python inference of business direction).
    polarity_by_measure = _polarity_map(specs)

    seen: set[tuple] = {_lattice_key(s) for s in specs}

    def _visual_count() -> int:
        return sum(1 for s in specs if s.question_type != "detail")

    def _try_add(s: WidgetSpec) -> bool:
        if len(specs) >= cap:
            return False
        key = _lattice_key(s)
        if key in seen:
            return False
        # Carry the LLM-authored polarity for this measure (None when unseen).
        if s.measure and s.polarity is None:
            s.polarity = polarity_by_measure.get(s.measure)
        seen.add(key)
        specs.append(s)
        return True

    tname = table.table_name
    measures = _measures(table)
    headline = measures[0] if measures else None
    temporal = list(getattr(table, "temporal", []) or [])

    # 1) KPI RIBBON — one headline KPI per measure (capped), each a real measure.
    for m in measures[:_MAX_RIBBON_KPIS]:
        if _visual_count() >= _MIN_BOARD_VISUALS and len(specs) >= 3:
            break
        _try_add(WidgetSpec(
            title=f"Total {m}", question_type="kpi", table=tname,
            measure=m, viz="kpi_card",
        ))

    # 2) TREND — one time series on the headline measure (only if a date exists).
    if headline and temporal and not any(s.question_type == "trend" for s in specs):
        _try_add(WidgetSpec(
            title=f"{headline} over time", question_type="trend", table=tname,
            measure=headline, dimension=temporal[0], viz="line_chart",
        ))

    # 3) BREAKDOWNS — break the headline measure across categorical dimensions,
    #    routed STRUCTURALLY by cardinality (mirrors the catalog rules, no fitted
    #    literals):
    #      card <= _SHARE_MAX_CARD  → part-of-whole share/donut (first such dim)
    #      card >= _RANK_MIN_CARD   → the "who's driving it" ENTITY dim → ranked_bar
    #                                  (top-N driver view; a full bar would clutter)
    #      mid                      → a plain category bar
    #    A high-cardinality entity dim is sought FIRST so the driver view is on the
    #    board; the recommender's scoring (bar caps at 50 card, ranked_bar at 1000)
    #    agrees with this shape end-to-end.
    if headline:
        used_dims: set[str] = {s.dimension for s in specs if s.dimension}
        added = 0
        share_used = any(s.question_type == "share" for s in specs)
        while added < _MAX_BREAKDOWNS and _visual_count() < _MIN_BOARD_VISUALS:
            # Prefer the entity ("who") dim — the highest-cardinality categorical
            # within the ranked ceiling — so the driver view is represented; else
            # fall back to the lowest-cardinality readable bar dim.
            dim = _highest_card_dimension(
                table, _RANK_MAX_CARD, min_card=_RANK_MIN_CARD, exclude=used_dims
            ) or _pick_dimension(table, _BREAKDOWN_MAX_CARD, exclude=used_dims)
            if not dim:
                break
            used_dims.add(dim)
            card = _card(table, dim)
            if card is not None and card >= _RANK_MIN_CARD:
                # Many distinct entities → top-N ranked driver view.
                qt, viz = "breakdown", "ranked_bar"
            elif card is not None and card <= _SHARE_MAX_CARD and not share_used:
                # A very-low-cardinality dimension reads best as a share/donut.
                qt, viz = "share", "pie_chart"
                share_used = True
            else:
                qt, viz = "breakdown", "bar_chart"
            if not _try_add(WidgetSpec(
                title=f"{headline} by {dim}", question_type=qt, table=tname,
                measure=headline, dimension=dim, viz=viz,
            )):
                break
            added += 1

    return specs[:cap]


# --------------------------------------------------------------------------
# S4 — specs → WidgetIntent (the existing run_widget contract)
# --------------------------------------------------------------------------

def _nl_query(spec: WidgetSpec, win_phrase: str) -> str:
    """Deterministic fallback instruction (explicit about aggregation, grouping,
    ordering and output shape) used only when the brain omits a detailed query."""
    m = spec.measure or "records"
    d = spec.dimension
    comp = f" Also compare {spec.comparison}." if spec.comparison else ""
    qt = spec.question_type
    if qt == "kpi":
        return (f"Return the single aggregated total of {m}{win_phrase} as one number "
                f"(the SUM of {m}).{comp}")
    if qt == "trend":
        return (f"Return a time series of {m}{win_phrase}: aggregate the SUM of {m} per "
                f"time period (use day or month granularity), output one row per period "
                f"with the period and the summed {m}, ordered chronologically.{comp}")
    if qt == "breakdown":
        return (f"Aggregate the SUM of {m} grouped by {d}{win_phrase}: output {d} and the "
                f"total {m}, returning the top 10 rows ordered by total {m} descending.{comp}")
    if qt == "share":
        return (f"Aggregate the SUM of {m} grouped by {d}{win_phrase}: output {d} and its "
                f"share of the overall total {m} (a part-of-whole breakdown).")
    if qt == "matrix":
        return (f"Aggregate the SUM of {m} grouped by BOTH {d} and {spec.dimension2}"
                f"{win_phrase}: output {d}, {spec.dimension2} and the total {m}.")
    # detail
    order = f", ordered by {spec.measure} descending" if spec.measure else ""
    return (f"Return the top 20 detail records from {spec.table}{win_phrase}{order}, "
            f"showing the most relevant columns.")


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

        # Prefer the brain's DETAILED instruction; fall back to the strict template.
        nl = (s.query or "").strip() or _nl_query(s, win_phrase)
        # Guarantee the shared window is present even if the brain forgot it.
        if win and win_phrase and not (str(win[0]) in nl or str(win[1]) in nl):
            nl = nl.rstrip(".") + f". Restrict to the period between {win[0]} and {win[1]}."

        # P0: pin the planner's validated lattice as the "planned" half of the
        # spec contract. nl_query is the exact re-run handle that goes to the agent.
        planned = {
            "question_type": s.question_type,
            "table": s.table,
            "measure": s.measure,
            "dimension": s.dimension,
            "dimension2": s.dimension2,
            "comparison": s.comparison,
            "viz": s.viz,
            "polarity": s.polarity,
            "chart_rationale": s.chart_rationale,
            # STRUCTURAL ratio ceiling (100) for an intrinsic 0-100% metric; the
            # recommender uses it as the ONLY honest gauge target (D.1).
            "metric_max": s.metric_max,
            "nl_query": nl,
        }

        intents.append(WidgetIntent(
            title=s.title,
            nl_query=nl,
            requested_viz=viz,
            hints=hints,
            spec={"schema_version": 1, "planned": planned},
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
    catalog = catalog or []
    catalog_size = len(catalog)

    # RC-004 GUARD: with NO grounding catalog there is nothing to plan against.
    # Running the ungrounded fallback here produced phantom widgets (the agent
    # resolved every vague question to the same wrong table). Surface ONE clear,
    # actionable state instead of a bag of misleading tiles + skip-warnings.
    if catalog_size == 0:
        chat_logger.warning(
            "dashboard_board_no_catalog",
            reason="empty_catalog",
            detail="no ingested, dashboard-ready data resolved for this scope",
        )
        return [], [
            "No analyzable data is available for this dashboard yet. Once files "
            "finish ingesting for this container, regenerate to build widgets."
        ]

    try:
        window = _board_window(catalog)
        raw = await _design_board(prompt, grounding_text, window, max_widgets)
        specs = _coerce_specs(raw, max_widgets)
    except Exception as exc:
        chat_logger.warning("dashboard_board_plan_error", error=str(exc)[:200])
        specs = []

    if specs:
        kept, reasons = feasibility_filter(specs, catalog)
        # S3.5: shape the feasible set into a dense PowerBI-style collage (KPI
        # ribbon + trend + breakdowns) using ONLY real catalog columns. Honest:
        # thin data stays sparse; this never fabricates or pads.
        dense = ensure_composition(kept, catalog, max_widgets=max_widgets)
        intents = specs_to_intents(dense[:max_widgets], catalog)
        if intents:
            # Only surface drop reasons when SOME widgets survived — a partial
            # plan. They describe THIS plan's discarded widgets, so they are
            # coherent with the rendered tiles.
            # Internal dedup bookkeeping ("duplicate of another widget") is not a
            # user-facing caveat — in the Analyst Notes panel it reads like an
            # error. Surface only genuine data-limitation reasons (no table / no
            # measure); the rest stay in telemetry below.
            warnings.extend(r for r in reasons if "duplicate" not in r.lower())
            chat_logger.info(
                "dashboard_board_planned",
                catalog_size=catalog_size,
                proposed=len(specs), feasible=len(intents), dropped=len(reasons),
            )
            return intents, warnings
        # Every proposed widget was infeasible against the catalog. Do NOT keep
        # the per-widget skip reasons AND run a different ungrounded fallback —
        # that is the RC-004 two-planner split. Log it as a degraded state.
        chat_logger.warning(
            "dashboard_board_all_infeasible",
            catalog_size=catalog_size, proposed=len(specs), dropped=len(reasons),
        )

    # Fallback: the original grounded single-pass decomposition. It still has the
    # full grounding_text, so it is grounded — just without the lattice/dry-run.
    # We deliberately do NOT carry forward the board planner's drop reasons here.
    chat_logger.warning(
        "dashboard_board_fallback_decompose",
        catalog_size=catalog_size, had_specs=bool(specs),
    )
    intents = await decompose_prompt(prompt, grounding_text, max_widgets=max_widgets)
    return intents, warnings

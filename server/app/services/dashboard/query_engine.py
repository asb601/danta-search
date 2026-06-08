"""
Dashboard Query Processing Engine (response.txt Section 6).

Responsibilities:
  1. decompose_prompt() - split ONE dashboard prompt into N discrete widget
     intents (LLM, grounded in the data catalog), honoring explicit chart-type
     requests. Deterministic guards bound N and degrade gracefully.
  2. run_widget() - thin wrapper over the EXISTING agent (run_agent_query).
     The agent does retrieval, planning, join resolution, SQL generation and
     execution. We do NOT re-implement any of that here.
  3. profile_dataset() - pure-Python profiling of a returned dataset into a
     DatasetShape that drives component recommendation.

This is the "fan-out coordinator" identified in the orchestrator review:
one prompt -> many agent calls -> many datasets.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from app.core.config import get_settings
from app.core.logger import chat_logger
from app.core.openai_client import get_client

# Render types we accept as explicit user requests (maps to ComponentType values).
_VIZ_KEYWORDS = {
    "kpi": "kpi_card",
    "kpi card": "kpi_card",
    "metric": "metric_tile",
    "tile": "metric_tile",
    "table": "table",
    "line": "line_chart",
    "line chart": "line_chart",
    "trend": "line_chart",
    "bar": "bar_chart",
    "bar chart": "bar_chart",
    "column chart": "bar_chart",
    "pie": "pie_chart",
    "pie chart": "pie_chart",
    "donut": "pie_chart",
    "area": "area_chart",
    "area chart": "area_chart",
    "heatmap": "heatmap",
    "heat map": "heatmap",
    "funnel": "funnel",
    "gauge": "gauge_ring",
    "gauge ring": "gauge_ring",
    "progress": "progress_kpi",
    "progress bar": "progress_kpi",
    "bullet": "bullet",
    "bullet chart": "bullet",
    "ranked bar": "ranked_bar",
    "top n": "ranked_bar",
    "leaderboard": "ranked_bar",
}

MAX_WIDGETS = 8


# --------------------------------------------------------------------------
# Transient models
# --------------------------------------------------------------------------

@dataclass
class WidgetIntent:
    title: str
    nl_query: str
    requested_viz: str | None = None   # e.g. "pie_chart" if user named one
    hints: dict = field(default_factory=dict)
    # P0: the planner's validated spec ({"schema_version", "planned": {...}}),
    # carried so the route can pin a planned+bound contract into the persisted
    # config. None on the decompose_prompt fallback path (no lattice available).
    spec: dict | None = None


@dataclass
class ColumnProfile:
    name: str
    dtype: str            # number | temporal | categorical | id | boolean
    kind: str             # measure | dimension | temporal | id
    cardinality: int
    null_ratio: float
    sample_values: list = field(default_factory=list)


@dataclass
class DatasetShape:
    row_count: int
    columns: list[ColumnProfile]
    measures: list[str]
    dimensions: list[str]
    temporal: list[str]
    aggregation: str            # SUM|COUNT|AVG|RAW|NONE
    is_time_series: bool
    is_single_value: bool
    is_distribution: bool
    intent: str                 # kpi|trend|comparison|distribution|detail|multi-dim


# --------------------------------------------------------------------------
# 1. Prompt decomposition
# --------------------------------------------------------------------------

def _detect_requested_viz(text: str) -> str | None:
    low = text.lower()
    # Longer keys first so "pie chart" wins over "pie".
    for kw in sorted(_VIZ_KEYWORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return _VIZ_KEYWORDS[kw]
    return None


def _safe_json(raw: str) -> dict | list | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Strip markdown fences / surrounding prose, retry on the first {...} or [...].
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


async def decompose_prompt(
    prompt: str,
    grounding_text: str,
    *,
    max_widgets: int = MAX_WIDGETS,
) -> list[WidgetIntent]:
    """
    Split a dashboard prompt into discrete widget intents. LLM-grounded with a
    compact catalog summary; never raises (deterministic fallback to a single
    widget so generation always proceeds).
    """
    max_widgets = max(1, min(int(max_widgets or MAX_WIDGETS), MAX_WIDGETS))

    def _run() -> list[dict]:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI

        sys = (
            "You are the planning stage of a dashboard generator. Split the user's "
            "dashboard request into a list of DISTINCT analytical questions, one per "
            "widget. Each question must be self-contained and answerable as a single "
            "analytical query against the available tables. Honor any explicitly "
            "requested chart type. Prefer 3-6 widgets for broad requests; use 1 for a "
            "single-question request. Never exceed the cap.\n\n"
            "GROUNDING RULES (critical — the catalog below is authoritative):\n"
            "- Prefer SINGLE-TABLE widgets. Only span two tables when a join between "
            "them is listed under KNOWN JOINS; never invent a join or join across "
            "unrelated business domains.\n"
            "- Reference ONLY columns shown for a table. Do NOT invent columns.\n"
            "- Use ONLY the categorical values shown under 'values:' for a column. "
            "Never invent a status/category literal (e.g. do not assume a status "
            "'Shipped' exists if the listed values are 'Open, In Process').\n"
            "- DATE WINDOWS: each temporal column shows its real 'date coverage'. If "
            "the user's requested period falls OUTSIDE that coverage, do NOT emit an "
            "impossible filter — either omit the time filter or use the covered range, "
            "so the widget returns data. Phrase the query against the data that exists.\n"
            "- For each widget also return the 'table' you intend to use and the "
            "'columns' it needs, drawn strictly from the catalog."
        )
        user = f"""Available tables (real columns, date coverage, observed values, and known joins):
{grounding_text or '(catalog unavailable - infer reasonable widgets from the prompt)'}

Dashboard request:
\"\"\"{prompt}\"\"\"

Return ONLY JSON of this shape (max {max_widgets} widgets):
{{"widgets": [
  {{"title": "short title",
    "query": "a single natural-language analytical question grounded in the tables above",
    "table": "the primary table name from the catalog (or null)",
    "columns": ["column names this widget needs, from the catalog"],
    "viz": "kpi_card|metric_tile|table|line_chart|bar_chart|pie_chart|area_chart|heatmap|funnel|null"}}
]}}"""

        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=900,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = _safe_json(raw) or {}
        widgets = parsed.get("widgets") if isinstance(parsed, dict) else None
        return widgets if isinstance(widgets, list) else []

    widgets_raw: list[dict] = []
    try:
        widgets_raw = await asyncio.to_thread(_run)
    except Exception as exc:
        chat_logger.warning("dashboard_decompose_error", error=str(exc)[:200])

    intents: list[WidgetIntent] = []
    for w in widgets_raw[:max_widgets]:
        if not isinstance(w, dict):
            continue
        nl = str(w.get("query") or w.get("title") or "").strip()
        if not nl:
            continue
        viz = w.get("viz")
        viz = str(viz).strip().lower() if viz and str(viz).lower() != "null" else None
        if viz and viz not in _VIZ_KEYWORDS.values():
            viz = None
        # If the model didn't tag a viz but the user clearly named one, capture it.
        if not viz:
            viz = _detect_requested_viz(nl)
        # Capture the grounded table/columns the planner intends to use so
        # run_widget can build a per-widget grounding block for the agent.
        hints: dict = {}
        tbl = w.get("table")
        if tbl and str(tbl).lower() != "null":
            hints["table"] = str(tbl)
        cols = w.get("columns")
        if isinstance(cols, list):
            hints["columns"] = [str(c) for c in cols if c][:24]
        intents.append(
            WidgetIntent(
                title=str(w.get("title") or nl)[:120],
                nl_query=nl,
                requested_viz=viz,
                hints=hints,
            )
        )

    if not intents:
        # Deterministic fallback: one widget from the raw prompt.
        intents = [
            WidgetIntent(
                title=prompt.strip()[:120] or "Analysis",
                nl_query=prompt.strip(),
                requested_viz=_detect_requested_viz(prompt),
            )
        ]
    return intents


# --------------------------------------------------------------------------
# 2. Run a widget through the existing agent
# --------------------------------------------------------------------------

def _build_widget_grounding(intent: WidgetIntent, catalog: list | None, *, applied_filters: list | None = None) -> str:
    """
    Build a short, agent-facing grounding block for ONE widget from the planner's
    hinted table + the matching catalog row. Passed via conversation_context
    (the only free-text channel into the agent's system prompt) so the agent does
    not hallucinate columns/joins/date windows. Kept compact and placed so it
    survives the prompt builder's tail truncation. No agent code changes.
    """
    if not catalog:
        return ""
    table = (intent.hints or {}).get("table")
    if not table:
        return ""
    match = None
    for t in catalog:
        if getattr(t, "table_name", None) == table:
            match = t
            break
    if match is None:
        return ""

    lines = [
        "DASHBOARD WIDGET DATA GROUNDING (authoritative — obey strictly):",
        f"- Primary table: {match.table_name}.",
    ]
    cols = [c.name for c in getattr(match, "columns", [])][:30]
    if cols:
        lines.append(f"- Valid columns (use ONLY these): {', '.join(cols)}.")
    coverage = match.date_coverage() if hasattr(match, "date_coverage") else []
    if coverage:
        lines.append(
            "- Date coverage: " + "; ".join(coverage)
            + ". If the question's period is outside this, query the covered range "
            "instead of returning no data."
        )
    # Real categorical values so the agent never invents a status literal.
    val_bits = []
    dims = set(getattr(match, "dimensions", []))
    for c in getattr(match, "columns", []):
        if c.name in dims and getattr(c, "top_values", None) and (c.cardinality or 99) <= 20:
            val_bits.append(f"{c.name} ∈ {{{', '.join(str(v) for v in c.top_values[:8])}}}")
    if val_bits:
        lines.append("- Real values (never invent others): " + " | ".join(val_bits[:6]) + ".")
    # P2 ADDITIVITY (G2): if the planned measure's ingestion role is non-additive
    # (ratio/rate/percentage), instruct the agent never to SUM it. Role-driven only
    # (never the column name); fail-closed (no role -> no directive, no claim).
    measure = ((intent.spec or {}).get("planned") or {}).get("measure")
    if measure:
        from app.services import semantic_roles as _sr
        from app.services.dashboard.data_catalog import role_map_for_table

        if _sr.is_non_additive_measure_role(role_map_for_table(match).get(measure)):
            lines.append(
                f"- ADDITIVITY: '{measure}' is a non-additive measure (ratio/rate/"
                f"percentage). NEVER SUM it across rows — AVERAGE it, or recompute it "
                f"from its numerator and denominator. Summing it is meaningless."
            )
    # P7 GLOBAL FILTER: a board-level slicer on a CONFORMED dimension, pre-resolved
    # to THIS widget's physical column. Text-only (the agent applies the predicate in
    # its own SQL); values come from the real observed members, never invented.
    for f in (applied_filters or []):
        vals = ", ".join(str(v) for v in (f.get("values") or []))
        lines.append(
            f"- GLOBAL FILTER (board-level, authoritative): restrict ALL results to rows "
            f"where {f.get('column')} ∈ {{{vals}}}. Apply this to every aggregation in this widget."
        )
    lines.append(
        "- Prefer this single table. Do not join to a different business domain. "
        "Do not invent columns or filter values."
    )
    return "\n".join(lines)


async def run_widget(
    intent: WidgetIntent, *, db, scope: dict, catalog: list | None = None,
    applied_filters: list | None = None,
) -> dict:
    """
    Execute one widget intent by REUSING the existing agent. Returns the agent
    result dict ({answer,data,chart,row_count,files_used,...}); never raises.
    Imported lazily to avoid import cycles at module load.
    """
    from app.agent import run_agent_query

    grounding = _build_widget_grounding(intent, catalog, applied_filters=applied_filters)

    try:
        result = await run_agent_query(
            intent.nl_query,
            db,
            conversation_context=grounding,
            user_id=scope.get("user_id", ""),
            is_admin=scope.get("is_admin", False),
            allowed_domains=scope.get("allowed_domains"),
            container_id=scope.get("container_id"),
            prior_files=None,
            actor_email=scope.get("actor_email", ""),
            actor_role=scope.get("actor_role", ""),
        )
        return result or {}
    except Exception as exc:
        chat_logger.warning(
            "dashboard_widget_error", title=intent.title, error=str(exc)[:200]
        )
        return {"answer": f"Could not generate '{intent.title}'.", "data": [], "chart": None,
                "row_count": 0, "files_used": [], "error": str(exc)[:200]}


async def run_widgets(intents: list, run_one, *, concurrency: int, parallel: bool) -> list:
    """Fan-out coordinator (P3). Run `run_one(intent)` for every intent and return
    the results IN INPUT ORDER.

    parallel=True -> bounded `asyncio.gather` under a semaphore of `concurrency`
    (each `run_one` must supply its own resources, e.g. a fresh DB session, since
    the request session is not concurrency-safe); else strictly sequential. Only the
    I/O-bound `run_one` runs concurrently — callers MUST keep every shared-state
    transform (profiling, recommend, warnings, aggregation) in a SEQUENTIAL pass over
    the returned list, so output is byte-identical regardless of the flag.

    In parallel mode exceptions are returned IN PLACE (`return_exceptions=True`) so a
    single failing widget never cancels its siblings; the caller maps them. Sequential
    mode preserves today's behavior (a raise propagates).
    """
    if not intents:
        return []
    if not parallel:
        return [await run_one(intent) for intent in intents]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(intent):
        async with sem:
            return await run_one(intent)

    return await asyncio.gather(*[_guarded(i) for i in intents], return_exceptions=True)


# --------------------------------------------------------------------------
# 3. Dataset profiling
# --------------------------------------------------------------------------

_TEMPORAL_NAME = re.compile(r"(date|day|week|month|quarter|year|period|time|ts|timestamp)", re.I)
_NUMERIC_TYPES = ("int", "float", "double", "decimal", "number", "numeric", "bigint")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _looks_temporal(name: str, values: list) -> bool:
    if _TEMPORAL_NAME.search(name or ""):
        return True
    for v in values[:8]:
        if isinstance(v, (date, datetime)):
            return True
        if isinstance(v, str) and re.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?", v):
            return True
    return False


def profile_dataset(rows: list[dict], chart_hint: dict | None = None) -> DatasetShape:
    """Pure profiling of a dataset into a DatasetShape (drives recommendation)."""
    rows = rows or []
    n = len(rows)
    if n == 0:
        return DatasetShape(
            row_count=0, columns=[], measures=[], dimensions=[], temporal=[],
            aggregation="NONE", is_time_series=False, is_single_value=False,
            is_distribution=False, intent="detail",
        )

    col_names: list[str] = list(rows[0].keys())
    columns: list[ColumnProfile] = []
    measures: list[str] = []
    dimensions: list[str] = []
    temporal: list[str] = []

    for name in col_names:
        values = [r.get(name) for r in rows]
        non_null = [v for v in values if v is not None]
        null_ratio = round(1 - (len(non_null) / n), 4) if n else 0.0
        distinct = len({v for v in non_null if not isinstance(v, (list, dict))})
        numeric = bool(non_null) and all(_is_number(v) for v in non_null)
        temporal_col = _looks_temporal(name, non_null)
        id_like = bool(re.search(r"(^id$|_id$|code$|number$|^key$)", name or "", re.I))

        if temporal_col:
            kind, dtype = "temporal", "temporal"
            temporal.append(name)
        elif numeric and not id_like:
            kind, dtype = "measure", "number"
            measures.append(name)
        elif id_like:
            kind, dtype = "id", "id"
        else:
            kind, dtype = "dimension", "categorical"
            dimensions.append(name)

        columns.append(
            ColumnProfile(
                name=name, dtype=dtype, kind=kind, cardinality=distinct,
                null_ratio=null_ratio, sample_values=non_null[:5],
            )
        )

    is_single_value = (n == 1 and len(measures) >= 1 and len(dimensions) == 0 and len(temporal) == 0)
    is_time_series = bool(temporal) and bool(measures)
    # Distribution (part-of-whole -> pie): one small-cardinality categorical dim,
    # one measure, AND a share-like measure name. Otherwise a category+measure
    # split is a COMPARISON (-> bar), which matches user expectations for
    # phrasings like "revenue by region".
    primary_dim_card = max([c.cardinality for c in columns if c.kind == "dimension"], default=0)
    _share_like = re.compile(r"(share|pct|percent|ratio|proportion|distribution|mix|split)", re.I)
    has_share_measure = any(_share_like.search(m or "") for m in measures)
    is_distribution = (
        len(dimensions) == 1
        and len(measures) == 1
        and 2 <= n <= 6
        and primary_dim_card <= 6
        and has_share_measure
    )

    aggregation = "RAW"
    if is_single_value or (len(measures) >= 1 and (dimensions or temporal)):
        aggregation = "SUM"

    # Intent classification (seeded by the agent's own chart hint when present).
    if is_single_value:
        intent = "kpi"
    elif is_time_series:
        intent = "trend"
    elif len(dimensions) >= 2 and measures:
        intent = "multi-dim"
    elif is_distribution:
        intent = "distribution"
    elif len(dimensions) == 1 and measures:
        intent = "comparison"
    else:
        intent = "detail"

    hint_type = (chart_hint or {}).get("type") if isinstance(chart_hint, dict) else None
    if hint_type == "line" and bool(temporal):
        intent = "trend"
    elif hint_type == "pie" and is_distribution:
        intent = "distribution"

    return DatasetShape(
        row_count=n, columns=columns, measures=measures, dimensions=dimensions,
        temporal=temporal, aggregation=aggregation, is_time_series=is_time_series,
        is_single_value=is_single_value, is_distribution=is_distribution, intent=intent,
    )

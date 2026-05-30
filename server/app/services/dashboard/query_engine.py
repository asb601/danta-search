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
            "single-question request. Never exceed the cap."
        )
        user = f"""Available tables (name [domain] rows: description; measures; dimensions):
{grounding_text or '(catalog unavailable - infer reasonable widgets from the prompt)'}

Dashboard request:
\"\"\"{prompt}\"\"\"

Return ONLY JSON of this shape (max {max_widgets} widgets):
{{"widgets": [
  {{"title": "short title",
    "query": "a single natural-language analytical question",
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
        intents.append(
            WidgetIntent(
                title=str(w.get("title") or nl)[:120],
                nl_query=nl,
                requested_viz=viz,
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

async def run_widget(intent: WidgetIntent, *, db, scope: dict) -> dict:
    """
    Execute one widget intent by REUSING the existing agent. Returns the agent
    result dict ({answer,data,chart,row_count,files_used,...}); never raises.
    Imported lazily to avoid import cycles at module load.
    """
    from app.agent import run_agent_query

    try:
        result = await run_agent_query(
            intent.nl_query,
            db,
            conversation_context="",
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

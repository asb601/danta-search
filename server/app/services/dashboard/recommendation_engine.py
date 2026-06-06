"""
Component Recommendation Engine (response.txt Section 7).

Given a DatasetShape + WidgetIntent, score every catalog component against the
component's visualization_rules (read as DATA), pick the best, and bind dataset
columns to the component's config schema. Explicit user-requested chart types
win. Deterministic and explainable — no LLM. A future ML ranker can replace
score_component() behind the same recommend() signature.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

from app.services.dashboard.component_catalog import (
    ComponentDefinition,
    ComponentType,
    fallback_component,
    list_components,
)
from app.services.dashboard.query_engine import DatasetShape, WidgetIntent


@dataclass
class ResolvedWidget:
    widget_id: str
    component_id: str
    component_type: str
    title: str
    dataset: list
    config: dict
    score: float
    rationale: str
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _n_measures(shape: DatasetShape) -> int:
    return len(shape.measures)


def _n_dims(shape: DatasetShape) -> int:
    return len(shape.dimensions) + len(shape.temporal)


def _max_dim_cardinality(shape: DatasetShape) -> int:
    cards = [c.cardinality for c in shape.columns if c.kind in ("dimension", "temporal")]
    return max(cards) if cards else 0


def score_component(comp: ComponentDefinition, shape: DatasetShape, intent: WidgetIntent) -> float:
    """Score how well a component fits a dataset. Higher is better; <=0 means unfit."""
    rules = comp.visualization_rules or {}
    score = 0.0

    # Hard structural gates -------------------------------------------------
    if "max_rows" in rules and shape.row_count > rules["max_rows"]:
        return -1.0
    if "min_rows" in rules and shape.row_count < rules["min_rows"]:
        return -1.0
    if rules.get("requires_temporal") and not shape.temporal:
        return -1.0

    nm, nd = _n_measures(shape), _n_dims(shape)
    if "n_measures" in rules and nm < rules["n_measures"]:
        # KPI/charts need at least one measure.
        return -1.0
    if "n_dimensions" in rules:
        need = rules["n_dimensions"]
        if need == 0 and nd != 0:
            return -1.0
        if need >= 1 and nd < need:
            return -1.0
    if "max_dimension_cardinality" in rules:
        if _max_dim_cardinality(shape) > rules["max_dimension_cardinality"]:
            return -1.0

    # Soft preference scoring ----------------------------------------------
    score += 1.0  # passed gates
    preferred = rules.get("preferred_intents", [])
    if shape.intent in preferred:
        score += 4.0
    # Exact dimensionality match is better than a loose one.
    if rules.get("n_dimensions") == nd:
        score += 1.0
    if rules.get("n_measures") == nm:
        score += 0.5

    # Best-practice nudges.
    if comp.component_type == ComponentType.PIE_CHART and _max_dim_cardinality(shape) > 6:
        score -= 1.0
    if comp.component_type == ComponentType.TABLE and shape.intent != "detail":
        score -= 0.5  # table is the fallback, not the first choice

    # Priority acts as a stable tie-breaker (scaled small).
    score += comp.priority / 1000.0
    return score


def _format_for_role(name: str, shape: DatasetShape) -> str:
    low = name.lower()
    if any(k in low for k in ("amount", "revenue", "cost", "price", "value", "sales", "spend")):
        return "currency"
    if any(k in low for k in ("pct", "percent", "rate", "ratio", "share")):
        return "percent"
    return "number"


def _bind_config(comp: ComponentDefinition, shape: DatasetShape) -> dict:
    """Bind dataset columns to the component's config schema."""
    ct = comp.component_type
    measure = shape.measures[0] if shape.measures else None
    dim = shape.dimensions[0] if shape.dimensions else None
    temporal = shape.temporal[0] if shape.temporal else None

    if ct in (ComponentType.KPI_CARD, ComponentType.METRIC_TILE):
        return {
            "value": measure,
            "label": comp.name if not measure else measure,
            "format": _format_for_role(measure or "", shape),
        }
    if ct in (ComponentType.LINE_CHART, ComponentType.AREA_CHART):
        x = temporal or dim
        series = shape.dimensions[0] if (temporal and shape.dimensions) else None
        return {
            "x": x,
            "y": shape.measures[:3] or [measure],
            "series": series,
            "format": _format_for_role(measure or "", shape),
        }
    if ct == ComponentType.BAR_CHART:
        return {
            "x": dim or temporal,
            "y": measure,
            "orientation": "vertical",
            "format": _format_for_role(measure or "", shape),
        }
    if ct == ComponentType.PIE_CHART:
        return {"label": dim, "value": measure, "format": _format_for_role(measure or "", shape)}
    if ct == ComponentType.HEATMAP:
        x = shape.dimensions[0] if len(shape.dimensions) >= 1 else temporal
        y = shape.dimensions[1] if len(shape.dimensions) >= 2 else (temporal or dim)
        return {"x": x, "y": y, "value": measure, "format": _format_for_role(measure or "", shape)}
    if ct == ComponentType.FUNNEL:
        return {"stage": dim, "value": measure, "format": _format_for_role(measure or "", shape)}
    # TABLE — show everything, with type-aware formatting handled by the renderer.
    return {"columns": [c.name for c in shape.columns]}


def _empty_widget(intent: WidgetIntent, provenance: dict) -> ResolvedWidget:
    return ResolvedWidget(
        widget_id=uuid.uuid4().hex[:12],
        component_id="table.detail.v1",
        component_type=ComponentType.TABLE.value,
        title=intent.title,
        dataset=[],
        config={"columns": []},
        score=0.0,
        rationale="No data was returned for this question.",
        provenance={**provenance, "empty": True},
    )


def build_pinned_spec(intent: WidgetIntent, widget: ResolvedWidget, shape: DatasetShape) -> dict:
    """
    P0: build the planned+bound contract pinned into the persisted config.

    `planned` = the planner's lattice (what was ASKED; None on the fallback path).
    `bound`   = what the recommender actually bound/rendered.

    Faithfulness rules (data-science gate):
    - aggregation is recorded as `aggregation_inferred` — it is profiled from the
      result shape, NOT the aggregate the agent's SQL actually applied. Never label
      it as executed.
    - No `sql` field: run_agent_query does not surface executed SQL, so a re-derived
      query would be an unfaithful claim. The honest re-run handle (nl_query,
      files_used, route, row_count) lives in the parent provenance dict.
    """
    planned = (intent.spec or {}).get("planned")
    bound = {
        "component_id": widget.component_id,
        "component_type": widget.component_type,
        "config": widget.config,
        "aggregation_inferred": getattr(shape, "aggregation", None),
        "score": widget.score,
        "rationale": widget.rationale,
    }
    return {
        "schema_version": 1,
        "planned": planned,
        "bound": bound,
        "empty": not widget.dataset,
    }


def recommend(
    shape: DatasetShape,
    intent: WidgetIntent,
    dataset: list,
    *,
    provenance: dict | None = None,
) -> ResolvedWidget:
    """Pick the best component for a dataset and bind its config."""
    provenance = provenance or {}

    if not dataset or shape.row_count == 0:
        return _empty_widget(intent, provenance)

    # STEP 1 — explicit user request wins if it can bind the dataset.
    if intent.requested_viz:
        candidates = [c for c in list_components() if c.component_type.value == intent.requested_viz]
        for comp in candidates:
            if score_component(comp, shape, intent) > 0 or comp.component_type == ComponentType.TABLE:
                return ResolvedWidget(
                    widget_id=uuid.uuid4().hex[:12],
                    component_id=comp.component_id,
                    component_type=comp.component_type.value,
                    title=intent.title,
                    dataset=dataset,
                    config=_bind_config(comp, shape),
                    score=99.0,
                    rationale=f"Used the explicitly requested {comp.name}.",
                    provenance=provenance,
                )

    # STEP 2 — rule scoring across the whole catalog.
    best: ComponentDefinition | None = None
    best_score = 0.0
    for comp in list_components():
        s = score_component(comp, shape, intent)
        if s > best_score:
            best, best_score = comp, s

    # STEP 3 — fallback to a table when nothing matched confidently.
    if best is None:
        best = fallback_component()
        best_score = 0.5
        rationale = "Defaulted to a data table — no chart confidently matched the dataset shape."
    else:
        rationale = (
            f"Selected {best.name} for a '{shape.intent}' dataset "
            f"({shape.row_count} rows, {len(shape.measures)} measure(s), "
            f"{len(shape.dimensions) + len(shape.temporal)} dimension(s))."
        )

    return ResolvedWidget(
        widget_id=uuid.uuid4().hex[:12],
        component_id=best.component_id,
        component_type=best.component_type.value,
        title=intent.title,
        dataset=dataset,
        config=_bind_config(best, shape),
        score=round(best_score, 3),
        rationale=rationale,
        provenance=provenance,
    )

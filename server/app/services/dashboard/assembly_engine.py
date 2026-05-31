"""
Dashboard Assembly Engine (response.txt Section 8).

Orders resolved widgets by information hierarchy (KPIs first, then trends,
comparisons, distributions, detail tables), packs them into a responsive
12-column grid using each component's default size, and emits a versioned
DashboardConfig — the persisted, render-ready contract consumed by the frontend
renderer.
"""
from __future__ import annotations

from app.services.dashboard.recommendation_engine import ResolvedWidget

# Masonry layout: the frontend renders a responsive, gap-filling column grid
# (CSS grid-auto-flow: dense). The backend does NOT compute absolute x/y — it
# assigns each widget a DATA-ADAPTIVE SIZE INTENT only:
#   - w = column span (1..MASONRY_COLS) → tile width
#   - h = row span (in renderer row units) → tile height
# Size = component type × data shape (rows / cols / points) × importance (hero).
# The renderer packs these into whatever column count the viewport allows.
MASONRY_COLS = 4

# Render order by component type (lower sorts first). Defines the narrative the
# dense packer flows through: KPIs → trends → comparisons → distributions → detail.
_TYPE_ORDER = {
    "kpi_card": 0,
    "metric_tile": 1,
    "line_chart": 2,
    "area_chart": 3,
    "bar_chart": 4,
    "pie_chart": 5,
    "funnel": 6,
    "heatmap": 7,
    "table": 8,
}


def _shape(widget: ResolvedWidget) -> tuple[int, int]:
    """(row_count, column_count) of the widget's bound dataset."""
    data = widget.dataset or []
    rows = len(data)
    if rows and isinstance(data[0], dict):
        cols = len(data[0])
    else:
        cols = len((widget.config or {}).get("columns") or [])
    return rows, cols


def _size_widget(widget: ResolvedWidget, *, is_hero: bool) -> dict:
    """
    Data-adaptive size intent for one widget → {x, y, w, h}.
    w = column span, h = row span. Driven by component type × data shape ×
    importance (NOT a static per-type default). x/y are unused by the masonry
    renderer (kept at 0 for contract compatibility).
    """
    t = widget.component_type
    rows, cols = _shape(widget)

    if t in ("kpi_card", "metric_tile"):
        col, row = (2, 3) if is_hero else (1, 2)
    elif t in ("line_chart", "area_chart"):
        if is_hero or rows >= 12:
            col, row = 4, 6          # hero / dense trend → full-width band
        elif rows <= 3:
            col, row = 2, 4          # sparse trend → compact
        else:
            col, row = 2, 5
    elif t == "bar_chart":
        col = 3 if rows >= 12 else 2
        row = 5
    elif t == "pie_chart":
        col, row = 1, 4
    elif t == "funnel":
        col, row = 1, 4
    elif t == "heatmap":
        col, row = 2, 5
    elif t == "table":
        col = 4 if cols >= 6 else 2
        row = 4 if rows <= 6 else (6 if rows <= 15 else 8)
    else:
        col, row = 2, 4

    col = max(1, min(MASONRY_COLS, col))
    return {"x": 0, "y": 0, "w": col, "h": row}


def assemble(
    widgets: list[ResolvedWidget],
    *,
    title: str,
    description: str | None,
    prompt: str,
    generated_at: str,
    warnings: list[str] | None = None,
) -> dict:
    """Build the versioned DashboardConfig from resolved widgets."""
    ordered = sorted(
        widgets,
        key=lambda w: (_TYPE_ORDER.get(w.component_type, 99), -w.score),
    )
    # The single highest-scoring widget is the board's hero (gets a larger tile).
    hero_id = max(ordered, key=lambda w: w.score).widget_id if ordered else None

    widget_payloads = []
    for w in ordered:
        widget_payloads.append(
            {
                "widget_id": w.widget_id,
                "component_id": w.component_id,
                "type": w.component_type,
                "title": w.title,
                "grid": _size_widget(w, is_hero=(w.widget_id == hero_id)),
                "config": w.config,
                "data": w.dataset,
                "rationale": w.rationale,
                "score": w.score,
                "provenance": w.provenance,
            }
        )

    return {
        "version": "1.1",
        "title": title,
        "description": description or "",
        "generated_at": generated_at,
        "prompt": prompt,
        "layout": "masonry",
        "columns": MASONRY_COLS,
        "widgets": widget_payloads,
        "warnings": warnings or [],
    }

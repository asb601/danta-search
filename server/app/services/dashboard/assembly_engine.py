"""
Dashboard Assembly Engine (response.txt Section 8).

Orders resolved widgets by information hierarchy (KPIs first, then trends,
comparisons, distributions, detail tables), packs them into a responsive
12-column grid using each component's default size, and emits a versioned
DashboardConfig — the persisted, render-ready contract consumed by the frontend
renderer.
"""
from __future__ import annotations

from app.services.dashboard.component_catalog import get_component
from app.services.dashboard.recommendation_engine import ResolvedWidget

GRID_COLS = 12

# Render order by component type (lower sorts first).
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


def _default_size(component_id: str) -> dict:
    comp = get_component(component_id)
    if comp:
        size = comp.rendering_metadata.get("default_size") or {}
        return {"w": int(size.get("w", 6)), "h": int(size.get("h", 4))}
    return {"w": 6, "h": 4}


def _flow_layout(widgets: list[ResolvedWidget]) -> list[dict]:
    """
    Left-to-right row-packing into a 12-column grid. Returns parallel list of
    {x,y,w,h} dicts aligned with `widgets`.
    """
    grids: list[dict] = []
    cursor_x = 0
    row_y = 0
    row_h = 0
    for w in widgets:
        size = _default_size(w.component_id)
        wd = min(size["w"], GRID_COLS)
        if cursor_x + wd > GRID_COLS:
            # wrap to next row
            row_y += row_h
            cursor_x = 0
            row_h = 0
        grids.append({"x": cursor_x, "y": row_y, "w": wd, "h": size["h"]})
        cursor_x += wd
        row_h = max(row_h, size["h"])
    return grids


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
    grids = _flow_layout(ordered)

    widget_payloads = []
    for w, grid in zip(ordered, grids):
        widget_payloads.append(
            {
                "widget_id": w.widget_id,
                "component_id": w.component_id,
                "type": w.component_type,
                "title": w.title,
                "grid": grid,
                "config": w.config,
                "data": w.dataset,
                "rationale": w.rationale,
                "score": w.score,
                "provenance": w.provenance,
            }
        )

    return {
        "version": "1.0",
        "title": title,
        "description": description or "",
        "generated_at": generated_at,
        "prompt": prompt,
        "layout": "grid-12",
        "widgets": widget_payloads,
        "warnings": warnings or [],
    }

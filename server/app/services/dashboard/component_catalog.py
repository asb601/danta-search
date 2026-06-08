"""
Dashboard Component Catalog — a metadata-driven registry of reusable
visualization components (response.txt Section 4).

Design principle: visualization logic is DATA, not code. Each component is a
ComponentDefinition describing what it can render (supported metrics/dimensions),
how it aggregates/filters, the MATCHING PREDICATE used by the recommendation
engine (visualization_rules), the config keys the renderer binds
(config_schema), and the frontend component + default layout size
(rendering_metadata).

Adding the 301st component is a registry change here (+ a React renderer only
for genuinely new visuals). The recommendation engine and the frontend both
consume this registry generically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class ComponentType(str, Enum):
    KPI_CARD = "kpi_card"
    METRIC_TILE = "metric_tile"
    TABLE = "table"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    AREA_CHART = "area_chart"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    GAUGE_RING = "gauge_ring"
    PROGRESS_KPI = "progress_kpi"
    RANKED_BAR = "ranked_bar"
    DELTA_KPI = "delta_kpi"
    BULLET = "bullet"


@dataclass
class ComponentDefinition:
    component_id: str
    name: str
    component_type: ComponentType
    description: str
    supported_metrics: list[str]          # measure roles or ["*"]
    supported_dimensions: list[str]       # ["*","temporal","categorical"]
    required_tables: int                  # min distinct tables (informational)
    required_joins: bool
    aggregation_rules: dict               # {allowed:[...], default:...}
    filtering_rules: dict                 # {supports_time_filter, supports_topn}
    visualization_rules: dict             # matching predicate (read as data)
    config_schema: dict                   # render config keys
    rendering_metadata: dict              # {frontend_component, icon, default_size}
    priority: int = 50                    # tie-breaker (higher wins)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["component_type"] = self.component_type.value
        return d


# --------------------------------------------------------------------------
# Seed catalog — the nine required component types. Each row is metadata only.
# Variants (stacked bar, multi-series line, ...) are added as additional rows.
# --------------------------------------------------------------------------

_CATALOG: list[ComponentDefinition] = [
    ComponentDefinition(
        component_id="kpi.single_value.v1",
        name="KPI Card",
        component_type=ComponentType.KPI_CARD,
        description="A single headline metric (e.g. total revenue).",
        supported_metrics=["*"],
        supported_dimensions=[],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG", "MAX", "MIN"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "max_rows": 1,
            "n_measures": 1,
            "n_dimensions": 0,
            "requires_temporal": False,
            "preferred_intents": ["kpi"],
        },
        config_schema={"value": "measure", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "KpiCard",
            "icon": "Hash",
            "default_size": {"w": 3, "h": 2},
            "palette": "chart",
        },
        priority=90,
    ),
    ComponentDefinition(
        component_id="metric.tile.v1",
        name="Metric Tile",
        component_type=ComponentType.METRIC_TILE,
        description="A compact metric with optional secondary/delta value.",
        supported_metrics=["*"],
        supported_dimensions=[],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "max_rows": 2,
            "n_measures": 1,
            "n_dimensions": 0,
            "requires_temporal": False,
            "preferred_intents": ["kpi"],
        },
        config_schema={"value": "measure", "delta": "measure?", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "MetricTile",
            "icon": "Gauge",
            "default_size": {"w": 3, "h": 2},
            "palette": "chart",
        },
        priority=70,
    ),
    ComponentDefinition(
        component_id="chart.line.v1",
        name="Line Chart",
        component_type=ComponentType.LINE_CHART,
        description="Trend of one or more measures over time.",
        supported_metrics=["*"],
        supported_dimensions=["temporal"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "AVG", "COUNT"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": True,
            "preferred_intents": ["trend"],
        },
        config_schema={"x": "temporal", "y": "measure[]", "series": "dimension?"},
        rendering_metadata={
            "frontend_component": "LineChart",
            "icon": "TrendingUp",
            "default_size": {"w": 8, "h": 4},
            "palette": "chart",
            "supports_legend": True,
        },
        priority=85,
    ),
    ComponentDefinition(
        component_id="chart.area.v1",
        name="Area Chart",
        component_type=ComponentType.AREA_CHART,
        description="Volume/cumulative trend over time.",
        supported_metrics=["*"],
        supported_dimensions=["temporal"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "AVG"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": True,
            "preferred_intents": ["trend"],
        },
        config_schema={"x": "temporal", "y": "measure[]", "series": "dimension?"},
        rendering_metadata={
            "frontend_component": "AreaChart",
            "icon": "AreaChart",
            "default_size": {"w": 8, "h": 4},
            "palette": "chart",
            "supports_legend": True,
        },
        priority=60,
    ),
    ComponentDefinition(
        component_id="chart.bar.v1",
        name="Bar Chart",
        component_type=ComponentType.BAR_CHART,
        description="Compare a measure across categories.",
        supported_metrics=["*"],
        supported_dimensions=["categorical"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": True},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": False,
            "max_dimension_cardinality": 50,
            "preferred_intents": ["comparison"],
        },
        config_schema={"x": "dimension", "y": "measure", "orientation": "vertical"},
        rendering_metadata={
            "frontend_component": "BarChart",
            "icon": "BarChart3",
            "default_size": {"w": 6, "h": 4},
            "palette": "chart",
            "supports_legend": False,
        },
        priority=80,
    ),
    ComponentDefinition(
        component_id="chart.pie.v1",
        name="Pie Chart",
        component_type=ComponentType.PIE_CHART,
        description="Show the share/distribution of a measure across few categories.",
        supported_metrics=["*"],
        supported_dimensions=["categorical"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": True},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": False,
            "max_dimension_cardinality": 8,
            "preferred_intents": ["distribution"],
        },
        config_schema={"label": "dimension", "value": "measure"},
        rendering_metadata={
            "frontend_component": "PieChart",
            "icon": "PieChart",
            "default_size": {"w": 4, "h": 4},
            "palette": "chart",
            "supports_legend": True,
        },
        priority=65,
    ),
    ComponentDefinition(
        component_id="chart.heatmap.v1",
        name="Heatmap",
        component_type=ComponentType.HEATMAP,
        description="Two-dimensional matrix of a measure across two dimensions.",
        supported_metrics=["*"],
        supported_dimensions=["categorical", "temporal"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "AVG", "COUNT"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 2,
            "requires_temporal": False,
            "max_dimension_cardinality": 40,
            "preferred_intents": ["multi-dim", "distribution"],
        },
        config_schema={"x": "dimension", "y": "dimension", "value": "measure"},
        rendering_metadata={
            "frontend_component": "Heatmap",
            "icon": "Grid3x3",
            "default_size": {"w": 6, "h": 4},
            "palette": "chart",
        },
        priority=75,
    ),
    ComponentDefinition(
        component_id="chart.funnel.v1",
        name="Funnel",
        component_type=ComponentType.FUNNEL,
        description="Stage-by-stage drop-off of a monotone measure.",
        supported_metrics=["*"],
        supported_dimensions=["categorical"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": False,
            "max_dimension_cardinality": 12,
            "preferred_intents": ["funnel"],
        },
        config_schema={"stage": "dimension", "value": "measure"},
        rendering_metadata={
            "frontend_component": "Funnel",
            "icon": "Filter",
            "default_size": {"w": 4, "h": 4},
            "palette": "chart",
        },
        priority=55,
    ),
    ComponentDefinition(
        component_id="kpi.gauge_ring.v1",
        name="Gauge Ring",
        component_type=ComponentType.GAUGE_RING,
        description="A single metric as a radial arc filling toward a target.",
        supported_metrics=["*"],
        supported_dimensions=[],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG", "MAX", "MIN"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "max_rows": 1,
            "n_measures": 1,
            "n_dimensions": 0,
            "requires_temporal": False,
            # Binding gate (NOT a score gate): the renderer fails closed to a plain
            # value when no target column is bound. Never fabricates a target.
            "requires_target": True,
            "preferred_intents": ["kpi"],
        },
        config_schema={"value": "measure", "target": "measure?", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "GaugeRing",
            "icon": "Gauge",
            "default_size": {"w": 3, "h": 3},
            "palette": "chart",
        },
        priority=50,
    ),
    ComponentDefinition(
        component_id="kpi.progress.v1",
        name="Progress KPI",
        component_type=ComponentType.PROGRESS_KPI,
        description="A headline metric with a progress bar toward a target.",
        supported_metrics=["*"],
        supported_dimensions=[],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG", "MAX", "MIN"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "max_rows": 1,
            "n_measures": 1,
            "n_dimensions": 0,
            "requires_temporal": False,
            "requires_target": True,
            "preferred_intents": ["kpi"],
        },
        config_schema={"value": "measure", "target": "measure?", "delta": "measure?", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "ProgressKpi",
            "icon": "BarChartHorizontal",
            "default_size": {"w": 3, "h": 2},
            "palette": "chart",
        },
        priority=50,
    ),
    ComponentDefinition(
        component_id="kpi.bullet.v1",
        name="Bullet",
        component_type=ComponentType.BULLET,
        description="An actual-vs-target measure bar over a qualitative band.",
        supported_metrics=["*"],
        supported_dimensions=[],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG", "MAX", "MIN"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "max_rows": 1,
            "n_measures": 1,
            "n_dimensions": 0,
            "requires_temporal": False,
            "requires_target": True,
            "preferred_intents": ["kpi"],
        },
        config_schema={"value": "measure", "target": "measure?", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "Bullet",
            "icon": "Target",
            "default_size": {"w": 4, "h": 2},
            "palette": "chart",
        },
        priority=50,
    ),
    ComponentDefinition(
        component_id="kpi.delta.v1",
        name="Delta KPI",
        component_type=ComponentType.DELTA_KPI,
        description="A headline metric with a period delta and a sparkline.",
        supported_metrics=["*"],
        supported_dimensions=["temporal"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG", "MAX", "MIN"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": False},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": True,
            "preferred_intents": ["trend", "kpi"],
        },
        config_schema={"value": "measure", "spark": "measure?", "delta": "measure?", "format": "auto", "label": "string"},
        rendering_metadata={
            "frontend_component": "DeltaKpi",
            "icon": "Activity",
            "default_size": {"w": 3, "h": 2},
            "palette": "chart",
        },
        priority=58,
    ),
    ComponentDefinition(
        component_id="chart.ranked_bar.v1",
        name="Ranked Bar",
        component_type=ComponentType.RANKED_BAR,
        description="Top-N categories by a measure, sorted descending — the driver view.",
        supported_metrics=["*"],
        supported_dimensions=["categorical"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["SUM", "COUNT", "AVG"], "default": "SUM"},
        filtering_rules={"supports_time_filter": True, "supports_topn": True},
        visualization_rules={
            "min_rows": 2,
            "n_measures": 1,
            "n_dimensions": 1,
            "requires_temporal": False,
            "max_dimension_cardinality": 1000,
            "default_top_n": 10,
            "preferred_intents": ["comparison"],
        },
        config_schema={"x": "dimension", "value": "measure", "top_n": "int", "orientation": "horizontal"},
        rendering_metadata={
            "frontend_component": "RankedBar",
            "icon": "ListOrdered",
            "default_size": {"w": 6, "h": 4},
            "palette": "chart",
            "supports_legend": False,
        },
        priority=78,
    ),
    ComponentDefinition(
        component_id="table.detail.v1",
        name="Data Table",
        component_type=ComponentType.TABLE,
        description="Detailed records — the universal fallback for any shape.",
        supported_metrics=["*"],
        supported_dimensions=["*"],
        required_tables=1,
        required_joins=False,
        aggregation_rules={"allowed": ["RAW", "SUM", "COUNT", "AVG"], "default": "RAW"},
        filtering_rules={"supports_time_filter": True, "supports_topn": True},
        visualization_rules={
            "min_rows": 1,
            "preferred_intents": ["detail"],
            "is_fallback": True,
        },
        config_schema={"columns": "all"},
        rendering_metadata={
            "frontend_component": "CatalogTable",
            "icon": "Table",
            "default_size": {"w": 12, "h": 6},
            "palette": "chart",
        },
        priority=40,
    ),
]

_BY_ID: dict[str, ComponentDefinition] = {c.component_id: c for c in _CATALOG}


def list_components() -> list[ComponentDefinition]:
    """All registered component definitions."""
    return list(_CATALOG)


def get_component(component_id: str) -> ComponentDefinition | None:
    return _BY_ID.get(component_id)


def components_for_type(component_type: ComponentType | str) -> list[ComponentDefinition]:
    ct = component_type.value if isinstance(component_type, ComponentType) else str(component_type)
    return [c for c in _CATALOG if c.component_type.value == ct]


def fallback_component() -> ComponentDefinition:
    """The component used when nothing else confidently matches (table)."""
    return _BY_ID["table.detail.v1"]


def catalog_as_metadata() -> list[dict]:
    """Serialize the catalog for the frontend Analytics Catalog surface."""
    return [c.as_dict() for c in _CATALOG]

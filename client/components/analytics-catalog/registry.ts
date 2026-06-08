// Maps a widget render type -> React component. Mirrors the backend
// rendering_metadata.frontend_component field. Adding a component type = one
// entry here + its renderer. The DashboardRenderer is generic over this map.

import { ComponentType } from "react";
import { DashboardWidget, WidgetType } from "./types";
import {
  KpiCard,
  MetricTile,
  CatalogTable,
  LineChart,
  AreaChart,
  BarChart,
  PieChart,
  Heatmap,
  Funnel,
  GaugeRing,
  ProgressKpi,
  RankedBar,
  DeltaKpi,
  Bullet,
} from "./components";

type WidgetComponent = ComponentType<{ widget: DashboardWidget }>;

export const WIDGET_REGISTRY: Record<WidgetType, WidgetComponent> = {
  kpi_card: KpiCard,
  metric_tile: MetricTile,
  table: CatalogTable,
  line_chart: LineChart,
  area_chart: AreaChart,
  bar_chart: BarChart,
  pie_chart: PieChart,
  heatmap: Heatmap,
  funnel: Funnel,
  gauge_ring: GaugeRing,
  progress_kpi: ProgressKpi,
  ranked_bar: RankedBar,
  delta_kpi: DeltaKpi,
  bullet: Bullet,
};

export function resolveWidgetComponent(type: WidgetType): WidgetComponent {
  return WIDGET_REGISTRY[type] ?? CatalogTable;
}

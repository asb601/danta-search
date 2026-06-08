// Shared types for the metadata-driven Analytics Catalog.
// These mirror the backend DashboardConfig (response.txt Section 8.3) so the
// renderer is a pure projection of persisted metadata.

export type WidgetType =
  | "kpi_card"
  | "metric_tile"
  | "table"
  | "line_chart"
  | "bar_chart"
  | "pie_chart"
  | "area_chart"
  | "heatmap"
  | "funnel"
  | "gauge_ring"
  | "progress_kpi"
  | "ranked_bar"
  | "delta_kpi"
  | "bullet";

export interface WidgetGrid {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WidgetProvenance {
  files_used?: string[];
  row_count?: number;
  route?: string;
  answer?: string;
  query?: string;
  empty?: boolean;
  error?: string;
  // P4 — honest empty-state classification + message.
  empty_reason?: "empty" | "missing" | "error";
  empty_message?: string;
  // P2 / P5 — per-widget correctness annotations (calm amber chips, not errors).
  join_warning?: "multi_table_no_validated_join";
  tie_out?: "over";
  // P2 — additive-measure flag; P0 — pinned spec (opaque to the renderer).
  summable?: boolean;
  spec?: Record<string, unknown>;
  // P7 — whether a board global filter applied to this widget, or it's not affected
  // (its table lacks the conformed dimension) — surfaced as an honest badge.
  filter_status?: { status: "applied" | "not_affected"; dimensions?: string[] };
}

// P7 — a conformed dimension the board can be sliced by (advertised to the slicer bar).
export interface ConformedFilter {
  dimension: string;
  label: string;
  values: string[];
  tables?: string[];
}
export interface ActiveFilter {
  dimension: string;
  values: string[];
}

export interface WidgetConfig {
  // Charts
  x?: string;
  y?: string | string[];
  series?: string | null;
  label?: string;
  value?: string;
  stage?: string;
  orientation?: "vertical" | "horizontal";
  columns?: string[] | "all";
  format?: "currency" | "percent" | "number" | "auto";
  // Target-driven tiles (gauge_ring / progress_kpi / bullet). `target` names a
  // result column; `target_value` is a literal. Both ABSENT => fail-closed to a
  // plain value (the tile never fabricates a target). Never hardcoded.
  target?: string;
  target_value?: number;
  // delta_kpi — `delta` names a precomputed delta column; `spark` names the
  // sibling-series column the Sparkline plots.
  delta?: string;
  spark?: string;
  // ranked_bar — top-N cap (default 10 in the renderer when absent).
  top_n?: number;
  // Delta-coloring polarity for headline/delta tiles: "inverse" => a rising value
  // is BAD (cost/aging/DSO/returns/overdue); "positive" => a rise is good.
  // Proposed by the planner; the DeltaBadge fails safe to neutral when absent.
  polarity?: "positive" | "inverse";
  // P4 — deterministic one-line analyst caption (absent when uncomputable).
  insight?: string;
  [key: string]: unknown;
}

export type WidgetRow = Record<string, unknown>;

export interface DashboardWidget {
  widget_id: string;
  component_id: string;
  type: WidgetType;
  title: string;
  grid: WidgetGrid;
  config: WidgetConfig;
  data: WidgetRow[];
  rationale?: string;
  score?: number;
  provenance?: WidgetProvenance;
}

export interface DashboardConfig {
  version: string;
  title: string;
  description?: string;
  generated_at?: string;
  prompt?: string;
  layout?: string;
  widgets: DashboardWidget[];
  warnings?: string[];
  // P7 — board-level slicers (conformed dimensions) + the currently-applied filters.
  available_filters?: ConformedFilter[];
  global_filters?: ActiveFilter[];
}

export interface DashboardSummary {
  id: string;
  title: string;
  description: string | null;
  folder_id: string | null;
  is_pinned: boolean;
  status: string;
  widget_count: number;
  created_at: string;
  updated_at: string;
}

export interface DashboardFull {
  id: string;
  title: string;
  description: string | null;
  folder_id: string | null;
  container_id: string | null;
  is_pinned: boolean;
  status: string;
  config: DashboardConfig | Record<string, never>;
  prompt_history: { prompt: string; created_at: string; widget_ids: string[] }[];
  source_file_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface DashboardFolder {
  id: string;
  name: string;
  parent_id: string | null;
  container_id: string | null;
  created_at: string;
}

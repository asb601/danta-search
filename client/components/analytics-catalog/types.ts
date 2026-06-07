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
  | "funnel";

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

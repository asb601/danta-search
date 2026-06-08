"use client";

// Generic renderer: a pure function of DashboardConfig. Lays the board out in
// narrative BANDS like a real BI dashboard — a tight KPI ribbon on top, a DENSE
// varied collage of charts in the middle (a packed 6-col masonry where each tile
// honors the server's persisted size intent), detail tables full-width at the
// bottom. KPI cards borrow a sibling trend (same measure) for a sparkline +
// period delta. Absolute persisted x/y are intentionally ignored; w/h are NOT —
// they drive the varied tile sizing that makes the board read like PowerBI.

import { DashboardConfig, DashboardWidget } from "./types";
import { resolveWidgetComponent } from "./registry";
import { KpiCard, CatalogTable } from "./components";

function plannedMeasure(w: DashboardWidget): string | undefined {
  const p = (w.provenance?.spec as { planned?: { measure?: string } } | undefined)?.planned;
  const m = p?.measure;
  if (m) return m;
  return typeof w.config?.value === "string" ? w.config.value : undefined;
}
function firstYKey(w: DashboardWidget): string | undefined {
  const y = w.config?.y;
  return Array.isArray(y) ? y[0] : typeof y === "string" ? y : undefined;
}

// ── Collage geometry ────────────────────────────────────────────────────────
// The chart band is a packed 6-column grid. Each tile's width/height come from
// the SERVER's size intent (widget.grid.w on its 12-col assembly grid, h in
// assembly row units) mapped onto our 6-col / compact-row collage. When the grid
// is absent or degenerate (legacy configs), we fall back to a per-TYPE default so
// donuts/gauges stay small, hero trends stay wide — never a uniform 2×2 wall.

const COLS = 6; // collage columns at the widest breakpoint

// Per-type fallback footprint (col span out of 6, row span in 90px rows) used
// when the server didn't persist a usable grid.w/h. Hero trends are wide; KPIs
// compact; donut/gauge/funnel small & square-ish.
function fallbackSpan(w: DashboardWidget, isHero: boolean): { col: number; row: number } {
  if (isHero) return { col: 4, row: 4 };
  switch (w.type) {
    case "pie_chart":
    case "gauge_ring":
      return { col: 2, row: 3 };
    case "funnel":
    case "bullet":
    case "progress_kpi":
      return { col: 2, row: 2 };
    case "heatmap":
      return { col: 4, row: 3 };
    case "ranked_bar":
      return { col: 3, row: 3 };
    case "bar_chart":
      return { col: 3, row: 3 };
    case "line_chart":
    case "area_chart":
      return { col: 4, row: 3 };
    default:
      return { col: 3, row: 3 };
  }
}

// Map the server's persisted size intent onto the collage grid. The assembly
// engine lays widgets on a 12-col grid; we proportionally rescale w → 6 cols and
// clamp to [2..6]. Height is taken from grid.h (assembly rows ≈ our rows) and
// clamped to [2..5]. A hero always gets the widest, tallest footprint so it
// anchors the board. Degenerate/absent grids fall back to the per-type default.
function spanFor(w: DashboardWidget, isHero: boolean): { col: number; row: number } {
  const fb = fallbackSpan(w, isHero);
  const g = w.grid;
  const sw = g && Number.isFinite(g.w) ? g.w : undefined;
  const sh = g && Number.isFinite(g.h) ? g.h : undefined;

  // No real size intent persisted → type default.
  if (!sw || sw <= 0) return fb;

  // Server grid is 12-col; rescale to our 6 and round. Clamp so nothing is a
  // sliver (<2) and nothing exceeds the row width.
  let col = Math.round((sw / 12) * COLS);
  if (col < 2) col = 2;
  if (col > COLS) col = COLS;

  // Height: assembly rows roughly correspond to our rows. Clamp to keep tiles
  // from being too short to read or so tall they leave dead space.
  let row = sh && sh > 0 ? Math.round(sh) : fb.row;
  if (row < 2) row = 2;
  if (row > 5) row = 5;

  // The hero earns at least its fallback footprint regardless of persisted size.
  if (isHero) {
    col = Math.max(col, fb.col);
    row = Math.max(row, fb.row);
  }
  return { col, row };
}

export function DashboardRenderer({ config }: { config: DashboardConfig | null | undefined }) {
  const widgets = config?.widgets ?? [];

  if (!widgets.length) {
    return (
      <div className="flex h-full min-h-[300px] flex-col items-center justify-center rounded-xl border border-dashed border-border/60 bg-muted/10 text-center p-8">
        <div className="w-14 h-14 rounded-2xl bg-muted/40 flex items-center justify-center mb-4">
          <svg className="w-7 h-7 text-muted-foreground/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <p className="text-sm font-medium text-foreground">No analytics yet</p>
        <p className="mt-1 text-xs text-muted-foreground max-w-xs">
          Describe the dashboard you want in the box below — KPIs, charts, and tables will be generated automatically.
        </p>
      </div>
    );
  }

  const kpis = widgets.filter((w) => w.type === "kpi_card" || w.type === "metric_tile");
  const tables = widgets.filter((w) => w.type === "table");
  const charts = widgets.filter((w) => w.type !== "table" && w.type !== "kpi_card" && w.type !== "metric_tile");

  // measure -> trend series, so a KPI can show a sparkline of its own measure.
  const trendByMeasure = new Map<string, number[]>();
  for (const c of charts) {
    if (c.type !== "line_chart" && c.type !== "area_chart") continue;
    const m = firstYKey(c) ?? plannedMeasure(c);
    if (!m || trendByMeasure.has(m)) continue;
    const vals = (c.data ?? []).map((r) => Number(r[m])).filter((v) => Number.isFinite(v));
    if (vals.length >= 2) trendByMeasure.set(m, vals);
  }
  const sparkFor = (w: DashboardWidget) => {
    const m = plannedMeasure(w);
    return m ? trendByMeasure.get(m) : undefined;
  };

  // The hero is the first trend (or the highest-scoring chart) — it gets the
  // widest, tallest tile so the board has a clear focal anchor.
  const hero =
    charts.find((c) => c.type === "line_chart" || c.type === "area_chart") ??
    [...charts].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0];

  return (
    <div className="space-y-3">
      {config?.warnings && config.warnings.length > 0 && (
        <div className="rounded-lg border border-warn-border bg-warn-bg px-3.5 py-2.5">
          <div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-warn-fg">
            <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            Analyst notes
          </div>
          <ul className="space-y-0.5 text-xs text-foreground/80">
            {config.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ── KPI ribbon: equal-height headline numbers, shoulder to shoulder ── */}
      {kpis.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
          {kpis.map((w) => (
            <div key={w.widget_id} className="collage-tile h-[112px] min-w-0">
              <KpiCard widget={w} spark={sparkFor(w)} />
            </div>
          ))}
        </div>
      )}

      {/* ── Chart band: a PACKED, varied collage. 6 dense columns, 90px rows,
            tight gaps; every tile honors the server's persisted size intent so
            the board reads like a balanced PowerBI page — never a 2-col wall. ── */}
      {charts.length > 0 && (
        <div
          className="grid grid-flow-dense grid-cols-2 gap-2.5 sm:grid-cols-4 lg:grid-cols-6"
          style={{ gridAutoRows: "90px" }}
        >
          {charts.map((w) => {
            const Comp = resolveWidgetComponent(w.type);
            const { col, row } = spanFor(w, w === hero);
            return (
              <div
                key={w.widget_id}
                className="collage-tile min-w-0"
                style={{
                  gridColumn: `span ${col} / span ${col}`,
                  gridRow: `span ${row} / span ${row}`,
                }}
              >
                <Comp widget={w} />
              </div>
            );
          })}
        </div>
      )}

      {/* ── Detail tables: full width, bottom ── */}
      {tables.length > 0 && (
        <div className="space-y-3">
          {tables.map((w) => (
            <div key={w.widget_id} className="collage-tile h-[440px] min-w-0">
              <CatalogTable widget={w} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

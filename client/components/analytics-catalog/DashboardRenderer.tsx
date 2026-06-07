"use client";

// Generic renderer: a pure function of DashboardConfig. Lays the board out in
// narrative BANDS like a real BI dashboard — a tight KPI ribbon on top, a dense
// chart grid (with a wide hero) in the middle, detail tables at the bottom — rather
// than one undifferentiated masonry. KPI cards borrow a sibling trend (same measure)
// for a sparkline + period delta. Absolute persisted x/y are intentionally ignored.

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

  // The hero is the first trend (or the highest-scoring chart) — it gets a wide tile.
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
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {kpis.map((w) => (
            <div key={w.widget_id} className="h-[116px] min-w-0">
              <KpiCard widget={w} spark={sparkFor(w)} />
            </div>
          ))}
        </div>
      )}

      {/* ── Chart band: dense 4-col grid, wide hero, dense back-fill ── */}
      {charts.length > 0 && (
        <div className="grid auto-rows-[150px] grid-cols-1 grid-flow-row-dense gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {charts.map((w) => {
            const Comp = resolveWidgetComponent(w.type);
            const isHero = w === hero;
            const narrow = w.type === "pie_chart" || w.type === "funnel";
            const colSpan = isHero ? 2 : narrow ? 1 : 2;
            const rowSpan = isHero ? 3 : 2;
            return (
              <div
                key={w.widget_id}
                className="min-w-0"
                style={{ gridColumn: `span ${colSpan} / span ${colSpan}`, gridRow: `span ${rowSpan} / span ${rowSpan}` }}
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
            <div key={w.widget_id} className="h-[440px] min-w-0">
              <CatalogTable widget={w} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

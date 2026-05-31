"use client";

// Generic renderer: a pure function of DashboardConfig. Lays widgets out as an
// adaptive MASONRY collage — a responsive column grid with gap-filling dense
// auto-flow. Each widget's persisted grid.w drives its column span (1..4) and
// grid.h drives its row span (data-adaptive height from the backend). Absolute
// x/y are intentionally ignored; the browser packs tiles for the viewport.

import { DashboardConfig } from "./types";
import { resolveWidgetComponent } from "./registry";

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

  return (
    <div className="space-y-4">
      {config?.warnings && config.warnings.length > 0 && (
        <div className="flex items-start gap-2 rounded-lg border border-accent-foreground/20 bg-accent px-3 py-2 text-xs text-accent-foreground">
          <svg className="mt-0.5 w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
          <span>{config.warnings.join(" · ")}</span>
        </div>
      )}

      {/* Adaptive masonry: 1 col (mobile) → 2 (sm) → 4 (lg). Dense auto-flow
          back-fills gaps so short tiles (KPIs) tuck beside tall ones (charts/
          tables) instead of leaving a ragged column. Row unit 80px + gap-4
          (16px): h=2 ≈ 176px (KPI), h=5 ≈ 464px (chart), h=8 ≈ 752px (table).
          A span wider than the current column count is clamped by the browser,
          so a w=4 table is full-width on every breakpoint. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 grid-flow-row-dense gap-4 auto-rows-[80px]">
        {widgets.map((w) => {
          const Comp = resolveWidgetComponent(w.type);
          const colSpan = Math.min(Math.max(w.grid?.w ?? 2, 1), 4);
          const rowSpan = Math.min(Math.max(w.grid?.h ?? 4, 2), 12);
          return (
            <div
              key={w.widget_id}
              className="min-w-0 h-full"
              style={{
                gridColumn: `span ${colSpan} / span ${colSpan}`,
                gridRow: `span ${rowSpan} / span ${rowSpan}`,
              }}
            >
              <Comp widget={w} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

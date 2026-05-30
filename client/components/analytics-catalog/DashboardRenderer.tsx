"use client";

// Generic renderer: a pure function of DashboardConfig. Lays widgets out on a
// responsive 12-column grid using each widget's persisted grid.{x,y,w,h}.

import { DashboardConfig } from "./types";
import { resolveWidgetComponent } from "./registry";

export function DashboardRenderer({ config }: { config: DashboardConfig | null | undefined }) {
  const widgets = config?.widgets ?? [];

  if (!widgets.length) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
        No analytics yet — describe the dashboard you want below to generate it.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {config?.warnings && config.warnings.length > 0 && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
          {config.warnings.join(" · ")}
        </div>
      )}
      <div className="grid grid-cols-12 gap-3 auto-rows-[88px]">
        {widgets.map((w) => {
          const Comp = resolveWidgetComponent(w.type);
          const colSpan = Math.min(Math.max(w.grid?.w ?? 6, 2), 12);
          const rowSpan = Math.max(w.grid?.h ?? 4, 2);
          return (
            <div
              key={w.widget_id}
              className="min-w-0"
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

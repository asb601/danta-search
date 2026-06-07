"use client";

// P7 — PowerBI-style global slicer bar. A pure projection of config.available_filters
// (the CONFORMED dimensions the backend deemed safe to slice across tables). Toggling
// values + Apply re-requests the whole board with the selected global_filters; the
// numbers are recomputed by the agent (the slicer never mutates client-side data).

import { useState } from "react";
import { DashboardConfig, ActiveFilter } from "./types";

export function SlicerBar({
  config,
  onApply,
  busy,
}: {
  config: DashboardConfig | null | undefined;
  onApply: (filters: ActiveFilter[]) => void;
  busy?: boolean;
}) {
  const available = config?.available_filters ?? [];
  const [sel, setSel] = useState<Record<string, Set<string>>>(() => {
    const m: Record<string, Set<string>> = {};
    for (const f of config?.global_filters ?? []) m[f.dimension] = new Set(f.values);
    return m;
  });

  if (!available.length) return null;

  const toggle = (dim: string, v: string) =>
    setSel((prev) => {
      const s = new Set(prev[dim] ?? []);
      s.has(v) ? s.delete(v) : s.add(v);
      return { ...prev, [dim]: s };
    });

  const apply = () =>
    onApply(
      available
        .map((d) => ({ dimension: d.dimension, values: Array.from(sel[d.dimension] ?? []) }))
        .filter((f) => f.values.length),
    );

  const hasSel = Object.values(sel).some((s) => s.size);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-border bg-muted/20 px-3 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Filters
      </span>
      {available.map((d) => (
        <div key={d.dimension} className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-foreground">{d.label}</span>
          <div className="flex flex-wrap gap-1">
            {d.values.slice(0, 12).map((v) => {
              const on = sel[d.dimension]?.has(v) ?? false;
              return (
                <button
                  key={v}
                  type="button"
                  onClick={() => toggle(d.dimension, v)}
                  className={`rounded-md border px-1.5 py-0.5 text-[11px] transition-colors ${
                    on
                      ? "border-chart-1 bg-chart-1/10 text-foreground"
                      : "border-border bg-card text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {String(v)}
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={apply}
        disabled={busy}
        className="ml-auto rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground transition-opacity disabled:opacity-50"
      >
        {busy ? "Applying…" : hasSel ? "Apply filters" : "Clear filters"}
      </button>
    </div>
  );
}

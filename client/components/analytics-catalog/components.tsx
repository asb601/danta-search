"use client";

// Pure-SVG/CSS, metadata-driven Analytics Catalog components.
// Each takes a DashboardWidget and renders from widget.config + widget.data.
// Zero external chart dependency — fully controlled by the OKLch token palette.

import { DashboardWidget, WidgetRow } from "./types";
import { WidgetFrame, EmptyState } from "./WidgetFrame";
import { colorAt, formatValue, compactNumber } from "./palette";

type Props = { widget: DashboardWidget };

// ---- helpers --------------------------------------------------------------

function num(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}
function str(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return String(v);
}
function firstY(widget: DashboardWidget): string | undefined {
  const y = widget.config.y;
  return Array.isArray(y) ? y[0] : y;
}
function yList(widget: DashboardWidget): string[] {
  const y = widget.config.y;
  if (Array.isArray(y)) return y.filter(Boolean) as string[];
  return y ? [y] : [];
}

// ---- KPI Card -------------------------------------------------------------

export function KpiCard({ widget }: Props) {
  const valueKey = widget.config.value;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="flex h-full flex-col items-start justify-center">
        <p className="text-3xl font-semibold tracking-tight text-foreground">
          {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
        </p>
        {valueKey && <p className="mt-1 text-xs text-muted-foreground">{valueKey}</p>}
      </div>
    </WidgetFrame>
  );
}

// ---- Metric Tile (value + optional delta) --------------------------------

export function MetricTile({ widget }: Props) {
  const valueKey = widget.config.value;
  const deltaKey = (widget.config as { delta?: string }).delta;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const delta = deltaKey && row ? num(row[deltaKey]) : undefined;
  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="flex h-full flex-col items-start justify-center">
        <p className="text-2xl font-semibold tracking-tight text-foreground">
          {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
        </p>
        {delta !== undefined && (
          <p className={`mt-1 text-xs font-medium ${delta >= 0 ? "text-emerald-500" : "text-red-500"}`}>
            {delta >= 0 ? "▲" : "▼"} {formatValue(Math.abs(delta), widget.config.format)}
          </p>
        )}
      </div>
    </WidgetFrame>
  );
}

// ---- Data Table -----------------------------------------------------------

export function CatalogTable({ widget }: Props) {
  const rows = widget.data || [];
  if (!rows.length) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState message={widget.provenance?.empty ? "No rows returned" : undefined} />
      </WidgetFrame>
    );
  }
  const cols =
    widget.config.columns && widget.config.columns !== "all"
      ? (widget.config.columns as string[])
      : Object.keys(rows[0]);
  const shown = rows.slice(0, 100);
  return (
    <WidgetFrame
      title={widget.title}
      rationale={widget.rationale}
      footer={rows.length > shown.length ? `Showing ${shown.length} of ${rows.length} rows` : undefined}
    >
      <div className="h-full overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-surface-raised">
            <tr>
              {cols.map((c) => (
                <th key={c} className="px-2 py-1.5 text-left font-medium text-muted-foreground whitespace-nowrap">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={i} className="border-t border-border/60 hover:bg-surface-raised/50">
                {cols.map((c) => (
                  <td key={c} className="px-2 py-1.5 text-foreground whitespace-nowrap">
                    {typeof r[c] === "number" ? formatValue(r[c], "number") : str(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </WidgetFrame>
  );
}

// ---- shared SVG cartesian frame ------------------------------------------

const PAD = { top: 10, right: 12, bottom: 28, left: 44 };
const VIEW_W = 480;
const VIEW_H = 240;

function axisTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round(step * i));
}

// ---- Line / Area Chart ----------------------------------------------------

function LineLike({ widget, area }: Props & { area: boolean }) {
  const rows = widget.data || [];
  const xKey = widget.config.x;
  const series = yList(widget);
  if (!rows.length || !xKey || !series.length) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const innerW = VIEW_W - PAD.left - PAD.right;
  const innerH = VIEW_H - PAD.top - PAD.bottom;
  const maxVal = Math.max(1, ...rows.flatMap((r) => series.map((s) => num(r[s]))));
  const xStep = rows.length > 1 ? innerW / (rows.length - 1) : 0;
  const xAt = (i: number) => PAD.left + (rows.length > 1 ? i * xStep : innerW / 2);
  const yAt = (v: number) => PAD.top + innerH - (v / maxVal) * innerH;

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full h-full" preserveAspectRatio="none">
        {axisTicks(maxVal).map((t, i) => {
          const y = yAt(t);
          return (
            <g key={i}>
              <line x1={PAD.left} y1={y} x2={VIEW_W - PAD.right} y2={y} stroke="var(--border)" strokeWidth={0.5} />
              <text x={PAD.left - 6} y={y + 3} textAnchor="end" fontSize={9} fill="var(--muted-foreground)">
                {compactNumber(t)}
              </text>
            </g>
          );
        })}
        {series.map((s, si) => {
          const pts = rows.map((r, i) => `${xAt(i)},${yAt(num(r[s]))}`).join(" ");
          const color = colorAt(si);
          return (
            <g key={s}>
              {area && (
                <polygon
                  points={`${PAD.left},${PAD.top + innerH} ${pts} ${xAt(rows.length - 1)},${PAD.top + innerH}`}
                  fill={color}
                  opacity={0.15}
                />
              )}
              <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
            </g>
          );
        })}
        {rows.map((r, i) => {
          if (rows.length > 12 && i % Math.ceil(rows.length / 8) !== 0) return null;
          return (
            <text key={i} x={xAt(i)} y={VIEW_H - 10} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
              {str(r[xKey]).slice(0, 10)}
            </text>
          );
        })}
      </svg>
    </WidgetFrame>
  );
}

export function LineChart({ widget }: Props) {
  return <LineLike widget={widget} area={false} />;
}
export function AreaChart({ widget }: Props) {
  return <LineLike widget={widget} area={true} />;
}

// ---- Bar Chart ------------------------------------------------------------

export function BarChart({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 30);
  const xKey = widget.config.x;
  const yKey = firstY(widget) || widget.config.value;
  if (!rows.length || !xKey || !yKey) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const innerW = VIEW_W - PAD.left - PAD.right;
  const innerH = VIEW_H - PAD.top - PAD.bottom;
  const maxVal = Math.max(1, ...rows.map((r) => num(r[yKey])));
  const slot = innerW / rows.length;
  const barW = Math.max(2, slot * 0.62);

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full h-full" preserveAspectRatio="none">
        {axisTicks(maxVal).map((t, i) => {
          const y = PAD.top + innerH - (t / maxVal) * innerH;
          return (
            <g key={i}>
              <line x1={PAD.left} y1={y} x2={VIEW_W - PAD.right} y2={y} stroke="var(--border)" strokeWidth={0.5} />
              <text x={PAD.left - 6} y={y + 3} textAnchor="end" fontSize={9} fill="var(--muted-foreground)">
                {compactNumber(t)}
              </text>
            </g>
          );
        })}
        {rows.map((r, i) => {
          const v = num(r[yKey]);
          const h = (v / maxVal) * innerH;
          const x = PAD.left + i * slot + (slot - barW) / 2;
          const y = PAD.top + innerH - h;
          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={h} rx={2} fill={colorAt(i)} />
              {rows.length <= 12 && (
                <text x={x + barW / 2} y={VIEW_H - 10} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
                  {str(r[xKey]).slice(0, 8)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </WidgetFrame>
  );
}

// ---- Pie Chart ------------------------------------------------------------

export function PieChart({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 8);
  const labelKey = widget.config.label || widget.config.x;
  const valueKey = widget.config.value || firstY(widget);
  if (!rows.length || !labelKey || !valueKey) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const total = rows.reduce((s, r) => s + num(r[valueKey]), 0) || 1;
  const cx = 80;
  const cy = 110;
  const radius = 70;
  let angle = -Math.PI / 2;
  const slices = rows.map((r, i) => {
    const frac = num(r[valueKey]) / total;
    const start = angle;
    const end = angle + frac * Math.PI * 2;
    angle = end;
    const large = end - start > Math.PI ? 1 : 0;
    const x1 = cx + radius * Math.cos(start);
    const y1 = cy + radius * Math.sin(start);
    const x2 = cx + radius * Math.cos(end);
    const y2 = cy + radius * Math.sin(end);
    const d = `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`;
    return { d, color: colorAt(i), label: str(r[labelKey]), pct: frac * 100, value: r[valueKey] };
  });

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="flex h-full items-center gap-3">
        <svg viewBox="0 0 160 220" className="h-full max-h-[200px]">
          {slices.map((s, i) => (
            <path key={i} d={s.d} fill={s.color} stroke="var(--background)" strokeWidth={1} />
          ))}
        </svg>
        <div className="flex-1 min-w-0 space-y-1 overflow-auto">
          {slices.map((s, i) => (
            <div key={i} className="flex items-center gap-1.5 text-[11px]">
              <span className="inline-block h-2.5 w-2.5 rounded-sm shrink-0" style={{ background: s.color }} />
              <span className="truncate text-foreground">{s.label}</span>
              <span className="ml-auto text-muted-foreground">{s.pct.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </WidgetFrame>
  );
}

// ---- Heatmap --------------------------------------------------------------

export function Heatmap({ widget }: Props) {
  const rows = widget.data || [];
  const xKey = widget.config.x;
  const yKey = (widget.config as { y?: string }).y;
  const valueKey = widget.config.value;
  if (!rows.length || !xKey || !yKey || !valueKey || typeof yKey !== "string") {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const xs = Array.from(new Set(rows.map((r) => str(r[xKey])))).slice(0, 16);
  const ys = Array.from(new Set(rows.map((r) => str(r[yKey as string])))).slice(0, 12);
  const cell = new Map<string, number>();
  let maxV = 1;
  for (const r of rows) {
    const k = `${str(r[xKey])}|${str(r[yKey as string])}`;
    const v = num(r[valueKey]);
    cell.set(k, v);
    if (v > maxV) maxV = v;
  }
  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="h-full overflow-auto">
        <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `auto repeat(${xs.length}, minmax(28px, 1fr))` }}>
          <div />
          {xs.map((x) => (
            <div key={x} className="px-1 text-[9px] text-muted-foreground truncate text-center">{x}</div>
          ))}
          {ys.map((y) => (
            <FragmentRow key={y} y={y} xs={xs} cell={cell} maxV={maxV} format={widget.config.format} />
          ))}
        </div>
      </div>
    </WidgetFrame>
  );
}

function FragmentRow({
  y, xs, cell, maxV, format,
}: {
  y: string; xs: string[]; cell: Map<string, number>; maxV: number;
  format?: "currency" | "percent" | "number" | "auto";
}) {
  return (
    <>
      <div className="pr-1 text-[9px] text-muted-foreground truncate self-center max-w-[80px]">{y}</div>
      {xs.map((x) => {
        const v = cell.get(`${x}|${y}`) ?? 0;
        const intensity = maxV ? v / maxV : 0;
        return (
          <div
            key={x}
            title={`${x} / ${y}: ${formatValue(v, format)}`}
            className="h-7 rounded-sm flex items-center justify-center text-[8px] text-foreground/80"
            style={{ background: `color-mix(in oklab, var(--chart-1) ${Math.round(intensity * 100)}%, transparent)` }}
          >
            {intensity > 0.15 ? compactNumber(v) : ""}
          </div>
        );
      })}
    </>
  );
}

// ---- Funnel ---------------------------------------------------------------

export function Funnel({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 12);
  const stageKey = widget.config.stage || widget.config.x;
  const valueKey = widget.config.value || firstY(widget);
  if (!rows.length || !stageKey || !valueKey) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const maxV = Math.max(1, ...rows.map((r) => num(r[valueKey])));
  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="flex h-full flex-col justify-center gap-1.5">
        {rows.map((r, i) => {
          const v = num(r[valueKey]);
          const pct = (v / maxV) * 100;
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="w-20 shrink-0 truncate text-[11px] text-muted-foreground">{str(r[stageKey])}</span>
              <div className="flex-1 h-6 bg-surface-raised rounded-sm overflow-hidden">
                <div
                  className="h-full flex items-center justify-end pr-2 text-[10px] text-white/90 rounded-sm"
                  style={{ width: `${Math.max(pct, 6)}%`, background: colorAt(i) }}
                >
                  {formatValue(v, widget.config.format)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </WidgetFrame>
  );
}

"use client";

// Pure-SVG/CSS, metadata-driven Analytics Catalog components.
// Each takes a DashboardWidget and renders from widget.config + widget.data.
// Zero external chart dependency — fully controlled by the OKLch token palette.

import { DashboardWidget } from "./types";
import { WidgetFrame, EmptyState } from "./WidgetFrame";
import { colorAt, formatValue, compactNumber } from "./palette";

type Props = { widget: DashboardWidget };

// ---- helpers ----------------------------------------------------------------

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
function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Catmull-Rom → cubic bezier spline for smooth chart lines
function catmullRomPath(pts: [number, number][]): string {
  if (!pts.length) return "";
  if (pts.length === 1) return `M ${pts[0][0]} ${pts[0][1]}`;
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

function areaPath(pts: [number, number][], baseY: number): string {
  if (!pts.length) return "";
  const line = catmullRomPath(pts);
  return `${line} L ${pts[pts.length - 1][0]},${baseY} L ${pts[0][0]},${baseY} Z`;
}

// ---- KPI Card ---------------------------------------------------------------

export function KpiCard({ widget }: Props) {
  const valueKey = widget.config.value;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const formatted = raw === undefined ? "—" : formatValue(raw, widget.config.format);
  const isPositive = raw !== undefined && num(raw) >= 0;

  return (
    <div className="flex flex-col h-full rounded-xl border border-border bg-card overflow-hidden">
      <div className="h-[3px] w-full shrink-0" style={{ background: "var(--chart-1)" }} />
      <div className="flex flex-col justify-between flex-1 px-4 py-3">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest leading-tight truncate">
            {widget.title}
          </p>
          {raw !== undefined && (
            <span
              className={`shrink-0 text-[11px] font-semibold px-1.5 py-0.5 rounded ${
                isPositive
                  ? "bg-emerald-500/10 text-emerald-500"
                  : "bg-red-500/10 text-red-500"
              }`}
            >
              {isPositive ? "↑" : "↓"}
            </span>
          )}
        </div>
        <div>
          <p className="text-[2rem] font-bold text-foreground leading-none tracking-tight">
            {formatted}
          </p>
          {valueKey && (
            <p className="mt-1 text-[11px] text-muted-foreground/70 truncate">
              {humanize(valueKey)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Metric Tile (value + optional delta) -----------------------------------

export function MetricTile({ widget }: Props) {
  const valueKey = widget.config.value;
  const deltaKey = (widget.config as { delta?: string }).delta;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const delta = deltaKey && row ? num(row[deltaKey]) : undefined;

  return (
    <div className="flex flex-col h-full rounded-xl border border-border bg-card overflow-hidden">
      <div className="h-[3px] w-full shrink-0" style={{ background: "var(--chart-2)" }} />
      <div className="flex flex-col justify-between flex-1 px-4 py-3">
        <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest truncate">
          {widget.title}
        </p>
        <div>
          <p className="text-[1.75rem] font-bold text-foreground leading-none tracking-tight">
            {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
          </p>
          {delta !== undefined && (
            <p
              className={`mt-1 text-xs font-semibold flex items-center gap-1 ${
                delta >= 0 ? "text-emerald-500" : "text-red-500"
              }`}
            >
              {delta >= 0 ? "▲" : "▼"}
              {formatValue(Math.abs(delta), widget.config.format)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Data Table -------------------------------------------------------------

export function CatalogTable({ widget }: Props) {
  const rows = widget.data || [];
  if (!rows.length) {
    const agentAnswer = widget.provenance?.answer;
    const msg = agentAnswer || (widget.provenance?.empty ? "No data was returned for this query." : undefined);
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState message={msg} />
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
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-border/60">
              {cols.map((c) => (
                <th
                  key={c}
                  className="px-3 py-2 text-left text-[10px] font-semibold text-muted-foreground uppercase tracking-wider whitespace-nowrap sticky top-0 bg-card"
                >
                  {humanize(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr
                key={i}
                className={`border-b border-border/30 hover:bg-muted/20 transition-colors ${
                  i % 2 !== 0 ? "bg-muted/[0.04]" : ""
                }`}
              >
                {cols.map((c) => (
                  <td key={c} className="px-3 py-1.5 text-foreground whitespace-nowrap">
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

// ---- Shared SVG frame -------------------------------------------------------

const PAD = { top: 14, right: 16, bottom: 32, left: 50 };
const VIEW_W = 480;
const VIEW_H = 220;

function axisTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round(step * i));
}

// ---- Line / Area Chart ------------------------------------------------------

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
  const xAt = (i: number) =>
    PAD.left + (rows.length > 1 ? (i / (rows.length - 1)) * innerW : innerW / 2);
  const yAt = (v: number) => PAD.top + innerH - (v / maxVal) * innerH;
  const baseY = PAD.top + innerH;
  const gradNs = `grad-${widget.widget_id}`;

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full h-full" preserveAspectRatio="none">
        <defs>
          {series.map((_, si) => (
            <linearGradient key={si} id={`${gradNs}-${si}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colorAt(si)} stopOpacity={0.3} />
              <stop offset="100%" stopColor={colorAt(si)} stopOpacity={0.01} />
            </linearGradient>
          ))}
        </defs>

        {/* Horizontal grid */}
        {axisTicks(maxVal).map((t, i) => {
          const y = yAt(t);
          return (
            <g key={i}>
              <line
                x1={PAD.left} y1={y} x2={VIEW_W - PAD.right} y2={y}
                stroke="var(--border)" strokeWidth={0.6} strokeDasharray="3 4"
              />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize={9} fill="var(--muted-foreground)">
                {compactNumber(t)}
              </text>
            </g>
          );
        })}

        {/* Series */}
        {series.map((s, si) => {
          const pts: [number, number][] = rows.map((r, i) => [xAt(i), yAt(num(r[s]))]);
          const color = colorAt(si);
          const line = catmullRomPath(pts);
          return (
            <g key={s}>
              {area && <path d={areaPath(pts, baseY)} fill={`url(#${gradNs}-${si})`} />}
              <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" />
              {rows.length <= 24 &&
                pts.map(([cx, cy], i) => (
                  <circle key={i} cx={cx} cy={cy} r={2.5} fill={color} stroke="var(--card)" strokeWidth={1.5} />
                ))}
            </g>
          );
        })}

        {/* X labels */}
        {rows.map((r, i) => {
          if (rows.length > 12 && i % Math.ceil(rows.length / 8) !== 0) return null;
          return (
            <text key={i} x={xAt(i)} y={VIEW_H - 9} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
              {str(r[xKey]).slice(0, 10)}
            </text>
          );
        })}

        {/* Multi-series legend */}
        {series.length > 1 &&
          series.map((s, si) => (
            <g key={s}>
              <rect x={PAD.left + si * 90} y={VIEW_H - 3} width={8} height={3} rx={2} fill={colorAt(si)} />
              <text x={PAD.left + si * 90 + 11} y={VIEW_H - 1} fontSize={8} fill="var(--muted-foreground)">
                {humanize(s)}
              </text>
            </g>
          ))}
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

// ---- Bar Chart --------------------------------------------------------------

export function BarChart({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 30);
  const xKey = widget.config.x;
  const yKey = firstY(widget) || (widget.config.value as string | undefined);
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
  const barW = Math.max(4, slot * 0.66);
  const gradId = `bgrad-${widget.widget_id}`;

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.9} />
            <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0.45} />
          </linearGradient>
        </defs>
        {axisTicks(maxVal).map((t, i) => {
          const y = PAD.top + innerH - (t / maxVal) * innerH;
          return (
            <g key={i}>
              <line
                x1={PAD.left} y1={y} x2={VIEW_W - PAD.right} y2={y}
                stroke="var(--border)" strokeWidth={0.6} strokeDasharray="3 4"
              />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize={9} fill="var(--muted-foreground)">
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
              <rect x={x} y={y} width={barW} height={Math.max(h, 1)} rx={3} fill={`url(#${gradId})`} />
              {rows.length <= 12 && (
                <text x={x + barW / 2} y={VIEW_H - 9} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
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

// ---- Pie / Donut Chart ------------------------------------------------------

export function PieChart({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 8);
  const labelKey = widget.config.label || (widget.config.x as string | undefined);
  const valueKey = widget.config.value || firstY(widget);
  if (!rows.length || !labelKey || !valueKey) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale}>
        <EmptyState />
      </WidgetFrame>
    );
  }
  const total = rows.reduce((s, r) => s + num(r[valueKey]), 0) || 1;
  const cx = 90, cy = 100, outerR = 72, innerR = 40;
  let angle = -Math.PI / 2;
  const slices = rows.map((r, i) => {
    const frac = num(r[valueKey]) / total;
    const start = angle;
    const end = angle + frac * Math.PI * 2;
    angle = end;
    const large = end - start > Math.PI ? 1 : 0;
    const x1o = cx + outerR * Math.cos(start), y1o = cy + outerR * Math.sin(start);
    const x2o = cx + outerR * Math.cos(end), y2o = cy + outerR * Math.sin(end);
    const x1i = cx + innerR * Math.cos(end), y1i = cy + innerR * Math.sin(end);
    const x2i = cx + innerR * Math.cos(start), y2i = cy + innerR * Math.sin(start);
    const d = `M ${x1o} ${y1o} A ${outerR} ${outerR} 0 ${large} 1 ${x2o} ${y2o} L ${x1i} ${y1i} A ${innerR} ${innerR} 0 ${large} 0 ${x2i} ${y2i} Z`;
    return { d, color: colorAt(i), label: str(r[labelKey]), pct: frac * 100 };
  });

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale}>
      <div className="flex h-full items-center gap-4">
        <svg viewBox="0 0 180 200" className="h-full max-h-[180px] shrink-0">
          {slices.map((s, i) => (
            <path key={i} d={s.d} fill={s.color} stroke="var(--card)" strokeWidth={2} />
          ))}
          <text x={cx} y={cy - 6} textAnchor="middle" fontSize={14} fontWeight="700" fill="var(--foreground)">
            {rows.length}
          </text>
          <text x={cx} y={cy + 10} textAnchor="middle" fontSize={9} fill="var(--muted-foreground)">
            items
          </text>
        </svg>
        <div className="flex-1 min-w-0 space-y-1.5 overflow-auto">
          {slices.map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px]">
              <span className="inline-block h-2.5 w-2.5 rounded-sm shrink-0" style={{ background: s.color }} />
              <span className="flex-1 min-w-0 truncate text-foreground">{s.label}</span>
              <span className="shrink-0 text-muted-foreground font-medium tabular-nums">
                {s.pct.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </WidgetFrame>
  );
}

// ---- Heatmap ----------------------------------------------------------------

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
        <div
          className="inline-grid gap-0.5"
          style={{ gridTemplateColumns: `auto repeat(${xs.length}, minmax(28px, 1fr))` }}
        >
          <div />
          {xs.map((x) => (
            <div key={x} className="px-1 text-[9px] text-muted-foreground truncate text-center">
              {x}
            </div>
          ))}
          {ys.map((y) => (
            <HeatmapRow key={y} y={y} xs={xs} cell={cell} maxV={maxV} format={widget.config.format} />
          ))}
        </div>
      </div>
    </WidgetFrame>
  );
}

function HeatmapRow({
  y, xs, cell, maxV, format,
}: {
  y: string;
  xs: string[];
  cell: Map<string, number>;
  maxV: number;
  format?: "currency" | "percent" | "number" | "auto";
}) {
  return (
    <>
      <div className="pr-2 text-[9px] text-muted-foreground truncate self-center max-w-[80px]">{y}</div>
      {xs.map((x) => {
        const v = cell.get(`${x}|${y}`) ?? 0;
        const intensity = maxV ? v / maxV : 0;
        return (
          <div
            key={x}
            title={`${x} / ${y}: ${formatValue(v, format)}`}
            className="h-7 rounded-sm flex items-center justify-center text-[8px] font-medium"
            style={{
              background: `color-mix(in oklab, var(--chart-1) ${Math.round(intensity * 90)}%, transparent)`,
              color: intensity > 0.5 ? "white" : "var(--muted-foreground)",
            }}
          >
            {intensity > 0.1 ? compactNumber(v) : ""}
          </div>
        );
      })}
    </>
  );
}

// ---- Funnel -----------------------------------------------------------------

export function Funnel({ widget }: Props) {
  const rows = (widget.data || []).slice(0, 12);
  const stageKey = widget.config.stage || (widget.config.x as string | undefined);
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
            <div key={i} className="flex items-center gap-2.5">
              <span className="w-20 shrink-0 truncate text-[11px] text-muted-foreground">
                {str(r[stageKey])}
              </span>
              <div className="flex-1 h-5 bg-muted/30 rounded overflow-hidden">
                <div
                  className="h-full rounded flex items-center justify-end pr-2 text-[10px] font-medium text-white/90 transition-all"
                  style={{
                    width: `${Math.max(pct, 6)}%`,
                    background: colorAt(i),
                  }}
                >
                  {pct > 15 ? formatValue(v, widget.config.format) : ""}
                </div>
              </div>
              <span className="w-14 shrink-0 text-right text-[11px] text-muted-foreground tabular-nums">
                {formatValue(v, widget.config.format)}
              </span>
            </div>
          );
        })}
      </div>
    </WidgetFrame>
  );
}

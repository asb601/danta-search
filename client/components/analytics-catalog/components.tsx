"use client";

// Pure-SVG/CSS, metadata-driven Analytics Catalog components built on the shared
// shadcn ui primitives (Card/Badge/Table) and the OKLch token palette. Zero
// external chart dependency; every color flows from --chart-* / semantic tokens
// so the catalog tracks the active theme.

import { useState, useRef, useEffect } from "react";
import { DashboardWidget } from "./types";
import { WidgetFrame, EmptyState, DeltaBadge, WarningChips, statusVariant } from "./WidgetFrame";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
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
function isNumeric(v: unknown): boolean {
  if (typeof v === "number") return Number.isFinite(v);
  if (typeof v === "string" && v.trim() !== "") return Number.isFinite(Number(v));
  return false;
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
  return key
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

// Catmull-Rom → cubic bezier spline for smooth chart lines.
function smoothPath(pts: [number, number][]): string {
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

// ---- KPI Card ---------------------------------------------------------------

export function KpiCard({ widget }: Props) {
  const valueKey = widget.config.value;
  const deltaKey = (widget.config as { delta?: string }).delta;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const delta = deltaKey && row && isNumeric(row[deltaKey]) ? num(row[deltaKey]) : undefined;
  const rowCount = widget.provenance?.row_count;

  return (
    <Card className="flex h-full flex-col justify-between gap-3 p-5">
      <div className="flex items-start justify-between gap-2">
        <p className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground leading-tight truncate">
          {widget.title}
        </p>
        <div className="flex items-center gap-1 shrink-0">
          <WarningChips provenance={widget.provenance} />
          {delta !== undefined && <DeltaBadge value={delta} format={widget.config.format} />}
        </div>
      </div>
      <div>
        <p className="text-3xl font-bold leading-none tracking-tight tabular-nums text-foreground">
          {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
        </p>
        {raw === undefined && widget.provenance?.empty_message ? (
          // Honest blank: explain WHY the KPI has no value (error / missing / no rows),
          // instead of an undiagnosable em-dash.
          <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
            {widget.provenance.empty_message}
          </p>
        ) : valueKey ? (
          <p className="mt-2 text-xs text-muted-foreground truncate">{humanize(valueKey)}</p>
        ) : null}
        {typeof rowCount === "number" && rowCount > 1 && (
          <p className="mt-0.5 text-[11px] text-subtle-foreground">
            across {compactNumber(rowCount)} records
          </p>
        )}
      </div>
    </Card>
  );
}

// ---- Metric Tile ------------------------------------------------------------

export function MetricTile({ widget }: Props) {
  const valueKey = widget.config.value;
  const deltaKey = (widget.config as { delta?: string }).delta;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const delta = deltaKey && row && isNumeric(row[deltaKey]) ? num(row[deltaKey]) : undefined;

  return (
    <Card className="flex h-full flex-col justify-between gap-2 p-4">
      <p className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground truncate">
        {widget.title}
      </p>
      <div className="flex items-end justify-between gap-2">
        <p className="text-2xl font-bold leading-none tracking-tight tabular-nums text-foreground">
          {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
        </p>
        <div className="flex items-center gap-1 shrink-0">
          <WarningChips provenance={widget.provenance} />
          {delta !== undefined && <DeltaBadge value={delta} format={widget.config.format} />}
        </div>
      </div>
      {raw === undefined && widget.provenance?.empty_message && (
        <p className="text-[10px] leading-snug text-muted-foreground">
          {widget.provenance.empty_message}
        </p>
      )}
    </Card>
  );
}

// ---- Data Table -------------------------------------------------------------

export function CatalogTable({ widget }: Props) {
  const rows = widget.data || [];
  if (!rows.length) {
    const msg =
      widget.provenance?.answer ||
      (widget.provenance?.error ? `Could not generate this widget: ${widget.provenance.error}` : undefined) ||
      (widget.provenance?.empty ? "No rows were returned for this query." : undefined);
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? msg}
        />
      </WidgetFrame>
    );
  }
  const cols =
    widget.config.columns && widget.config.columns !== "all"
      ? (widget.config.columns as string[])
      : Object.keys(rows[0]);
  const shown = rows.slice(0, 100);

  // A column is "numeric" if every present value parses as a number.
  const numericCols = new Set(
    cols.filter((c) => shown.every((r) => r[c] === null || r[c] === undefined || isNumeric(r[c]))),
  );

  return (
    <WidgetFrame
      title={widget.title}
      rationale={widget.rationale}
      provenance={widget.provenance}
      insight={widget.config.insight}
      footer={rows.length > shown.length ? `Showing ${shown.length} of ${rows.length} rows` : undefined}
    >
      <div className="h-full overflow-auto rounded-lg">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              {cols.map((c) => (
                <TableHead
                  key={c}
                  className={`sticky top-0 z-10 bg-card ${numericCols.has(c) ? "text-right" : ""}`}
                >
                  {humanize(c)}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((r, i) => (
              <TableRow key={i} className="even:bg-muted/30">
                {cols.map((c) => {
                  const val = r[c];
                  if (numericCols.has(c)) {
                    return (
                      <TableCell key={c} className="text-right tabular-nums text-foreground">
                        {val === null || val === undefined ? "—" : formatValue(val, "number")}
                      </TableCell>
                    );
                  }
                  const sv = typeof val === "string" ? statusVariant(val) : null;
                  return (
                    <TableCell key={c} className="text-foreground">
                      {sv ? <Badge variant={sv}>{str(val)}</Badge> : str(val)}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </WidgetFrame>
  );
}

// ---- Shared SVG frame -------------------------------------------------------

const PAD = { top: 14, right: 16, bottom: 30, left: 48 };
const STROKE = { vectorEffect: "non-scaling-stroke" } as const;

function axisTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round(step * i));
}

/**
 * Measure the chart container so the SVG coordinate system equals real pixels
 * (1 unit = 1px). This lets us fill the cell WITHOUT preserveAspectRatio="none"
 * distortion — text glyphs and rounded corners stay crisp at any aspect ratio.
 */
function useChartSize() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 480, h: 220 });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 8 && height > 8) {
          setSize({ w: Math.round(width), h: Math.round(height) });
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, ...size };
}

// ---- Line / Area Chart (hero) ----------------------------------------------

function LineLike({ widget, area }: Props & { area: boolean }) {
  const allRows = widget.data || [];
  const xKey = widget.config.x;
  const series = yList(widget);
  const { ref, w: W, h: H } = useChartSize();

  // Time-range toggle: slice to the most recent N points for long series.
  const ranges = [
    { label: "All", n: allRows.length },
    { label: "30", n: 30 },
    { label: "12", n: 12 },
  ].filter((r, idx) => idx === 0 || r.n < allRows.length);
  const [rangeN, setRangeN] = useState<number>(allRows.length);
  const rows = rangeN >= allRows.length ? allRows : allRows.slice(allRows.length - rangeN);

  if (!allRows.length || !xKey || !series.length) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? widget.provenance?.answer}
        />
      </WidgetFrame>
    );
  }

  const innerW = Math.max(1, W - PAD.left - PAD.right);
  const innerH = Math.max(1, H - PAD.top - PAD.bottom);
  const maxVal = Math.max(1, ...rows.flatMap((r) => series.map((s) => num(r[s]))));
  const xAt = (i: number) =>
    PAD.left + (rows.length > 1 ? (i / (rows.length - 1)) * innerW : innerW / 2);
  const yAt = (v: number) => PAD.top + innerH - (v / maxVal) * innerH;
  const baseY = PAD.top + innerH;
  const gradNs = `grad-${widget.widget_id}`;

  const toggle =
    ranges.length > 1 ? (
      <div className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5">
        {ranges.map((r) => {
          const active = rangeN === r.n;
          return (
            <button
              key={r.label}
              onClick={() => setRangeN(r.n)}
              className={`rounded-md px-2 py-0.5 text-[10px] font-medium transition-colors ${
                active ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {r.label}
            </button>
          );
        })}
      </div>
    ) : undefined;

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale} action={toggle} provenance={widget.provenance} insight={widget.config.insight}>
      <div className="flex h-full flex-col">
        <div ref={ref} className="w-full flex-1 min-h-0">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%">
          <defs>
            {series.map((_, si) => (
              <linearGradient key={si} id={`${gradNs}-${si}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={colorAt(si)} stopOpacity={0.35} />
                <stop offset="100%" stopColor={colorAt(si)} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>

          {axisTicks(maxVal).map((t, i) => {
            const y = yAt(t);
            return (
              <g key={i}>
                <line
                  x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                  stroke="var(--chart-grid)" strokeWidth={1} {...STROKE}
                />
                <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize={10} fill="var(--chart-axis)">
                  {compactNumber(t)}
                </text>
              </g>
            );
          })}

          {series.map((s, si) => {
            const pts: [number, number][] = rows.map((r, i) => [xAt(i), yAt(num(r[s]))]);
            const color = colorAt(si);
            const line = smoothPath(pts);
            return (
              <g key={s}>
                {area && (
                  <path
                    d={`${line} L ${pts[pts.length - 1][0]},${baseY} L ${pts[0][0]},${baseY} Z`}
                    fill={`url(#${gradNs}-${si})`}
                  />
                )}
                <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" {...STROKE} />
              </g>
            );
          })}

          {rows.map((r, i) => {
            if (rows.length > 12 && i % Math.ceil(rows.length / 8) !== 0) return null;
            return (
              <text key={i} x={xAt(i)} y={H - 9} textAnchor="middle" fontSize={10} fill="var(--chart-axis)">
                {str(r[xKey]).slice(0, 10)}
              </text>
            );
          })}
        </svg>
        </div>

        {series.length > 1 && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-2 pb-1 pt-1">
            {series.map((s, si) => (
              <span key={s} className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <span className="h-2 w-2 rounded-full" style={{ background: colorAt(si) }} />
                {humanize(s)}
              </span>
            ))}
          </div>
        )}
      </div>
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
  const { ref, w: W, h: H } = useChartSize();
  if (!rows.length || !xKey || !yKey) {
    return (
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? widget.provenance?.answer}
        />
      </WidgetFrame>
    );
  }
  const innerW = Math.max(1, W - PAD.left - PAD.right);
  const innerH = Math.max(1, H - PAD.top - PAD.bottom);
  const maxVal = Math.max(1, ...rows.map((r) => num(r[yKey])));
  const slot = innerW / rows.length;
  const barW = Math.max(4, slot * 0.64);
  const gradId = `bgrad-${widget.widget_id}`;

  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance} insight={widget.config.insight}>
      <div ref={ref} className="h-full w-full">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.95} />
            <stop offset="100%" stopColor="var(--chart-2)" stopOpacity={0.6} />
          </linearGradient>
        </defs>
        {axisTicks(maxVal).map((t, i) => {
          const y = PAD.top + innerH - (t / maxVal) * innerH;
          return (
            <g key={i}>
              <line
                x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                stroke="var(--chart-grid)" strokeWidth={1} {...STROKE}
              />
              <text x={PAD.left - 6} y={y + 3.5} textAnchor="end" fontSize={10} fill="var(--chart-axis)">
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
            <g key={i} className="transition-opacity hover:opacity-80">
              <title>{`${str(r[xKey])}: ${formatValue(v, widget.config.format)}`}</title>
              <rect x={x} y={y} width={barW} height={Math.max(h, 1)} rx={3} fill={`url(#${gradId})`} />
              {rows.length <= 12 && (
                <text x={x + barW / 2} y={H - 9} textAnchor="middle" fontSize={10} fill="var(--chart-axis)">
                  {str(r[xKey]).slice(0, 8)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      </div>
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
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? widget.provenance?.answer}
        />
      </WidgetFrame>
    );
  }
  const total = rows.reduce((s, r) => s + num(r[valueKey]), 0) || 1;
  const cx = 90, cy = 100, outerR = 74, innerR = 46;
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
    <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance} insight={widget.config.insight}>
      <div className="flex h-full items-center gap-4 px-2">
        <svg viewBox="0 0 180 200" className="h-full max-h-[180px] shrink-0">
          {slices.map((s, i) => (
            <path key={i} d={s.d} fill={s.color} stroke="var(--card)" strokeWidth={2} />
          ))}
          <text x={cx} y={cy - 4} textAnchor="middle" fontSize={15} fontWeight="700" fill="var(--foreground)">
            {compactNumber(total)}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" fontSize={10} fill="var(--chart-axis)">
            total
          </text>
        </svg>
        <div className="flex-1 min-w-0 space-y-1.5 overflow-auto">
          {slices.map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px]">
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: s.color }} />
              <span className="flex-1 min-w-0 truncate text-foreground">{s.label}</span>
              <span className="shrink-0 font-medium tabular-nums text-muted-foreground">
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
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? widget.provenance?.answer}
        />
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
    <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance} insight={widget.config.insight}>
      <div className="h-full overflow-auto px-2">
        <div
          className="inline-grid gap-0.5"
          style={{ gridTemplateColumns: `auto repeat(${xs.length}, minmax(30px, 1fr))` }}
        >
          <div />
          {xs.map((x) => (
            <div key={x} className="px-1 text-[10px] text-muted-foreground truncate text-center">
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
      <div className="self-center max-w-[90px] truncate pr-2 text-[10px] text-muted-foreground">{y}</div>
      {xs.map((x) => {
        const v = cell.get(`${x}|${y}`) ?? 0;
        const intensity = maxV ? v / maxV : 0;
        return (
          <div
            key={x}
            title={`${x} / ${y}: ${formatValue(v, format)}`}
            className="flex h-7 items-center justify-center rounded-sm text-[8px] font-medium"
            style={{
              background: `color-mix(in oklab, var(--chart-1) ${Math.round(intensity * 90)}%, transparent)`,
              color: intensity > 0.5 ? "var(--primary-foreground)" : "var(--muted-foreground)",
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
      <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance}>
        <EmptyState
          reason={widget.provenance?.empty_reason}
          message={widget.provenance?.empty_message ?? widget.provenance?.answer}
        />
      </WidgetFrame>
    );
  }
  const maxV = Math.max(1, ...rows.map((r) => num(r[valueKey])));
  const firstV = num(rows[0]?.[valueKey]) || maxV;
  return (
    <WidgetFrame title={widget.title} rationale={widget.rationale} provenance={widget.provenance} insight={widget.config.insight}>
      <div className="flex h-full flex-col justify-center gap-2 px-2">
        {rows.map((r, i) => {
          const v = num(r[valueKey]);
          const pct = (v / maxV) * 100;
          const conv = firstV ? (v / firstV) * 100 : 0;
          return (
            <div key={i} className="flex items-center gap-2.5">
              <span className="w-20 shrink-0 truncate text-[11px] text-muted-foreground">
                {str(r[stageKey])}
              </span>
              <div className="relative h-5 flex-1 overflow-hidden rounded bg-muted/40">
                <div
                  className="flex h-full items-center justify-end rounded pr-2 text-[10px] font-medium text-primary-foreground"
                  style={{ width: `${Math.max(pct, 4)}%`, background: colorAt(i) }}
                >
                  {pct > 22 ? formatValue(v, widget.config.format) : ""}
                </div>
              </div>
              <span className="w-12 shrink-0 text-right text-[10px] tabular-nums text-muted-foreground">
                {i === 0 ? formatValue(v, widget.config.format) : `${conv.toFixed(0)}%`}
              </span>
            </div>
          );
        })}
      </div>
    </WidgetFrame>
  );
}

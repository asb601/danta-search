"use client";

// SummaryView — the tie-out artifact a CFO/investor exports. Where the Board is a
// dense visual collage, the Summary is a linear, auditable ledger: it walks
// config.widgets IN BOARD ORDER and, for each, prints a labelled section with the
// widget title, its provenance line (metric · grain · source) when present, and
// the underlying data as a plain table. KPI/gauge/target tiles render as a single
// value + delta row instead of a one-row table. It is a pure projection of the
// persisted config — it never recomputes a number.

import { DashboardConfig, DashboardWidget, WidgetRow } from "./types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { DeltaBadge } from "./WidgetFrame";
import { formatValue } from "./palette";

// Single-value tiles render as a headline value + optional delta, not a table.
const VALUE_TYPES = new Set([
  "kpi_card",
  "metric_tile",
  "delta_kpi",
  "gauge_ring",
  "progress_kpi",
  "bullet",
]);

function isNumeric(v: unknown): boolean {
  if (typeof v === "number") return Number.isFinite(v);
  if (typeof v === "string" && v.trim() !== "") return Number.isFinite(Number(v));
  return false;
}
function num(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}
function str(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return String(v);
}
function humanize(key: string): string {
  return key
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

// The provenance line: metric · grain · source. Each part is shown only when the
// backend persisted it (planned spec for metric/grain; files_used for source).
function provenanceParts(w: DashboardWidget): { label: string; value: string }[] {
  const out: { label: string; value: string }[] = [];
  const planned = (w.provenance?.spec as { planned?: { measure?: string; grain?: string; dimension?: string } } | undefined)
    ?.planned;
  const metric = planned?.measure ?? (typeof w.config?.value === "string" ? w.config.value : undefined);
  const grain = planned?.grain ?? planned?.dimension ?? (typeof w.config?.x === "string" ? w.config.x : undefined);
  if (metric) out.push({ label: "Metric", value: humanize(metric) });
  if (grain) out.push({ label: "Grain", value: humanize(grain) });
  const files = w.provenance?.files_used;
  if (Array.isArray(files) && files.length) {
    out.push({ label: "Source", value: files.map((f) => f.split("/").pop() ?? f).join(", ") });
  }
  if (typeof w.provenance?.row_count === "number") {
    out.push({ label: "Rows", value: w.provenance.row_count.toLocaleString() });
  }
  return out;
}

export function SummaryView({ config }: { config: DashboardConfig | null | undefined }) {
  const widgets = config?.widgets ?? [];

  if (!widgets.length) {
    return (
      <div className="flex h-full min-h-[300px] flex-col items-center justify-center rounded-xl border border-dashed border-border/60 bg-muted/10 p-8 text-center">
        <p className="text-sm font-medium text-foreground">Nothing to summarize yet</p>
        <p className="mt-1 max-w-xs text-xs text-muted-foreground">
          Generate the board first — the summary is a line-by-line tie-out of every widget.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Title block — the cover line of the export. */}
      <div className="rounded-xl border border-border bg-card px-5 py-4">
        <div className="flex items-baseline justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              Tie-out summary
            </p>
            <h2 className="mt-0.5 truncate text-lg font-bold tracking-tight text-foreground">
              {config?.title || "Dashboard summary"}
            </h2>
          </div>
          <Badge variant="muted" className="shrink-0">
            {widgets.length} section{widgets.length === 1 ? "" : "s"}
          </Badge>
        </div>
        {config?.description && (
          <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{config.description}</p>
        )}
      </div>

      {widgets.map((w, i) => (
        <SummarySection key={w.widget_id} widget={w} index={i + 1} />
      ))}
    </div>
  );
}

function SummarySection({ widget, index }: { widget: DashboardWidget; index: number }) {
  const parts = provenanceParts(widget);
  const isValue = VALUE_TYPES.has(widget.type);
  const rows = widget.data ?? [];

  return (
    <Card className="overflow-hidden">
      {/* Section header: ordinal + title + provenance line */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-border/60 px-5 py-3">
        <span className="text-[11px] font-semibold tabular-nums text-muted-foreground/70">
          {String(index).padStart(2, "0")}
        </span>
        <h3 className="text-sm font-semibold tracking-tight text-foreground">{widget.title}</h3>
        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground/60">
          {widget.type.replace(/_/g, " ")}
        </span>
        {parts.length > 0 && (
          <div className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-0.5">
            {parts.map((p) => (
              <span key={p.label} className="text-[11px] text-muted-foreground">
                <span className="text-muted-foreground/55">{p.label}:</span>{" "}
                <span className="font-medium text-foreground/80">{p.value}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="px-5 py-4">
        {isValue ? (
          <ValueRow widget={widget} />
        ) : rows.length ? (
          <SummaryTable widget={widget} rows={rows} />
        ) : (
          <p className="text-xs text-muted-foreground">
            {widget.provenance?.empty_message ||
              widget.provenance?.answer ||
              "No rows were returned for this widget."}
          </p>
        )}
      </div>
    </Card>
  );
}

// A single value + delta for KPI/gauge/target tiles — the headline number, audited.
function ValueRow({ widget }: { widget: DashboardWidget }) {
  const valueKey =
    widget.config.value ||
    (Array.isArray(widget.config.y) ? widget.config.y[0] : (widget.config.y as string | undefined));
  const deltaKey = (widget.config as { delta?: string }).delta;
  const row = widget.data?.[0];
  const raw = valueKey && row ? row[valueKey] : undefined;
  const delta = deltaKey && row && isNumeric(row[deltaKey]) ? num(row[deltaKey]) : undefined;
  const targetKey = (widget.config as { target?: string }).target;
  const targetVal = (widget.config as { target_value?: number }).target_value;
  const target =
    targetKey && row && isNumeric(row[targetKey])
      ? num(row[targetKey])
      : typeof targetVal === "number" && Number.isFinite(targetVal)
        ? targetVal
        : undefined;

  return (
    <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
      <div>
        <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          {valueKey ? humanize(valueKey) : "Value"}
        </p>
        <p className="mt-0.5 text-3xl font-bold leading-none tracking-tight tabular-nums text-foreground">
          {raw === undefined ? "—" : formatValue(raw, widget.config.format)}
        </p>
      </div>
      {delta !== undefined && (
        <div className="pb-1">
          <DeltaBadge value={delta} format={widget.config.format} />
        </div>
      )}
      {target !== undefined && (
        <div className="pb-1">
          <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Target</p>
          <p className="mt-0.5 text-sm font-semibold tabular-nums text-foreground/80">
            {formatValue(target, widget.config.format)}
          </p>
        </div>
      )}
    </div>
  );
}

// A light, capped table of the widget's underlying rows — the audit trail.
function SummaryTable({ widget, rows }: { widget: DashboardWidget; rows: WidgetRow[] }) {
  const cols =
    widget.config.columns && widget.config.columns !== "all"
      ? (widget.config.columns as string[])
      : Object.keys(rows[0] ?? {});
  const shown = rows.slice(0, 50);
  const numericCols = new Set(
    cols.filter((c) => shown.every((r) => r[c] === null || r[c] === undefined || isNumeric(r[c]))),
  );

  if (!cols.length) {
    return <p className="text-xs text-muted-foreground">No columns to display.</p>;
  }

  return (
    <div className="overflow-auto rounded-lg border border-border/60">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {cols.map((c) => (
              <TableHead key={c} className={`bg-card ${numericCols.has(c) ? "text-right" : ""}`}>
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
                return (
                  <TableCell key={c} className="text-foreground">
                    {str(val)}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {rows.length > shown.length && (
        <p className="px-3 py-2 text-[10px] text-muted-foreground/70">
          Showing {shown.length} of {rows.length} rows
        </p>
      )}
    </div>
  );
}

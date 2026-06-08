"use client";

import { useState } from "react";
import { Info, TrendingUp, TrendingDown, Inbox, SearchX, Unplug, CircleAlert, TriangleAlert } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge, BadgeVariant } from "@/components/ui/badge";
import { formatValue } from "./palette";
import { WidgetProvenance } from "./types";

export function WidgetFrame({
  title,
  rationale,
  children,
  footer,
  action,
  provenance,
  insight,
}: {
  title: string;
  rationale?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  action?: React.ReactNode;
  provenance?: WidgetProvenance;
  insight?: string;
}) {
  const [showInfo, setShowInfo] = useState(false);
  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-2 p-4 pb-2">
        <CardTitle className="truncate">{title}</CardTitle>
        <div className="flex items-center gap-1 shrink-0">
          <WarningChips provenance={provenance} />
          {action}
          {rationale && (
            <div className="relative">
              <button
                onMouseEnter={() => setShowInfo(true)}
                onMouseLeave={() => setShowInfo(false)}
                className="p-0.5 rounded text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                aria-label="Why this visualization"
              >
                <Info className="w-3.5 h-3.5" />
              </button>
              {showInfo && (
                <div className="absolute right-0 top-6 z-30 w-60 rounded-lg border border-border bg-popover p-2.5 text-[11px] leading-relaxed text-muted-foreground shadow-lg">
                  {rationale}
                </div>
              )}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 p-2 pt-0">{children}</CardContent>
      {insight && <InsightStrip text={insight} />}
      {footer && (
        <CardFooter className="px-4 pb-3 pt-0 text-[10px] text-muted-foreground/70">
          {footer}
        </CardFooter>
      )}
    </Card>
  );
}

/** Calm amber annotation chips for per-widget correctness caveats — visible but
 *  never alarmist (amber, not red). Surfaces the backend's tie_out / join_warning
 *  verbatim; the renderer never recomputes them. */
export function WarningChips({ provenance }: { provenance?: WidgetProvenance }) {
  if (!provenance) return null;
  const chips: { key: string; label: string; title: string; tone: "warn" | "muted" }[] = [];
  if (provenance.tie_out === "over") {
    chips.push({
      key: "tie",
      tone: "warn",
      label: "May double-count",
      title: "The breakdown sums to more than the headline total — a double-counting symptom.",
    });
  }
  if (provenance.join_warning === "multi_table_no_validated_join") {
    chips.push({
      key: "join",
      tone: "warn",
      label: "Unvalidated join",
      title: "This widget combined tables without a validated relationship; the number may double-count.",
    });
  }
  // P7 — honest "not filtered" note (informational, neutral — not an alarm).
  const fs = provenance.filter_status;
  if (fs?.status === "not_affected" && Array.isArray(fs.dimensions) && fs.dimensions.length) {
    chips.push({
      key: "na",
      tone: "muted",
      label: `Not filtered by ${fs.dimensions.join(", ")}`,
      title: "This widget's table doesn't carry the slicer dimension, so the board filter doesn't apply here.",
    });
  }
  if (!chips.length) return null;
  return (
    <>
      {chips.map((c) => (
        <span
          key={c.key}
          title={c.title}
          className={
            c.tone === "warn"
              ? "inline-flex items-center gap-1 rounded-md border border-warn-border bg-warn-bg px-1.5 py-0.5 text-[10px] font-medium text-warn-fg"
              : "inline-flex items-center gap-1 rounded-md border border-border bg-muted/40 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
          }
        >
          {c.tone === "warn" && <TriangleAlert className="h-3 w-3" />}
          {c.label}
        </span>
      ))}
    </>
  );
}

/** The analyst caption — a quiet, declarative takeaway under the chart. The amber
 *  dot marks it as the pull-quote; no "Insight:" prefix, no emoji. */
function InsightStrip({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 border-t border-border/60 px-4 py-2">
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-chart-1" />
      <p className="truncate text-[12px] leading-relaxed text-muted-foreground" title={text}>
        {text}
      </p>
    </div>
  );
}

/** Honest 3-way empty state — distinct, calm visuals per reason; error is amber,
 *  never a red wall. Falls back to the generic look for legacy (1.1) configs. */
export function EmptyState({ message, reason }: { message?: string; reason?: "empty" | "missing" | "error" }) {
  const variants = {
    error: { Icon: CircleAlert, ring: "bg-warn-bg", tint: "text-warn-fg" },
    missing: { Icon: Unplug, ring: "bg-muted/50", tint: "text-muted-foreground/70" },
    empty: { Icon: SearchX, ring: "bg-muted/50", tint: "text-muted-foreground/60" },
  } as const;
  const v = (reason && variants[reason]) || { Icon: Inbox, ring: "bg-muted/50", tint: "text-muted-foreground/60" };
  const Icon = v.Icon;
  return (
    <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2 px-5 text-center">
      <div className={`flex h-9 w-9 items-center justify-center rounded-full ${v.ring}`}>
        <Icon className={`h-4 w-4 ${v.tint}`} />
      </div>
      <p className="max-w-[280px] text-[11px] leading-relaxed text-muted-foreground">
        {message || "No data available"}
      </p>
    </div>
  );
}

/**
 * A +/- delta pill. The arrow + sign carry the NUMERIC direction; the color
 * carries the BUSINESS meaning via `polarity`:
 *   - "positive" (growth metric): a rise is GOOD  → success token.
 *   - "inverse"  (cost/aging/DSO/returns/overdue): a rise is BAD → danger token.
 * Polarity is resolved by the planner (the LLM proposes the metric semantics) and
 * carried in widget.config.polarity; it fails safe to "positive". Without a
 * polarity hint the badge stays neutral gray (today's behavior).
 */
export function DeltaBadge({
  value,
  format,
  polarity,
}: {
  value: number;
  format?: "currency" | "percent" | "number" | "auto";
  polarity?: "positive" | "inverse";
}) {
  const rising = value >= 0;
  const Icon = rising ? TrendingUp : TrendingDown;
  // Good when a growth metric rises OR an inverse metric falls.
  const good = polarity === "inverse" ? !rising : rising;
  const variant: BadgeVariant = polarity ? (good ? "success" : "danger") : "muted";
  return (
    <Badge variant={variant}>
      <Icon className="h-3 w-3" />
      {rising ? "+" : "−"}
      {formatValue(Math.abs(value), format)}
    </Badge>
  );
}

// Exact-match status vocabularies — substring matching painted ordinary text
// (e.g. "Operations" → "op", "class" → "cl") as colored badges, so we match the
// whole trimmed value only. Danger is checked first so negations ("not
// delivered") win over the success token they contain.
const _DANGER = new Set([
  "failed", "fail", "error", "cancelled", "canceled", "rejected", "overdue",
  "lost", "blocked", "not delivered", "not completed", "declined", "expired", "void",
]);
const _SUCCESS = new Set([
  "active", "success", "successful", "paid", "completed", "complete", "approved",
  "closed", "cl", "done", "won", "delivered", "fulfilled", "shipped", "settled",
]);
const _WARNING = new Set([
  "pending", "processing", "in process", "in progress", "open", "op", "draft",
  "partial", "partially delivered", "partially", "review", "on hold", "hold", "queued",
]);

/** Map a known status value to a Badge variant; null for ordinary text. */
export function statusVariant(raw: string): BadgeVariant | null {
  const v = raw.trim().toLowerCase();
  if (!v) return null;
  // Monochrome: a known status gets a neutral gray badge (no semantic color).
  if (_DANGER.has(v) || _SUCCESS.has(v) || _WARNING.has(v)) return "muted";
  return null;
}

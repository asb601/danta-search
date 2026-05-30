"use client";

import { useState } from "react";
import { Info, TrendingUp, TrendingDown, Inbox } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge, BadgeVariant } from "@/components/ui/badge";
import { formatValue } from "./palette";

export function WidgetFrame({
  title,
  rationale,
  children,
  footer,
  action,
}: {
  title: string;
  rationale?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  action?: React.ReactNode;
}) {
  const [showInfo, setShowInfo] = useState(false);
  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-2 p-4 pb-2">
        <CardTitle className="truncate">{title}</CardTitle>
        <div className="flex items-center gap-1 shrink-0">
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
      {footer && (
        <CardFooter className="px-4 pb-3 pt-0 text-[10px] text-muted-foreground/70">
          {footer}
        </CardFooter>
      )}
    </Card>
  );
}

export function EmptyState({ message }: { message?: string }) {
  return (
    <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2 px-5 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted/50">
        <Inbox className="h-4 w-4 text-muted-foreground/60" />
      </div>
      <p className="max-w-[280px] text-[11px] leading-relaxed text-muted-foreground">
        {message || "No data available"}
      </p>
    </div>
  );
}

/** A +/- delta pill, colored by sign via the success/danger tokens. */
export function DeltaBadge({
  value,
  format,
}: {
  value: number;
  format?: "currency" | "percent" | "number" | "auto";
}) {
  const positive = value >= 0;
  const Icon = positive ? TrendingUp : TrendingDown;
  return (
    <Badge variant={positive ? "success" : "danger"}>
      <Icon className="h-3 w-3" />
      {positive ? "+" : "−"}
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
  if (_DANGER.has(v)) return "danger";
  if (_SUCCESS.has(v)) return "success";
  if (_WARNING.has(v)) return "warning";
  return null;
}

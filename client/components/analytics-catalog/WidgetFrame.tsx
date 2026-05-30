"use client";

import { useState } from "react";
import { Info } from "lucide-react";

export function WidgetFrame({
  title,
  rationale,
  children,
  footer,
}: {
  title: string;
  rationale?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  const [showInfo, setShowInfo] = useState(false);
  return (
    <div className="flex flex-col h-full rounded-xl border border-border bg-card overflow-hidden">
      <div className="flex items-center justify-between gap-2 px-4 pt-3 pb-2 shrink-0">
        <h3 className="text-sm font-semibold text-foreground truncate leading-tight">{title}</h3>
        {rationale && (
          <div className="relative shrink-0">
            <button
              onMouseEnter={() => setShowInfo(true)}
              onMouseLeave={() => setShowInfo(false)}
              className="p-0.5 rounded text-muted-foreground/50 hover:text-muted-foreground transition-colors"
              aria-label="Why this visualization"
            >
              <Info className="w-3.5 h-3.5" />
            </button>
            {showInfo && (
              <div className="absolute right-0 top-6 z-20 w-60 rounded-lg border border-border bg-popover p-2.5 text-[11px] leading-relaxed text-muted-foreground shadow-xl">
                {rationale}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0 px-1 pb-1">{children}</div>
      {footer && (
        <div className="px-4 pb-2 text-[10px] text-muted-foreground/60">{footer}</div>
      )}
    </div>
  );
}

export function EmptyState({ message }: { message?: string }) {
  return (
    <div className="flex h-full min-h-[80px] flex-col items-center justify-center gap-2 px-4 text-center">
      <div className="w-8 h-8 rounded-full bg-muted/40 flex items-center justify-center">
        <span className="text-muted-foreground text-sm">—</span>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed max-w-[260px]">
        {message || "No data available"}
      </p>
    </div>
  );
}

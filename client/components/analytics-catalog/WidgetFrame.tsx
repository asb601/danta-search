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
    <div className="flex flex-col h-full bg-surface border border-border rounded-lg p-4 overflow-hidden">
      <div className="flex items-start justify-between gap-2 mb-3">
        <h3 className="text-sm font-semibold text-foreground truncate">{title}</h3>
        {rationale && (
          <div className="relative shrink-0">
            <button
              onMouseEnter={() => setShowInfo(true)}
              onMouseLeave={() => setShowInfo(false)}
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Why this visualization"
            >
              <Info className="w-3.5 h-3.5" />
            </button>
            {showInfo && (
              <div className="absolute right-0 top-5 z-20 w-56 rounded-md border border-border bg-card p-2 text-[11px] leading-snug text-muted-foreground shadow-lg">
                {rationale}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0">{children}</div>
      {footer && <div className="mt-2 text-[11px] text-muted-foreground">{footer}</div>}
    </div>
  );
}

export function EmptyState({ message }: { message?: string }) {
  return (
    <div className="flex h-full min-h-[120px] items-center justify-center text-xs text-muted-foreground">
      {message || "No data available"}
    </div>
  );
}

"use client";

import { useState, useRef, useEffect } from "react";
import { Zap, Hammer, Globe, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/** The reingest-all scopes, mapped to the backend query params they send. */
export type ReingestAllScope = "quick" | "full" | "org_wide";

const OPTIONS: {
  scope: ReingestAllScope;
  label: string;
  desc: string;
  Icon: typeof Zap;
}[] = [
  {
    scope: "quick",
    label: "Reingest (reuse converted files)",
    desc: "Redo AI analysis & relationships. Reuses parquet — fast.",
    Icon: Zap,
  },
  {
    scope: "full",
    label: "Preprocess + Reingest (full rebuild)",
    desc: "Re-clean & re-convert (picks up xlsx / not-yet-converted). Slow.",
    Icon: Hammer,
  },
  {
    scope: "org_wide",
    label: "Reingest ALL containers (org-wide)",
    desc: "Rebuilds the ERP knowledge base across every container in the org.",
    Icon: Globe,
  },
];

interface ReingestAllMenuProps {
  /** Quick reingest of the current container (force_preprocess=false). */
  onQuick?: () => void;
  /** Full rebuild of the current container (force_preprocess=true). */
  onFull?: () => void;
  /** Org-wide reingest across every container (all_containers=true). */
  onOrgWide?: () => void;
  quickLoading?: boolean;
  fullLoading?: boolean;
  orgWideLoading?: boolean;
}

/**
 * Single dropdown that consolidates the folder-level reingest actions.
 * Matches ReprocessMenu's visual pattern (button + ChevronDown + popover list
 * with icon / label / description per option). Self-contains open state with
 * Escape + click-outside close. Loading state is owned by the parent (page).
 */
export function ReingestAllMenu({
  onQuick,
  onFull,
  onOrgWide,
  quickLoading,
  fullLoading,
  orgWideLoading,
}: ReingestAllMenuProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const busy = !!(quickLoading || fullLoading || orgWideLoading);

  // Click-outside + Escape close, consistent with ReprocessMenu.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const handlers: Record<ReingestAllScope, (() => void) | undefined> = {
    quick: onQuick,
    full: onFull,
    org_wide: onOrgWide,
  };
  const loadingFor: Record<ReingestAllScope, boolean | undefined> = {
    quick: quickLoading,
    full: fullLoading,
    org_wide: orgWideLoading,
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={busy}
        title="Reingest options"
        className={cn(
          "h-7 px-2.5 flex items-center gap-1.5 rounded-md text-xs font-medium transition-colors",
          busy
            ? "text-amber-400/70 bg-amber-400/10 cursor-not-allowed"
            : "text-amber-400 hover:bg-amber-400/10"
        )}
      >
        <Zap className={cn("w-3.5 h-3.5", busy && "animate-pulse")} />
        {busy ? "Reingesting…" : "Reingest All"}
        <ChevronDown className="w-3 h-3 opacity-70" />
      </button>

      {open && (
        <div className="absolute right-0 top-9 z-[200] w-72 bg-surface border border-border rounded-lg shadow-md py-1">
          {OPTIONS.map(({ scope, label, desc, Icon }) => {
            const handler = handlers[scope];
            const itemLoading = loadingFor[scope];
            return (
              <button
                key={scope}
                onClick={() => {
                  setOpen(false);
                  handler?.();
                }}
                disabled={busy || !handler}
                className={cn(
                  "w-full px-3 py-2 flex items-start gap-2.5 text-left transition-colors",
                  busy || !handler
                    ? "cursor-not-allowed opacity-60"
                    : "hover:bg-surface-raised"
                )}
              >
                <Icon
                  className={cn(
                    "w-3.5 h-3.5 mt-0.5 text-amber-400 shrink-0",
                    itemLoading && "animate-pulse"
                  )}
                />
                <span className="flex flex-col">
                  <span className="text-xs font-medium text-foreground">{label}</span>
                  <span className="text-[11px] text-muted-foreground leading-snug">{desc}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

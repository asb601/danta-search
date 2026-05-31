"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { RefreshCw, Wand2, Hammer, ChevronDown } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

type Scope = "refresh_rules" | "re_analyze" | "full_rebuild";

const OPTIONS: { scope: Scope; label: string; desc: string; Icon: typeof RefreshCw }[] = [
  {
    scope: "refresh_rules",
    label: "Refresh Business Rules",
    desc: "Re-apply ERP rules & rebuild the data contract. Fast (~minutes).",
    Icon: RefreshCw,
  },
  {
    scope: "re_analyze",
    label: "Re-analyze Files",
    desc: "Redo all AI analysis. Reuses converted files (skips preprocessing).",
    Icon: Wand2,
  },
  {
    scope: "full_rebuild",
    label: "Full Rebuild",
    desc: "Reprocess everything from scratch (clean → convert → analyze).",
    Icon: Hammer,
  },
];

interface ReprocessMenuProps {
  containerId?: string;
  /** Called after a successful queue so the parent can refresh/poll. */
  onQueued?: () => void;
}

/**
 * The 3 scoped re-ingestion actions. One control, one backend endpoint
 * (POST /api/chat/reprocess). Self-contained: own open/loading/message state.
 */
interface Progress {
  in_progress: boolean;
  done: number;
  total: number;
}

export function ReprocessMenu({ containerId, onQueued }: ReprocessMenuProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<Scope | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const disabled = !containerId;

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  // Poll progress while a re-ingestion is running, so we can tell the user to
  // wait (chat may be slower while ingestion uses its reserved cores).
  useEffect(() => {
    if (!containerId || !progress?.in_progress) return;
    let alive = true;
    const tick = async () => {
      try {
        const res = await apiFetch(`/api/chat/reprocess-status?container_id=${encodeURIComponent(containerId)}`);
        const data = (await res.json()) as Progress;
        if (alive) setProgress(data);
      } catch {
        /* keep last known state */
      }
    };
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [containerId, progress?.in_progress]);

  const run = useCallback(
    async (scope: Scope) => {
      if (!containerId) return;
      setBusy(scope);
      setMessage(null);
      setOpen(false);
      try {
        const res = await apiFetch("/api/chat/reprocess", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ container_id: containerId, scope }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          setMessage(data?.detail || "Failed to start.");
        } else {
          setMessage(null);
          // Kick off progress polling so the wait banner appears immediately.
          setProgress({ in_progress: true, done: 0, total: data?.queued ?? 0 });
          onQueued?.();
        }
      } catch {
        setMessage("Failed to start.");
      } finally {
        setBusy(null);
        if (message) setTimeout(() => setMessage(null), 6000);
      }
    },
    [containerId, onQueued, message]
  );

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => !disabled && setOpen((o) => !o)}
        disabled={disabled || busy !== null}
        title={disabled ? "Select a container first" : "Re-ingest options"}
        className={cn(
          "h-7 px-2.5 flex items-center gap-1.5 rounded-md text-xs font-medium transition-colors",
          disabled || busy
            ? "text-muted-foreground/60 cursor-not-allowed"
            : "text-amber-400 hover:bg-amber-400/10"
        )}
      >
        <RefreshCw className={cn("w-3.5 h-3.5", busy && "animate-spin")} />
        {busy ? "Queuing…" : "Re-ingest"}
        <ChevronDown className="w-3 h-3 opacity-70" />
      </button>

      {open && (
        <div className="absolute right-0 top-9 z-[200] w-72 bg-surface border border-border rounded-lg shadow-md py-1">
          {OPTIONS.map(({ scope, label, desc, Icon }) => (
            <button
              key={scope}
              onClick={() => run(scope)}
              className="w-full px-3 py-2 flex items-start gap-2.5 text-left hover:bg-surface-raised transition-colors"
            >
              <Icon className="w-3.5 h-3.5 mt-0.5 text-amber-400 shrink-0" />
              <span className="flex flex-col">
                <span className="text-xs font-medium text-foreground">{label}</span>
                <span className="text-[11px] text-muted-foreground leading-snug">{desc}</span>
              </span>
            </button>
          ))}
        </div>
      )}

      {message && !progress?.in_progress && (
        <div className="absolute right-0 top-9 z-[200] w-72 bg-surface border border-border rounded-lg shadow-md px-3 py-2 text-[11px] text-muted-foreground">
          {message}
        </div>
      )}

      {progress?.in_progress && (
        <div className="absolute right-0 top-9 z-[200] w-72 bg-surface border border-amber-400/40 rounded-lg shadow-md px-3 py-2">
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-amber-400">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Re-ingestion in progress
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground leading-snug">
            {progress.total > 0 ? `${progress.done}/${progress.total} files done. ` : ""}
            Chat may be slower until this finishes — please wait.
          </div>
        </div>
      )}
    </div>
  );
}

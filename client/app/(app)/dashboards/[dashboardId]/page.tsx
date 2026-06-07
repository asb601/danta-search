"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback, use } from "react";
import { ArrowLeft, Sparkles, Loader2, RefreshCw, LayoutDashboard } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { DashboardFull, DashboardConfig } from "@/components/analytics-catalog/types";
import { DashboardRenderer } from "@/components/analytics-catalog/DashboardRenderer";
import { SlicerBar } from "@/components/analytics-catalog/SlicerBar";
import { ActiveFilter } from "@/components/analytics-catalog/types";
import { Button } from "@/components/ui/button";
import { Badge, BadgeVariant } from "@/components/ui/badge";

export default function DashboardDetailPage({
  params,
}: {
  params: Promise<{ dashboardId: string }>;
}) {
  const { dashboardId } = use(params);
  const router = useRouter();
  const [dashboard, setDashboard] = useState<DashboardFull | null>(null);
  const [loading, setLoading] = useState(true);
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const res = await apiFetch(`/api/dashboards/${dashboardId}`);
    if (res.ok) setDashboard(await res.json());
    else if (res.status === 404) router.replace("/dashboards");
    setLoading(false);
  }, [dashboardId, router]);

  useEffect(() => { load(); }, [load]);

  const config = dashboard?.config && "widgets" in dashboard.config
    ? (dashboard.config as DashboardConfig)
    : null;

  const widgetCount = config?.widgets?.length ?? 0;

  const generate = async (opts?: { promptOverride?: string; filters?: ActiveFilter[] }) => {
    const p = (opts?.promptOverride ?? prompt).trim();
    if (!p || generating) return;
    setGenerating(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/dashboards/${dashboardId}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // P7: re-applying a slicer regenerates the SAME board prompt (not append) with
        // the selected global_filters; numbers are recomputed by the agent.
        body: JSON.stringify({
          prompt: p,
          append: opts?.filters ? false : widgetCount > 0,
          global_filters: opts?.filters ?? [],
        }),
      });
      if (res.ok) {
        setDashboard(await res.json());
        if (!opts?.promptOverride) setPrompt("");
      } else {
        const body = await res.json().catch(() => ({}));
        setError(body.detail || "Generation failed — check your data files are ingested.");
      }
    } catch {
      setError("Network error during generation.");
    } finally {
      setGenerating(false);
    }
  };

  // P7: apply the board slicer — regenerate the current board prompt with filters.
  const applyFilters = (filters: ActiveFilter[]) => {
    const lastPrompt = config?.prompt || prompt;
    if (lastPrompt) generate({ promptOverride: lastPrompt, filters });
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading dashboard…</p>
        </div>
      </div>
    );
  }
  if (!dashboard) return null;

  return (
    <div className="flex h-full flex-col bg-background">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="shrink-0 border-b border-border/60 bg-card/40 backdrop-blur-sm px-5 py-3">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/dashboards")}
            aria-label="Back to dashboards"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>

          {/* Title */}
          <div className="flex-1 min-w-0 flex items-center gap-2">
            <LayoutDashboard className="w-4 h-4 text-muted-foreground shrink-0" />
            <input
              defaultValue={dashboard.title}
              onBlur={async (e) => {
                const t = e.target.value.trim();
                if (t && t !== dashboard.title) {
                  await apiFetch(`/api/dashboards/${dashboardId}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title: t }),
                  });
                }
              }}
              className="bg-transparent text-base font-semibold text-foreground focus:outline-none min-w-0 flex-1"
              placeholder="Untitled dashboard"
            />
          </div>

          {/* Meta chips */}
          <div className="flex items-center gap-2 shrink-0">
            {widgetCount > 0 && (
              <Badge variant="muted" className="hidden sm:inline-flex">
                {widgetCount} widget{widgetCount !== 1 ? "s" : ""}
              </Badge>
            )}
            <StatusBadge status={dashboard.status} generating={generating} />
            {config?.generated_at && (
              <span className="hidden md:inline text-[11px] text-muted-foreground/60">
                {new Date(config.generated_at).toLocaleDateString(undefined, {
                  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
                })}
              </span>
            )}
            {widgetCount > 0 && !generating && (
              <Button variant="ghost" size="icon" onClick={load} title="Refresh dashboard">
                <RefreshCw className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* ── Canvas ─────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[1400px] mx-auto px-5 py-6">

          {/* Generating overlay */}
          {generating && (
            <div className="mb-5 flex items-center gap-3 rounded-xl border border-primary/20 bg-primary/[0.04] px-4 py-3 text-sm">
              <div className="relative shrink-0">
                <Loader2 className="w-4 h-4 animate-spin text-primary" />
              </div>
              <div>
                <p className="font-medium text-foreground">Generating dashboard…</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Retrieving data, planning widgets, and running analytics. This may take 30–90 seconds.
                </p>
              </div>
            </div>
          )}

          <div className="space-y-4">
            <SlicerBar config={config} onApply={applyFilters} busy={generating} />
            <DashboardRenderer config={config} />
          </div>
        </div>
      </main>

      {/* ── Composer ───────────────────────────────────────────────────── */}
      <footer className="shrink-0 border-t border-border/60 bg-card/40 backdrop-blur-sm px-5 py-4">
        <div className="max-w-[1400px] mx-auto">
          {error && (
            <div className="mb-3 flex items-start gap-2 rounded-lg border border-danger/20 bg-danger-bg px-3 py-2 text-xs text-danger">
              <span className="mt-0.5 shrink-0">✕</span>
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-end gap-3">
            {/* Input area */}
            <div className="flex-1 relative">
              <div className="absolute left-3 top-2.5 pointer-events-none">
                <Sparkles className="w-4 h-4 text-muted-foreground/60" />
              </div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate();
                }}
                rows={2}
                disabled={generating}
                placeholder='e.g. "Total revenue KPI, monthly trend as area chart, top 10 customers by revenue as bar chart"'
                className="w-full resize-none rounded-xl border border-border bg-muted/20 pl-9 pr-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60 focus:bg-card transition-colors disabled:opacity-60"
              />
            </div>

            {/* Generate button */}
            <Button
              onClick={() => generate()}
              disabled={generating || !prompt.trim()}
              size="lg"
              className="shrink-0 self-stretch"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              {widgetCount > 0 ? "Add widgets" : "Generate"}
            </Button>
          </div>

          <p className="mt-2 text-[11px] text-muted-foreground/50">
            ⌘+Enter to generate · Analytics are persisted — return anytime without regenerating
          </p>
        </div>
      </footer>
    </div>
  );
}

function StatusBadge({ status, generating }: { status: string; generating: boolean }) {
  const s = generating ? "generating" : status;
  const map: Record<string, { label: string; variant: BadgeVariant; dot?: "static" | "pulse" }> = {
    ready:      { label: "Ready",      variant: "success", dot: "static" },
    generating: { label: "Generating", variant: "default", dot: "pulse" },
    draft:      { label: "Draft",      variant: "muted" },
    error:      { label: "Error",      variant: "danger" },
  };
  const { label, variant, dot } = map[s] ?? map.draft;
  return (
    <Badge variant={variant}>
      {dot && (
        <span className={`h-1.5 w-1.5 rounded-full bg-current ${dot === "pulse" ? "animate-pulse" : ""}`} />
      )}
      {label}
    </Badge>
  );
}

"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback, use } from "react";
import {
  ArrowLeft, Sparkles, Loader2, RefreshCw, LayoutDashboard, Copy, Trash2,
  LayoutGrid, ListChecks, MessageSquareText, ChevronDown,
} from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { DashboardFull, DashboardConfig } from "@/components/analytics-catalog/types";
import { DashboardRenderer } from "@/components/analytics-catalog/DashboardRenderer";
import { SummaryView } from "@/components/analytics-catalog/SummaryView";
import { SlicerBar } from "@/components/analytics-catalog/SlicerBar";
import { ActiveFilter } from "@/components/analytics-catalog/types";
import { DomainPicker } from "@/app/(app)/chat/_components/DomainPicker";
import { ContainerPicker } from "@/app/(app)/chat/_components/ContainerPicker";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Badge, BadgeVariant } from "@/components/ui/badge";

type BoardTab = "board" | "summary";

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
  const [tab, setTab] = useState<BoardTab>("board");
  // Container + domain pickers (mirror chat): the domain picker needs a container
  // in scope to list its domain folders. Transient — re-sent on every generate.
  const [selectedContainerId, setSelectedContainerId] = useState<string | null>(null);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);

  // Default the container scope so the domain picker can populate without a click:
  // prefer the dashboard's own container, else the user's sole container.
  const { data: myContainers } = useSWR<{ id: string; name: string }[]>(
    "containers-list",
    async () => {
      const res = await apiFetch("/api/containers");
      return res.ok ? res.json() : [];
    },
    { revalidateOnFocus: false },
  );
  useEffect(() => {
    if (selectedContainerId) return;
    const def =
      dashboard?.container_id ??
      (myContainers && myContainers.length === 1 ? myContainers[0].id : null);
    if (def) setSelectedContainerId(def);
  }, [myContainers, dashboard?.container_id, selectedContainerId]);

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
          container_id: selectedContainerId,
          folder_id: selectedFolderId,
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

  const handleDuplicate = async () => {
    const res = await apiFetch(`/api/dashboards/${dashboardId}/duplicate`, { method: "POST" });
    if (res.ok) {
      const dup = await res.json().catch(() => null);
      if (dup?.id) router.push(`/dashboards/${dup.id}`);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm(`Delete "${dashboard?.title || "this dashboard"}"? This can't be undone.`)) return;
    const res = await apiFetch(`/api/dashboards/${dashboardId}`, { method: "DELETE" });
    if (res.ok || res.status === 404) router.push("/dashboards");
    else setError("Failed to delete dashboard.");
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
                  // Update local state so the live title (header + Summary cover)
                  // reflects the rename immediately, without a reload.
                  setDashboard((d) => (d ? { ...d, title: t } : d));
                }
              }}
              className="bg-transparent text-base font-semibold text-foreground focus:outline-none min-w-0 flex-1"
              placeholder="Untitled dashboard"
            />
          </div>

          {/* Board / Summary segmented control */}
          {widgetCount > 0 && (
            <div className="seg-track shrink-0">
              <button
                type="button"
                data-active={tab === "board"}
                onClick={() => setTab("board")}
                className="seg-tab"
              >
                <LayoutGrid className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Board</span>
              </button>
              <button
                type="button"
                data-active={tab === "summary"}
                onClick={() => setTab("summary")}
                className="seg-tab"
              >
                <ListChecks className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Summary</span>
              </button>
            </div>
          )}

          {/* Meta chips */}
          <div className="flex items-center gap-2 shrink-0">
            {widgetCount > 0 && (
              <Badge variant="muted" className="hidden lg:inline-flex">
                {widgetCount} widget{widgetCount !== 1 ? "s" : ""}
              </Badge>
            )}
            <QuestionsMenu dashboard={dashboard} config={config} />
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
            <Button variant="ghost" size="icon" onClick={handleDuplicate} title="Duplicate dashboard">
              <Copy className="w-3.5 h-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDelete}
              title="Delete dashboard"
              className="text-muted-foreground hover:text-foreground"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
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
            {tab === "board" ? (
              <>
                <SlicerBar config={config} onApply={applyFilters} busy={generating} />
                <DashboardRenderer
                  config={config}
                  onAskQuestion={(q) => generate({ promptOverride: q })}
                />
              </>
            ) : (
              <SummaryView config={config} title={dashboard.title} />
            )}
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

          {/* Scope — pick a container, then a domain within it (mirrors chat) */}
          <div className="mb-2 flex items-center gap-2">
            <ContainerPicker
              value={selectedContainerId}
              onChange={setSelectedContainerId}
            />
            <DomainPicker
              containerId={selectedContainerId}
              value={selectedFolderId}
              onChange={setSelectedFolderId}
            />
          </div>

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

/* ── Questions dropdown ───────────────────────────────────────────────────────
   The board's prompt history — every natural-language question that shaped it,
   newest first. Reuses the kebab-menu language from the dashboards list page
   (click-outside overlay + rounded popover on tokens). Falls back to the single
   config.prompt when no history was persisted. */
function QuestionsMenu({
  dashboard,
  config,
}: {
  dashboard: DashboardFull;
  config: DashboardConfig | null;
}) {
  const [open, setOpen] = useState(false);

  // Newest first. Fall back to the seed prompt when history is empty.
  const history = [...(dashboard.prompt_history ?? [])].reverse();
  const items: { prompt: string; created_at?: string }[] =
    history.length > 0
      ? history.map((h) => ({ prompt: h.prompt, created_at: h.created_at }))
      : config?.prompt
        ? [{ prompt: config.prompt, created_at: config.generated_at }]
        : [];

  if (!items.length) return null;

  const fmt = (iso?: string) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  };

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 text-[12px] font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <MessageSquareText className="w-3.5 h-3.5" />
        <span className="hidden md:inline">Questions</span>
        <span className="tabular-nums text-muted-foreground/60">{items.length}</span>
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div
            className="dash-pop absolute right-0 top-10 z-40 w-[340px] max-w-[calc(100vw-32px)] p-1.5"
            role="menu"
          >
            <p className="px-2.5 pb-1.5 pt-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70">
              Prompt history
            </p>
            <div className="max-h-[320px] overflow-y-auto">
              {items.map((it, i) => (
                <div key={i} className="dash-pop-item">
                  <p className="text-[12.5px] leading-snug text-foreground">{it.prompt}</p>
                  {it.created_at && (
                    <p className="mt-0.5 text-[10.5px] tabular-nums text-muted-foreground/60">
                      {fmt(it.created_at)}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
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

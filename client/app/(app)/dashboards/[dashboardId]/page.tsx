"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback, use } from "react";
import { ArrowLeft, Sparkles, Loader2, Clock } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { DashboardFull, DashboardConfig } from "@/components/analytics-catalog/types";
import { DashboardRenderer } from "@/components/analytics-catalog/DashboardRenderer";

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

  useEffect(() => {
    load();
  }, [load]);

  const generate = async () => {
    const p = prompt.trim();
    if (!p || generating) return;
    setGenerating(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/dashboards/${dashboardId}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: p,
          append: !!(dashboard?.config && (dashboard.config as DashboardConfig).widgets?.length),
        }),
      });
      if (res.ok) {
        setDashboard(await res.json());
        setPrompt("");
      } else {
        const body = await res.json().catch(() => ({}));
        setError(body.detail || "Generation failed.");
      }
    } catch {
      setError("Network error during generation.");
    } finally {
      setGenerating(false);
    }
  };

  if (loading) {
    return <div className="flex h-full items-center justify-center text-sm text-muted-foreground">Loading…</div>;
  }
  if (!dashboard) return null;

  const config = (dashboard.config && "widgets" in dashboard.config
    ? (dashboard.config as DashboardConfig)
    : null);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-6 py-4 flex items-center gap-3">
        <button onClick={() => router.push("/dashboards")} className="text-muted-foreground hover:text-foreground">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="min-w-0">
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
            className="bg-transparent text-lg font-semibold text-foreground focus:outline-none w-full"
          />
          {config?.generated_at && (
            <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
              <Clock className="w-3 h-3" /> Generated {new Date(config.generated_at).toLocaleString()}
            </p>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="max-w-7xl mx-auto">
          {generating && (
            <div className="mb-4 flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-sm text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing data, planning widgets, and generating analytics…
            </div>
          )}
          <DashboardRenderer config={config} />
        </div>
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t border-border bg-background px-6 py-4">
        <div className="max-w-7xl mx-auto">
          {error && <p className="mb-2 text-xs text-red-500">{error}</p>}
          <div className="flex items-end gap-2">
            <div className="flex-1 relative">
              <Sparkles className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate();
                }}
                rows={2}
                placeholder='Describe the dashboard, e.g. "Sales overview: total revenue KPI, monthly revenue trend, revenue by region as a bar chart, and top 10 products"'
                className="w-full resize-none rounded-md border border-border bg-surface pl-9 pr-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <button
              onClick={generate}
              disabled={generating || !prompt.trim()}
              className="flex items-center gap-2 px-4 py-2.5 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
              {config?.widgets?.length ? "Add widgets" : "Generate"}
            </button>
          </div>
          <p className="mt-1.5 text-[11px] text-muted-foreground">
            Cmd/Ctrl + Enter to generate. Analytics are persisted — you can return anytime without regenerating.
          </p>
        </div>
      </div>
    </div>
  );
}

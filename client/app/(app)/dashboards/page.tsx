"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  Plus, Search, Pin, PinOff, Copy, Trash2, FolderPlus, LayoutDashboard,
  Folder as FolderIcon, MoreVertical, Pencil,
} from "lucide-react";
import { useDashboards } from "./_hooks/useDashboards";
import { DashboardSummary } from "@/components/analytics-catalog/types";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, BadgeVariant } from "@/components/ui/badge";

export default function DashboardsWorkspace() {
  const router = useRouter();
  const {
    dashboards, folders, loading, search, setSearch, activeFolder, setActiveFolder,
    createDashboard, createFolder, updateDashboard, deleteDashboard, duplicateDashboard,
  } = useDashboards();
  const [creating, setCreating] = useState(false);

  const handleCreate = async () => {
    setCreating(true);
    const d = await createDashboard("Untitled dashboard");
    setCreating(false);
    if (d?.id) router.push(`/dashboards/${d.id}`);
  };

  const handleNewFolder = async () => {
    const name = window.prompt("Folder name");
    if (name && name.trim()) await createFolder(name.trim());
  };

  return (
    <div className="flex h-full">
      {/* Folder rail */}
      <aside className="hidden lg:flex w-56 shrink-0 flex-col border-r border-border bg-sidebar/40 p-3">
        <div className="flex items-center justify-between mb-2 px-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Workspaces</span>
          <button onClick={handleNewFolder} className="text-muted-foreground hover:text-foreground" aria-label="New folder">
            <FolderPlus className="w-4 h-4" />
          </button>
        </div>
        <button
          onClick={() => setActiveFolder(null)}
          className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors ${
            activeFolder === null ? "bg-primary/[0.09] text-foreground font-medium" : "text-muted-foreground hover:bg-surface-raised"
          }`}
        >
          <LayoutDashboard className="w-4 h-4" /> All dashboards
        </button>
        {folders.map((f) => (
          <button
            key={f.id}
            onClick={() => setActiveFolder(f.id)}
            className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors ${
              activeFolder === f.id ? "bg-primary/[0.09] text-foreground font-medium" : "text-muted-foreground hover:bg-surface-raised"
            }`}
          >
            <FolderIcon className="w-4 h-4" /> <span className="truncate">{f.name}</span>
          </button>
        ))}
      </aside>

      {/* Main */}
      <div className="flex-1 min-w-0 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between gap-4 mb-6">
            <div>
              <h1 className="text-xl font-semibold text-foreground">Dashboards</h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                Create dashboards from natural-language prompts.
              </p>
            </div>
            <Button onClick={handleCreate} disabled={creating}>
              <Plus className="w-4 h-4" /> New dashboard
            </Button>
          </div>

          <div className="relative mb-5">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search dashboards…"
              className="w-full pl-9 pr-3 py-2 rounded-md bg-surface border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>

          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : dashboards.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <LayoutDashboard className="w-10 h-10 text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">No dashboards yet.</p>
              <button onClick={handleCreate} className="mt-3 text-sm text-primary hover:underline">
                Create your first dashboard
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {dashboards.map((d) => (
                <DashboardCard
                  key={d.id}
                  d={d}
                  onOpen={() => router.push(`/dashboards/${d.id}`)}
                  onPin={() => updateDashboard(d.id, { is_pinned: !d.is_pinned })}
                  onRename={async () => {
                    const title = window.prompt("Rename dashboard", d.title);
                    if (title && title.trim()) await updateDashboard(d.id, { title: title.trim() });
                  }}
                  onDuplicate={() => duplicateDashboard(d.id)}
                  onDelete={() => {
                    if (window.confirm(`Delete "${d.title}"?`)) deleteDashboard(d.id);
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DashboardCard({
  d, onOpen, onPin, onRename, onDuplicate, onDelete,
}: {
  d: DashboardSummary;
  onOpen: () => void;
  onPin: () => void;
  onRename: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const statusVariant: BadgeVariant =
    d.status === "ready" ? "success" : d.status === "error" ? "danger" : "muted";
  return (
    <Card className="group relative p-4 transition-colors hover:border-primary/40">
      <div className="flex items-start justify-between gap-2">
        <button onClick={onOpen} className="text-left min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            {d.is_pinned && <Pin className="w-3.5 h-3.5 text-primary fill-primary" />}
            <h3 className="text-sm font-medium text-foreground truncate">{d.title}</h3>
          </div>
          {d.description && <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{d.description}</p>}
        </button>
        <div className="relative">
          <button
            onClick={() => setMenu((m) => !m)}
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-surface-raised"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
          {menu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setMenu(false)} />
              <div className="absolute right-0 top-7 z-20 w-40 rounded-md border border-border bg-card py-1 shadow-lg text-sm">
                <MenuItem icon={d.is_pinned ? PinOff : Pin} label={d.is_pinned ? "Unpin" : "Pin"} onClick={() => { setMenu(false); onPin(); }} />
                <MenuItem icon={Pencil} label="Rename" onClick={() => { setMenu(false); onRename(); }} />
                <MenuItem icon={Copy} label="Duplicate" onClick={() => { setMenu(false); onDuplicate(); }} />
                <MenuItem icon={Trash2} label="Delete" danger onClick={() => { setMenu(false); onDelete(); }} />
              </div>
            </>
          )}
        </div>
      </div>
      <button onClick={onOpen} className="mt-3 flex w-full items-center justify-between text-[11px] text-muted-foreground">
        <span>{d.widget_count} widget{d.widget_count === 1 ? "" : "s"}</span>
        <Badge variant={statusVariant} className="capitalize">{d.status}</Badge>
      </button>
    </Card>
  );
}

function MenuItem({
  icon: Icon, label, onClick, danger,
}: {
  icon: typeof Pin; label: string; onClick: () => void; danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-2 px-3 py-1.5 hover:bg-surface-raised ${danger ? "text-danger" : "text-foreground"}`}
    >
      <Icon className="w-3.5 h-3.5" /> {label}
    </button>
  );
}

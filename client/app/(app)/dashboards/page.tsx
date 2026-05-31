"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { motion } from "framer-motion";
import {
  Plus, Search, Pin, PinOff, Copy, Trash2, FolderPlus, LayoutDashboard,
  Folder as FolderIcon, MoreVertical, Pencil, ArrowRight,
} from "lucide-react";
import { useDashboards } from "./_hooks/useDashboards";
import { DashboardSummary } from "@/components/analytics-catalog/types";
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
    if (name?.trim()) await createFolder(name.trim());
  };

  return (
    <div className="flex h-full">
      {/* Folder rail */}
      <aside className="hidden lg:flex w-52 shrink-0 flex-col border-r border-[#e5e5e5] bg-[#f9f9f9] p-3">
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="section-label">Workspaces</span>
          <button onClick={handleNewFolder} className="btn-ghost p-1 rounded-md" aria-label="New folder">
            <FolderPlus className="w-3.5 h-3.5" />
          </button>
        </div>

        <button
          onClick={() => setActiveFolder(null)}
          className={`sidebar-item w-full ${activeFolder === null ? "active" : ""}`}
        >
          {activeFolder === null && (
            <motion.span layoutId="folder-pill" className="absolute inset-0 bg-[#f0f0f0] rounded-[6px]" transition={{ type: "spring", stiffness: 420, damping: 36 }} />
          )}
          <LayoutDashboard className="w-3.5 h-3.5 relative z-10 text-[#a3a3a3]" />
          <span className="relative z-10">All dashboards</span>
        </button>

        {folders.map((f) => (
          <button
            key={f.id}
            onClick={() => setActiveFolder(f.id)}
            className={`sidebar-item w-full ${activeFolder === f.id ? "active" : ""}`}
          >
            {activeFolder === f.id && (
              <motion.span layoutId="folder-pill" className="absolute inset-0 bg-[#f0f0f0] rounded-[6px]" transition={{ type: "spring", stiffness: 420, damping: 36 }} />
            )}
            <FolderIcon className="w-3.5 h-3.5 relative z-10 text-[#a3a3a3]" />
            <span className="relative z-10 truncate">{f.name}</span>
          </button>
        ))}
      </aside>

      {/* Main */}
      <div className="flex-1 min-w-0 overflow-y-auto bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-5 sm:py-7">

          {/* Header */}
          <div className="flex items-start sm:items-center justify-between gap-3 mb-5 sm:mb-7">
            <div>
              <h1 className="text-[20px] sm:text-[22px] font-bold text-[#0a0a0a]" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.028em" }}>
                Dashboards
              </h1>
              <p className="text-[12px] sm:text-[13px] text-[#737373] mt-0.5 hidden sm:block">
                Create dashboards from natural-language prompts.
              </p>
            </div>
            <button
              onClick={handleCreate}
              disabled={creating}
              className="btn-black px-3 sm:px-4 h-9 rounded-lg text-[12.5px] sm:text-[13px] gap-1.5 shrink-0"
            >
              <Plus className="w-3.5 h-3.5" /> New dashboard
            </button>
          </div>

          {/* Search */}
          <div className="relative mb-4 sm:mb-6 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#a3a3a3]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search dashboards…"
              className="search-input"
            />
          </div>

          {loading ? (
            <p className="text-[13px] text-[#a3a3a3]">Loading…</p>
          ) : dashboards.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="flex flex-col items-center justify-center py-24 text-center"
            >
              <div className="w-12 h-12 rounded-2xl bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center mb-4">
                <LayoutDashboard className="w-5 h-5 text-[#0a0a0a]" />
              </div>
              <p className="text-[15px] font-semibold text-[#0a0a0a] mb-1" style={{ fontFamily: "var(--font-display)" }}>
                No dashboards yet
              </p>
              <p className="text-[13px] text-[#737373] mb-5">Create your first dashboard from a natural-language prompt.</p>
              <button onClick={handleCreate} className="btn-black px-5 h-9 rounded-lg text-[13px] gap-1.5">
                Create dashboard <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          ) : (
            <motion.div
              initial="hidden"
              animate="show"
              variants={{ hidden: {}, show: { transition: { staggerChildren: 0.07 } } }}
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
            >
              {dashboards.map((d) => (
                <DashboardCard
                  key={d.id}
                  d={d}
                  onOpen={() => router.push(`/dashboards/${d.id}`)}
                  onPin={() => updateDashboard(d.id, { is_pinned: !d.is_pinned })}
                  onRename={async () => {
                    const title = window.prompt("Rename dashboard", d.title);
                    if (title?.trim()) await updateDashboard(d.id, { title: title.trim() });
                  }}
                  onDuplicate={() => duplicateDashboard(d.id)}
                  onDelete={() => {
                    if (window.confirm(`Delete "${d.title}"?`)) deleteDashboard(d.id);
                  }}
                />
              ))}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Dashboard card with animated bar preview ── */
const BAR_HEIGHTS = [35, 55, 42, 68, 52, 78, 60, 85, 65, 90, 72, 88];

function DashboardCard({ d, onOpen, onPin, onRename, onDuplicate, onDelete }: {
  d: DashboardSummary;
  onOpen: () => void; onPin: () => void; onRename: () => void;
  onDuplicate: () => void; onDelete: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const [hovered, setHovered] = useState(false);
  const statusVariant: BadgeVariant =
    d.status === "ready" ? "success" : d.status === "error" ? "danger" : "muted";

  return (
    <motion.div
      variants={{ hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.42, ease: "easeOut" } } }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      className="dash-card"
    >
      {/* Animated bar chart preview */}
      <button onClick={onOpen} className="block w-full h-28 bg-[#f9f9f9] border-b border-[#e5e5e5] px-5 py-4">
        <div className="flex items-end gap-1 h-full">
          {BAR_HEIGHTS.map((h, i) => (
            <motion.div
              key={i}
              animate={{ height: hovered ? `${Math.min(h + 8, 100)}%` : `${h}%`, opacity: hovered ? (i >= 10 ? 1 : 0.18 + i * 0.06) : (i >= 10 ? 0.85 : 0.12 + i * 0.05) }}
              transition={{ duration: 0.35, delay: i * 0.02, ease: "easeOut" }}
              className="flex-1 rounded-sm bg-[#0a0a0a]"
            />
          ))}
        </div>
      </button>

      {/* Card footer */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <button onClick={onOpen} className="text-left min-w-0 flex-1">
            <div className="flex items-center gap-1.5 mb-0.5">
              {d.is_pinned && <Pin className="w-3 h-3 text-[#0a0a0a] fill-[#0a0a0a] shrink-0" />}
              <h3 className="text-[13.5px] font-semibold text-[#0a0a0a] truncate" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.01em" }}>
                {d.title}
              </h3>
            </div>
            {d.description && <p className="text-[12px] text-[#737373] line-clamp-1">{d.description}</p>}
          </button>

          <div className="relative shrink-0">
            <button
              onClick={() => setMenu((m) => !m)}
              className="btn-ghost p-1.5 rounded-lg"
            >
              <MoreVertical className="w-3.5 h-3.5" />
            </button>
            {menu && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setMenu(false)} />
                <div className="absolute right-0 top-8 z-20 w-40 rounded-xl border border-[#e5e5e5] bg-white py-1 text-[13px]" style={{ boxShadow: "0 8px 24px rgba(0,0,0,0.10)", maxWidth: "calc(100vw - 32px)" }}>
                  <MenuItem icon={d.is_pinned ? PinOff : Pin} label={d.is_pinned ? "Unpin" : "Pin"} onClick={() => { setMenu(false); onPin(); }} />
                  <MenuItem icon={Pencil} label="Rename" onClick={() => { setMenu(false); onRename(); }} />
                  <MenuItem icon={Copy} label="Duplicate" onClick={() => { setMenu(false); onDuplicate(); }} />
                  <MenuItem icon={Trash2} label="Delete" danger onClick={() => { setMenu(false); onDelete(); }} />
                </div>
              </>
            )}
          </div>
        </div>

        <button onClick={onOpen} className="mt-3 flex w-full items-center justify-between">
          <span className="text-[11.5px] text-[#a3a3a3]">{d.widget_count} widget{d.widget_count === 1 ? "" : "s"}</span>
          <Badge variant={statusVariant} className="capitalize text-[11px]">{d.status}</Badge>
        </button>
      </div>
    </motion.div>
  );
}

function MenuItem({ icon: Icon, label, onClick, danger }: {
  icon: typeof Pin; label: string; onClick: () => void; danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-2.5 px-3 py-1.5 hover:bg-[#f4f4f4] transition-colors ${danger ? "text-[#dc2626]" : "text-[#0a0a0a]"}`}
    >
      <Icon className="w-3.5 h-3.5" /> {label}
    </button>
  );
}

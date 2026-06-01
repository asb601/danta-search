"use client";

import { useState } from "react";
import {
  Plus, MessageSquare, Trash2, Pencil, Check, X, Clock,
  PanelLeftClose, Search,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "./types";

export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

const GROUP_ORDER = ["Today", "Yesterday", "Previous 7 days", "This month", "Older"] as const;
type GroupLabel = (typeof GROUP_ORDER)[number];

function getGroupLabel(isoDate: string): GroupLabel {
  const days = Math.floor((Date.now() - new Date(isoDate).getTime()) / 86400000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days <= 7) return "Previous 7 days";
  if (days <= 30) return "This month";
  return "Older";
}

function groupByTime(
  convs: ConversationSummary[]
): Array<{ label: GroupLabel; items: ConversationSummary[] }> {
  const map = new Map<GroupLabel, ConversationSummary[]>();
  for (const conv of convs) {
    const label = getGroupLabel(conv.updated_at || conv.created_at);
    if (!map.has(label)) map.set(label, []);
    map.get(label)!.push(conv);
  }
  return GROUP_ORDER.filter((l) => map.has(l)).map((l) => ({ label: l, items: map.get(l)! }));
}

const sidebarVariants = {
  hidden: { x: -16, opacity: 0 },
  show: {
    x: 0, opacity: 1,
    transition: { duration: 0.22, ease: "easeOut" as const },
  },
  exit: {
    x: -16, opacity: 0,
    transition: { duration: 0.16, ease: "easeIn" as const },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -8 },
  show: (i: number) => ({
    opacity: 1, x: 0,
    transition: { delay: i * 0.03, duration: 0.22, ease: "easeOut" as const },
  }),
};

export function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
  isOpen,
  onToggle,
  searchQuery,
  onSearchChange,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [searchFocused, setSearchFocused] = useState(false);

  const startRename = (conv: ConversationSummary) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const commitRename = () => {
    if (editingId && editTitle.trim()) onRename(editingId, editTitle.trim());
    setEditingId(null);
  };

  const grouped = groupByTime(conversations);
  let globalIdx = 0;

  return (
    <AnimatePresence>
      {isOpen && (
      <>
        {/* Mobile backdrop */}
        <motion.div
          key="sidebar-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-30 bg-black/30 backdrop-blur-[2px] sm:hidden"
          onClick={onToggle}
        />
      <motion.div
        key="conv-sidebar"
        variants={sidebarVariants}
        initial="hidden"
        animate="show"
        exit="exit"
        className="fixed sm:relative inset-y-0 left-0 z-40 sm:z-auto w-[260px] sm:w-[240px] shrink-0 border-r border-[#e5e5e5] bg-[#f9f9f9] flex flex-col h-full overflow-hidden shadow-[4px_0_24px_rgba(0,0,0,0.08)] sm:shadow-none"
      >
        {/* Header */}
        <div className="px-3 pt-3.5 pb-2.5 border-b border-[#e5e5e5] flex items-center justify-between gap-2">
          <span
            className="text-[13px] font-semibold text-[#0a0a0a] truncate"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.01em" }}
          >
            Conversations
          </span>
          <div className="flex items-center gap-0.5 shrink-0">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.92 }}
              onClick={onNew}
              className="p-1.5 rounded-md text-[#a3a3a3] hover:text-[#0a0a0a] hover:bg-[#ebebeb] transition-colors"
              title="New chat"
            >
              <Plus className="w-3.5 h-3.5" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.92 }}
              onClick={onToggle}
              className="p-1.5 rounded-md text-[#a3a3a3] hover:text-[#0a0a0a] hover:bg-[#ebebeb] transition-colors"
              title="Close sidebar"
            >
              <PanelLeftClose className="w-3.5 h-3.5" />
            </motion.button>
          </div>
        </div>

        {/* Search */}
        <div className="px-3 py-2 border-b border-[#e5e5e5]">
          <motion.div
            animate={{ boxShadow: searchFocused ? "0 0 0 2px rgba(10,10,10,0.10)" : "none" }}
            transition={{ duration: 0.15 }}
            className="relative rounded-lg overflow-hidden"
          >
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-[#a3a3a3] pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setSearchFocused(false)}
              placeholder="Search conversations…"
              className="w-full pl-7 pr-3 py-1.5 text-[12px] bg-white border border-[#e5e5e5] rounded-lg text-[#0a0a0a] placeholder:text-[#a3a3a3] focus:outline-none transition-colors"
            />
          </motion.div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto py-1.5 scrollbar-thin">
          <AnimatePresence mode="wait">
            {conversations.length === 0 ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="px-4 py-10 text-center"
              >
                <div className="w-9 h-9 rounded-xl bg-[#f0f0f0] border border-[#e5e5e5] flex items-center justify-center mx-auto mb-3">
                  <MessageSquare className="w-4 h-4 text-[#a3a3a3]" />
                </div>
                <p className="text-[12px] text-[#a3a3a3]">
                  {searchQuery ? "No matching conversations" : "No conversations yet"}
                </p>
              </motion.div>
            ) : (
              <motion.div key="list" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {grouped.map(({ label, items }) => (
                  <div key={label}>
                    <div className="px-3 pt-3 pb-1">
                      <span className="text-[10px] font-semibold text-[#c4c4c4] uppercase tracking-[0.08em]">
                        {label}
                      </span>
                    </div>
                    {items.map((conv) => {
                      const idx = globalIdx++;
                      const isActive = activeId === conv.id;
                      return (
                        <motion.div
                          key={conv.id}
                          custom={idx}
                          variants={itemVariants}
                          initial="hidden"
                          animate="show"
                          layout
                          className={cn(
                            "group relative mx-1.5 mb-0.5 px-2.5 py-2 rounded-lg cursor-pointer transition-colors duration-150",
                            isActive
                              ? "bg-white border border-[#e5e5e5] shadow-[0_1px_4px_rgba(0,0,0,0.06)]"
                              : "hover:bg-white/70 border border-transparent"
                          )}
                          onClick={() => {
                            if (editingId !== conv.id) onSelect(conv.id);
                          }}
                        >
                          {/* Active indicator */}
                          {isActive && (
                            <motion.div
                              layoutId="conv-active-bar"
                              className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r bg-[#0a0a0a]"
                              transition={{ type: "spring", stiffness: 400, damping: 36 }}
                            />
                          )}

                          {editingId === conv.id ? (
                            <div className="flex items-center gap-1">
                              <input
                                autoFocus
                                value={editTitle}
                                onChange={(e) => setEditTitle(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") commitRename();
                                  if (e.key === "Escape") setEditingId(null);
                                }}
                                className="flex-1 text-[12px] bg-[#f4f4f4] border border-[#e5e5e5] rounded px-2 py-1 text-[#0a0a0a] focus:outline-none focus:border-[#a3a3a3]"
                                onClick={(e) => e.stopPropagation()}
                              />
                              <button onClick={(e) => { e.stopPropagation(); commitRename(); }} className="p-0.5 rounded text-green-500 hover:bg-green-50">
                                <Check className="w-3.5 h-3.5" />
                              </button>
                              <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }} className="p-0.5 rounded text-[#a3a3a3] hover:bg-[#f4f4f4]">
                                <X className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          ) : confirmDeleteId === conv.id ? (
                            <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                              <span className="text-[11px] text-[#dc2626] truncate flex-1">Delete?</span>
                              <button
                                onClick={() => { onDelete(conv.id); setConfirmDeleteId(null); }}
                                className="px-2 py-0.5 text-[11px] font-semibold rounded-md bg-[#dc2626] text-white hover:opacity-90"
                              >Yes</button>
                              <button
                                onClick={() => setConfirmDeleteId(null)}
                                className="px-2 py-0.5 text-[11px] font-medium rounded-md border border-[#e5e5e5] text-[#737373] hover:text-[#0a0a0a]"
                              >No</button>
                            </div>
                          ) : (
                            <>
                              <p className="text-[12px] font-medium text-[#0a0a0a] truncate pr-10 leading-snug">
                                {conv.title}
                              </p>
                              <div className="flex items-center gap-1.5 mt-0.5">
                                <Clock className="w-2.5 h-2.5 text-[#c4c4c4] shrink-0" />
                                <p className="text-[10.5px] text-[#a3a3a3]">
                                  {relativeTime(conv.updated_at)}
                                </p>
                                {conv.message_count > 0 && (
                                  <>
                                    <span className="text-[#d4d4d4]">·</span>
                                    <p className="text-[10.5px] text-[#a3a3a3]">
                                      {conv.message_count} msg{conv.message_count !== 1 ? "s" : ""}
                                    </p>
                                  </>
                                )}
                              </div>

                              {/* Hover actions */}
                              <div className="absolute right-1.5 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center gap-0.5 bg-white border border-[#e5e5e5] rounded-md px-0.5 shadow-sm">
                                <button
                                  onClick={(e) => { e.stopPropagation(); startRename(conv); }}
                                  className="p-1 rounded text-[#a3a3a3] hover:text-[#0a0a0a] transition-colors"
                                  title="Rename"
                                >
                                  <Pencil className="w-3 h-3" />
                                </button>
                                <button
                                  onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(conv.id); }}
                                  className="p-1 rounded text-[#a3a3a3] hover:text-[#dc2626] transition-colors"
                                  title="Delete"
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            </>
                          )}
                        </motion.div>
                      );
                    })}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
      </>
      )}
    </AnimatePresence>
  );
}

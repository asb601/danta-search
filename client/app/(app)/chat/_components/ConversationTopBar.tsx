"use client";

import { useRef, useEffect } from "react";
import { Plus, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "./types";

export function ConversationTopBar({
  conversations,
  activeId,
  onSelect,
  onNew,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll active tab into view
  useEffect(() => {
    if (!activeId || !scrollRef.current) return;
    const activeEl = scrollRef.current.querySelector(`[data-conv-id="${activeId}"]`);
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    }
  }, [activeId]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="hidden sm:flex items-center border-b border-[#e5e5e5] bg-white px-3 gap-2.5 h-[46px] overflow-hidden relative"
    >
      {/* New chat button */}
      <motion.button
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.94 }}
        onClick={onNew}
        className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[12px] font-semibold border border-[#e5e5e5] text-[#0a0a0a] hover:bg-[#0a0a0a] hover:text-white hover:border-[#0a0a0a] transition-all duration-150"
        style={{ boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}
      >
        <Plus className="w-3 h-3" />
        <span>New</span>
      </motion.button>

      {/* Divider */}
      <div className="w-px h-4 bg-[#e5e5e5] shrink-0" />

      {/* Scrollable conversation tabs */}
      <div
        ref={scrollRef}
        className="flex-1 flex items-center gap-0.5 overflow-x-auto min-w-0 py-1"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        <AnimatePresence initial={false}>
          {conversations.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-1.5 px-2"
            >
              <MessageSquare className="w-3.5 h-3.5 text-[#d4d4d4]" />
              <p className="text-[12px] text-[#c4c4c4] whitespace-nowrap">
                No conversations yet — start one below
              </p>
            </motion.div>
          ) : (
            conversations.map((conv, i) => {
              const isActive = activeId === conv.id;
              return (
                <motion.button
                  key={conv.id}
                  data-conv-id={conv.id}
                  initial={{ opacity: 0, scale: 0.88, x: -8 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.88, x: 4 }}
                  transition={{ delay: Math.min(i * 0.018, 0.18), duration: 0.22, ease: "easeOut" }}
                  onClick={() => onSelect(conv.id)}
                  className={cn(
                    "relative shrink-0 flex items-center px-3 py-1.5 rounded-full text-[12.5px] font-medium cursor-pointer whitespace-nowrap transition-colors duration-150",
                    isActive
                      ? "text-white"
                      : "text-[#737373] hover:text-[#0a0a0a] hover:bg-[#f4f4f4]"
                  )}
                >
                  {/* Animated active pill background */}
                  {isActive && (
                    <motion.div
                      layoutId="conv-top-active"
                      className="absolute inset-0 rounded-full bg-[#0a0a0a]"
                      style={{
                        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12), 0 1px 4px rgba(0,0,0,0.15)",
                      }}
                      transition={{ type: "spring", stiffness: 420, damping: 34 }}
                    />
                  )}
                  <span className="relative z-10 max-w-[160px] truncate">{conv.title}</span>
                </motion.button>
              );
            })
          )}
        </AnimatePresence>
      </div>

      {/* Right fade — signals scrollable overflow */}
      <div
        className="absolute right-0 top-0 bottom-0 w-10 pointer-events-none"
        style={{ background: "linear-gradient(to right, transparent, white)" }}
      />
    </motion.div>
  );
}

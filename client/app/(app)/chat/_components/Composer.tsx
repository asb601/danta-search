"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Send, X } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The redesigned chat composer — a presentational shell. All state/handlers are
 * passed in so the page can drive it with either the Excel (`useChat`) or PDF
 * (`usePdfChat`) hook. Layout:
 *
 *   [ mode switcher (drop-up) ] [ scope control ]        ← controls row
 *   ┌──────────────────────────────────────────────┐
 *   │ auto-growing textarea               [ send ]  │    ← input shell
 *   └──────────────────────────────────────────────┘
 *   sub-hint (mode-aware)
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  onKeyDown,
  isLoading,
  placeholder,
  modeSwitcher,
  scopeControl,
  subHint,
  canSend = true,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  isLoading: boolean;
  placeholder: string;
  modeSwitcher: React.ReactNode;
  scopeControl: React.ReactNode;
  subHint: React.ReactNode;
  canSend?: boolean;
}) {
  const [focused, setFocused] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // ── Auto-grow: reset to auto then snap to scrollHeight (capped by max-h CSS) ──
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, [value]);

  const sendable = canSend && !!value.trim();

  return (
    <div className="border-t border-[var(--border)] bg-[var(--bg)] px-3 sm:px-4 pt-3 pb-3">
      <form onSubmit={onSubmit} className="max-w-3xl mx-auto">
        {/* ── controls row ── */}
        <div className="flex items-center gap-2 mb-2">
          {modeSwitcher}
          {scopeControl}
        </div>

        {/* ── input shell ── */}
        <div
          className={cn(
            "composer-shell",
            focused && "composer-shell-focused",
          )}
        >
          <textarea
            ref={taRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={placeholder}
            rows={1}
            aria-label="Message"
            className="composer-textarea"
          />
          <AnimatePresence mode="wait" initial={false}>
            {isLoading ? (
              <motion.button
                key="stop"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ duration: 0.15 }}
                type="button"
                onClick={onStop}
                aria-label="Stop generating"
                className="composer-stop-btn"
              >
                <X className="w-3.5 h-3.5" />
              </motion.button>
            ) : (
              <motion.button
                key="send"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ duration: 0.15 }}
                type="submit"
                disabled={!sendable}
                aria-label="Send message"
                whileHover={sendable ? { scale: 1.06 } : undefined}
                whileTap={sendable ? { scale: 0.94 } : undefined}
                className={cn(
                  "composer-send-btn",
                  sendable ? "composer-send-btn-on" : "composer-send-btn-off",
                )}
              >
                <Send className="w-3.5 h-3.5" />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </form>

      {subHint}
    </div>
  );
}

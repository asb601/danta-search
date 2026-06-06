"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  ChevronUp,
  FileSpreadsheet,
  FileText,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatMode } from "./types.pdf";

interface ModeOption {
  value: ChatMode;
  label: string;
  hint: string;
  icon: typeof FileText;
  disabled?: boolean;
}

const MODES: ModeOption[] = [
  {
    value: "excel",
    label: "Excel chat",
    hint: "Query your datasets",
    icon: FileSpreadsheet,
  },
  {
    value: "pdf",
    label: "PDF chat",
    hint: "Ask across your documents",
    icon: FileText,
  },
  {
    value: "combined",
    label: "Combined",
    hint: "Data + documents together",
    icon: Layers,
    disabled: true,
  },
];

/**
 * The mode selector. Opens UPWARD (drop-up) from a pill anchored at the left of
 * the composer input shell. "Combined" is shown with a "Soon" badge and is NOT
 * selectable — it never sends and never calls the backend.
 */
export function ModeSwitcher({
  mode,
  onChange,
}: {
  mode: ChatMode;
  onChange: (m: ChatMode) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const current = MODES.find((m) => m.value === mode) ?? MODES[0];
  const CurrentIcon = current.icon;

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Chat mode: ${current.label}. Change mode`}
        className="mode-trigger"
      >
        <CurrentIcon className="w-3.5 h-3.5 shrink-0 opacity-70" />
        <span className="truncate max-w-[110px]">{current.label}</span>
        <ChevronUp
          className={cn(
            "w-3 h-3 shrink-0 opacity-60 transition-transform duration-150",
            open ? "rotate-0" : "rotate-180",
          )}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            aria-label="Chat mode"
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="mode-dropup-panel"
          >
            {MODES.map((m) => {
              const Icon = m.icon;
              const active = m.value === mode;
              if (m.disabled) {
                return (
                  <div
                    key={m.value}
                    role="menuitem"
                    aria-disabled="true"
                    className="mode-option mode-option-disabled"
                  >
                    <span className="flex items-center gap-2.5 min-w-0">
                      <Icon className="w-4 h-4 shrink-0" />
                      <span className="flex flex-col min-w-0 text-left">
                        <span className="truncate">{m.label}</span>
                        <span className="mode-option-hint truncate">
                          {m.hint}
                        </span>
                      </span>
                    </span>
                    <span className="badge-soon shrink-0">Soon</span>
                  </div>
                );
              }
              return (
                <button
                  key={m.value}
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onChange(m.value);
                    setOpen(false);
                  }}
                  className={cn(
                    "mode-option",
                    active && "mode-option-active",
                  )}
                >
                  <span className="flex items-center gap-2.5 min-w-0">
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="flex flex-col min-w-0 text-left">
                      <span className="truncate">{m.label}</span>
                      <span className="mode-option-hint truncate">{m.hint}</span>
                    </span>
                  </span>
                  {active && <Check className="w-4 h-4 shrink-0" />}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

"use client";

import { useState } from "react";
import {
  ChevronDown,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  FileText,
  GitMerge,
  Gauge,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { blobToLabel } from "./DataTable";
import type { Governance } from "./types";

/* ── per-mode visual config (calm green / amber / red) ───────────────────── */
const MODE_META = {
  answer: {
    label: "Answered",
    Icon: ShieldCheck,
    badge:
      "bg-gov-answer-bg text-gov-answer-fg border border-gov-answer-border",
    accent: "text-gov-answer-fg",
  },
  caveat: {
    label: "Caveated",
    Icon: ShieldAlert,
    badge:
      "bg-gov-caveat-bg text-gov-caveat-fg border border-gov-caveat-border",
    accent: "text-gov-caveat-fg",
  },
  refusal: {
    label: "Refused",
    Icon: ShieldX,
    badge:
      "bg-gov-refusal-bg text-gov-refusal-fg border border-gov-refusal-border",
    accent: "text-gov-refusal-fg",
  },
} as const;

const CONFIDENCE_LABEL: Record<Governance["confidence"]["level"], string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
};

/* trust-state dot — green=trusted, red=quarantined, gray=unknown/absent */
function trustDotClass(trust?: string): string {
  const t = (trust ?? "").toLowerCase();
  if (t === "trusted" || t === "trust" || t === "verified")
    return "bg-gov-trust-ok";
  if (t === "quarantined" || t === "quarantine" || t === "untrusted")
    return "bg-gov-trust-bad";
  return "bg-gov-trust-unknown";
}

export function GovernancePanel({ governance }: { governance: Governance }) {
  const [isOpen, setIsOpen] = useState(false);

  const mode = MODE_META[governance.mode] ?? MODE_META.answer;
  const { Icon } = mode;
  const score = Math.round((governance.confidence.score ?? 0) * 100);
  const fileCount = governance.files?.length ?? 0;
  const joinCount = governance.approved_joins?.length ?? 0;

  return (
    <div className="mt-3 border border-gov-strip-border rounded-lg overflow-hidden">
      {/* ── strip header ───────────────────────────────────────────────── */}
      <button
        onClick={() => setIsOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2.5 bg-gov-strip-bg hover:bg-surface-raised/70 transition-colors text-left select-none"
      >
        <div className="flex items-center gap-2 min-w-0">
          <Icon className={cn("w-3.5 h-3.5 shrink-0", mode.accent)} />
          <span className="text-xs font-medium text-foreground">
            Governance
          </span>
          <span
            className={cn(
              "inline-flex items-center text-[11px] font-semibold rounded px-1.5 py-0.5",
              mode.badge
            )}
          >
            {mode.label}
          </span>
          <span className="text-[11px] text-muted-foreground hidden xs:inline">
            · {CONFIDENCE_LABEL[governance.confidence.level]} ({score}%)
          </span>
          <span className="text-[11px] text-muted-foreground hidden sm:inline">
            · {fileCount} file{fileCount !== 1 ? "s" : ""}
            {joinCount > 0 &&
              ` · ${joinCount} join${joinCount !== 1 ? "s" : ""}`}
          </span>
        </div>
        <ChevronDown
          className={cn(
            "w-4 h-4 text-muted-foreground shrink-0 transition-transform duration-200",
            isOpen && "rotate-180"
          )}
        />
      </button>

      {/* ── strip body ─────────────────────────────────────────────────── */}
      {isOpen && (
        <div className="p-3 space-y-3 border-t border-gov-strip-border">
          {/* confidence + mode reason */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Gauge className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
              <span className="text-[11px] font-medium text-foreground">
                {CONFIDENCE_LABEL[governance.confidence.level]}
              </span>
              {/* confidence meter */}
              <div className="flex-1 h-1.5 rounded-full bg-surface-raised overflow-hidden max-w-[180px]">
                <div
                  className={cn("h-full rounded-full", mode.badge)}
                  style={{ width: `${Math.max(0, Math.min(100, score))}%` }}
                />
              </div>
              <span className="text-[11px] font-mono text-muted-foreground">
                {score}%
              </span>
            </div>
            {governance.reason && (
              <p className={cn("text-[12px] leading-relaxed", mode.accent)}>
                {governance.reason}
              </p>
            )}
          </div>

          {/* feasibility */}
          <div className="flex items-start gap-2">
            <span
              className={cn(
                "mt-1 w-1.5 h-1.5 rounded-full shrink-0",
                governance.feasibility.answerable
                  ? "bg-gov-trust-ok"
                  : "bg-gov-trust-bad"
              )}
            />
            <div className="min-w-0">
              <span className="text-[11px] font-medium text-foreground">
                {governance.feasibility.answerable
                  ? "Answerable against available data"
                  : "Not fully answerable against available data"}
              </span>
              {governance.feasibility.note && (
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  {governance.feasibility.note}
                </p>
              )}
            </div>
          </div>

          {/* files used */}
          {fileCount > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-[11px] font-medium text-foreground">
                  Files considered
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {governance.files.map((f, i) => (
                  <span
                    key={`${f.name}-${i}`}
                    className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-surface-raised border border-border text-[11px] text-foreground max-w-[220px]"
                    title={f.trust_state ? `${f.name} · ${f.trust_state}` : f.name}
                  >
                    {f.trust_state && (
                      <span
                        className={cn(
                          "w-1.5 h-1.5 rounded-full shrink-0",
                          trustDotClass(f.trust_state)
                        )}
                      />
                    )}
                    <span className="truncate">{blobToLabel(f.name)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* validated joins */}
          {joinCount > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <GitMerge className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-[11px] font-medium text-foreground">
                  Validated joins available
                </span>
              </div>
              <div className="flex flex-col gap-1">
                {governance.approved_joins.map((j, i) => (
                  <div
                    key={`${j.from}-${j.to}-${i}`}
                    className="flex items-center flex-wrap gap-1 text-[11px] text-muted-foreground"
                  >
                    <span className="font-medium text-foreground">
                      {blobToLabel(j.from)}
                    </span>
                    <span aria-hidden className="text-muted-foreground">
                      →
                    </span>
                    <span className="font-medium text-foreground">
                      {blobToLabel(j.to)}
                    </span>
                    <span>on</span>
                    <span className="font-mono text-foreground bg-surface-raised border border-border rounded px-1 py-0.5">
                      {j.on}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

"use client";

// Progress checklist reflecting GET /api/onboarding/state.
// Steps are gated: a step is clickable only if it is completed or is the
// current active step (cannot skip ahead).

import { Check, Lock } from "lucide-react";
import { cn } from "@/lib/utils";
import { STEPS, type StepKey } from "../_lib/types";

export function ChecklistSidebar({
  activeStep,
  completedSteps,
  onSelect,
}: {
  activeStep: StepKey;
  completedSteps: StepKey[];
  onSelect: (step: StepKey) => void;
}) {
  const completedCount = STEPS.filter((s) => completedSteps.includes(s.key)).length;
  const pct = Math.round((completedCount / STEPS.length) * 100);

  return (
    <aside className="w-full md:w-[280px] shrink-0 md:border-r border-[#e5e5e5] bg-[color:var(--surface)] md:h-full flex flex-col">
      <div className="p-5 md:p-6">
        <p className="section-label mb-2">Organization setup</p>
        <h2 className="display-md text-[22px] md:text-[26px] mb-4">Get started</h2>

        {/* Progress bar */}
        <div className="mb-1.5 flex items-center justify-between text-[11.5px] text-[color:var(--fg-muted)]">
          <span>{completedCount} of {STEPS.length} complete</span>
          <span>{pct}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-[color:var(--surface-raised)] overflow-hidden">
          <div
            className="h-full bg-[#0a0a0a] transition-[width] duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-5 space-y-0.5">
        {STEPS.map((step, i) => {
          const isDone = completedSteps.includes(step.key);
          const isActive = step.key === activeStep;
          const reachable = isDone || isActive;

          return (
            <button
              key={step.key}
              type="button"
              disabled={!reachable}
              onClick={() => reachable && onSelect(step.key)}
              aria-current={isActive ? "step" : undefined}
              className={cn(
                "w-full text-left flex items-start gap-3 rounded-[var(--radius)] px-3 py-2.5 transition-colors",
                isActive
                  ? "bg-white border border-[#e5e5e5] shadow-[var(--shadow-xs)]"
                  : reachable
                  ? "hover:bg-[color:var(--surface-raised)] border border-transparent"
                  : "opacity-55 cursor-not-allowed border border-transparent",
              )}
            >
              <span
                className={cn(
                  "mt-0.5 w-5 h-5 rounded-full shrink-0 flex items-center justify-center text-[11px] font-semibold border",
                  isDone
                    ? "bg-[color:var(--success)] border-[color:var(--success)] text-white"
                    : isActive
                    ? "bg-[#0a0a0a] border-[#0a0a0a] text-white"
                    : "bg-transparent border-[#c4c4c4] text-[color:var(--fg-subtle)]",
                )}
              >
                {isDone ? (
                  <Check className="w-3 h-3" strokeWidth={3} />
                ) : !reachable ? (
                  <Lock className="w-2.5 h-2.5" />
                ) : (
                  i + 1
                )}
              </span>
              <span className="min-w-0">
                <span
                  className={cn(
                    "block text-[13px] font-medium leading-tight",
                    isActive || isDone
                      ? "text-[color:var(--fg)]"
                      : "text-[color:var(--fg-muted)]",
                  )}
                >
                  {step.title}
                  {step.optional && (
                    <span className="badge-muted ml-2 align-middle">optional</span>
                  )}
                </span>
                <span className="block text-[11.5px] text-[color:var(--fg-subtle)] leading-snug mt-0.5">
                  {step.description}
                </span>
              </span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}

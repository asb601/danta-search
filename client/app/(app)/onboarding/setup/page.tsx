"use client";

// Organization onboarding wizard.
//
// Route: /onboarding/setup
// Owner / org-admin / platform-admin only (gated via role capabilities).
// Driven by the server-side onboarding_state (GET /api/onboarding/state).
// Steps are strictly ordered and gated — a step can't be opened until every
// step before it is marked complete on the server. The final action POSTs
// /api/onboarding/complete and redirects into the app.

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, ArrowRight, CheckCircle2, Loader2, LogOut, ShieldAlert } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";
import { useRole } from "@/components/role-guard";
import { roleBadge } from "@/lib/roles";
import { cn } from "@/lib/utils";
import {
  STEPS,
  STEP_ORDER,
  type OnboardingState,
  type StepKey,
} from "./_lib/types";
import { safeError } from "./_lib/api";
import { ChecklistSidebar } from "./_components/ChecklistSidebar";
import { StepOwnerSignin } from "./_components/StepOwnerSignin";
import { StepAiSettings } from "./_components/StepAiSettings";
import { StepStorage } from "./_components/StepStorage";
import { StepDomains } from "./_components/StepDomains";
import { StepUsers } from "./_components/StepUsers";
import { StepAccessControl } from "./_components/StepAccessControl";

export default function OnboardingWizardPage() {
  const router = useRouter();
  const { user, loading: authLoading, logout } = useAuth();
  const { role, can } = useRole();

  const [state, setState] = useState<OnboardingState | null>(null);
  const [loadingState, setLoadingState] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState<StepKey>("owner_signin");
  const [completing, setCompleting] = useState(false);
  const [completeError, setCompleteError] = useState<string | null>(null);

  const completedSteps: StepKey[] = useMemo(
    () => state?.completed_steps ?? [],
    [state],
  );

  /** Fetch server state. After auth or the user advances we refresh. */
  const loadState = useCallback(
    async (selectFromServer = false) => {
      setLoadError(null);
      try {
        const res = await apiFetch("/api/onboarding/state");
        if (!res.ok) throw new Error(await safeError(res));
        const data: OnboardingState = await res.json();
        setState(data);
        if (data.completed) {
          router.replace("/chat");
          return;
        }
        if (selectFromServer && data.current_step) {
          setActiveStep(data.current_step);
        }
      } catch (err) {
        // Backend may not be ready yet — degrade to the signed-in step so the
        // owner can still proceed once endpoints exist.
        setLoadError(
          err instanceof Error ? err.message : "Could not load onboarding state.",
        );
        setState((prev) => prev ?? { completed_steps: [], current_step: "owner_signin" });
      } finally {
        setLoadingState(false);
      }
    },
    [router],
  );

  useEffect(() => {
    if (authLoading) return;
    void loadState(true);
  }, [authLoading, loadState]);

  const isStepComplete = useCallback(
    (key: StepKey) => {
      if (key === "owner_signin") return !!user || completedSteps.includes(key);
      return completedSteps.includes(key);
    },
    [user, completedSteps],
  );

  const isStepReachable = useCallback(
    (key: StepKey) => {
      const idx = STEP_ORDER.indexOf(key);
      if (idx <= 0) return true;
      // reachable iff every prior step is complete
      return STEP_ORDER.slice(0, idx).every((k) => isStepComplete(k));
    },
    [isStepComplete],
  );

  const goToStep = (key: StepKey) => {
    if (isStepReachable(key)) setActiveStep(key);
  };

  /** Called by a step after it persisted successfully → refresh + advance. */
  const handleStepSaved = useCallback(async () => {
    await loadState(false);
    const idx = STEP_ORDER.indexOf(activeStep);
    const next = STEP_ORDER[idx + 1];
    if (next) {
      setActiveStep(next);
    } else {
      setActiveStep(activeStep); // last step → stay; user clicks Finish
    }
  }, [activeStep, loadState]);

  const allRequiredDone = STEPS.filter((s) => !s.optional).every((s) =>
    isStepComplete(s.key),
  );

  const handleComplete = async () => {
    setCompleteError(null);
    setCompleting(true);
    try {
      const res = await apiFetch("/api/onboarding/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(await safeError(res));
      router.replace("/chat");
    } catch (err) {
      setCompleteError(
        err instanceof Error ? err.message : "Failed to complete onboarding.",
      );
      setCompleting(false);
    }
  };

  // ── Loading ──────────────────────────────────────────────
  if (authLoading || loadingState) {
    return (
      <div className="flex h-screen items-center justify-center bg-white">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-[#0a0a0a]/8 flex items-center justify-center">
            <Loader2 className="w-4 h-4 animate-spin text-[#0a0a0a]" />
          </div>
          <p className="text-[13px] text-[color:var(--fg-subtle)]">Loading setup…</p>
        </div>
      </div>
    );
  }

  // ── Role gate ────────────────────────────────────────────
  if (user && !can.canRunOnboarding) {
    return (
      <div className="flex h-screen items-center justify-center bg-white px-6">
        <div className="max-w-md text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-[var(--radius-lg)] bg-[color:var(--surface-raised)] flex items-center justify-center">
            <ShieldAlert className="w-6 h-6 text-[color:var(--fg-muted)]" />
          </div>
          <h1 className="display-md text-[24px]">Not available</h1>
          <p className="body-lead text-[13.5px]">
            Organization setup is reserved for owners and administrators. Ask
            your organization owner to complete it.
          </p>
          <button onClick={() => router.replace("/chat")} className="btn-outline h-10 px-5">
            Go to app
          </button>
        </div>
      </div>
    );
  }

  const badge = roleBadge(role);
  const activeMeta = STEPS.find((s) => s.key === activeStep)!;
  const activeIdx = STEP_ORDER.indexOf(activeStep);
  const isLastStep = activeIdx === STEP_ORDER.length - 1;
  const prevStep = STEP_ORDER[activeIdx - 1];

  return (
    <div className="h-screen bg-white flex flex-col md:flex-row overflow-hidden">
      <ChecklistSidebar
        activeStep={activeStep}
        completedSteps={STEP_ORDER.filter(isStepComplete)}
        onSelect={goToStep}
      />

      {/* ── Main content ──────────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
        {/* header */}
        <header className="app-topbar">
          <div className="flex items-center gap-2 min-w-0">
            <span className="section-label">Step {activeIdx + 1} of {STEP_ORDER.length}</span>
          </div>
          <div className="flex items-center gap-2.5">
            {badge && (
              <span className={cn("text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full", badge.cls)}>
                {badge.text}
              </span>
            )}
            {user && (
              <button
                onClick={logout}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[12px] text-[#a3a3a3] hover:text-[#dc2626] hover:bg-[#dc2626]/6 transition-colors"
                title="Sign out"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Sign out</span>
              </button>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <div className="max-w-3xl mx-auto px-5 md:px-10 py-8 md:py-12">
            {loadError && (
              <div className="mb-6 flex items-start gap-2 px-3.5 py-2.5 rounded-[var(--radius)] bg-[color:var(--danger-bg)] border border-[color:var(--danger)]/20 text-[12px] text-[color:var(--danger)]">
                <ShieldAlert className="w-4 h-4 shrink-0 mt-px" />
                <span>{loadError} Showing best-effort state.</span>
              </div>
            )}

            <div className="mb-7">
              <h1 className="display-md text-[26px] md:text-[30px] mb-2">
                {activeMeta.title}
              </h1>
              <p className="body-lead text-[14px]">{activeMeta.description}</p>
            </div>

            {/* Step body */}
            {activeStep === "owner_signin" && (
              <StepOwnerSignin onContinue={() => goToStep("ai_settings")} />
            )}
            {activeStep === "ai_settings" && (
              <StepAiSettings state={state} onSaved={handleStepSaved} />
            )}
            {activeStep === "storage" && (
              <StepStorage state={state} onSaved={handleStepSaved} />
            )}
            {activeStep === "domains" && (
              <StepDomains state={state} onSaved={handleStepSaved} />
            )}
            {activeStep === "users" && (
              <StepUsers state={state} onSaved={handleStepSaved} />
            )}
            {activeStep === "access_control" && (
              <StepAccessControl state={state} onSaved={handleStepSaved} />
            )}

            {/* Footer nav */}
            <div className="mt-10 pt-5 border-t border-[#e5e5e5] flex items-center justify-between">
              <button
                type="button"
                onClick={() => prevStep && goToStep(prevStep)}
                disabled={!prevStep}
                className="btn-ghost gap-1.5 px-2 disabled:opacity-30 disabled:pointer-events-none"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Back
              </button>

              {isLastStep ? (
                <div className="flex flex-col items-end gap-2">
                  {completeError && (
                    <span className="text-[11.5px] text-[color:var(--danger)]">
                      {completeError}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={handleComplete}
                    disabled={completing || !allRequiredDone}
                    className="btn-black h-10 px-5 gap-2"
                    title={allRequiredDone ? undefined : "Complete the required steps first"}
                  >
                    {completing ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4" />
                    )}
                    Finish setup
                  </button>
                </div>
              ) : (
                <span className="text-[11.5px] text-[color:var(--fg-subtle)] inline-flex items-center gap-1">
                  Use the form above to continue
                  <ArrowRight className="w-3 h-3" />
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

// Step 6 (optional) — Access control.
// POST /api/onboarding/platform-admin-grant
// Toggle granting Platform Admin access to this organization.

import { useState } from "react";
import { ShieldCheck } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { FormError, FormSuccess, SubmitButton } from "./FormBits";
import { safeError } from "../_lib/api";
import type { OnboardingState } from "../_lib/types";

export function StepAccessControl({
  state,
  onSaved,
}: {
  state: OnboardingState | null;
  onSaved: () => void;
}) {
  const [granted, setGranted] = useState<boolean>(
    () => !!state?.platform_admin_granted,
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const save = async () => {
    setError(null);
    setSuccess(null);
    setSubmitting(true);
    try {
      const res = await apiFetch("/api/onboarding/platform-admin-grant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ granted }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      setSuccess("Access control preference saved.");
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preference.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-xl">
      <FormError message={error} />
      <FormSuccess message={success} />

      <p className="body-lead text-[13.5px]">
        This step is optional. You can grant the platform team administrative
        access to your organization for support and maintenance. You can change
        this later in settings.
      </p>

      <button
        type="button"
        role="switch"
        aria-checked={granted}
        onClick={() => setGranted((g) => !g)}
        className={cn(
          "w-full flex items-center gap-4 rounded-[var(--radius-lg)] border px-4 py-4 text-left transition-colors",
          granted
            ? "border-[#0a0a0a] bg-[color:var(--surface)]"
            : "border-[#e5e5e5] bg-white hover:border-[#c4c4c4]",
        )}
      >
        <div className="w-10 h-10 rounded-[var(--radius)] bg-[color:var(--surface-raised)] flex items-center justify-center shrink-0">
          <ShieldCheck className="w-5 h-5 text-[color:var(--fg)]" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[13.5px] font-medium text-[color:var(--fg)]">
            Grant Platform Admin access
          </p>
          <p className="text-[12px] text-[color:var(--fg-muted)]">
            Allow platform administrators to manage this organization.
          </p>
        </div>
        {/* toggle track */}
        <span
          className={cn(
            "relative w-10 h-6 rounded-full transition-colors shrink-0",
            granted ? "bg-[#0a0a0a]" : "bg-[color:var(--border-strong)]",
          )}
        >
          <span
            className={cn(
              "absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-[var(--shadow-xs)] transition-transform",
              granted && "translate-x-4",
            )}
          />
        </span>
      </button>

      <div className="flex items-center gap-3 pt-1">
        <SubmitButton type="button" onClick={save} loading={submitting}>
          Save &amp; continue
        </SubmitButton>
        <button
          type="button"
          onClick={onSaved}
          disabled={submitting}
          className="btn-ghost px-3 h-10"
        >
          Skip
        </button>
      </div>
    </div>
  );
}

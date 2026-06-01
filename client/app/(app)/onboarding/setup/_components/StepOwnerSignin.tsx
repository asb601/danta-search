"use client";

// Step 1 — Owner Google sign-in. Owner is Google-SSO only, so this reuses the
// existing Google login flow (getGoogleLoginUrl). When a session already exists
// (the user reached this page authenticated) the step is satisfied and we just
// confirm + advance.

import { CheckCircle2 } from "lucide-react";
import { getGoogleLoginUrl } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";
import { SubmitButton } from "./FormBits";

export function StepOwnerSignin({ onContinue }: { onContinue: () => void }) {
  const { user } = useAuth();

  if (user) {
    return (
      <div className="space-y-5 max-w-md">
        <div className="flex items-center gap-3 px-4 py-3 rounded-[var(--radius)] bg-[color:var(--success-bg)] border border-[color:var(--success)]/20">
          <CheckCircle2 className="w-5 h-5 text-[color:var(--success)] shrink-0" />
          <div className="min-w-0">
            <p className="text-[13px] font-medium text-[color:var(--fg)]">
              Signed in as owner
            </p>
            <p className="text-[12px] text-[color:var(--fg-muted)] truncate">
              {user.email}
            </p>
          </div>
        </div>
        <p className="body-lead text-[13.5px]">
          You are signed in with the owner Google account. Continue to configure
          your organization.
        </p>
        <SubmitButton type="button" onClick={onContinue}>
          Continue
        </SubmitButton>
      </div>
    );
  }

  return (
    <div className="space-y-5 max-w-md">
      <p className="body-lead text-[13.5px]">
        The organization owner signs in with Google. This account will own the
        organization and complete the remaining setup steps.
      </p>
      <a
        href={getGoogleLoginUrl()}
        className="btn-black h-11 px-5 gap-3 inline-flex w-full sm:w-auto"
      >
        <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" aria-hidden="true">
          <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="rgba(255,255,255,0.9)" />
          <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="rgba(255,255,255,0.9)" />
          <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="rgba(255,255,255,0.9)" />
          <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="rgba(255,255,255,0.9)" />
        </svg>
        Sign in with Google
      </a>
    </div>
  );
}

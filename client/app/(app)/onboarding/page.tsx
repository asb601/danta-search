"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Building2, Clock, CheckCircle2, XCircle, Loader2, Send, LogOut } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";

type AccessStatus = "loading" | "none" | "pending" | "approved" | "declined" | "no_domains";

export default function OnboardingPage() {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  const [status, setStatus] = useState<AccessStatus>("loading");
  const [orgName, setOrgName] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Admins go straight to chat
  useEffect(() => {
    if (!loading && user?.is_admin) {
      router.replace("/chat");
    }
  }, [loading, user, router]);

  // Check current access status
  useEffect(() => {
    if (loading || !user || user.is_admin) return;
    apiFetch("/api/access-requests/me/status")
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "approved") {
          // Only redirect if the user actually has domains assigned.
          // If an admin removed all domains the request status stays "approved"
          // but allowed_domains is empty — redirecting would cause an infinite
          // loop (layout sends them back here because they have no domains).
          if (user.allowed_domains && user.allowed_domains.length > 0) {
            // Force a full page reload so the auth provider re-fetches /me
            // and picks up the updated allowed_domains set by the approval.
            window.location.replace("/chat");
          } else {
            // Approved but no domains assigned (admin may have removed them).
            setStatus("no_domains");
          }
        } else {
          setStatus(data.status as AccessStatus);
        }
      })
      .catch(() => setStatus("none"));
  }, [loading, user, router]);

  const handleSubmit = async () => {
    setSubmitError(null);
    if (!orgName.trim()) {
      setSubmitError("Organization name is required.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await apiFetch("/api/access-requests/me", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          org_name: orgName.trim(),
          message: message.trim() || null,
        }),
      });
      if (!res.ok) {
        let detail = "Failed to submit request.";
        try {
          const body = await res.json();
          if (typeof body?.detail === "string") detail = body.detail;
        } catch {
          /* ignore parse errors */
        }
        setSubmitError(detail);
        return;
      }
      setStatus("pending");
    } catch {
      setSubmitError("Network error. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading || status === "loading" || (user?.is_admin)) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background items-center justify-center p-6">
      <div className="w-full max-w-md space-y-8">

        {/* ── Request form ── */}
        {status === "none" && (
          <>
            <div className="text-center space-y-2">
              <div className="flex justify-center">
                <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                  <Building2 className="w-7 h-7 text-primary" />
                </div>
              </div>
              <h1 className="text-2xl font-semibold text-foreground">
                Request Access
              </h1>
              <p className="text-sm text-muted-foreground">
                This workspace is invite-only. Submit a request and an admin will review it.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">
                  Your email
                </label>
                <div className="px-3 py-2 rounded-xl border border-border bg-surface text-sm text-muted-foreground">
                  {user?.email}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">
                  Organization name
                </label>
                <input
                  type="text"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                  placeholder="e.g. Acme Corp"
                  className="w-full px-3 py-2 rounded-xl border border-border bg-surface text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-foreground mb-1.5">
                  Message <span className="text-muted-foreground font-normal">(optional)</span>
                </label>
                <textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Why do you need access? Which team are you from?"
                  rows={3}
                  className="w-full px-3 py-2 rounded-xl border border-border bg-surface text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
            </div>

            {submitError && (
              <p className="text-sm text-destructive">{submitError}</p>
            )}

            <button
              onClick={handleSubmit}
              disabled={submitting || !orgName.trim()}
              className="w-full py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              {submitting ? "Sending…" : "Submit Request"}
            </button>
          </>
        )}

        {/* ── Pending ── */}
        {status === "pending" && (
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="w-14 h-14 rounded-2xl bg-yellow-500/10 flex items-center justify-center">
                <Clock className="w-7 h-7 text-yellow-500" />
              </div>
            </div>
            <h1 className="text-2xl font-semibold text-foreground">Request Sent</h1>
            <p className="text-sm text-muted-foreground">
              Your access request is pending review. You will receive an email at{" "}
              <span className="text-foreground font-medium">{user?.email}</span> once
              an admin approves or declines it.
            </p>
            <p className="text-xs text-muted-foreground">
              You can close this tab. Check your inbox for the decision.
            </p>
          </div>
        )}

        {/* ── Approved but no domains assigned ── */}
        {status === "no_domains" && (
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="w-14 h-14 rounded-2xl bg-yellow-500/10 flex items-center justify-center">
                <Clock className="w-7 h-7 text-yellow-500" />
              </div>
            </div>
            <h1 className="text-2xl font-semibold text-foreground">Awaiting Domain Assignment</h1>
            <p className="text-sm text-muted-foreground">
              Your account has been approved but no data domains have been assigned yet.
              Please contact your administrator to complete the setup.
            </p>
            <p className="text-xs text-muted-foreground">
              Signed in as{" "}
              <span className="text-foreground font-medium">{user?.email}</span>
            </p>
          </div>
        )}

        {/* ── Declined ── */}
        {status === "declined" && (
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="w-14 h-14 rounded-2xl bg-destructive/10 flex items-center justify-center">
                <XCircle className="w-7 h-7 text-destructive" />
              </div>
            </div>
            <h1 className="text-2xl font-semibold text-foreground">Request Declined</h1>
            <p className="text-sm text-muted-foreground">
              Your access request was not approved. Contact your administrator if you
              believe this is a mistake.
            </p>
          </div>
        )}

        {/* ── Approved (brief flash before redirect) ── */}
        {status === "approved" && (
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="w-14 h-14 rounded-2xl bg-green-500/10 flex items-center justify-center">
                <CheckCircle2 className="w-7 h-7 text-green-500" />
              </div>
            </div>
            <h1 className="text-2xl font-semibold text-foreground">Access Approved</h1>
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground mx-auto" />
          </div>
        )}

        {/* ── Sign out ── */}
        {status !== "approved" && (
          <div className="text-center">
            <button
              onClick={logout}
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </button>
          </div>
        )}

      </div>
    </div>
  );
}


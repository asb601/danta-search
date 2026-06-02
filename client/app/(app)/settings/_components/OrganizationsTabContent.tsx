"use client";

import { useState, useCallback } from "react";
import useSWR from "swr";
import { motion } from "framer-motion";
import {
  Building2,
  Loader2,
  Trash2,
  Users,
  Database,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

/* ── types ───────────────────────────────────────────────────────────────── */

// Backend contract: GET /api/organizations -> OrgOut[]
interface OrgItem {
  id: string;
  name: string;
  container_id: string | null;
  container_name: string | null;
  user_count: number;
  onboarding_state: string | null;
  slug: string | null;
  owner_email: string | null;
  created_at: string;
}

/* ── fetcher ─────────────────────────────────────────────────────────────── */

const orgsFetcher = async (): Promise<OrgItem[]> => {
  const res = await apiFetch("/api/organizations");
  if (!res.ok) return [];
  return res.json();
};

/* ── onboarding-state badge ──────────────────────────────────────────────── */

// "completed" is the only terminal/done state; everything else is in-progress.
function OnboardingBadge({ state }: { state: string | null }) {
  const value = state ?? "created";
  const done = value === "completed";
  const label = value
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded-full border",
        done
          ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-600"
          : "bg-amber-500/10 border-amber-500/25 text-amber-600"
      )}
    >
      {done ? <CheckCircle2 className="w-3 h-3" /> : <Clock className="w-3 h-3" />}
      {label}
    </span>
  );
}

/* ── date formatting ─────────────────────────────────────────────────────── */

const fmtDate = (iso: string): string => {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

/* ── OrganizationsTabContent (platform-admin only) ───────────────────────── */

export default function OrganizationsTabContent() {
  const { data: orgs, mutate } = useSWR("organizations-list", orgsFetcher, {
    revalidateOnFocus: false,
  });
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const extractError = async (res: Response): Promise<string> => {
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") return body.detail;
      if (Array.isArray(body?.detail)) {
        return body.detail
          .map((d: { msg?: string }) => d?.msg)
          .filter(Boolean)
          .join("; ");
      }
    } catch {
      /* ignore parse errors */
    }
    return `Request failed (${res.status}).`;
  };

  const handleDelete = useCallback(
    async (orgId: string) => {
      setDeleteError(null);
      setDeletingId(orgId);
      try {
        const res = await apiFetch(`/api/organizations/${orgId}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          setDeleteError(await extractError(res));
          return;
        }
        mutate();
      } finally {
        setDeletingId(null);
        setConfirmDeleteId(null);
      }
    },
    [mutate]
  );

  if (!orgs) {
    return (
      <div className="flex items-center justify-center h-40">
        <Loader2 className="w-5 h-5 text-[#a3a3a3] animate-spin" />
      </div>
    );
  }

  if (orgs.length === 0) {
    return (
      <div className="max-w-xl flex flex-col items-center justify-center gap-3 py-16 text-center">
        <div className="w-12 h-12 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center">
          <Building2 className="w-6 h-6 text-[#a3a3a3]" />
        </div>
        <p className="text-[13px] font-medium text-[#0a0a0a]">No organizations yet</p>
        <p className="text-[12px] text-[#737373]">
          Organizations appear here once they are created.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-xl">
      <div className="space-y-3">
        {deleteError && <p className="text-[12px] text-[#dc2626]">{deleteError}</p>}

        {orgs.map((o, idx) => (
          <motion.div
            key={o.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.04, duration: 0.22 }}
            className="flex items-center justify-between gap-4 px-4 py-3 rounded-xl border border-[#e5e5e5] bg-[#f9f9f9]"
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-9 h-9 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center shrink-0">
                <Building2 className="w-5 h-5 text-[#a3a3a3]" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <p className="text-[13px] font-medium text-[#0a0a0a] truncate">
                    {o.name}
                  </p>
                  {o.slug && (
                    <span className="text-[11px] text-[#a3a3a3] truncate">
                      {o.slug}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-[#737373] truncate">
                  {o.owner_email || "No owner"}
                </p>
                <div className="flex items-center gap-3 mt-1 text-[11px] text-[#a3a3a3]">
                  <span className="inline-flex items-center gap-1">
                    <Users className="w-3 h-3" />
                    {o.user_count} user{o.user_count !== 1 && "s"}
                  </span>
                  {o.container_name && (
                    <span className="inline-flex items-center gap-1 truncate">
                      <Database className="w-3 h-3" />
                      {o.container_name}
                    </span>
                  )}
                  <span>{fmtDate(o.created_at)}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <OnboardingBadge state={o.onboarding_state} />

              {confirmDeleteId === o.id ? (
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-[#dc2626] font-medium">Delete?</span>
                  <button
                    onClick={() => handleDelete(o.id)}
                    disabled={deletingId === o.id}
                    className="px-2 py-0.5 text-[11px] rounded-lg bg-[#dc2626] text-white hover:bg-[#b91c1c] transition-colors disabled:opacity-50"
                  >
                    {deletingId === o.id ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      "Yes"
                    )}
                  </button>
                  <button
                    onClick={() => setConfirmDeleteId(null)}
                    className="px-2 py-0.5 text-[11px] rounded-lg bg-[#f4f4f4] text-[#737373] hover:text-[#0a0a0a] transition-colors"
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    setDeleteError(null);
                    setConfirmDeleteId(o.id);
                  }}
                  title="Delete organization"
                  className="p-1.5 rounded-lg text-[#a3a3a3] hover:text-[#dc2626] hover:bg-[#dc2626]/8 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import useSWR from "swr";
import { motion } from "framer-motion";
import {
  UserCircle,
  Shield,
  ShieldOff,
  Loader2,
  ChevronDown,
  CheckCircle2,
  Trash2,
  Clock,
  UserCheck,
  UserX,
} from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

/* ── types ───────────────────────────────────────────────────────────────── */

interface UserItem {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  is_admin: boolean;
  role: string;
  created_at: string;
  file_count: number;
  allowed_domains: string[] | null;
}

interface AccessRequestItem {
  id: string;
  user_id: string;
  user_email: string;
  user_name: string | null;
  user_picture: string | null;
  status: string;
  message: string | null;
  org_name: string | null;
  requested_at: string;
}

/** Pending requester awaiting a role grant. Backend contract:
 *  GET /api/users/pending -> [{id,email,name,org_name,status}] */
interface PendingUser {
  id: string;
  email: string;
  name: string | null;
  org_name: string | null;
  status: string;
}

/** The two grantable roles, shown verbatim in the grant dropdown. */
const GRANT_ROLES: { value: "owner" | "admin"; label: string }[] = [
  { value: "owner", label: "Organization owner" },
  { value: "admin", label: "Application admin" },
];

/* ── fetchers ────────────────────────────────────────────────────────────── */

const usersFetcher = async (): Promise<UserItem[]> => {
  const res = await apiFetch("/api/users");
  if (!res.ok) return [];
  return res.json();
};

const accessRequestsFetcher = async (): Promise<AccessRequestItem[]> => {
  const res = await apiFetch("/api/access-requests");
  if (!res.ok) return [];
  return res.json();
};

const pendingUsersFetcher = async (): Promise<PendingUser[]> => {
  const res = await apiFetch("/api/users/pending");
  if (!res.ok) return [];
  return res.json();
};

/* ── Role dropdown component ─────────────────────────────────────────────── */

const ROLES: { value: string; label: string; color: string }[] = [
  { value: "admin",     label: "Admin",     color: "text-primary" },
  { value: "developer", label: "Developer", color: "text-violet-400" },
  { value: "manager",   label: "Manager",   color: "text-cyan-400" },
  { value: "user",      label: "Member",    color: "text-muted-foreground" },
];

function RoleDropdown({
  currentRole,
  disabled,
  onChange,
}: {
  currentRole: string;
  disabled: boolean;
  onChange: (role: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const current = ROLES.find((r) => r.value === currentRole) ?? ROLES[3];

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((p) => !p)}
        className={cn(
          "inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-medium rounded-full border transition-colors",
          currentRole === "admin"     ? "bg-[#0a0a0a]/8 border-[#0a0a0a]/20 text-[#0a0a0a]" :
          currentRole === "developer" ? "bg-violet-500/10 border-violet-500/25 text-violet-600" :
          currentRole === "manager"   ? "bg-cyan-500/10 border-cyan-500/25 text-cyan-600" :
          "bg-[#f4f4f4] border-[#e5e5e5] text-[#737373]",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        {disabled ? (
          <Loader2 className="w-3 h-3 animate-spin" />
        ) : (
          currentRole === "admin" && <Shield className="w-3 h-3" />
        )}
        {current.label}
        {!disabled && <ChevronDown className="w-2.5 h-2.5 ml-0.5" />}
      </button>

      {open && (
        <div className="absolute z-30 mt-1 right-0 w-36 rounded-xl border border-[#e5e5e5] bg-white shadow-[0_4px_20px_rgba(0,0,0,0.1)] overflow-hidden">
          {ROLES.map((r) => (
            <button
              key={r.value}
              type="button"
              onClick={() => {
                onChange(r.value);
                setOpen(false);
              }}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2 text-[12px] hover:bg-[#f4f4f4] transition-colors",
                r.value === "admin" ? "text-[#0a0a0a]" :
                r.value === "developer" ? "text-violet-600" :
                r.value === "manager" ? "text-cyan-600" :
                "text-[#737373]",
                r.value === currentRole && "font-semibold"
              )}
            >
              <span>{r.label}</span>
              {r.value === currentRole && <CheckCircle2 className="w-3 h-3" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Grant-role dropdown (pending requesters) ────────────────────────────── */
/* Exactly two options: "owner" (Organization owner) and "admin" (Application
 * admin). Selecting one fires the grant callback. */

function GrantRoleDropdown({
  disabled,
  onSelect,
}: {
  disabled: boolean;
  onSelect: (role: "owner" | "admin") => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((p) => !p)}
        className={cn(
          "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-colors",
          "bg-[#0a0a0a] text-white border-[#0a0a0a] hover:opacity-90",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        {disabled ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <UserCheck className="w-3.5 h-3.5" />
        )}
        Grant role
        {!disabled && <ChevronDown className="w-2.5 h-2.5 ml-0.5" />}
      </button>

      {open && (
        <div className="absolute z-30 mt-1 right-0 w-52 rounded-xl border border-[#e5e5e5] bg-white shadow-[0_4px_20px_rgba(0,0,0,0.1)] overflow-hidden">
          {GRANT_ROLES.map((r) => (
            <button
              key={r.value}
              type="button"
              onClick={() => {
                onSelect(r.value);
                setOpen(false);
              }}
              className="w-full flex items-center gap-2 px-3 py-2 text-[12px] text-left text-[#0a0a0a] hover:bg-[#f4f4f4] transition-colors"
            >
              {r.value === "owner" ? (
                <Shield className="w-3.5 h-3.5 text-[#0a0a0a] shrink-0" />
              ) : (
                <ShieldOff className="w-3.5 h-3.5 text-[#737373] shrink-0" />
              )}
              <span>{r.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── UsersTabContent (platform-admin only) ───────────────────────────────── */

export default function UsersTabContent({ currentUserId }: { currentUserId: string }) {
  const { data: users, mutate } = useSWR("users-list", usersFetcher, {
    revalidateOnFocus: false,
  });
  // Pending requesters awaiting a role grant (GET /api/users/pending).
  const { data: pendingUsers, mutate: mutatePending } = useSWR(
    "users-pending",
    pendingUsersFetcher,
    { revalidateOnFocus: true, refreshInterval: 30000 },
  );
  // Access requests kept only to map a pending user → its request id for Decline.
  const { data: accessRequests, mutate: mutateRequests } = useSWR(
    "access-requests",
    accessRequestsFetcher,
    { revalidateOnFocus: true, refreshInterval: 30000 },
  );
  const [changingRoleId, setChangingRoleId] = useState<string | null>(null);
  const [reviewingId, setReviewingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);

  const handleSetRole = useCallback(
    async (userId: string, role: string) => {
      setChangingRoleId(userId);
      try {
        const res = await apiFetch(`/api/users/${userId}/role`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role }),
        });
        if (res.ok) mutate();
      } finally {
        setChangingRoleId(null);
      }
    },
    [mutate]
  );

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

  // Grant a role to a pending requester. PATCH /api/users/{id}/grant with a
  // JSON body {role, org_name?}. role is one of "owner" | "admin".
  const handleGrant = useCallback(
    async (userId: string, role: "owner" | "admin", orgName: string | null) => {
      setReviewError(null);
      setReviewingId(userId);
      try {
        const res = await apiFetch(`/api/users/${userId}/grant`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role,
            org_name: orgName ?? undefined,
          }),
        });
        if (!res.ok) {
          setReviewError(await extractError(res));
          return;
        }
        mutatePending();
        mutateRequests();
        mutate(); // refresh users list too
      } finally {
        setReviewingId(null);
      }
    },
    [mutate, mutatePending, mutateRequests]
  );

  // Decline stays a bodyless PATCH on the access-request id.
  const handleDecline = useCallback(
    async (requestId: string) => {
      setReviewError(null);
      setReviewingId(requestId);
      try {
        const res = await apiFetch(`/api/access-requests/${requestId}/decline`, {
          method: "PATCH",
        });
        if (!res.ok) {
          setReviewError(await extractError(res));
          return;
        }
        mutatePending();
        mutateRequests();
        mutate();
      } finally {
        setReviewingId(null);
      }
    },
    [mutate, mutatePending, mutateRequests]
  );

  const handleDeleteUser = useCallback(
    async (userId: string) => {
      setDeletingId(userId);
      try {
        const res = await apiFetch(`/api/users/${userId}`, { method: "DELETE" });
        if (res.ok) mutate();
      } finally {
        setDeletingId(null);
        setConfirmDeleteId(null);
      }
    },
    [mutate]
  );

  if (!users) {
    return (
      <div className="flex items-center justify-center h-40">
        <Loader2 className="w-5 h-5 text-[#a3a3a3] animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-xl">

      {/* ── Pending requesters → grant a role ── */}
      {pendingUsers && pendingUsers.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="w-3.5 h-3.5 text-amber-500" />
            <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">
              Pending requests
            </p>
            <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded-full bg-amber-500/12 text-amber-600">
              {pendingUsers.length}
            </span>
          </div>

          {reviewError && (
            <p className="text-[12px] text-[#dc2626]">{reviewError}</p>
          )}

          {pendingUsers.map((p) => {
            const reviewing = reviewingId === p.id;
            // Map this pending user → its access-request id (for Decline).
            const request = accessRequests?.find((r) => r.user_id === p.id);
            const declining = request ? reviewingId === request.id : false;
            return (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start justify-between gap-4 px-4 py-3 rounded-xl border border-amber-200 bg-amber-50/60"
              >
                <div className="flex items-start gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center shrink-0 mt-0.5">
                    <UserCircle className="w-5 h-5 text-[#a3a3a3]" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-[13px] font-medium text-[#0a0a0a] truncate">
                      {p.name || p.email}
                    </p>
                    <p className="text-[11px] text-[#737373] truncate">{p.email}</p>
                    {p.org_name && (
                      <p className="text-[11px] text-[#737373] truncate mt-0.5">
                        Org: <span className="font-medium text-[#0a0a0a]">{p.org_name}</span>
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0 mt-0.5">
                  <GrantRoleDropdown
                    disabled={reviewing || declining}
                    onSelect={(role) => handleGrant(p.id, role, p.org_name)}
                  />
                  {request && (
                    <button
                      onClick={() => handleDecline(request.id)}
                      disabled={reviewing || declining}
                      title="Decline"
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium bg-[#dc2626]/8 text-[#dc2626] hover:bg-[#dc2626]/15 transition-colors disabled:opacity-40"
                    >
                      {declining ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <UserX className="w-3.5 h-3.5" />
                      )}
                      Decline
                    </button>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* ── Existing users ── */}
      <div className="space-y-3">
        {pendingUsers && pendingUsers.length > 0 && (
          <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">Members</p>
        )}
      {users.map((u, idx) => {
        const isCurrent = u.id === currentUserId;
        const changing = changingRoleId === u.id;
        const roleLabel = u.role === "admin" ? "Admin" : u.role === "developer" ? "Developer" : u.role === "manager" ? "Manager" : "Member";

        return (
          <motion.div
            key={u.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.04, duration: 0.22 }}
            className="flex items-center justify-between gap-4 px-4 py-3 rounded-xl border border-[#e5e5e5] bg-[#f9f9f9]"
          >
            <div className="flex items-center gap-3 min-w-0">
              {u.picture ? (
                <img
                  src={u.picture}
                  alt=""
                  className="w-9 h-9 rounded-full border-2 border-white shadow-sm"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <div className="w-9 h-9 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center">
                  <UserCircle className="w-5 h-5 text-[#a3a3a3]" />
                </div>
              )}
              <div className="min-w-0">
                <p className="text-[13px] font-medium text-[#0a0a0a] truncate">
                  {u.name || u.email}
                </p>
                <p className="text-[11px] text-[#737373] truncate">{u.email}</p>
              </div>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <span className="text-[11px] text-[#a3a3a3]">
                {u.file_count} file{u.file_count !== 1 && "s"}
              </span>

              {isCurrent ? (
                <span className={cn(
                  "inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded-full border",
                  u.role === "admin" ? "bg-[#0a0a0a]/8 border-[#0a0a0a]/15 text-[#0a0a0a]" :
                  u.role === "developer" ? "bg-violet-500/10 border-violet-500/20 text-violet-600" :
                  u.role === "manager" ? "bg-cyan-500/10 border-cyan-500/20 text-cyan-600" :
                  "bg-[#f4f4f4] border-[#e5e5e5] text-[#737373]"
                )}>
                  {u.role === "admin" && <Shield className="w-3 h-3" />}
                  {roleLabel}
                </span>
              ) : (
                <RoleDropdown
                  currentRole={u.role || (u.is_admin ? "admin" : "user")}
                  disabled={changing}
                  onChange={(role) => handleSetRole(u.id, role)}
                />
              )}

              {!isCurrent && (
                confirmDeleteId === u.id ? (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] text-[#dc2626] font-medium">Delete?</span>
                    <button
                      onClick={() => handleDeleteUser(u.id)}
                      disabled={deletingId === u.id}
                      className="px-2 py-0.5 text-[11px] rounded-lg bg-[#dc2626] text-white hover:bg-[#b91c1c] transition-colors disabled:opacity-50"
                    >
                      {deletingId === u.id ? <Loader2 className="w-3 h-3 animate-spin" /> : "Yes"}
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
                    onClick={() => setConfirmDeleteId(u.id)}
                    title="Delete user"
                    className="p-1.5 rounded-lg text-[#a3a3a3] hover:text-[#dc2626] hover:bg-[#dc2626]/8 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )
              )}
            </div>
          </motion.div>
        );
      })}
      </div>
    </div>
  );
}

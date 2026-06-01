"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import useSWR from "swr";
import { motion, AnimatePresence } from "framer-motion";
import { UserCircle, Users, Shield, ShieldOff, Loader2, Database, RefreshCw, CheckCircle2, AlertTriangle, Tag, X, Plus, Sparkles, FolderOpen, FileText, ChevronDown, ChevronRight, Trash2, Clock, UserCheck, UserX, Mail } from "lucide-react";
import { useAuth } from "@/components/auth-provider";
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

interface EligibleFile {
  file_id: string;
  name: string;
  folder_id: string | null;
  current_domain: string | null;
  ai_description: string | null;
  good_for: string[];
  ingest_status: string;
}

interface DeptFile {
  file_id: string;
  name: string;
  ingest_status: string;
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

const domainsFetcher = async (): Promise<string[]> => {
  const res = await apiFetch("/api/admin/domains");
  if (!res.ok) return [];
  const data = await res.json();
  return data.domains ?? [];
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

/* ── tabs ────────────────────────────────────────────────────────────────── */

type Tab = "profile" | "users" | "domains" | "parquet";

/* ── page ────────────────────────────────────────────────────────────────── */

export default function ProfilePage() {
  const { user } = useAuth();
  const [tab, setTab] = useState<Tab>("profile");

  if (!user) return null;

  const tabs: { id: Tab; label: string; icon: typeof UserCircle; adminOnly?: boolean }[] = [
    { id: "profile", label: "Profile", icon: UserCircle },
    { id: "users", label: "Users", icon: Users, adminOnly: true },
    { id: "domains", label: "Domains", icon: Tag, adminOnly: true },
    { id: "parquet", label: "Parquet Status", icon: Database, adminOnly: true },
  ];

  const visibleTabs = tabs.filter((t) => !t.adminOnly || user.is_admin);

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="shrink-0 px-6 pt-6 pb-0 border-b border-[#e5e5e5]">
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-7 h-7 rounded-lg bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center">
            <UserCircle className="w-4 h-4 text-[#737373]" />
          </div>
          <h1
            className="text-[17px] font-bold text-[#0a0a0a]"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}
          >
            Profile
          </h1>
        </div>

        {/* Animated tab bar */}
        <div className="flex gap-0.5">
          {visibleTabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "relative flex items-center gap-1.5 px-3 py-2 text-[13px] font-medium transition-colors rounded-t-lg",
                tab === t.id ? "text-[#0a0a0a]" : "text-[#a3a3a3] hover:text-[#737373]"
              )}
            >
              <t.icon className="w-3.5 h-3.5" />
              {t.label}
              {tab === t.id && (
                <motion.div
                  layoutId="profile-tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#0a0a0a] rounded-full"
                  transition={{ type: "spring", stiffness: 400, damping: 36 }}
                />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="p-6"
          >
            {tab === "profile" && <ProfileTab />}
            {tab === "users" && user.is_admin && <UsersTab currentUserId={user.id} />}
            {tab === "domains" && user.is_admin && <DomainsTab />}
            {tab === "parquet" && user.is_admin && <ParquetTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ── Profile tab ─────────────────────────────────────────────────────────── */

function ProfileTab() {
  const { user } = useAuth();
  const [editingDomains, setEditingDomains] = useState(false);

  if (!user) return null;

  const roleLabel = user.is_admin ? "Admin" : user.role === "developer" ? "Developer" : user.role === "manager" ? "Manager" : "Member";
  const roleCls = user.is_admin
    ? "bg-[#0a0a0a] text-white"
    : user.role === "developer"
    ? "bg-violet-100 text-violet-700"
    : user.role === "manager"
    ? "bg-cyan-100 text-cyan-700"
    : "bg-[#f4f4f4] text-[#737373]";

  return (
    <div className="max-w-lg space-y-6">
      {/* Profile hero card */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="relative overflow-hidden rounded-2xl border border-[#e5e5e5] bg-[#f9f9f9] p-6"
      >
        {/* Subtle grid background */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: "linear-gradient(to right, #e5e5e5 1px, transparent 1px), linear-gradient(to bottom, #e5e5e5 1px, transparent 1px)",
            backgroundSize: "28px 28px",
            opacity: 0.4,
            maskImage: "radial-gradient(ellipse 80% 80% at 50% 0%, black, transparent)",
          }}
        />
        <div className="relative flex items-center gap-5">
          {/* Avatar */}
          <div className="relative shrink-0">
            {user.picture ? (
              <img
                src={user.picture}
                alt=""
                className="w-16 h-16 rounded-2xl border-2 border-white shadow-md"
                referrerPolicy="no-referrer"
              />
            ) : (
              <div className="w-16 h-16 rounded-2xl bg-[#ebebeb] border-2 border-white shadow-md flex items-center justify-center">
                <UserCircle className="w-8 h-8 text-[#a3a3a3]" />
              </div>
            )}
            <span className={cn("absolute -bottom-1 -right-1 text-[10px] font-bold px-1.5 py-0.5 rounded-full border-2 border-white", roleCls)}>
              {roleLabel}
            </span>
          </div>

          <div className="min-w-0">
            <p
              className="text-[18px] font-bold text-[#0a0a0a] truncate"
              style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}
            >
              {user.name || "—"}
            </p>
            <p className="text-[13px] text-[#737373] truncate mt-0.5">{user.email}</p>
          </div>
        </div>
      </motion.div>

      {/* Info fields */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08, duration: 0.32 }}
        className="space-y-2"
      >
        <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest mb-3">Account details</p>
        <Field label="Full name" value={user.name || "—"} />
        <Field label="Email address" value={user.email} />
        <Field label="Role" value={roleLabel} />
      </motion.div>

      {/* Departments (non-admin) */}
      {!user.is_admin && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.14, duration: 0.32 }}
          className="space-y-3"
        >
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">My departments</p>
            <button
              onClick={() => setEditingDomains(true)}
              className="text-[12px] font-medium text-[#0a0a0a] hover:text-[#525252] transition-colors underline underline-offset-2"
            >
              Edit
            </button>
          </div>
          {user.allowed_domains && user.allowed_domains.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {user.allowed_domains.map((d) => (
                <span
                  key={d}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[12px] font-medium bg-[#f4f4f4] text-[#0a0a0a] border border-[#e5e5e5]"
                >
                  <Tag className="w-3 h-3 text-[#a3a3a3]" />
                  {d}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-[13px] text-[#a3a3a3]">No departments assigned.</p>
          )}
        </motion.div>
      )}

      {editingDomains && (
        <DomainPickerModal
          current={user.allowed_domains ?? []}
          onClose={() => setEditingDomains(false)}
        />
      )}
    </div>
  );
}

/* ── Domain picker modal (for regular users editing their own) ───────────── */

function DomainPickerModal({
  current,
  onClose,
}: {
  current: string[];
  onClose: () => void;
}) {
  const [domains, setDomains] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>(current);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiFetch("/api/users/domains")
      .then((r) => r.json())
      .then((d) => setDomains(d.domains ?? []))
      .catch(() => {});
  }, []);

  const toggle = (d: string) =>
    setSelected((prev) =>
      prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]
    );

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiFetch("/api/users/me/domains", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_domains: selected.length > 0 ? selected : null }),
      });
      // Reload page to refresh auth context
      window.location.reload();
    } catch {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 4 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="w-full max-w-sm bg-white border border-[#e5e5e5] rounded-2xl shadow-xl p-6 space-y-4"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-[15px] font-bold text-[#0a0a0a]" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>Edit Departments</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg text-[#a3a3a3] hover:text-[#0a0a0a] hover:bg-[#f4f4f4] transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {domains.map((d) => {
            const active = selected.includes(d);
            return (
              <button
                key={d}
                onClick={() => toggle(d)}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded-lg border text-[13px] text-left transition-all",
                  active
                    ? "border-[#0a0a0a] bg-[#0a0a0a]/5 text-[#0a0a0a] font-medium"
                    : "border-[#e5e5e5] text-[#737373] hover:text-[#0a0a0a] hover:border-[#a3a3a3]"
                )}
              >
                {active && <CheckCircle2 className="w-3.5 h-3.5 text-[#0a0a0a] shrink-0" />}
                <span className="truncate">{d}</span>
              </button>
            );
          })}
        </div>
        <div className="flex gap-2 pt-1">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg border border-[#e5e5e5] text-[13px] text-[#737373] hover:text-[#0a0a0a] hover:border-[#a3a3a3] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 py-2 rounded-lg text-[13px] font-medium text-white disabled:opacity-50 hover:opacity-90 transition-opacity flex items-center justify-center gap-1.5"
            style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
          >
            {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            Save
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-[#a3a3a3]">{label}</span>
      <span className="text-[13px] text-[#0a0a0a] px-3 py-2 rounded-lg bg-[#f9f9f9] border border-[#e5e5e5]">
        {value}
      </span>
    </div>
  );
}

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

/* ── Users tab (admin only) ──────────────────────────────────────────────── */

function UsersTab({ currentUserId }: { currentUserId: string }) {
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

  /** Keep toggle-admin working for old code paths (unused in new UI, but safe to keep) */
  const handleToggleAdmin = useCallback(
    async (userId: string, currentRole: string) => {
      const next = currentRole === "admin" ? "user" : "admin";
      await handleSetRole(userId, next);
    },
    [handleSetRole]
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

/* ── Parquet status tab (admin only) ─────────────────────────────────────── */

interface MissingFile {
  file_id: string;
  name: string;
  blob_path: string;
  has_analytics: boolean;
  job_status: string | null;
  job_error: string | null;
  last_attempt: string | null;
}

function ParquetTab() {
  const { data, error, isLoading, mutate } = useSWR<{ files: MissingFile[]; count: number }>(
    "/api/admin/missing-parquet",
    (url: string) => apiFetch(url).then((r) => r.json()),
    { refreshInterval: 0 },
  );
  const [retrying, setRetrying] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);

  // Auto-refresh after retry: poll at 5s, 12s, 25s, 45s intervals
  useEffect(() => {
    if (pollCount === 0) return;
    const delays = [5000, 12000, 25000, 45000];
    const timers = delays.map((d) => setTimeout(() => mutate(), d));
    return () => timers.forEach(clearTimeout);
  }, [pollCount, mutate]);

  const retryAll = useCallback(async () => {
    setRetrying(true);
    setResult(null);
    try {
      const res = await apiFetch("/api/admin/retry-parquet", { method: "POST" });
      const body = await res.json();
      const total = body.total ?? body.missing_parquet ?? body.count ?? 0;
      setResult((body.message ?? "Started") + ` (${total} files)`);
      setPollCount((n) => n + 1);
    } catch {
      setResult("Failed to start retry");
    } finally {
      setRetrying(false);
    }
  }, [mutate]);

  if (isLoading) return <div className="flex justify-center py-12"><Loader2 className="w-6 h-6 animate-spin text-[#a3a3a3]" /></div>;
  if (error) return <p className="text-[#dc2626] p-4 text-[13px]">Failed to load parquet status.</p>;

  const files = data?.files ?? [];

  return (
    <div className="space-y-4 max-w-xl">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest mb-1">Parquet status</p>
          <p className="text-[13px] text-[#737373]">{files.length} file{files.length !== 1 ? "s" : ""} without parquet</p>
        </div>
        {files.length > 0 && (
          <button
            onClick={retryAll}
            disabled={retrying}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] font-medium text-white disabled:opacity-50 hover:opacity-90 transition-opacity"
            style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
          >
            {retrying ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            Retry All
          </button>
        )}
      </div>

      {result && (
        <div className="flex items-center gap-2 p-3 bg-[#0a0a0a]/5 border border-[#0a0a0a]/10 rounded-xl text-[13px] text-[#0a0a0a]">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          {result}
        </div>
      )}

      {files.length === 0 ? (
        <div className="flex flex-col items-center py-12 text-[#a3a3a3]">
          <CheckCircle2 className="w-8 h-8 mb-2 text-green-500" />
          <p className="text-[13px]">All files have parquet conversions.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {files.map((f) => (
            <div key={f.file_id} className="p-3 bg-[#f9f9f9] rounded-xl border border-[#e5e5e5] space-y-1">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[13px] font-medium text-[#0a0a0a] truncate min-w-0">{f.name}</p>
                {f.job_status === "failed" && (
                  <span className="shrink-0 text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-[#dc2626]/8 text-[#dc2626] border border-[#dc2626]/15">failed</span>
                )}
                {f.job_status === "running" && (
                  <span className="shrink-0 flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-600 border border-amber-500/20">
                    <Loader2 className="w-2.5 h-2.5 animate-spin" />running
                  </span>
                )}
                {!f.job_status && (
                  <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
                )}
              </div>
              <p className="text-[11px] text-[#737373] truncate">{f.blob_path}</p>
              {f.job_error && (
                <p className="text-[11px] text-[#dc2626] break-all">{f.job_error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Domains tab (admin only) ────────────────────────────────────────────── */

function DomainsTab() {
  const { data: domains, mutate: mutateDomains } = useSWR<string[]>(
    "admin-domains",
    domainsFetcher,
    { revalidateOnFocus: false }
  );
  const { data: users, mutate: mutateUsers } = useSWR<UserItem[]>(
    "users-list",
    usersFetcher,
    { revalidateOnFocus: false }
  );

  // Create domain
  const [newDomain, setNewDomain] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  // Per-user domain assignment
  const [pendingDomains, setPendingDomains] = useState<Record<string, string[]>>({});
  const [savingUser, setSavingUser] = useState<string | null>(null);

  const handleCreateDomain = useCallback(async () => {
    const name = newDomain.trim();
    if (!name) return;
    setCreating(true);
    setCreateError(null);
    try {
      const res = await apiFetch("/api/admin/domains", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setCreateError(body.detail ?? "Failed to create domain");
      } else {
        setNewDomain("");
        mutateDomains();
      }
    } catch {
      setCreateError("Network error");
    } finally {
      setCreating(false);
    }
  }, [newDomain, mutateDomains]);

  const getUserDomains = (userId: string, fallback: string[] | null) =>
    pendingDomains[userId] ?? fallback ?? [];

  const toggleUserDomain = (userId: string, domain: string, current: string[] | null) => {
    const base = pendingDomains[userId] ?? current ?? [];
    const next = base.includes(domain)
      ? base.filter((d) => d !== domain)
      : [...base, domain];
    setPendingDomains((p) => ({ ...p, [userId]: next }));
  };

  const handleSaveUserDomains = useCallback(
    async (userId: string) => {
      setSavingUser(userId);
      const domains = pendingDomains[userId] ?? [];
      try {
        await apiFetch(`/api/admin/users/${userId}/domains`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ allowed_domains: domains.length > 0 ? domains : null }),
        });
        setPendingDomains((p) => {
          const next = { ...p };
          delete next[userId];
          return next;
        });
        mutateUsers();
      } finally {
        setSavingUser(null);
      }
    },
    [pendingDomains, mutateUsers]
  );

  const isLoading = !domains || !users;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-40">
        <Loader2 className="w-5 h-5 text-[#a3a3a3] animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-2xl">

      {/* ── Create domain ── */}
      <section className="space-y-3">
        <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">Create Department</p>
        <p className="text-[12px] text-[#737373] leading-relaxed">
          Adding a department creates a top-level folder tagged with that name. Users can then be assigned to it.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={newDomain}
            onChange={(e) => setNewDomain(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreateDomain()}
            placeholder="e.g. Finance, HR, Engineering"
            className="flex-1 px-3 py-2 text-[13px] rounded-xl border border-[#e5e5e5] bg-[#f9f9f9] text-[#0a0a0a] placeholder:text-[#a3a3a3] focus:outline-none focus:ring-2 focus:ring-[#0a0a0a]/10 focus:border-[#a3a3a3] transition-colors"
          />
          <button
            onClick={handleCreateDomain}
            disabled={!newDomain.trim() || creating}
            className="flex items-center gap-1.5 px-4 py-2 text-[13px] font-medium text-white rounded-xl disabled:opacity-40 hover:opacity-90 transition-opacity"
            style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
          >
            {creating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
            Add
          </button>
        </div>
        {createError && <p className="text-[12px] text-[#dc2626]">{createError}</p>}

        {domains.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {domains.map((d) => (
              <span
                key={d}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[12px] font-medium bg-[#f4f4f4] text-[#0a0a0a] border border-[#e5e5e5]"
              >
                <Tag className="w-3 h-3 text-[#a3a3a3]" />
                {d}
              </span>
            ))}
          </div>
        )}
      </section>

      {/* ── User domain assignments ── */}
      <section className="space-y-3">
        <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">User Access</p>
        <p className="text-[12px] text-[#737373] leading-relaxed">
          Assign departments to users. Users with no departments are unrestricted.
        </p>

        {domains.length === 0 ? (
          <p className="text-[13px] text-[#a3a3a3] py-4 text-center">
            No departments yet. Create one above first.
          </p>
        ) : (
          <div className="space-y-3">
            {users.filter((u) => !u.is_admin).map((u) => {
              const current = getUserDomains(u.id, u.allowed_domains);
              const isDirty = !!pendingDomains[u.id];

              return (
                <div
                  key={u.id}
                  className="p-4 rounded-xl border border-[#e5e5e5] bg-[#f9f9f9] space-y-3"
                >
                  <div className="flex items-center gap-3">
                    {u.picture ? (
                      <img src={u.picture} alt="" className="w-8 h-8 rounded-full border-2 border-white shadow-sm" referrerPolicy="no-referrer" />
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-[#f4f4f4] border border-[#e5e5e5] flex items-center justify-center">
                        <UserCircle className="w-4 h-4 text-[#a3a3a3]" />
                      </div>
                    )}
                    <div className="min-w-0">
                      <p className="text-[13px] font-medium text-[#0a0a0a] truncate">{u.name || u.email}</p>
                      <p className="text-[11px] text-[#737373] truncate">{u.email}</p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1.5">
                    {domains.map((d) => {
                      const active = current.includes(d);
                      return (
                        <button
                          key={d}
                          onClick={() => toggleUserDomain(u.id, d, u.allowed_domains)}
                          className={cn(
                            "inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[12px] font-medium border transition-all",
                            active
                              ? "bg-[#0a0a0a]/8 text-[#0a0a0a] border-[#0a0a0a]/20"
                              : "bg-white text-[#737373] border-[#e5e5e5] hover:border-[#a3a3a3] hover:text-[#0a0a0a]"
                          )}
                        >
                          {active && <CheckCircle2 className="w-3 h-3" />}
                          {d}
                        </button>
                      );
                    })}
                  </div>

                  {current.length === 0 && (
                    <p className="text-[11px] text-amber-600">No departments — user sees all data</p>
                  )}

                  {isDirty && (
                    <button
                      onClick={() => handleSaveUserDomains(u.id)}
                      disabled={savingUser === u.id}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-white rounded-lg disabled:opacity-50 hover:opacity-90 transition-opacity"
                      style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
                    >
                      {savingUser === u.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle2 className="w-3 h-3" />}
                      Save changes
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* ── Department file assignment ── */}
      <section className="space-y-3">
        <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">File Assignment by Department</p>
        <p className="text-[12px] text-[#737373] leading-relaxed">
          Select a department to see its files. Use AI Sort to auto-assign based on file content, or add files manually.
        </p>

        {domains.length === 0 ? (
          <p className="text-[13px] text-[#a3a3a3] py-4 text-center">
            No departments yet. Create one above first.
          </p>
        ) : (
          <div className="space-y-2">
            {domains.map((domain) => (
              <DepartmentFilePanel key={domain} domain={domain} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

/* ── Department file panel ────────────────────────────────────────────────── */

function DepartmentFilePanel({ domain }: { domain: string }) {
  const [expanded, setExpanded] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResult, setAiResult] = useState<{ assigned_count: number; assigned_files: string[] } | null>(null);
  const [showPicker, setShowPicker] = useState(false);

  const { data, mutate, isLoading } = useSWR<{ files: DeptFile[]; count: number }>(
    expanded ? `/api/admin/departments/${encodeURIComponent(domain)}/files` : null,
    (url: string) => apiFetch(url).then((r) => r.json()),
  );

  const handleAiSort = async () => {
    setAiLoading(true);
    setAiResult(null);
    try {
      const res = await apiFetch(`/api/admin/departments/${encodeURIComponent(domain)}/ai-assign`, {
        method: "POST",
      });
      const body = await res.json();
      setAiResult(body);
      mutate();
    } catch {
      setAiResult({ assigned_count: -1, assigned_files: [] });
    } finally {
      setAiLoading(false);
    }
  };

  const handleRemoveFile = async (fileId: string) => {
    await apiFetch(`/api/admin/departments/${encodeURIComponent(domain)}/files/${fileId}`, {
      method: "DELETE",
    });
    mutate();
  };

  return (
    <div className="rounded-xl border border-[#e5e5e5] bg-[#f9f9f9] overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f4f4f4] transition-colors"
      >
        <div className="flex items-center gap-2">
          <FolderOpen className="w-4 h-4 text-[#0a0a0a]" />
          <span className="text-[13px] font-medium text-[#0a0a0a]">{domain}</span>
          {data && (
            <span className="text-[11px] text-[#a3a3a3]">({data.count} file{data.count !== 1 ? "s" : ""})</span>
          )}
        </div>
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-[#a3a3a3]" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[#a3a3a3]" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-[#e5e5e5]">
          <div className="flex gap-2 pt-3">
            <button
              onClick={handleAiSort}
              disabled={aiLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-white rounded-lg disabled:opacity-50 hover:opacity-90 transition-opacity"
              style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
            >
              {aiLoading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              AI Sort
            </button>
            <button
              onClick={() => setShowPicker(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] border border-[#e5e5e5] text-[#737373] rounded-lg hover:text-[#0a0a0a] hover:border-[#a3a3a3] transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              Add Files
            </button>
          </div>

          {aiResult && (
            <div className={cn(
              "flex items-start gap-2 p-3 rounded-xl text-[12px] border",
              aiResult.assigned_count === -1
                ? "bg-[#dc2626]/5 border-[#dc2626]/15 text-[#dc2626]"
                : "bg-[#0a0a0a]/5 border-[#0a0a0a]/10 text-[#0a0a0a]"
            )}>
              {aiResult.assigned_count === -1 ? (
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              ) : (
                <Sparkles className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              )}
              <div>
                {aiResult.assigned_count === -1
                  ? "AI sort failed — please try again."
                  : aiResult.assigned_count === 0
                  ? "No matching files found for this department."
                  : (
                    <>
                      <span className="font-medium">AI assigned {aiResult.assigned_count} file{aiResult.assigned_count !== 1 ? "s" : ""}:</span>{" "}
                      {aiResult.assigned_files.join(", ")}
                    </>
                  )}
              </div>
            </div>
          )}

          {isLoading ? (
            <div className="flex justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin text-[#a3a3a3]" />
            </div>
          ) : data && data.files.length === 0 ? (
            <p className="text-[12px] text-[#a3a3a3] py-2">No files assigned yet.</p>
          ) : (
            <div className="space-y-1.5">
              {(data?.files ?? []).map((f) => (
                <div
                  key={f.file_id}
                  className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-white border border-[#e5e5e5]"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="w-3.5 h-3.5 text-[#a3a3a3] shrink-0" />
                    <span className="text-[12px] text-[#0a0a0a] truncate">{f.name}</span>
                    {f.ingest_status !== "ingested" && (
                      <span className="text-[10px] text-amber-600 shrink-0">{f.ingest_status}</span>
                    )}
                  </div>
                  <button
                    onClick={() => handleRemoveFile(f.file_id)}
                    className="p-1 text-[#a3a3a3] hover:text-[#dc2626] transition-colors shrink-0"
                    title="Remove from department"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {showPicker && (
        <FilePickerModal
          domain={domain}
          onClose={() => setShowPicker(false)}
          onAssigned={() => { mutate(); setShowPicker(false); }}
        />
      )}
    </div>
  );
}

/* ── File picker modal ────────────────────────────────────────────────────── */

function FilePickerModal({
  domain,
  onClose,
  onAssigned,
}: {
  domain: string;
  onClose: () => void;
  onAssigned: () => void;
}) {
  const [files, setFiles] = useState<EligibleFile[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    apiFetch("/api/admin/files/eligible")
      .then((r) => r.json())
      .then((d) => {
        // Show only files not already in this domain, sorted by name
        const eligible = (d.files as EligibleFile[]).filter(
          (f) => f.current_domain !== domain
        );
        setFiles(eligible);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [domain]);

  const toggle = (fileId: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(fileId)) next.delete(fileId);
      else next.add(fileId);
      return next;
    });

  const handleSave = async () => {
    if (selected.size === 0) return;
    setSaving(true);
    try {
      await apiFetch(`/api/admin/departments/${encodeURIComponent(domain)}/assign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_ids: Array.from(selected) }),
      });
      onAssigned();
    } catch {
      setSaving(false);
    }
  };

  const filtered = files.filter((f) =>
    f.name.toLowerCase().includes(search.toLowerCase()) ||
    (f.ai_description ?? "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 4 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="w-full max-w-lg bg-white border border-[#e5e5e5] rounded-2xl shadow-xl flex flex-col max-h-[80vh]"
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#e5e5e5]">
          <div>
            <h2 className="text-[15px] font-bold text-[#0a0a0a]" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>Add Files to {domain}</h2>
            <p className="text-[11px] text-[#a3a3a3] mt-0.5">
              {selected.size} file{selected.size !== 1 ? "s" : ""} selected
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-[#a3a3a3] hover:text-[#0a0a0a] hover:bg-[#f4f4f4] transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-5 py-3 border-b border-[#e5e5e5]">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or description..."
            className="w-full px-3 py-1.5 text-[13px] rounded-xl border border-[#e5e5e5] bg-[#f9f9f9] text-[#0a0a0a] placeholder:text-[#a3a3a3] focus:outline-none focus:ring-2 focus:ring-[#0a0a0a]/10 focus:border-[#a3a3a3] transition-colors"
          />
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-1.5">
          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="w-5 h-5 animate-spin text-[#a3a3a3]" />
            </div>
          ) : filtered.length === 0 ? (
            <p className="text-[13px] text-[#a3a3a3] text-center py-8">
              {search ? "No files match your search." : "All files are already assigned to this department."}
            </p>
          ) : (
            filtered.map((f) => {
              const active = selected.has(f.file_id);
              return (
                <button
                  key={f.file_id}
                  onClick={() => toggle(f.file_id)}
                  className={cn(
                    "w-full flex items-start gap-3 px-3 py-2.5 rounded-xl border text-left transition-all",
                    active
                      ? "border-[#0a0a0a]/20 bg-[#0a0a0a]/5"
                      : "border-[#e5e5e5] hover:border-[#a3a3a3] bg-[#f9f9f9]"
                  )}
                >
                  <div className={cn(
                    "w-4 h-4 mt-0.5 rounded border-2 shrink-0 flex items-center justify-center",
                    active ? "bg-[#0a0a0a] border-[#0a0a0a]" : "border-[#a3a3a3]"
                  )}>
                    {active && <CheckCircle2 className="w-3 h-3 text-white" />}
                  </div>
                  <div className="min-w-0">
                    <p className="text-[12px] font-medium text-[#0a0a0a] truncate">{f.name}</p>
                    {f.current_domain && (
                      <p className="text-[10px] text-amber-600">Currently in: {f.current_domain}</p>
                    )}
                    {f.ai_description && (
                      <p className="text-[10px] text-[#737373] line-clamp-1 mt-0.5">{f.ai_description}</p>
                    )}
                    {f.good_for && f.good_for.length > 0 && (
                      <p className="text-[10px] text-[#737373] mt-0.5 line-clamp-1">
                        Good for: {f.good_for.slice(0, 3).join(", ")}
                      </p>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="flex gap-2 px-5 py-4 border-t border-[#e5e5e5]">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-xl border border-[#e5e5e5] text-[13px] text-[#737373] hover:text-[#0a0a0a] hover:border-[#a3a3a3] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={selected.size === 0 || saving}
            className="flex-1 py-2 rounded-xl text-[13px] font-medium text-white disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center justify-center gap-1.5"
            style={{ background: "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)", boxShadow: "inset 0 1px 0 rgba(255,255,255,0.12)" }}
          >
            {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            Assign {selected.size > 0 ? `(${selected.size})` : ""}
          </button>
        </div>
      </motion.div>
    </div>
  );
}

"use client";

import { useState, useCallback, useEffect } from "react";
import useSWR from "swr";
import { motion } from "framer-motion";
import {
  UserCircle,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Tag,
  X,
  Plus,
  Sparkles,
  FolderOpen,
  FileText,
  ChevronDown,
  ChevronRight,
  Trash2,
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
            <h2 className="text-[15px] font-bold text-[#0a0a0a] tracking-tight">Add Files to {domain}</h2>
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

/* ── DomainsTabContent (admin only) ──────────────────────────────────────── */

export default function DomainsTabContent() {
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
      const userDomains = pendingDomains[userId] ?? [];
      try {
        await apiFetch(`/api/admin/users/${userId}/domains`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ allowed_domains: userDomains.length > 0 ? userDomains : null }),
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

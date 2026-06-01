"use client";

import { useState, useEffect } from "react";
import { UserCircle, Tag, CheckCircle2, Loader2, X } from "lucide-react";
import { motion } from "framer-motion";
import { useAuth } from "@/components/auth-provider";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";

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

function DomainPickerModal({ current, onClose }: { current: string[]; onClose: () => void }) {
  const [domains, setDomains] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>(current);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiFetch("/api/users/domains").then((r) => r.json()).then((d) => setDomains(d.domains ?? [])).catch(() => {});
  }, []);

  const toggle = (d: string) =>
    setSelected((prev) => prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiFetch("/api/users/me/domains", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ allowed_domains: selected.length > 0 ? selected : null }),
      });
      window.location.reload();
    } catch { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
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
              <button key={d} onClick={() => toggle(d)}
                className={cn("flex items-center gap-2 px-3 py-2 rounded-lg border text-[13px] text-left transition-all",
                  active ? "border-[#0a0a0a] bg-[#0a0a0a]/5 text-[#0a0a0a] font-medium" : "border-[#e5e5e5] text-[#737373] hover:text-[#0a0a0a] hover:border-[#a3a3a3]"
                )}
              >
                {active && <CheckCircle2 className="w-3.5 h-3.5 text-[#0a0a0a] shrink-0" />}
                <span className="truncate">{d}</span>
              </button>
            );
          })}
        </div>
        <div className="flex gap-2 pt-1">
          <button onClick={onClose} className="flex-1 py-2 rounded-lg border border-[#e5e5e5] text-[13px] text-[#737373] hover:text-[#0a0a0a] transition-colors">Cancel</button>
          <button onClick={handleSave} disabled={saving}
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

export default function ProfileTabContent() {
  const { user } = useAuth();
  const [editingDomains, setEditingDomains] = useState(false);

  if (!user) return null;

  const roleLabel = user.is_admin ? "Admin" : user.role === "developer" ? "Developer" : user.role === "manager" ? "Manager" : "Member";
  const roleCls = user.is_admin ? "bg-[#0a0a0a] text-white"
    : user.role === "developer" ? "bg-violet-100 text-violet-700"
    : user.role === "manager" ? "bg-cyan-100 text-cyan-700"
    : "bg-[#f4f4f4] text-[#737373]";

  return (
    <div className="max-w-lg space-y-6">
      {/* Hero card */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}
        className="relative overflow-hidden rounded-2xl border border-[#e5e5e5] bg-[#f9f9f9] p-6"
      >
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: "linear-gradient(to right, #e5e5e5 1px, transparent 1px), linear-gradient(to bottom, #e5e5e5 1px, transparent 1px)",
          backgroundSize: "28px 28px", opacity: 0.4,
          maskImage: "radial-gradient(ellipse 80% 80% at 50% 0%, black, transparent)",
        }} />
        <div className="relative flex items-center gap-5">
          <div className="relative shrink-0">
            {user.picture
              ? <img src={user.picture} alt="" className="w-16 h-16 rounded-2xl border-2 border-white shadow-md" referrerPolicy="no-referrer" />
              : <div className="w-16 h-16 rounded-2xl bg-[#ebebeb] border-2 border-white shadow-md flex items-center justify-center"><UserCircle className="w-8 h-8 text-[#a3a3a3]" /></div>
            }
            <span className={cn("absolute -bottom-1 -right-1 text-[10px] font-bold px-1.5 py-0.5 rounded-full border-2 border-white", roleCls)}>{roleLabel}</span>
          </div>
          <div className="min-w-0">
            <p className="text-[18px] font-bold text-[#0a0a0a] truncate" style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.02em" }}>{user.name || "—"}</p>
            <p className="text-[13px] text-[#737373] truncate mt-0.5">{user.email}</p>
          </div>
        </div>
      </motion.div>

      {/* Fields */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08, duration: 0.32 }} className="space-y-2">
        <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest mb-3">Account details</p>
        <Field label="Full name" value={user.name || "—"} />
        <Field label="Email address" value={user.email} />
        <Field label="Role" value={roleLabel} />
      </motion.div>

      {/* Departments */}
      {!user.is_admin && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14, duration: 0.32 }} className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold text-[#a3a3a3] uppercase tracking-widest">My departments</p>
            <button onClick={() => setEditingDomains(true)} className="text-[12px] font-medium text-[#0a0a0a] hover:text-[#525252] transition-colors underline underline-offset-2">Edit</button>
          </div>
          {user.allowed_domains && user.allowed_domains.length > 0
            ? <div className="flex flex-wrap gap-1.5">
                {user.allowed_domains.map((d) => (
                  <span key={d} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[12px] font-medium bg-[#f4f4f4] text-[#0a0a0a] border border-[#e5e5e5]">
                    <Tag className="w-3 h-3 text-[#a3a3a3]" />{d}
                  </span>
                ))}
              </div>
            : <p className="text-[13px] text-[#a3a3a3]">No departments assigned.</p>
          }
        </motion.div>
      )}

      {editingDomains && <DomainPickerModal current={user.allowed_domains ?? []} onClose={() => setEditingDomains(false)} />}
    </div>
  );
}

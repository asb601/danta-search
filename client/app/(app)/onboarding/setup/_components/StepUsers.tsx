"use client";

// Step 5 — User management.
//   POST /api/onboarding/users          (single user)
//   POST /api/onboarding/users/bulk     (multipart .xlsx)
// Add users (email, role in {admin, manager, user}, domains for manager/user).

import { useRef, useState } from "react";
import {
  Plus,
  Trash2,
  Loader2,
  UploadCloud,
  FileSpreadsheet,
  Mail,
} from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { Field, FormError, FormSuccess, SubmitButton } from "./FormBits";
import { safeError } from "../_lib/api";
import type {
  OnboardingDomain,
  OnboardingState,
  OnboardingUser,
} from "../_lib/types";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const ROLES: OnboardingUser["role"][] = ["admin", "manager", "user"];

export function StepUsers({
  state,
  onSaved,
}: {
  state: OnboardingState | null;
  onSaved: () => void;
}) {
  const availableDomains: OnboardingDomain[] = state?.domains ?? [];

  const [users, setUsers] = useState<OnboardingUser[]>(
    () =>
      (state?.users ?? []).map((u) => ({
        email: u.email,
        role: (ROLES.includes(u.role as OnboardingUser["role"])
          ? u.role
          : "user") as OnboardingUser["role"],
        domains: u.domains ?? [],
      })),
  );

  const [email, setEmail] = useState("");
  const [role, setRole] = useState<OnboardingUser["role"]>("user");
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // bulk upload
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const needsDomains = role === "manager" || role === "user";

  const toggleDomain = (name: string) => {
    setSelectedDomains((prev) =>
      prev.includes(name) ? prev.filter((d) => d !== name) : [...prev, name],
    );
  };

  const addUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const em = email.trim().toLowerCase();
    if (!EMAIL_RE.test(em)) {
      setError("Enter a valid email address.");
      return;
    }
    if (users.some((u) => u.email === em)) {
      setError(`${em} is already added.`);
      return;
    }
    if (needsDomains && selectedDomains.length === 0) {
      setError(`Assign at least one domain for a ${role}.`);
      return;
    }

    setAdding(true);
    try {
      const payload: OnboardingUser = {
        email: em,
        role,
        domains: needsDomains ? selectedDomains : [],
      };
      const res = await apiFetch("/api/onboarding/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await safeError(res));
      setUsers((prev) => [...prev, payload]);
      setEmail("");
      setSelectedDomains([]);
      setRole("user");
      setSuccess(`Invited ${em}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add user.");
    } finally {
      setAdding(false);
    }
  };

  const handleFile = async (file: File) => {
    setError(null);
    setSuccess(null);
    if (!/\.xlsx$/i.test(file.name)) {
      setError("Please upload a .xlsx file.");
      return;
    }
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await apiFetch("/api/onboarding/users/bulk", {
        method: "POST",
        body: form, // browser sets multipart boundary
      });
      if (!res.ok) throw new Error(await safeError(res));
      let imported: OnboardingUser[] = [];
      try {
        const data = await res.json();
        if (Array.isArray(data?.users)) imported = data.users;
        else if (Array.isArray(data)) imported = data;
      } catch {
        /* server may return a count only */
      }
      if (imported.length) {
        setUsers((prev) => {
          const seen = new Set(prev.map((u) => u.email));
          const merged = [...prev];
          for (const u of imported) {
            if (u.email && !seen.has(u.email.toLowerCase())) {
              merged.push({
                email: u.email.toLowerCase(),
                role: (ROLES.includes(u.role) ? u.role : "user"),
                domains: u.domains ?? [],
              });
              seen.add(u.email.toLowerCase());
            }
          }
          return merged;
        });
      }
      setSuccess(
        imported.length
          ? `Imported ${imported.length} user${imported.length === 1 ? "" : "s"}.`
          : "Bulk upload processed.",
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Bulk upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const removeLocal = (em: string) =>
    setUsers((prev) => prev.filter((u) => u.email !== em));

  return (
    <div className="space-y-7 max-w-2xl">
      <FormError message={error} />
      <FormSuccess message={success} />

      {/* ── Single add ─────────────────────────────── */}
      <form onSubmit={addUser} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field label="Email" htmlFor="user-email" required>
            <input
              id="user-email"
              type="email"
              className="field-input"
              placeholder="teammate@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </Field>
          <Field label="Role" htmlFor="user-role" required>
            <select
              id="user-role"
              className="field-input appearance-none cursor-pointer"
              value={role}
              onChange={(e) =>
                setRole(e.target.value as OnboardingUser["role"])
              }
            >
              <option value="admin">Admin — full org access</option>
              <option value="manager">Manager — assigned domains</option>
              <option value="user">User — assigned domains</option>
            </select>
          </Field>
        </div>

        {needsDomains && (
          <Field
            label="Assigned domains"
            required
            hint="This user can only see data in the selected domains."
          >
            {availableDomains.length === 0 ? (
              <p className="text-[12px] text-[color:var(--fg-subtle)]">
                No domains created yet — go back to the Domains step first.
              </p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {availableDomains.map((d) => {
                  const on = selectedDomains.includes(d.name);
                  return (
                    <button
                      key={d.name}
                      type="button"
                      onClick={() => toggleDomain(d.name)}
                      aria-pressed={on}
                      className={cn(
                        "px-3 py-1.5 rounded-full text-[12px] font-medium border transition-colors",
                        on
                          ? "bg-[#0a0a0a] text-white border-[#0a0a0a]"
                          : "bg-white text-[color:var(--fg-muted)] border-[#e5e5e5] hover:border-[#c4c4c4]",
                      )}
                    >
                      {d.label || d.name}
                    </button>
                  );
                })}
              </div>
            )}
          </Field>
        )}

        <button type="submit" className="btn-outline h-9 px-4 gap-1.5" disabled={adding}>
          {adding ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Plus className="w-3.5 h-3.5" />
          )}
          Add user
        </button>
      </form>

      {/* ── Bulk upload ────────────────────────────── */}
      <div>
        <p className="section-label mb-2">Bulk import</p>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
          className={cn(
            "rounded-[var(--radius-lg)] border-2 border-dashed px-5 py-6 text-center transition-colors",
            dragging
              ? "border-[#0a0a0a] bg-[color:var(--surface)]"
              : "border-[#e5e5e5] bg-[color:var(--surface)]",
          )}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
          <div className="flex flex-col items-center gap-2">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-white border border-[#e5e5e5] flex items-center justify-center">
              {uploading ? (
                <Loader2 className="w-4 h-4 animate-spin text-[color:var(--fg-muted)]" />
              ) : (
                <FileSpreadsheet className="w-4 h-4 text-[color:var(--fg-muted)]" />
              )}
            </div>
            <p className="text-[12.5px] text-[color:var(--fg-muted)]">
              Drag an Excel file here, or
            </p>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="btn-outline h-8 px-3.5 gap-1.5"
            >
              <UploadCloud className="w-3.5 h-3.5" />
              {uploading ? "Uploading…" : "Choose .xlsx"}
            </button>
            <p className="text-[11px] text-[color:var(--fg-subtle)]">
              Columns: email, role, domains (comma-separated)
            </p>
          </div>
        </div>
      </div>

      {/* ── List ───────────────────────────────────── */}
      <div>
        <p className="section-label mb-2">Users ({users.length})</p>
        {users.length === 0 ? (
          <div className="rounded-[var(--radius)] border border-dashed border-[#e5e5e5] px-4 py-6 text-center text-[12.5px] text-[color:var(--fg-subtle)]">
            No users added yet.
          </div>
        ) : (
          <ul className="space-y-2">
            {users.map((u) => (
              <li
                key={u.email}
                className="flex items-center gap-3 rounded-[var(--radius)] border border-[#e5e5e5] bg-white px-3.5 py-2.5"
              >
                <Mail className="w-4 h-4 text-[color:var(--fg-subtle)] shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-[color:var(--fg)] truncate">
                    {u.email}
                  </p>
                  {u.domains && u.domains.length > 0 && (
                    <p className="text-[11.5px] text-[color:var(--fg-subtle)] truncate">
                      {u.domains.join(", ")}
                    </p>
                  )}
                </div>
                <span className="badge-muted capitalize">{u.role}</span>
                <button
                  type="button"
                  onClick={() => removeLocal(u.email)}
                  aria-label={`Remove ${u.email}`}
                  className="btn-ghost p-1.5"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="pt-1">
        <SubmitButton type="button" onClick={onSaved}>
          Continue
        </SubmitButton>
      </div>
    </div>
  );
}

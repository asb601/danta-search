"use client";

// Step 4 — Create domains. POST /api/onboarding/domains
// Add one or more data domains (finance / hr / …).

import { useState } from "react";
import { Plus, Trash2, Tag, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { Field, FormError, FormSuccess, SubmitButton } from "./FormBits";
import { safeError } from "../_lib/api";
import type { OnboardingDomain, OnboardingState } from "../_lib/types";

const NAME_RE = /^[a-z0-9][a-z0-9_-]*$/;

export function StepDomains({
  state,
  onSaved,
}: {
  state: OnboardingState | null;
  onSaved: () => void;
}) {
  const [domains, setDomains] = useState<OnboardingDomain[]>(
    () => state?.domains ?? [],
  );
  const [name, setName] = useState("");
  const [label, setLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [finishing, setFinishing] = useState(false);

  const addDomain = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const n = name.trim().toLowerCase();
    if (!n) {
      setError("Enter a domain name.");
      return;
    }
    if (!NAME_RE.test(n)) {
      setError(
        "Use lowercase letters, numbers, hyphens or underscores (e.g. finance).",
      );
      return;
    }
    if (domains.some((d) => d.name === n)) {
      setError(`Domain "${n}" already added.`);
      return;
    }

    setAdding(true);
    try {
      const res = await apiFetch("/api/onboarding/domains", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: n, label: label.trim() || null }),
      });
      if (!res.ok) throw new Error(await safeError(res));
      let created: OnboardingDomain = { name: n, label: label.trim() || null };
      try {
        const data = await res.json();
        if (data?.name) created = data as OnboardingDomain;
      } catch {
        /* server may return 204 */
      }
      setDomains((prev) => [...prev, created]);
      setName("");
      setLabel("");
      setSuccess(`Added "${n}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add domain.");
    } finally {
      setAdding(false);
    }
  };

  const removeLocal = (n: string) => {
    setDomains((prev) => prev.filter((d) => d.name !== n));
  };

  const finish = () => {
    if (domains.length === 0) {
      setError("Add at least one domain before continuing.");
      return;
    }
    setFinishing(true);
    onSaved();
  };

  return (
    <div className="space-y-6 max-w-xl">
      <FormError message={error} />
      <FormSuccess message={success} />

      {/* Add form */}
      <form onSubmit={addDomain} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field
            label="Domain name"
            htmlFor="domain-name"
            required
            hint="Lowercase identifier, e.g. finance."
          >
            <input
              id="domain-name"
              type="text"
              className="field-input"
              placeholder="finance"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>
          <Field label="Display label" htmlFor="domain-label">
            <input
              id="domain-label"
              type="text"
              className="field-input"
              placeholder="Finance & Accounting"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </Field>
        </div>
        <button type="submit" className="btn-outline h-9 px-4 gap-1.5" disabled={adding}>
          {adding ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Plus className="w-3.5 h-3.5" />
          )}
          Add domain
        </button>
      </form>

      {/* List */}
      <div>
        <p className="section-label mb-2">
          Domains ({domains.length})
        </p>
        {domains.length === 0 ? (
          <div className="rounded-[var(--radius)] border border-dashed border-[#e5e5e5] px-4 py-6 text-center text-[12.5px] text-[color:var(--fg-subtle)]">
            No domains yet. Add at least one to continue.
          </div>
        ) : (
          <ul className="space-y-2">
            {domains.map((d) => (
              <li
                key={d.name}
                className="flex items-center gap-3 rounded-[var(--radius)] border border-[#e5e5e5] bg-white px-3.5 py-2.5"
              >
                <Tag className="w-4 h-4 text-[color:var(--fg-subtle)] shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-[color:var(--fg)] truncate">
                    {d.label || d.name}
                  </p>
                  <p className="text-[11.5px] text-[color:var(--fg-subtle)] font-mono truncate">
                    {d.name}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => removeLocal(d.name)}
                  aria-label={`Remove ${d.name}`}
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
        <SubmitButton
          type="button"
          onClick={finish}
          loading={finishing}
          disabled={domains.length === 0}
        >
          Continue
        </SubmitButton>
      </div>
    </div>
  );
}

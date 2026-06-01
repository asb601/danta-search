"use client";

// Step 3 — Connect storage. POST /api/onboarding/storage
// Azure storage connection string + container name.

import { useState } from "react";
import { apiFetch } from "@/lib/auth";
import { Field, FormError, FormSuccess, SubmitButton } from "./FormBits";
import { safeError } from "../_lib/api";
import type { OnboardingState } from "../_lib/types";

export function StepStorage({
  state,
  onSaved,
}: {
  state: OnboardingState | null;
  onSaved: () => void;
}) {
  const pre = state?.storage ?? null;

  const [connectionString, setConnectionString] = useState("");
  const [containerName, setContainerName] = useState(pre?.container_name ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const alreadyConfigured = !!pre?.configured;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!alreadyConfigured && !connectionString.trim()) {
      setError("A storage connection string is required.");
      return;
    }
    if (!containerName.trim()) {
      setError("A container name is required.");
      return;
    }

    setSubmitting(true);
    try {
      const body: Record<string, string> = {
        container_name: containerName.trim(),
      };
      if (connectionString.trim())
        body.connection_string = connectionString.trim();

      const res = await apiFetch("/api/onboarding/storage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await safeError(res));
      setSuccess("Storage connected.");
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect storage.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5 max-w-xl">
      <FormError message={error} />
      <FormSuccess message={success} />

      <Field
        label="Azure connection string"
        htmlFor="conn-str"
        required={!alreadyConfigured}
        hint="Found under Storage account → Access keys. Stored encrypted; never logged."
      >
        <textarea
          id="conn-str"
          className="field-textarea font-mono text-[12px]"
          rows={3}
          autoComplete="off"
          spellCheck={false}
          placeholder={
            alreadyConfigured
              ? "•••••••• (unchanged)"
              : "DefaultEndpointsProtocol=https;AccountName=…;AccountKey=…;EndpointSuffix=core.windows.net"
          }
          value={connectionString}
          onChange={(e) => setConnectionString(e.target.value)}
        />
      </Field>

      <Field
        label="Container name"
        htmlFor="container-name"
        required
        hint="The blob container that holds this organization's datasets."
      >
        <input
          id="container-name"
          type="text"
          className="field-input"
          placeholder="datasets"
          value={containerName}
          onChange={(e) => setContainerName(e.target.value)}
        />
      </Field>

      <div className="pt-1">
        <SubmitButton loading={submitting}>Connect &amp; continue</SubmitButton>
      </div>
    </form>
  );
}

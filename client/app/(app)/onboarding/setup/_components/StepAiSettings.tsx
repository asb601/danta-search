"use client";

// Step 2 — AI settings. PUT /api/onboarding/ai-settings
// chat / embeddings / fallback API keys + postgres_url
// (+ optional chat_endpoint/deployment/api_version).

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { Field, FormError, FormSuccess, SubmitButton } from "./FormBits";
import { safeError } from "../_lib/api";
import type { OnboardingState } from "../_lib/types";

export function StepAiSettings({
  state,
  onSaved,
}: {
  state: OnboardingState | null;
  onSaved: () => void;
}) {
  const pre = state?.ai_settings ?? null;

  const [chatKey, setChatKey] = useState("");
  const [embeddingsKey, setEmbeddingsKey] = useState("");
  const [fallbackKey, setFallbackKey] = useState("");
  const [chatEndpoint, setChatEndpoint] = useState(pre?.chat_endpoint ?? "");
  const [postgresUrl, setPostgresUrl] = useState(pre?.postgres_url ?? "");
  const [deployment, setDeployment] = useState(pre?.chat_deployment ?? "");
  const [apiVersion, setApiVersion] = useState(pre?.api_version ?? "");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  // Inline error attached to the Postgres URL field (e.g. 422 connect failure).
  const [postgresError, setPostgresError] = useState<string | null>(null);

  const alreadyConfigured = !!pre?.configured;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setPostgresError(null);

    // When not yet configured, keys are required. When re-editing, allow leaving
    // a key blank to keep the existing stored value.
    if (!alreadyConfigured && (!chatKey.trim() || !embeddingsKey.trim())) {
      setError("Chat and embeddings API keys are required.");
      return;
    }
    if (!alreadyConfigured && !postgresUrl.trim()) {
      setError("Postgres URL is required.");
      return;
    }

    setSubmitting(true);
    try {
      const body: Record<string, string> = {};
      if (chatKey.trim()) body.chat_api_key = chatKey.trim();
      if (embeddingsKey.trim()) body.embeddings_api_key = embeddingsKey.trim();
      if (fallbackKey.trim()) body.fallback_api_key = fallbackKey.trim();
      if (chatEndpoint.trim()) body.chat_endpoint = chatEndpoint.trim();
      if (postgresUrl.trim()) body.postgres_url = postgresUrl.trim();
      if (deployment.trim()) body.chat_deployment = deployment.trim();
      if (apiVersion.trim()) body.api_version = apiVersion.trim();

      const res = await apiFetch("/api/onboarding/ai-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const msg = await safeError(res);
        // A 422 means the postgres_url could not connect — surface it inline on
        // the Postgres URL field and do NOT advance.
        if (res.status === 422) {
          setPostgresError(msg);
          return;
        }
        throw new Error(msg);
      }
      setSuccess("AI settings saved.");
      onSaved();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to save AI settings.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5 max-w-xl">
      <FormError message={error} />
      <FormSuccess message={success} />

      {alreadyConfigured && (
        <p className="text-[12px] text-[color:var(--fg-muted)]">
          Keys are already configured. Leave a field blank to keep its current
          value.
        </p>
      )}

      <Field
        label="Chat API key"
        htmlFor="chat-key"
        required={!alreadyConfigured}
        hint="Used for chat completions (Azure OpenAI gpt-4o / gpt-4o-mini)."
      >
        <input
          id="chat-key"
          type="password"
          autoComplete="off"
          className="field-input"
          placeholder={alreadyConfigured ? "•••••••• (unchanged)" : "sk-…"}
          value={chatKey}
          onChange={(e) => setChatKey(e.target.value)}
        />
      </Field>

      <Field
        label="Embeddings API key"
        htmlFor="emb-key"
        required={!alreadyConfigured}
        hint="Used for text-embedding-3-large indexing."
      >
        <input
          id="emb-key"
          type="password"
          autoComplete="off"
          className="field-input"
          placeholder={alreadyConfigured ? "•••••••• (unchanged)" : "sk-…"}
          value={embeddingsKey}
          onChange={(e) => setEmbeddingsKey(e.target.value)}
        />
      </Field>

      <Field
        label="Fallback API key"
        htmlFor="fallback-key"
        hint="Optional secondary key used if the primary key fails."
      >
        <input
          id="fallback-key"
          type="password"
          autoComplete="off"
          className="field-input"
          placeholder="sk-…"
          value={fallbackKey}
          onChange={(e) => setFallbackKey(e.target.value)}
        />
      </Field>

      <Field
        label="Postgres URL"
        htmlFor="postgres-url"
        required={!alreadyConfigured}
        hint="PostgreSQL connection string for this organization's metadata database."
        error={postgresError}
      >
        <input
          id="postgres-url"
          type="text"
          autoComplete="off"
          className="field-input"
          placeholder={
            alreadyConfigured
              ? "•••••••• (unchanged)"
              : "postgresql://user:pass@host:5432/db"
          }
          value={postgresUrl}
          onChange={(e) => {
            setPostgresUrl(e.target.value);
            if (postgresError) setPostgresError(null);
          }}
        />
      </Field>

      {/* Advanced (endpoint / deployment / api version) */}
      <button
        type="button"
        onClick={() => setShowAdvanced((s) => !s)}
        className="btn-ghost gap-1.5 px-1"
        aria-expanded={showAdvanced}
      >
        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? "rotate-180" : ""}`}
        />
        Advanced options
      </button>

      {showAdvanced && (
        <div className="space-y-5 pl-1 border-l-2 border-[#e5e5e5] ml-1 pt-1">
          <div className="pl-4 space-y-5">
            <Field label="OpenAI base URL" htmlFor="chat-ep">
              <input
                id="chat-ep"
                type="url"
                className="field-input"
                placeholder="https://<resource>.openai.azure.com"
                value={chatEndpoint}
                onChange={(e) => setChatEndpoint(e.target.value)}
              />
            </Field>
            <Field label="Deployment name" htmlFor="deployment">
              <input
                id="deployment"
                type="text"
                className="field-input"
                placeholder="gpt-4o"
                value={deployment}
                onChange={(e) => setDeployment(e.target.value)}
              />
            </Field>
            <Field label="API version" htmlFor="api-version">
              <input
                id="api-version"
                type="text"
                className="field-input"
                placeholder="2024-02-15-preview"
                value={apiVersion}
                onChange={(e) => setApiVersion(e.target.value)}
              />
            </Field>
          </div>
        </div>
      )}

      <div className="pt-1">
        <SubmitButton loading={submitting}>Save &amp; continue</SubmitButton>
      </div>
    </form>
  );
}

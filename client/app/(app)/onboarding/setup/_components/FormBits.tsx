"use client";

// Small presentational form helpers shared across wizard steps.
// Styling comes exclusively from globals.css utility classes + Tailwind layout.

import { type ReactNode } from "react";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

export function Field({
  label,
  htmlFor,
  hint,
  required,
  children,
  error,
}: {
  label: string;
  htmlFor?: string;
  hint?: string;
  required?: boolean;
  children: ReactNode;
  error?: string | null;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={htmlFor}
        className="block text-[13px] font-medium text-[color:var(--fg)]"
      >
        {label}
        {required && <span className="text-[color:var(--danger)] ml-0.5">*</span>}
        {!required && (
          <span className="text-[color:var(--fg-subtle)] font-normal ml-1.5">
            optional
          </span>
        )}
      </label>
      {children}
      {hint && !error && (
        <p className="text-[11.5px] text-[color:var(--fg-subtle)] leading-relaxed">
          {hint}
        </p>
      )}
      {error && (
        <p className="flex items-center gap-1 text-[11.5px] text-[color:var(--danger)]">
          <AlertCircle className="w-3 h-3 shrink-0" />
          {error}
        </p>
      )}
    </div>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className="flex items-start gap-2 px-3.5 py-2.5 rounded-[var(--radius)] bg-[color:var(--danger-bg)] border border-[color:var(--danger)]/20 text-[12.5px] text-[color:var(--danger)]"
    >
      <AlertCircle className="w-4 h-4 shrink-0 mt-px" />
      <span>{message}</span>
    </div>
  );
}

export function FormSuccess({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="status"
      className="flex items-start gap-2 px-3.5 py-2.5 rounded-[var(--radius)] bg-[color:var(--success-bg)] border border-[color:var(--success)]/20 text-[12.5px] text-[color:var(--success)]"
    >
      <CheckCircle2 className="w-4 h-4 shrink-0 mt-px" />
      <span>{message}</span>
    </div>
  );
}

export function SubmitButton({
  loading,
  children,
  disabled,
  type = "submit",
  onClick,
}: {
  loading?: boolean;
  children: ReactNode;
  disabled?: boolean;
  type?: "submit" | "button";
  onClick?: () => void;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={loading || disabled}
      className="btn-black h-10 px-5 gap-2"
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" />}
      {children}
    </button>
  );
}

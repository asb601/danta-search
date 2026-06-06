"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  ChevronDown,
  FileText,
  Loader2,
  RefreshCw,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  PDF_SELECTABLE_STATUSES,
  type PdfDocStatus,
  type PdfDocument,
} from "./types.pdf";
import type { UploadProgress } from "../_hooks/usePdfChat";

// ── status → human label + pill class ───────────────────────────────────────
const STATUS_LABEL: Record<PdfDocStatus, string> = {
  uploaded: "Queued",
  splitting: "Splitting",
  processing: "Processing",
  indexed: "Indexed",
  partially_indexed: "Partial",
  failed: "Failed",
};

function statusPillClass(status: PdfDocStatus): string {
  if (status === "indexed" || status === "partially_indexed")
    return "pdf-doc-status-indexed";
  if (status === "failed") return "pdf-doc-status-failed";
  return "pdf-doc-status-progress";
}

/** Short, stable label for an upload_id (the only identifier we have). */
function shortId(id: string): string {
  return id.length > 14 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

interface PdfDocumentPickerProps {
  documents: PdfDocument[];
  docsLoading: boolean;
  docsError: string | null;
  selectedDocIds: Set<string> | null; // null = "all documents"
  onSelectAll: () => void;
  onToggle: (id: string) => void;
  onRefresh: () => void;
  upload: UploadProgress | null;
  onUpload: (file: File) => void;
  onClearUpload: () => void;
}

export function PdfDocumentPicker({
  documents,
  docsLoading,
  docsError,
  selectedDocIds,
  onSelectAll,
  onToggle,
  onRefresh,
  upload,
  onUpload,
  onClearUpload,
}: PdfDocumentPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // Sort: usable (indexed) docs first, then in-progress, then failed — newest
  // within each group.
  const sorted = useMemo(() => {
    const rank = (s: PdfDocStatus) =>
      PDF_SELECTABLE_STATUSES.has(s) ? 0 : s === "failed" ? 2 : 1;
    return [...documents].sort((a, b) => {
      const r = rank(a.status) - rank(b.status);
      if (r !== 0) return r;
      return (b.created_at ?? "").localeCompare(a.created_at ?? "");
    });
  }, [documents]);

  const allMode = selectedDocIds === null;
  const selectedCount = selectedDocIds?.size ?? 0;
  const label = allMode
    ? "All documents"
    : selectedCount === 1
      ? "1 document"
      : `${selectedCount} documents`;

  return (
    <div className="relative min-w-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Choose which PDF documents to search"
        className="mode-trigger"
        title="Scope the question to specific documents"
      >
        <FileText className="w-3.5 h-3.5 shrink-0 opacity-70" />
        <span className="truncate max-w-[140px]">{label}</span>
        <ChevronDown
          className={cn(
            "w-3 h-3 shrink-0 opacity-60 transition-transform duration-150",
            open && "rotate-180",
          )}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="pdf-doc-panel"
            role="listbox"
            aria-label="PDF documents"
          >
            {/* ── header: title + refresh ── */}
            <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--color-subtle-foreground)]">
                Documents
              </span>
              <button
                type="button"
                onClick={onRefresh}
                aria-label="Refresh document list"
                className="btn-ghost p-1 rounded-md"
              >
                <RefreshCw
                  className={cn("w-3 h-3", docsLoading && "animate-spin")}
                />
              </button>
            </div>

            {/* ── all documents ── */}
            <button
              type="button"
              role="option"
              aria-selected={allMode}
              onClick={() => onSelectAll()}
              className={cn("pdf-doc-row", allMode && "pdf-doc-row-active")}
            >
              <span className="truncate">All documents</span>
              {allMode && <Check className="w-3.5 h-3.5 shrink-0" />}
            </button>

            <div className="pdf-doc-divider" />

            {/* ── per-document rows ── */}
            {docsLoading && documents.length === 0 ? (
              <div className="px-3 py-3 flex items-center gap-2 text-[12px] text-[var(--color-subtle-foreground)]">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading…
              </div>
            ) : sorted.length === 0 ? (
              <div className="px-3 py-3 text-[12px] text-[var(--color-subtle-foreground)]">
                {docsError ?? "No documents yet. Add a PDF below."}
              </div>
            ) : (
              sorted.map((d) => {
                const selectable = PDF_SELECTABLE_STATUSES.has(d.status);
                const checked = !allMode && !!selectedDocIds?.has(d.upload_id);
                return (
                  <button
                    key={d.upload_id}
                    type="button"
                    role="option"
                    aria-selected={checked}
                    disabled={!selectable}
                    onClick={() => onToggle(d.upload_id)}
                    className={cn(
                      "pdf-doc-row",
                      checked && "pdf-doc-row-active",
                      !selectable && "pdf-doc-row-disabled",
                    )}
                    title={d.upload_id}
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <span
                        className={cn(
                          "pdf-doc-check",
                          checked && "pdf-doc-check-on",
                        )}
                        aria-hidden="true"
                      >
                        {checked && <Check className="w-2.5 h-2.5" />}
                      </span>
                      <span className="flex flex-col min-w-0 text-left">
                        <span className="truncate font-[450]">
                          {shortId(d.upload_id)}
                        </span>
                        {d.page_count != null && (
                          <span className="text-[10.5px] text-[var(--color-subtle-foreground)]">
                            {d.page_count} page{d.page_count === 1 ? "" : "s"}
                          </span>
                        )}
                      </span>
                    </span>
                    <span className={statusPillClass(d.status)}>
                      {STATUS_LABEL[d.status]}
                    </span>
                  </button>
                );
              })
            )}

            {/* ── add PDF ── */}
            <div className="pdf-doc-divider" />
            <div className="px-3 py-2.5">
              <input
                ref={fileRef}
                type="file"
                accept="application/pdf"
                className="sr-only"
                aria-label="Upload a PDF"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUpload(f);
                  // reset so the same file can be re-selected
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="pdf-upload-btn w-full justify-center"
                disabled={!!upload && !upload.done}
              >
                {upload && !upload.done ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Upload className="w-3.5 h-3.5" />
                )}
                {upload && !upload.done ? "Uploading…" : "Add PDF"}
              </button>

              {upload && (
                <p
                  className={cn(
                    "pdf-upload-progress mt-1.5",
                    upload.error && "text-[var(--color-danger)]",
                  )}
                >
                  <span className="truncate inline-block max-w-full align-bottom">
                    {upload.filename}
                  </span>
                  {" — "}
                  {upload.error
                    ? upload.error
                    : upload.done
                      ? "ready"
                      : upload.status}
                  {upload.done && (
                    <button
                      type="button"
                      onClick={onClearUpload}
                      className="ml-1.5 underline opacity-70 hover:opacity-100"
                    >
                      dismiss
                    </button>
                  )}
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";
import type {
  ChatMode,
  PdfChatRequestBody,
  PdfChatResponse,
  PdfDocument,
  PdfMessage,
  PdfStatusResponse,
  PdfUploadResponse,
} from "../_components/types.pdf";

// ── Upload polling bounds — terminal states + an attempt cap so a not-yet-mounted
//    backend never spins forever. ~2s × 90 ≈ 3 minutes of polling. ──────────────
const POLL_INTERVAL_MS = 2_000;
const POLL_MAX_ATTEMPTS = 90;
const TERMINAL_STATUSES = new Set(["indexed", "partially_indexed", "failed"]);

export interface UploadProgress {
  upload_id: string;
  filename: string;
  status: string;
  done: boolean;
  error?: string;
}

/**
 * Self-contained hook owning ALL PDF-mode state. The Excel `useChat` hook is
 * untouched; the page picks which hook drives the composer by `mode`. PDF turns
 * are EPHEMERAL (held only here, never written to the conversation sidebar — the
 * /api/pdf/chat endpoint is stateless and has no conversation_id).
 */
export function usePdfChat({ mode }: { mode: ChatMode }) {
  const { user } = useAuth();

  // ── Composer input + ephemeral thread ──────────────────────────────────────
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<PdfMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── Ownership guard ─────────────────────────────────────────────────────────
  // A PDF turn belongs to PDF mode. Mode can change mid-request (the user switches
  // back to Excel chat), so a still-pending /api/pdf/chat response must NOT mutate
  // this ephemeral thread once we've left PDF mode — otherwise the answer leaks in
  // when the user returns. A synchronous ref mirrors `mode` so the in-flight
  // request sees the switch immediately (state reads inside an async closure are
  // stale). We do NOT abort — the request just completes silently.
  const modeRef = useRef<ChatMode>(mode);
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  // ── Document picker state ───────────────────────────────────────────────────
  const [documents, setDocuments] = useState<PdfDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState<string | null>(null);
  // null = "All documents" (whole tenant). A non-empty set = explicit scope.
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string> | null>(null);

  // ── Upload state ────────────────────────────────────────────────────────────
  const [upload, setUpload] = useState<UploadProgress | null>(null);
  const pollAbort = useRef<{ cancelled: boolean } | null>(null);

  // ── Fetch the tenant's documents (defensive: 503/404/401 → empty list) ──────
  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    setDocsError(null);
    try {
      const res = await apiFetch("/api/pdf/documents");
      if (!res.ok) {
        // 503 = router not mounted yet; 401 = auth; either way → empty + soft note.
        setDocuments([]);
        setDocsError(
          res.status === 503
            ? "PDF service is not available yet."
            : "Couldn’t load documents.",
        );
        return;
      }
      const data: PdfDocument[] = await res.json();
      setDocuments(Array.isArray(data) ? data : []);
    } catch {
      setDocuments([]);
      setDocsError("Couldn’t reach the PDF service.");
    } finally {
      setDocsLoading(false);
    }
  }, []);

  // Load documents the first time PDF mode becomes active.
  const loadedOnce = useRef(false);
  useEffect(() => {
    if (mode === "pdf" && !loadedOnce.current) {
      loadedOnce.current = true;
      loadDocuments();
    }
  }, [mode, loadDocuments]);

  // Auto-scroll the ephemeral thread.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // ── Send a PDF chat turn ────────────────────────────────────────────────────
  const send = useCallback(
    async (raw: string) => {
      const trimmed = raw.trim();
      if (!trimmed || isLoading) return;

      // Bind this turn to the mode it was started in. If the user switches away
      // from PDF mode before the response lands, suppress the write so it can't
      // leak into the (now-hidden) thread and surprise the user on return.
      const ownerMode = modeRef.current;
      const isStale = () => modeRef.current !== ownerMode;

      const userMsg: PdfMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const body: PdfChatRequestBody = {
          query: trimmed,
          // tenant_id is optional on the backend (the route derives the trusted
          // tenant from the JWT). We pass user.organization_id — which equals the
          // token's tenant — for explicitness; an empty value falls back to the
          // principal's tenant server-side.
          tenant_id: user?.organization_id ?? "",
        };
        // "All documents" (null) → omit doc_ids entirely (None = whole tenant).
        if (selectedDocIds && selectedDocIds.size > 0) {
          body.doc_ids = Array.from(selectedDocIds);
        }

        const res = await apiFetch("/api/pdf/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res
            .json()
            .catch(() => ({ detail: `HTTP ${res.status}` }));
          throw new Error(
            typeof err?.detail === "string" ? err.detail : `HTTP ${res.status}`,
          );
        }

        const data: PdfChatResponse = await res.json();
        if (isStale()) return; // user left PDF mode — drop the answer (no leak)
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.answer,
            citations: data.citations ?? [],
            chunksUsed: data.chunks_used,
            cached: data.cached,
          },
        ]);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // user stopped — no error bubble
        } else if (isStale()) {
          // user left PDF mode — don't surface a background error in the thread
        } else {
          const msg =
            err instanceof Error ? err.message : "Something went wrong.";
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: msg,
              error: true,
            },
          ]);
        }
      } finally {
        abortRef.current = null;
        setIsLoading(false);
      }
    },
    [isLoading, selectedDocIds, user?.organization_id],
  );

  // ── Composer adapters (parity with useChat's public surface) ────────────────
  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      void send(input);
    },
    [input, send],
  );

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void send(input);
      }
    },
    [input, send],
  );

  // ── Document selection helpers ──────────────────────────────────────────────
  const selectAllDocuments = useCallback(() => setSelectedDocIds(null), []);

  const toggleDocument = useCallback((id: string) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev ?? []);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next.size === 0 ? null : next;
    });
  }, []);

  // ── Upload a PDF, then poll status until terminal, then refresh the picker ──
  const uploadPdf = useCallback(
    async (file: File) => {
      // Reset any previous poll loop.
      if (pollAbort.current) pollAbort.current.cancelled = true;
      const token = { cancelled: false };
      pollAbort.current = token;

      setUpload({
        upload_id: "",
        filename: file.name,
        status: "uploading",
        done: false,
      });

      try {
        const fd = new FormData();
        fd.append("file", file);
        // Do NOT set Content-Type — the browser sets the multipart boundary.
        // Do NOT append tenant_id — /upload derives it from the JWT (optional Form).
        const res = await apiFetch("/api/pdf/upload", {
          method: "POST",
          body: fd,
        });
        if (!res.ok) {
          const err = await res
            .json()
            .catch(() => ({ detail: `HTTP ${res.status}` }));
          throw new Error(
            typeof err?.detail === "string" ? err.detail : `HTTP ${res.status}`,
          );
        }
        const up: PdfUploadResponse = await res.json();

        setUpload({
          upload_id: up.upload_id,
          filename: file.name,
          status: up.deduplicated ? "indexed" : up.status,
          done: up.deduplicated,
        });

        // Already indexed (deduplicated) → just refresh and we're done.
        if (up.deduplicated || up.status === "indexed") {
          await loadDocuments();
          return;
        }

        // Poll until a terminal status (or the attempt cap).
        for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
          if (token.cancelled) return;
          await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
          if (token.cancelled) return;

          let st: PdfStatusResponse | null = null;
          try {
            const sres = await apiFetch(`/api/pdf/status/${up.upload_id}`);
            if (sres.ok) st = await sres.json();
          } catch {
            // transient — keep polling
          }
          if (!st) continue;

          setUpload({
            upload_id: up.upload_id,
            filename: file.name,
            status: st.status,
            done: TERMINAL_STATUSES.has(st.status),
            error:
              st.status === "failed"
                ? st.error_message ?? "Processing failed."
                : undefined,
          });

          if (TERMINAL_STATUSES.has(st.status)) {
            await loadDocuments();
            return;
          }
        }
        // Hit the cap without a terminal status — surface a gentle note.
        setUpload((prev) =>
          prev
            ? { ...prev, done: true, error: "Still processing — check back later." }
            : prev,
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Upload failed.";
        setUpload({
          upload_id: "",
          filename: file.name,
          status: "failed",
          done: true,
          error: msg,
        });
      }
    },
    [loadDocuments],
  );

  const clearUpload = useCallback(() => {
    if (pollAbort.current) pollAbort.current.cancelled = true;
    setUpload(null);
  }, []);

  return {
    // composer surface (parity with useChat where the page needs it)
    input,
    setInput,
    isLoading,
    scrollRef,
    handleSubmit,
    handleStop,
    handleKeyDown,
    // ephemeral thread
    messages,
    // documents
    documents,
    docsLoading,
    docsError,
    loadDocuments,
    selectedDocIds,
    selectAllDocuments,
    toggleDocument,
    setSelectedDocIds,
    // upload
    upload,
    uploadPdf,
    clearUpload,
  };
}

"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { apiFetch } from "@/lib/auth";
import type { Message, ConversationSummary, AssistantPayload } from "../_components/types";

export function useChat() {
  // ── Message state ──────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [expandedMsgId, setExpandedMsgId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // ── Conversation state ─────────────────────────────────────────────────────
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingConv, setLoadingConv] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // ── Container scope (null = all containers) ────────────────────────────────
  const [selectedContainerId, setSelectedContainerId] = useState<string | null>(null);

  // ── Fetch conversation list ────────────────────────────────────────────────
  const fetchConversations = useCallback(async (search = "") => {
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (search.trim()) params.set("search", search.trim());
      const res = await apiFetch(`/api/chat/conversations?${params}`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data.conversations || []);
      }
    } catch {
      // silent — sidebar just won't load
    }
  }, []);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => fetchConversations(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery, fetchConversations]);

  // ── Load a conversation's messages ─────────────────────────────────────────
  const loadConversation = useCallback(async (convId: string) => {
    setLoadingConv(true);
    setActiveConvId(convId);
    setMessages([]);
    setExpandedMsgId(null);
    try {
      const res = await apiFetch(`/api/chat/conversations/${convId}`);
      if (!res.ok) return;
      const data = await res.json();
      const loaded: Message[] = (data.messages || []).map(
        (m: { id: string; role: string; content: string; payload?: AssistantPayload }) => ({
          id: m.id,
          role: m.role as "user" | "assistant",
          content: m.content,
          payload: m.role === "assistant" ? m.payload : undefined,
        })
      );
      setMessages(loaded);
    } catch {
      // silent
    } finally {
      setLoadingConv(false);
    }
  }, []);

  // ── New / delete / rename conversations ───────────────────────────────────
  const startNewChat = useCallback(() => {
    setActiveConvId(null);
    setMessages([]);
    setExpandedMsgId(null);
    setInput("");
  }, []);

  const deleteConversation = useCallback(
    async (convId: string) => {
      try {
        await apiFetch(`/api/chat/conversations/${convId}`, { method: "DELETE" });
        setConversations((prev) => prev.filter((c) => c.id !== convId));
        if (activeConvId === convId) startNewChat();
      } catch {
        // silent
      }
    },
    [activeConvId, startNewChat]
  );

  const renameConversation = useCallback(async (convId: string, title: string) => {
    try {
      await apiFetch(`/api/chat/conversations/${convId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, title } : c))
      );
    } catch {
      // silent
    }
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // ── Send a message ─────────────────────────────────────────────────────────
  // POST /api/chat/message/stream  ← this triggers the full pipeline:
  //   retrieval → agent reasoning → tool calls → SSE token stream → done
  // ──────────────────────────────────────────────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setExpandedMsgId(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // ── THE CHAT API CALL ─────────────────────────────────────────────────
      const res = await apiFetch("/api/chat/message/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,           // the user's question
          conversation_id: activeConvId, // null = start a new conversation
          container_id: selectedContainerId, // null = search all containers
        }),
        signal: controller.signal,
      });
      // ─────────────────────────────────────────────────────────────────────

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      // ── Parse SSE stream ──────────────────────────────────────────────────
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let streamedContent = "";
      let streamMsgId: string | null = null;
      let finalResult: (AssistantPayload & { conversation_id?: string; warning?: string }) | null = null;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.event === "started" && event.conversation_id) {
              if (!activeConvId || activeConvId !== event.conversation_id) {
                setActiveConvId(event.conversation_id);
              }
            } else if (event.event === "pipeline_step" && event.step === "retrieval") {
              const r: number = event.retrieved_files ?? 0;
              const t: number = event.total_files ?? 0;
              const label =
                t > 0 && r < t
                  ? `Searching ${r} of ${t} relevant files…`
                  : t > 0
                  ? `Searching ${t} files…`
                  : "Searching files…";
              if (!streamMsgId) {
                streamMsgId = crypto.randomUUID();
                setMessages((prev) => [
                  ...prev,
                  { id: streamMsgId!, role: "assistant", content: label },
                ]);
              }
            } else if (event.event === "thinking") {
              const toolName = event.tool || "tools";
              if (!streamMsgId) {
                streamMsgId = crypto.randomUUID();
                setMessages((prev) => [
                  ...prev,
                  { id: streamMsgId!, role: "assistant", content: `Running ${toolName}...` },
                ]);
              } else {
                const currentId = streamMsgId;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === currentId ? { ...m, content: `Running ${toolName}...` } : m
                  )
                );
              }
            } else if (event.event === "token") {
              streamedContent += event.content;
              if (!streamMsgId) {
                streamMsgId = crypto.randomUUID();
                setMessages((prev) => [
                  ...prev,
                  { id: streamMsgId!, role: "assistant", content: streamedContent },
                ]);
              } else {
                const currentContent = streamedContent;
                const currentId = streamMsgId;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === currentId ? { ...m, content: currentContent } : m
                  )
                );
              }
            } else if (event.event === "done") {
              finalResult = event.result;
              if (streamMsgId && finalResult) {
                const fResult = finalResult;
                const sId = streamMsgId;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === sId ? { ...m, content: fResult.answer, payload: fResult } : m
                  )
                );
                if (fResult.data && fResult.data.length > 0) {
                  setExpandedMsgId(sId);
                }
              }
              if (finalResult?.warning) {
                const warnMsg = finalResult.warning;
                setMessages((prev) => [
                  ...prev,
                  { id: crypto.randomUUID(), role: "assistant", content: warnMsg },
                ]);
              }
            } else if (event.event === "error") {
              throw new Error(event.detail || "Stream error");
            }
          } catch (parseErr) {
            // SyntaxError = malformed SSE line → safe to skip.
            // Any other error was thrown deliberately above → re-throw.
            if (!(parseErr instanceof SyntaxError)) throw parseErr;
          }
        }
      }

      // Optimistic sidebar update — no full refetch needed on existing conversations
      const resultConvId = finalResult?.conversation_id || activeConvId;
      if (resultConvId) {
        setConversations((prev) => {
          const exists = prev.some((c) => c.id === resultConvId);
          if (exists) {
            return prev.map((c) =>
              c.id === resultConvId
                ? {
                    ...c,
                    message_count: (c.message_count || 0) + 2,
                    updated_at: new Date().toISOString(),
                  }
                : c
            );
          }
          // New conversation — do one background refresh to get the server title
          fetchConversations(searchQuery);
          return prev;
        });
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // User clicked stop — don't show an error
      } else {
        const errMsg = err instanceof Error ? err.message : "Something went wrong.";
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "assistant", content: errMsg, error: true },
        ]);
      }
    } finally {
      abortRef.current = null;
      setIsLoading(false);
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return {
    // message area
    messages,
    input,
    setInput,
    isLoading,
    expandedMsgId,
    setExpandedMsgId,
    scrollRef,
    handleSubmit,
    handleStop,
    handleKeyDown,
    // conversations
    conversations,
    activeConvId,
    sidebarOpen,
    setSidebarOpen,
    loadingConv,
    searchQuery,
    setSearchQuery,
    loadConversation,
    startNewChat,
    deleteConversation,
    renameConversation,
    // container scope
    selectedContainerId,
    setSelectedContainerId,
  };
}

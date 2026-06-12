"use client";

import { RefreshCw, PanelLeft, Sparkles, FileText } from "lucide-react";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useChat } from "./_hooks/useChat";
import { usePdfChat } from "./_hooks/usePdfChat";
import { AssistantMessage } from "./_components/AssistantMessage";
import { PdfMessage } from "./_components/PdfMessage";
import { ConversationSidebar } from "./_components/ConversationSidebar";
import { ContainerPicker } from "./_components/ContainerPicker";
import { DomainPicker } from "./_components/DomainPicker";
import { ModeSwitcher } from "./_components/ModeSwitcher";
import { PdfDocumentPicker } from "./_components/PdfDocumentPicker";
import { Composer } from "./_components/Composer";
import type { ChatMode } from "./_components/types.pdf";

const PROMPTS = ["Summarise sales last month", "Show top 10 customers", "Compare Q1 vs Q2"];
const PDF_PROMPTS = ["Summarise this document", "What are the key terms?", "List the main findings"];

const msgVariants = {
  hidden: { opacity: 0, y: 14, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.28, ease: "easeOut" as const } },
};

export default function ChatPage() {
  // ── Mode (default "excel"; persisted; never restore the disabled "combined") ──
  const [mode, setMode] = useState<ChatMode>(() => {
    if (typeof window === "undefined") return "excel";
    return localStorage.getItem("gchat_mode") === "pdf" ? "pdf" : "excel";
  });
  useEffect(() => {
    if (mode === "combined") return; // never persist the disabled mode
    localStorage.setItem("gchat_mode", mode);
  }, [mode]);
  // Guard: "combined" is dummy/disabled — never let it become the active driver.
  const handleModeChange = (m: ChatMode) => {
    if (m === "combined") return;
    setMode(m);
  };

  // ── The Excel hook is UNCHANGED and remains the default driver. ──
  const {
    messages, input, setInput, isLoading, expandedMsgId, setExpandedMsgId,
    scrollRef, handleSubmit, handleStop, handleKeyDown,
    conversations, activeConvId, sidebarOpen, setSidebarOpen,
    loadingConv, searchQuery, setSearchQuery,
    loadConversation, startNewChat, deleteConversation, renameConversation,
    selectedContainerId, setSelectedContainerId,
    selectedFolderId, setSelectedFolderId,
  } = useChat();

  // ── The PDF hook owns its own ephemeral state (never touches conversations). ──
  const {
    input: pdfInput, setInput: setPdfInput, isLoading: pdfLoading,
    scrollRef: pdfScrollRef, handleSubmit: pdfSubmit, handleStop: pdfStop,
    handleKeyDown: pdfKeyDown, messages: pdfMessages,
    documents: pdfDocuments, docsLoading: pdfDocsLoading, docsError: pdfDocsError,
    loadDocuments: pdfLoadDocuments, selectedDocIds: pdfSelectedDocIds,
    selectAllDocuments: pdfSelectAll, toggleDocument: pdfToggleDocument,
    upload: pdfUpload, uploadPdf: pdfUploadPdf, clearUpload: pdfClearUpload,
  } = usePdfChat({ mode });

  const isPdf = mode === "pdf";

  // Active composer driver (the page picks by mode). Combined never reaches send.
  const composerInput = isPdf ? pdfInput : input;
  const composerSetInput = isPdf ? setPdfInput : setInput;
  const composerLoading = isPdf ? pdfLoading : isLoading;
  const composerSubmit = isPdf ? pdfSubmit : handleSubmit;
  const composerStop = isPdf ? pdfStop : handleStop;
  const composerKeyDown = isPdf ? pdfKeyDown : handleKeyDown;

  return (
    <div className="flex h-full bg-white overflow-hidden">

      {/* ── Conversation sidebar (Excel-bound; PDF turns are NOT persisted) ── */}
      <ConversationSidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={(id) => { loadConversation(id); setSidebarOpen(false); }}
        onNew={() => { startNewChat(); setSidebarOpen(false); }}
        onDelete={deleteConversation}
        onRename={renameConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(false)}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      {/* ── Main chat column ─────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 w-full overflow-hidden">

        {/* ── Top bar: mobile sidebar toggle (scope controls moved into composer) ── */}
        <div className="app-topbar px-3 sm:px-4">
          <div className="flex items-center gap-2 min-w-0">
            <AnimatePresence>
              {!sidebarOpen && (
                <motion.button
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -6 }}
                  transition={{ duration: 0.18 }}
                  onClick={() => setSidebarOpen(true)}
                  className="sm:hidden btn-ghost p-1.5 rounded-lg shrink-0"
                  aria-label="Open conversations"
                >
                  <PanelLeft className="w-4 h-4" />
                </motion.button>
              )}
            </AnimatePresence>
            <span className="text-[13px] font-semibold text-[#0a0a0a] truncate">
              {isPdf ? "PDF chat" : "Excel chat"}
            </span>
          </div>
        </div>

        {/* ── Messages area (branches by mode) ───────────────────────── */}
        {isPdf ? (
          // ─── PDF thread (ephemeral) ───
          <div ref={pdfScrollRef} className="flex-1 overflow-y-auto">
            {pdfMessages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="flex flex-col items-center justify-center h-full px-4 text-center"
              >
                <div className="relative mb-6">
                  <motion.div
                    animate={{ scale: [1, 1.12, 1], opacity: [0.15, 0.3, 0.15] }}
                    transition={{ repeat: Infinity, duration: 2.8, ease: "easeInOut" }}
                    className="absolute inset-0 rounded-2xl bg-[#0a0a0a]"
                    style={{ margin: "-6px" }}
                  />
                  <div className="bubble-avatar w-12 h-12 rounded-2xl relative z-10">
                    <FileText className="w-5 h-5 text-[#0a0a0a]" />
                  </div>
                </div>
                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1, duration: 0.4 }}
                  className="text-[20px] font-bold text-[#0a0a0a] mb-2"
                >
                  Chat with your documents
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.17, duration: 0.4 }}
                  className="text-[13.5px] text-[#737373] max-w-xs leading-relaxed mb-7"
                >
                  Ask questions across your uploaded PDFs. Answers cite the exact document and page.
                </motion.p>
                <div className="flex gap-2 flex-wrap justify-center">
                  {PDF_PROMPTS.map((s, i) => (
                    <motion.button
                      key={s}
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ delay: 0.24 + i * 0.08, duration: 0.32, ease: "easeOut" }}
                      whileHover={{ scale: 1.03, y: -1 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => setPdfInput(s)}
                      className="btn-outline px-3.5 py-2 rounded-xl text-[12.5px]"
                    >
                      {s}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <div className="max-w-3xl mx-auto px-3 sm:px-4 py-4 sm:py-6 space-y-4 sm:space-y-5">
                <p className="pdf-ephemeral-hint !mt-0 !mb-2">
                  PDF answers aren’t saved to your conversation history.
                </p>
                <AnimatePresence initial={false}>
                  {pdfMessages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      variants={msgVariants}
                      initial="hidden"
                      animate="show"
                      className={cn("flex gap-3", msg.role === "user" ? "justify-end" : "justify-start")}
                    >
                      {msg.role === "user" ? (
                        <div className="bubble-user">{msg.content}</div>
                      ) : (
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start gap-2.5">
                            <div className="bubble-avatar mt-0.5 shrink-0">
                              <FileText className="w-3 h-3 text-[#0a0a0a]" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <PdfMessage msg={msg} />
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {pdfLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 4 }}
                    transition={{ duration: 0.22 }}
                    className="flex items-start gap-2.5"
                  >
                    <div className="bubble-avatar shrink-0">
                      <FileText className="w-3 h-3 text-[#0a0a0a]" />
                    </div>
                    <div className="bubble-assistant inline-flex">
                      <div className="flex gap-1.5 items-center py-0.5">
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        ) : (
          // ─── Excel thread (UNCHANGED — existing SSE-driven path) ───
          <div ref={scrollRef} className="flex-1 overflow-y-auto">
            {loadingConv ? (
              <div className="flex items-center justify-center h-full">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                >
                  <RefreshCw className="w-5 h-5 text-[#a3a3a3]" />
                </motion.div>
              </div>
            ) : messages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="flex flex-col items-center justify-center h-full px-4 text-center"
              >
                {/* Avatar with pulse ring */}
                <div className="relative mb-6">
                  <motion.div
                    animate={{ scale: [1, 1.12, 1], opacity: [0.15, 0.3, 0.15] }}
                    transition={{ repeat: Infinity, duration: 2.8, ease: "easeInOut" }}
                    className="absolute inset-0 rounded-2xl bg-[#0a0a0a]"
                    style={{ margin: "-6px" }}
                  />
                  <div className="bubble-avatar w-12 h-12 rounded-2xl relative z-10">
                    <Sparkles className="w-5 h-5 text-[#0a0a0a]" />
                  </div>
                </div>

                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1, duration: 0.4 }}
                  className="text-[20px] font-bold text-[#0a0a0a] mb-2"
                >
                  Start a conversation
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.17, duration: 0.4 }}
                  className="text-[13.5px] text-[#737373] max-w-xs leading-relaxed mb-7"
                >
                  Ask anything about your data. The AI will search, query, and analyse it for you.
                </motion.p>

                {/* Prompt chips */}
                <div className="flex gap-2 flex-wrap justify-center">
                  {PROMPTS.map((s, i) => (
                    <motion.button
                      key={s}
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ delay: 0.24 + i * 0.08, duration: 0.32, ease: "easeOut" }}
                      whileHover={{ scale: 1.03, y: -1 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => setInput(s)}
                      className="btn-outline px-3.5 py-2 rounded-xl text-[12.5px]"
                    >
                      {s}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <div className="max-w-3xl mx-auto px-3 sm:px-4 py-4 sm:py-6 space-y-4 sm:space-y-5">
                <AnimatePresence initial={false}>
                  {messages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      variants={msgVariants}
                      initial="hidden"
                      animate="show"
                      className={cn("flex gap-3", msg.role === "user" ? "justify-end" : "justify-start")}
                    >
                      {msg.role === "user" ? (
                        <motion.div
                          whileHover={{ scale: 1.005 }}
                          className="bubble-user"
                        >
                          {msg.content}
                        </motion.div>
                      ) : (
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start gap-2.5">
                            <motion.div
                              initial={{ scale: 0.7, opacity: 0 }}
                              animate={{ scale: 1, opacity: 1 }}
                              transition={{ delay: 0.05, duration: 0.22, ease: "easeOut" }}
                              className="bubble-avatar mt-0.5 shrink-0"
                            >
                              <Sparkles className="w-3 h-3 text-[#0a0a0a]" />
                            </motion.div>
                            <motion.div
                              initial={{ opacity: 0, x: -6 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.08, duration: 0.24 }}
                              className="flex-1 min-w-0"
                            >
                              <AssistantMessage
                                msg={msg}
                                isExpanded={expandedMsgId === msg.id}
                                onToggle={() => setExpandedMsgId((p) => p === msg.id ? null : msg.id)}
                              />
                            </motion.div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {/* Typing indicator */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 4 }}
                    transition={{ duration: 0.22 }}
                    className="flex items-start gap-2.5"
                  >
                    <div className="bubble-avatar shrink-0">
                      <Sparkles className="w-3 h-3 text-[#0a0a0a]" />
                    </div>
                    <div className="bubble-assistant inline-flex">
                      <div className="flex gap-1.5 items-center py-0.5">
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Composer (redesigned; mode switcher + mode-aware scope) ───── */}
        <Composer
          value={composerInput}
          onChange={composerSetInput}
          onSubmit={composerSubmit}
          onStop={composerStop}
          onKeyDown={composerKeyDown}
          isLoading={composerLoading}
          placeholder={
            isPdf ? "Ask a question about your documents…" : "Ask a question about your data…"
          }
          modeSwitcher={<ModeSwitcher mode={mode} onChange={handleModeChange} />}
          scopeControl={
            isPdf ? (
              <PdfDocumentPicker
                documents={pdfDocuments}
                docsLoading={pdfDocsLoading}
                docsError={pdfDocsError}
                selectedDocIds={pdfSelectedDocIds}
                onSelectAll={pdfSelectAll}
                onToggle={pdfToggleDocument}
                onRefresh={pdfLoadDocuments}
                upload={pdfUpload}
                onUpload={pdfUploadPdf}
                onClearUpload={pdfClearUpload}
              />
            ) : (
              <div className="flex items-center gap-2">
                <ContainerPicker value={selectedContainerId} onChange={setSelectedContainerId} />
                <DomainPicker
                  containerId={selectedContainerId}
                  value={selectedFolderId}
                  onChange={setSelectedFolderId}
                />
              </div>
            )
          }
          subHint={
            isPdf ? (
              <p className="pdf-ephemeral-hint">
                PDF answers are session-only and aren’t saved to your conversation history.
              </p>
            ) : (
              <p className="text-center text-[11px] text-[#c4c4c4] mt-2">
                AI may make mistakes — verify important information.
              </p>
            )
          }
        />
      </div>
    </div>
  );
}

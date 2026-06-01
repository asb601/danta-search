"use client";

import { Send, RefreshCw, X, PanelLeft, Sparkles } from "lucide-react";
import { useState as useInputFocusState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useChat } from "./_hooks/useChat";
import { AssistantMessage } from "./_components/AssistantMessage";
import { ConversationSidebar } from "./_components/ConversationSidebar";
import { ContainerPicker } from "./_components/ContainerPicker";

const PROMPTS = ["Summarise sales last month", "Show top 10 customers", "Compare Q1 vs Q2"];

const msgVariants = {
  hidden: { opacity: 0, y: 14, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.28, ease: "easeOut" as const } },
};

export default function ChatPage() {
  const [inputFocused, setInputFocused] = useInputFocusState(false);
  const {
    messages, input, setInput, isLoading, expandedMsgId, setExpandedMsgId,
    scrollRef, handleSubmit, handleStop, handleKeyDown,
    conversations, activeConvId, sidebarOpen, setSidebarOpen,
    loadingConv, searchQuery, setSearchQuery,
    loadConversation, startNewChat, deleteConversation, renameConversation,
    selectedContainerId, setSelectedContainerId,
  } = useChat();

  return (
    <div className="flex h-full bg-white overflow-hidden">
      {/* Conversation sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={loadConversation}
        onNew={startNewChat}
        onDelete={deleteConversation}
        onRename={renameConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(false)}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      <div className="flex-1 flex flex-col min-w-0 w-full">
        {/* Top bar */}
        <div className="app-topbar px-3 sm:px-4">
          <div className="flex items-center gap-2 min-w-0">
            <AnimatePresence>
              {/* Show toggle always on mobile, only when closed on desktop */}
              {(!sidebarOpen) && (
                <motion.button
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -6 }}
                  transition={{ duration: 0.18 }}
                  onClick={() => setSidebarOpen(true)}
                  className="btn-ghost p-1.5 rounded-lg shrink-0"
                >
                  <PanelLeft className="w-4 h-4" />
                </motion.button>
              )}
            </AnimatePresence>
            <ContainerPicker value={selectedContainerId} onChange={setSelectedContainerId} />
          </div>
        </div>

        {/* Messages area */}
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
                style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.025em" }}
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
                {messages.map((msg, i) => (
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

        {/* Input bar */}
        <div className="border-t border-[#e5e5e5] bg-white px-3 sm:px-4 pt-3 pb-3">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <motion.div
              animate={{
                borderColor: inputFocused ? "#c4c4c4" : "#e5e5e5",
                boxShadow: inputFocused ? "0 0 0 3px rgba(10,10,10,0.06)" : "0 1px 3px rgba(0,0,0,0.06)",
                backgroundColor: inputFocused ? "#ffffff" : "#f9f9f9",
              }}
              transition={{ duration: 0.15 }}
              className="flex items-end gap-2 border rounded-2xl px-3 py-2"
            >
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder="Ask a question about your data…"
                rows={1}
                className="flex-1 bg-transparent border-none outline-none resize-none text-[13.5px] text-[#0a0a0a] placeholder:text-[#a3a3a3] leading-relaxed py-1 font-[family-name:var(--font-sans)] max-h-36 overflow-y-auto"
              />
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.button
                    key="stop"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    type="button"
                    onClick={handleStop}
                    className="shrink-0 w-8 h-8 rounded-xl bg-[#dc2626] flex items-center justify-center text-white hover:opacity-90 transition-opacity"
                  >
                    <X className="w-3.5 h-3.5" />
                  </motion.button>
                ) : (
                  <motion.button
                    key="send"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    type="submit"
                    disabled={!input.trim()}
                    whileHover={input.trim() ? { scale: 1.06 } : {}}
                    whileTap={input.trim() ? { scale: 0.94 } : {}}
                    className="shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150"
                    style={{
                      background: input.trim()
                        ? "linear-gradient(180deg, #1f1f1f 0%, #080808 100%)"
                        : "#f0f0f0",
                      boxShadow: input.trim() ? "inset 0 1px 0 rgba(255,255,255,0.12)" : "none",
                    }}
                  >
                    <Send className="w-3.5 h-3.5" style={{ color: input.trim() ? "#ffffff" : "#c4c4c4" }} />
                  </motion.button>
                )}
              </AnimatePresence>
            </motion.div>
          </form>
          <p className="text-center text-[11px] text-[#c4c4c4] mt-2">
            AI may make mistakes — verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}

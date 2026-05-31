"use client";

import { Send, RefreshCw, X, PanelLeft, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useChat } from "./_hooks/useChat";
import { AssistantMessage } from "./_components/AssistantMessage";
import { ConversationSidebar } from "./_components/ConversationSidebar";
import { ContainerPicker } from "./_components/ContainerPicker";

const PROMPTS = ["Summarise sales last month", "Show top 10 customers", "Compare Q1 vs Q2"];

export default function ChatPage() {
  const {
    messages, input, setInput, isLoading, expandedMsgId, setExpandedMsgId,
    scrollRef, handleSubmit, handleStop, handleKeyDown,
    conversations, activeConvId, sidebarOpen, setSidebarOpen,
    loadingConv, searchQuery, setSearchQuery,
    loadConversation, startNewChat, deleteConversation, renameConversation,
    selectedContainerId, setSelectedContainerId,
  } = useChat();

  return (
    <div className="flex h-full bg-white">
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

      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="app-topbar px-4">
          {!sidebarOpen ? (
            <button onClick={() => setSidebarOpen(true)} className="btn-ghost p-1.5 rounded-lg">
              <PanelLeft className="w-4 h-4" />
            </button>
          ) : <span />}
          <ContainerPicker value={selectedContainerId} onChange={setSelectedContainerId} />
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {loadingConv ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-5 h-5 text-[#a3a3a3] animate-spin" />
            </div>
          ) : messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="flex flex-col items-center justify-center h-full px-4 text-center"
            >
              <div className="bubble-avatar w-12 h-12 rounded-2xl mb-5">
                <Sparkles className="w-5 h-5 text-[#0a0a0a]" />
              </div>
              <h2
                className="text-[21px] font-bold text-[#0a0a0a] mb-2"
                style={{ fontFamily: "var(--font-display)", letterSpacing: "-0.025em" }}
              >
                Start a conversation
              </h2>
              <p className="text-[13.5px] text-[#737373] max-w-xs leading-relaxed mb-6">
                Ask anything about your data. The AI will search, query, and analyse it for you.
              </p>
              <div className="flex gap-2 flex-wrap justify-center">
                {PROMPTS.map((s, i) => (
                  <motion.button
                    key={s}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 + i * 0.07, duration: 0.35 }}
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
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.32, ease: "easeOut" }}
                    className={cn("flex gap-3", msg.role === "user" ? "justify-end" : "justify-start")}
                  >
                    {msg.role === "user" ? (
                      <div className="bubble-user">{msg.content}</div>
                    ) : (
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start gap-2.5">
                          <div className="bubble-avatar mt-0.5">
                            <Sparkles className="w-3 h-3 text-[#0a0a0a]" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <AssistantMessage
                              msg={msg}
                              isExpanded={expandedMsgId === msg.id}
                              onToggle={() => setExpandedMsgId((p) => p === msg.id ? null : msg.id)}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-start gap-2.5"
                >
                  <div className="bubble-avatar">
                    <Sparkles className="w-3 h-3 text-[#0a0a0a]" />
                  </div>
                  <div className="bubble-assistant">
                    <div className="flex gap-1.5 items-center">
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
        <div className="border-t border-[#e5e5e5] bg-white p-3 sm:p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data…"
              rows={1}
              className="field-textarea flex-1 shadow-xs"
            />
            {isLoading ? (
              <button type="button" onClick={handleStop} className="btn-danger shrink-0 h-10 w-10 rounded-xl">
                <X className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="btn-black shrink-0 h-10 w-10 rounded-xl"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </form>
          <p className="text-center text-[11px] text-[#a3a3a3] mt-2">
            AI may make mistakes — verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}

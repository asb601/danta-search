"use client";

import { Send, RefreshCw, X, PanelLeft, MessageSquare, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChat } from "./_hooks/useChat";
import { AssistantMessage } from "./_components/AssistantMessage";
import { ConversationSidebar } from "./_components/ConversationSidebar";
import { ContainerPicker } from "./_components/ContainerPicker";

export default function ChatPage() {
  const {
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
    selectedContainerId,
    setSelectedContainerId,
  } = useChat();

  return (
    <div className="flex h-full bg-background">
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
        <div className="px-4 py-2.5 border-b border-border bg-card flex items-center justify-between gap-3">
          {!sidebarOpen ? (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              title="Open sidebar"
            >
              <PanelLeft className="w-4 h-4" />
            </button>
          ) : (
            <span />
          )}
          <ContainerPicker
            value={selectedContainerId}
            onChange={setSelectedContainerId}
          />
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {loadingConv ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-5 h-5 text-muted-foreground animate-spin" />
            </div>
          ) : messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full px-4 text-center">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-5 shadow-sm">
                <Sparkles className="w-6 h-6 text-primary" />
              </div>
              <h2 className="text-lg font-bold tracking-tight text-foreground mb-2">
                Start a conversation
              </h2>
              <p className="text-sm text-muted-foreground max-w-xs leading-relaxed">
                Ask anything about your data. The AI will search, query, and analyse it for you.
              </p>
              <div className="flex gap-2 mt-6 flex-wrap justify-center">
                {["Summarise sales last month", "Show top 10 customers", "Compare Q1 vs Q2"].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="px-3 py-1.5 rounded-lg border border-border bg-card text-xs text-muted-foreground hover:text-foreground hover:border-primary/40 hover:bg-accent transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-5">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-3",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "user" ? (
                    <div className="max-w-[75%] sm:max-w-[65%] rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed shadow-xs"
                      style={{ background: "var(--gradient-primary)", color: "white" }}
                    >
                      {msg.content}
                    </div>
                  ) : (
                    <div className="flex-1 min-w-0 max-w-full">
                      {/* Assistant avatar */}
                      <div className="flex items-start gap-2.5">
                        <div className="w-6 h-6 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                          <MessageSquare className="w-3 h-3 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <AssistantMessage
                            msg={msg}
                            isExpanded={expandedMsgId === msg.id}
                            onToggle={() =>
                              setExpandedMsgId((prev) =>
                                prev === msg.id ? null : msg.id
                              )
                            }
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex items-start gap-2.5">
                  <div className="w-6 h-6 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                    <MessageSquare className="w-3 h-3 text-primary" />
                  </div>
                  <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3 shadow-xs">
                    <div className="flex gap-1.5 items-center">
                      <span className="w-2 h-2 rounded-full bg-primary/60 animate-pulse" />
                      <span className="w-2 h-2 rounded-full bg-primary/60 animate-pulse [animation-delay:150ms]" />
                      <span className="w-2 h-2 rounded-full bg-primary/60 animate-pulse [animation-delay:300ms]" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="border-t border-border bg-card p-4">
          <form
            onSubmit={handleSubmit}
            className="max-w-3xl mx-auto flex items-end gap-2"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data…"
              rows={1}
              className="flex-1 resize-none bg-background border border-border rounded-xl px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground input-focus shadow-xs"
            />
            {isLoading ? (
              <button
                type="button"
                onClick={handleStop}
                className="shrink-0 h-10 w-10 flex items-center justify-center rounded-xl bg-danger/90 text-white transition-opacity hover:opacity-90 shadow-xs"
                title="Stop generating"
              >
                <X className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="shrink-0 h-10 w-10 flex items-center justify-center rounded-xl disabled:opacity-40 transition-opacity hover:opacity-90 shadow-xs btn-gradient"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </form>
          <p className="text-center text-[11px] text-subtle-foreground mt-2">
            AI may make mistakes — verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}

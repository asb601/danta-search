"use client";

import { Send, RefreshCw, X, PanelLeft, MessageSquare } from "lucide-react";
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
    <div className="flex h-full">
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
        <div className="px-3 py-2 border-b border-border flex items-center justify-between gap-2">
          {!sidebarOpen ? (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-md bg-surface border border-border text-muted-foreground hover:text-foreground transition-colors"
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
              <div className="w-11 h-11 rounded-xl bg-foreground/[0.07] flex items-center justify-center mb-5">
                <MessageSquare className="w-5 h-5 text-foreground/60" />
              </div>
              <h2 className="text-base font-semibold tracking-tight text-foreground mb-1.5">Start a conversation</h2>
              <p className="text-sm text-muted-foreground max-w-sm leading-relaxed">
                Ask anything about your data. The AI will search, query, and analyse it for you.
              </p>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto px-3 sm:px-4 py-6 space-y-6">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-3",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {msg.role === "user" ? (
                    <div className="max-w-[75%] sm:max-w-[65%] rounded-xl px-4 py-3 text-sm leading-relaxed bg-foreground text-background">
                      {msg.content}
                    </div>
                  ) : (
                    <div className="flex-1 min-w-0 max-w-full">
                      <AssistantMessage
                        msg={msg}
                        isExpanded={expandedMsgId === msg.id}
                        onToggle={() =>
                          setExpandedMsgId((prev) => (prev === msg.id ? null : msg.id))
                        }
                      />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-surface border border-border rounded-xl px-4 py-3">
                    <div className="flex gap-1.5 items-center">
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse" />
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse [animation-delay:150ms]" />
                      <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse [animation-delay:300ms]" />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-border bg-surface p-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data..."
              rows={1}
              className="flex-1 resize-none bg-surface-raised border border-border rounded-lg px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
            />
            {isLoading ? (
              <button
                type="button"
                onClick={handleStop}
                className="shrink-0 h-10 w-10 flex items-center justify-center rounded-lg bg-danger/90 text-white transition-opacity hover:opacity-90"
                title="Stop generating"
              >
                <X className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="shrink-0 h-10 w-10 flex items-center justify-center rounded-lg bg-primary text-primary-foreground disabled:opacity-40 transition-opacity hover:opacity-90"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}

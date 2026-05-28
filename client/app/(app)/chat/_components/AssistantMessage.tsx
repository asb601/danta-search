"use client";

import { AlertCircle } from "lucide-react";
import { AnswerText, stripTabularContent } from "./AnswerText";
import { ResultsAccordion } from "./ResultsAccordion";
import type { Message } from "./types";

export function AssistantMessage({
  msg,
  isExpanded,
  onToggle,
}: {
  msg: Message;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const hasData = !!(
    (msg.payload?.data && msg.payload.data.length > 0) ||
    (msg.payload?.result_sets && msg.payload.result_sets.some((set) => Array.isArray(set.data) && set.data.length > 0))
  );
  const displayText = hasData ? stripTabularContent(msg.content) : msg.content;

  return (
    <div className="bg-surface border border-border rounded-xl px-4 py-3 w-full min-w-0">
      {msg.error ? (
        <span className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {msg.content}
        </span>
      ) : (
        <>
          <AnswerText text={displayText} />
          {msg.payload?.suggested_rephrase && (
            <p className="mt-2 text-xs text-muted-foreground italic border-t border-border pt-2">
              Try: &ldquo;{msg.payload.suggested_rephrase}&rdquo;
            </p>
          )}
          {msg.payload && hasData && (
            <ResultsAccordion
              payload={msg.payload}
              isOpen={isExpanded}
              onToggle={onToggle}
            />
          )}
        </>
      )}
    </div>
  );
}

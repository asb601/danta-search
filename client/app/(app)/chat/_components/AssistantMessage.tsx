"use client";

import { AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { AnswerText, stripTabularContent } from "./AnswerText";
import { ResultsAccordion } from "./ResultsAccordion";
import { GovernancePanel } from "./GovernancePanel";
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
  const hasData = !!(msg.payload?.data && msg.payload.data.length > 0);
  const displayText = hasData ? stripTabularContent(msg.content) : msg.content;

  if (msg.error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2.5 px-4 py-3 rounded-xl border border-[#fecaca] bg-[#fff5f5] text-[#dc2626] text-[13px]"
      >
        <AlertCircle className="w-4 h-4 shrink-0" />
        {msg.content}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white border border-[#e5e5e5] rounded-2xl rounded-tl-md px-4 py-3.5 w-full min-w-0 shadow-[0_1px_4px_rgba(0,0,0,0.05)]"
    >
      <AnswerText text={displayText} />
      {msg.payload?.suggested_rephrase && (
        <p className="mt-2.5 text-[12px] text-[#a3a3a3] italic border-t border-[#f0f0f0] pt-2.5">
          Try: &ldquo;{msg.payload.suggested_rephrase}&rdquo;
        </p>
      )}
      {msg.payload && msg.payload.data && msg.payload.data.length > 0 && (
        <ResultsAccordion
          payload={msg.payload}
          isOpen={isExpanded}
          onToggle={onToggle}
        />
      )}
      {msg.payload?.governance && (
        <GovernancePanel governance={msg.payload.governance} />
      )}
    </motion.div>
  );
}

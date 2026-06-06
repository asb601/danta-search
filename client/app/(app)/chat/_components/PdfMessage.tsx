"use client";

import { AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { AnswerText } from "./AnswerText";
import type { PdfMessage as PdfMessageType } from "./types.pdf";

/** Short, stable label for a doc_id inside a citation chip. */
function shortDoc(id: string): string {
  return id.length > 14 ? `${id.slice(0, 8)}…` : id;
}

/** Renders one PDF assistant turn: markdown answer + citation chips. The Excel
 *  ResultsAccordion / data-table path is intentionally NOT reused here. */
export function PdfMessage({ msg }: { msg: PdfMessageType }) {
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

  const citations = msg.citations ?? [];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white border border-[#e5e5e5] rounded-2xl rounded-tl-md px-4 py-3.5 w-full min-w-0 shadow-[0_1px_4px_rgba(0,0,0,0.05)]"
    >
      <AnswerText text={msg.content} />

      {citations.length > 0 && (
        <div className="mt-3 pt-2.5 border-t border-[#f0f0f0] flex flex-wrap gap-1.5">
          {citations.map((c) => (
            <span
              key={`${c.n}-${c.doc_id}-${c.page}`}
              className="pdf-citation-chip"
              title={`${c.doc_id} · page ${c.page}`}
            >
              <span className="font-semibold">[{c.n}]</span>
              <span className="truncate max-w-[140px]">{shortDoc(c.doc_id)}</span>
              <span className="opacity-70">p.{c.page}</span>
            </span>
          ))}
        </div>
      )}
    </motion.div>
  );
}

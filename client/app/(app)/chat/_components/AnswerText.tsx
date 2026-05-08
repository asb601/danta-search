"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Strip markdown pipe tables and tab-separated data dumps from LLM text
 *  when the UI renders the table separately in the ResultsAccordion. */
export function stripTabularContent(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|")) continue;
    if ((line.match(/\t/g) ?? []).length >= 2) continue;
    out.push(line);
  }
  return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function AnswerText({ text }: { text: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-li:my-0.5 prose-headings:mt-3 prose-headings:mb-1 prose-hr:my-2 prose-pre:bg-surface-raised prose-pre:border prose-pre:border-border prose-code:text-primary prose-code:before:content-none prose-code:after:content-none prose-table:text-xs prose-th:px-2 prose-th:py-1 prose-td:px-2 prose-td:py-1">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { downloadCsv, blobToLabel } from "./DataTable";

export function DownloadPanel({
  data,
  filesUsed,
}: {
  data: Record<string, unknown>[];
  filesUsed: string[];
}) {
  const [open, setOpen] = useState(false);
  const sources = filesUsed.map(blobToLabel);
  const filename =
    sources.length === 1
      ? `${sources[0].replace(/ /g, "_").toLowerCase()}.csv`
      : "query_result.csv";

  return (
    <div className="relative">
      <div className="flex items-center gap-1">
        <button
          onClick={() => downloadCsv(data, filename)}
          className="flex items-center gap-1.5 text-[11px] font-medium text-primary hover:text-primary/80 transition-colors px-2 py-1 rounded-md hover:bg-primary/8 border border-primary/20"
        >
          <svg
            className="w-3 h-3"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          >
            <path
              d="M8 2v8m0 0L5 7m3 3 3-3M2 12h12"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Download CSV
        </button>
        {filesUsed.length > 1 && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-0.5 text-[11px] text-muted-foreground hover:text-foreground px-1.5 py-1 rounded-md hover:bg-surface-raised transition-colors"
          >
            {filesUsed.length} sources
            <ChevronDown
              className={cn("w-3 h-3 transition-transform", open && "rotate-180")}
            />
          </button>
        )}
      </div>
      {open && filesUsed.length > 1 && (
        <div className="absolute left-0 top-full mt-1.5 bg-surface border border-border rounded-lg p-2.5 shadow-lg z-20 min-w-52">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1.5 font-semibold">
            Source files
          </p>
          {sources.map((name, i) => (
            <p key={i} className="text-xs text-foreground py-0.5 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-primary/50 shrink-0" />
              {name}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

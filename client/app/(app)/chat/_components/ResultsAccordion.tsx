"use client";

import { ChevronDown, Table2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { blobToLabel, DataTable } from "./DataTable";
import { DownloadPanel } from "./DownloadPanel";
import type { AssistantPayload } from "./types";

export function ResultsAccordion({
  payload,
  isOpen,
  onToggle,
}: {
  payload: AssistantPayload;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const hasData = payload.data && payload.data.length > 0;
  if (!hasData) return null;

  const totalRows = payload.row_count ?? payload.data.length;
  const displayedRows = payload.data.length;

  return (
    <div className="mt-3 border border-border rounded-lg overflow-hidden">
      {/* Accordion header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-raised hover:bg-surface-raised/70 transition-colors text-left select-none"
      >
        <div className="flex items-center gap-2">
          <Table2 className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-foreground">Results</span>
          <span className="bg-primary/10 text-primary text-[11px] font-mono rounded px-1.5 py-0.5">
            {totalRows.toLocaleString()} row{totalRows !== 1 ? "s" : ""}
          </span>
          {totalRows > displayedRows && (
            <span className="text-[11px] text-muted-foreground">
              · showing {displayedRows}
            </span>
          )}
          {payload.files_used && payload.files_used.length > 0 && (
            <span className="text-[11px] text-muted-foreground hidden sm:inline">
              · {payload.files_used.map(blobToLabel).join(", ")}
            </span>
          )}
        </div>
        <ChevronDown
          className={cn(
            "w-4 h-4 text-muted-foreground transition-transform duration-200",
            isOpen && "rotate-180"
          )}
        />
      </button>

      {/* Accordion body */}
      {isOpen && (
        <div className="p-3 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[11px] text-muted-foreground">
              Displaying {displayedRows.toLocaleString()} of{" "}
              {totalRows.toLocaleString()} total rows
            </p>
            {payload.files_used && payload.files_used.length > 0 && (
              <DownloadPanel
                data={payload.data}
                filesUsed={payload.files_used}
              />
            )}
          </div>
          <DataTable data={payload.data} totalRows={totalRows} />
        </div>
      )}
    </div>
  );
}

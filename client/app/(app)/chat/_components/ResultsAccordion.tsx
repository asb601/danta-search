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
  const resultSets = (payload.result_sets ?? []).filter((set) => set.data && set.data.length > 0);
  const hasData = resultSets.length > 0 || (payload.data && payload.data.length > 0);
  if (!hasData) return null;

  const primaryData = resultSets[0]?.data ?? payload.data;
  const totalRows = resultSets.length > 1
    ? resultSets.reduce((sum, set) => sum + (set.row_count ?? set.data.length), 0)
    : payload.row_count ?? primaryData.length;
  const displayedRows = resultSets.length > 1
    ? resultSets.reduce((sum, set) => sum + set.data.length, 0)
    : primaryData.length;

  return (
    <div className="mt-3 border border-border rounded-lg overflow-hidden">
      {/* Accordion header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-surface-raised hover:bg-surface-raised/70 transition-colors text-left select-none"
      >
        <div className="flex items-center gap-2">
          <Table2 className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-foreground">
            {resultSets.length > 1 ? `${resultSets.length} result tables` : "Results"}
          </span>
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
          {resultSets.length > 1 ? (
            <div className="space-y-3">
              {resultSets.map((set, index) => {
                const setTotalRows = set.row_count ?? set.data.length;
                const setFiles = set.files_used ?? [];
                return (
                  <details key={`${set.title ?? "result"}-${index}`} open={index === 0} className="border border-border rounded-md overflow-hidden">
                    <summary className="cursor-pointer list-none px-3 py-2 bg-surface-raised text-xs font-medium text-foreground flex items-center justify-between gap-2">
                      <span>{set.title || `Result ${index + 1}`}</span>
                      <span className="text-[11px] text-muted-foreground font-normal">
                        {setTotalRows.toLocaleString()} row{setTotalRows !== 1 ? "s" : ""}
                      </span>
                    </summary>
                    <div className="p-3 space-y-3">
                      <div className="flex items-center justify-between">
                        <p className="text-[11px] text-muted-foreground">
                          Displaying {set.data.length.toLocaleString()} of {setTotalRows.toLocaleString()} total rows
                        </p>
                        {setFiles.length > 0 && <DownloadPanel data={set.data} filesUsed={setFiles} />}
                      </div>
                      <DataTable data={set.data} totalRows={setTotalRows} />
                    </div>
                  </details>
                );
              })}
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between">
                <p className="text-[11px] text-muted-foreground">
                  Displaying {displayedRows.toLocaleString()} of{" "}
                  {totalRows.toLocaleString()} total rows
                </p>
                {payload.files_used && payload.files_used.length > 0 && (
                  <DownloadPanel
                    data={primaryData}
                    filesUsed={payload.files_used}
                  />
                )}
              </div>
              <DataTable data={primaryData} totalRows={totalRows} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

"use client";

import { cn } from "@/lib/utils";

export function downloadCsv(data: Record<string, unknown>[], filename = "query_result.csv") {
  if (!data.length) return;
  const cols = Object.keys(data[0]);
  const escape = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? `"${s.replace(/"/g, '""')}"`
      : s;
  };
  const csv = [
    cols.join(","),
    ...data.map((r) => cols.map((c) => escape(r[c])).join(",")),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function blobToLabel(blob: string): string {
  // Strip az://container/ URI prefix if present, then take only the filename part
  const stripped = blob.replace(/^az:\/\/[^/]+\//, "");
  const filename = stripped.split("/").pop() || stripped;
  const base = filename
    .replace(/\.[^.]+$/, "")        // strip extension
    .replace(/^[0-9a-f]{8}_/i, "") // strip upload hash prefix
    .replace(/\.sample$/, "");      // strip .sample suffix
  return base
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  const s = String(value);
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(s)) {
    try {
      const d = new Date(s);
      return d.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return s;
    }
  }
  const num = Number(s);
  if (!isNaN(num) && s.trim() !== "" && Math.abs(num) >= 1000) {
    return num % 1 === 0
      ? num.toLocaleString("en-US")
      : num.toLocaleString("en-US", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        });
  }
  return s;
}

export function DataTable({
  data,
  totalRows,
}: {
  data: Record<string, unknown>[];
  totalRows?: number;
}) {
  if (!data.length)
    return (
      <p className="text-xs text-muted-foreground py-6 text-center">
        No rows returned.
      </p>
    );

  const cols = Object.keys(data[0]);
  const rows = data.slice(0, 100);
  const displayCount = rows.length;
  const actualTotal = totalRows ?? data.length;

  return (
    <div className="overflow-x-auto rounded-md border border-border text-xs">
      <table className="min-w-full divide-y divide-border">
        <thead className="bg-surface-raised">
          <tr>
            <th className="px-2 py-2 text-center font-medium text-muted-foreground w-10">
              #
            </th>
            {cols.map((c) => (
              <th
                key={c}
                className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap"
              >
                {c.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border bg-surface">
          {rows.map((row, i) => (
            <tr
              key={i}
              className={cn(
                "hover:bg-surface-raised/60 transition-colors",
                i % 2 === 1 && "bg-surface-raised/30"
              )}
            >
              <td className="px-2 py-1.5 text-center text-muted-foreground tabular-nums">
                {i + 1}
              </td>
              {cols.map((c) => {
                const formatted = formatCell(row[c]);
                const isNum =
                  !isNaN(Number(row[c])) && String(row[c]).trim() !== "";
                return (
                  <td
                    key={c}
                    className={cn(
                      "px-3 py-1.5 text-foreground max-w-[320px] break-words",
                      isNum ? "tabular-nums text-right" : "text-left"
                    )}
                    title={String(row[c] ?? "")}
                  >
                    {formatted}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {actualTotal > displayCount && (
        <p className="px-3 py-2 text-xs text-muted-foreground border-t border-border bg-surface-raised text-center">
          Showing {displayCount.toLocaleString()} of{" "}
          {actualTotal.toLocaleString()} total rows
        </p>
      )}
    </div>
  );
}

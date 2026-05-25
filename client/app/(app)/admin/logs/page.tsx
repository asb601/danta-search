"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  FileText,
  Search,
  RefreshCw,
  AlertCircle,
  Info,
  AlertTriangle,
  ChevronDown,
  Clock,
  Upload,
  Zap,
  CheckCircle,
  XCircle,
  Loader2,
  Activity,
} from "lucide-react";
import { apiFetch } from "@/lib/auth";
import { useAuth } from "@/components/auth-provider";
import { cn } from "@/lib/utils";

/* ── types ───────────────────────────────────────────────────────────────── */

interface LogEntry {
  [key: string]: unknown;
  event?: string;
  level?: string;
  timestamp?: string;
  raw?: string;
}

interface LogResponse {
  log_type: string;
  returned: number;
  lines: LogEntry[];
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

const LEVEL_COLORS: Record<string, string> = {
  error: "bg-red-50 text-red-700 border-red-200",
  warning: "bg-yellow-50 text-yellow-700 border-yellow-200",
  info: "bg-blue-50 text-blue-700 border-blue-200",
  debug: "bg-zinc-100 text-zinc-600 border-zinc-300",
};

const LEVEL_ICONS: Record<string, typeof Info> = {
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
  debug: Info,
};

function formatTimestamp(ts: string | undefined): string {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

function formatDuration(ms: number | undefined): string {
  if (ms === undefined || ms === null) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/* ── log line component ──────────────────────────────────────────────────── */

function LogLine({ entry }: { entry: LogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const level = (entry.level || "info") as string;
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.info;
  const Icon = LEVEL_ICONS[level] || Info;

  // Raw text line
  if (entry.raw) {
    return (
      <div className="px-3 py-1.5 font-mono text-xs text-muted-foreground border-b border-border/50">
        {entry.raw}
      </div>
    );
  }

  const event = entry.event || "";
  const timestamp = formatTimestamp(entry.timestamp as string);
  const durationMs = entry.duration_ms as number | undefined;
  const step = entry.step as string | undefined;
  const status = entry.status as string | undefined;
  const traceId = entry.trace_id as string | undefined;

  // Keys to hide from detail view
  const hideKeys = new Set([
    "event",
    "level",
    "timestamp",
    "duration_ms",
    "step",
    "status",
    "trace_id",
    "pipeline",
  ]);
  const extraKeys = Object.keys(entry).filter(
    (k) => !hideKeys.has(k) && entry[k] !== undefined && entry[k] !== null
  );

  return (
    <div
      className={cn(
        "border-b border-border/50 hover:bg-surface-raised/50 transition-colors cursor-pointer",
        expanded && "bg-surface-raised/30"
      )}
      onClick={() => extraKeys.length > 0 && setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2 px-3 py-1.5">
        {/* Time */}
        <span className="text-[10px] text-muted-foreground font-mono w-16 shrink-0">
          {timestamp}
        </span>

        {/* Level badge */}
        <span
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border shrink-0",
            colors
          )}
        >
          <Icon className="w-3 h-3" />
          {level}
        </span>

        {/* Step badge */}
        {step && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-primary/10 text-primary border border-primary/20 shrink-0">
            {step}
          </span>
        )}

        {/* Event */}
        <span className="text-xs font-medium text-foreground truncate">
          {event}
        </span>

        {/* Status */}
        {status && (
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0",
              status === "done" || status === "started"
                ? "bg-green-50 text-green-700"
                : status === "failed"
                ? "bg-red-50 text-red-700"
                : "bg-zinc-100 text-zinc-600"
            )}
          >
            {status}
          </span>
        )}

        {/* Duration */}
        {durationMs !== undefined && (
          <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground shrink-0 ml-auto">
            <Clock className="w-3 h-3" />
            {formatDuration(durationMs)}
          </span>
        )}

        {/* Trace ID */}
        {traceId && (
          <span className="text-[10px] text-muted-foreground font-mono shrink-0 hidden lg:block">
            {traceId}
          </span>
        )}

        {/* Expand indicator */}
        {extraKeys.length > 0 && (
          <ChevronDown
            className={cn(
              "w-3 h-3 text-muted-foreground transition-transform shrink-0",
              expanded && "rotate-180"
            )}
          />
        )}
      </div>

      {/* Expanded details */}
      {expanded && extraKeys.length > 0 && (
        <div className="px-3 pb-2 pl-[7.5rem]">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
            {extraKeys.map((key) => (
              <div key={key} className="contents">
                <span className="text-muted-foreground font-mono">{key}</span>
                <span className="text-foreground font-mono break-all">
                  {typeof entry[key] === "object"
                    ? JSON.stringify(entry[key])
                    : String(entry[key])}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── main page ───────────────────────────────────────────────────────────── */

type PageView = "audit" | "logs" | "performance" | "pipeline" | "ingestion";

interface FileTiming {
  file_id: string;
  name: string;
  size: number;
  ingest_status: string;
  uploaded_at: string | null;
  upload_secs: number | null;
  ingested_at: string | null;
  ingestion_secs: number | null;
  parquet_status: string | null;
  parquet_secs: number | null;
  processing_secs: number | null;
  total_secs: number | null;
  parquet_error: string | null;
}

interface AuditEntry {
  id: string;
  created_at: string | null;
  log_type: string;
  event: string;
  level: string;
  actor: {
    user_id: string | null;
    email: string | null;
    role: string | null;
  };
  domain_tag: string | null;
  trace_id: string | null;
  file_id: string | null;
  file_name: string | null;
  request: {
    method: string | null;
    path: string | null;
    route_template: string | null;
    status_code: number | null;
    duration_ms: number | null;
    ip_address: string | null;
    user_agent: string | null;
  } | null;
  details: Record<string, unknown> | null;
}

interface AuditResponse {
  scope: "admin" | "domain" | "self";
  returned: number;
  lines: AuditEntry[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatSecs(secs: number | null): string {
  if (secs === null || secs === undefined) return "—";
  if (secs < 1) return `${Math.round(secs * 1000)}ms`;
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-US", {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; icon: typeof CheckCircle }> = {
    ingested: { color: "bg-green-50 text-green-700 border-green-200", icon: CheckCircle },
    done: { color: "bg-green-50 text-green-700 border-green-200", icon: CheckCircle },
    failed: { color: "bg-red-50 text-red-700 border-red-200", icon: XCircle },
    running: { color: "bg-blue-50 text-blue-700 border-blue-200", icon: Loader2 },
    pending: { color: "bg-yellow-50 text-yellow-700 border-yellow-200", icon: Clock },
    not_ingested: { color: "bg-zinc-100 text-zinc-600 border-zinc-300", icon: Clock },
  };
  const conf = map[status] || map.not_ingested;
  const Icon = conf.icon;
  return (
    <span className={cn("inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border", conf.color)}>
      <Icon className={cn("w-3 h-3", status === "running" && "animate-spin")} />
      {status}
    </span>
  );
}

/* ── AI Pipeline panel ───────────────────────────────────────────────────── */

/**
 * Map a pipeline event name to a UI step (number, label, colour).
 * Order in the UI mirrors the actual flow of a query.
 */
type StepConfig = {
  num: string;
  label: string;
  color: string; // tailwind text + bg combo for the badge
  short: string; // one-line summary builder key
};

const PIPELINE_STEPS: Record<string, StepConfig> = {
  query_received:        { num: "1", label: "Query Received",   color: "bg-cyan-50 text-cyan-800 border-cyan-200",       short: "query" },
  catalog_loaded:        { num: "2", label: "Catalog Loaded",   color: "bg-blue-50 text-blue-800 border-blue-200",       short: "catalog" },
  catalog_empty:         { num: "2", label: "Catalog Empty",    color: "bg-red-50 text-red-700 border-red-200",          short: "catalog_empty" },
  system_prompt_built:   { num: "3", label: "System Prompt",    color: "bg-violet-50 text-violet-800 border-violet-200", short: "prompt" },
  search_catalog:        { num: "3", label: "Catalog Search",   color: "bg-violet-50 text-violet-800 border-violet-200", short: "search" },
  get_file_schema:       { num: "3", label: "File Schema",      color: "bg-violet-50 text-violet-800 border-violet-200", short: "schema" },
  llm_input:             { num: "4", label: "LLM Input",        color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "llm_in" },
  llm_stream_input:      { num: "4", label: "LLM Input",        color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "llm_in" },
  llm_output:            { num: "4", label: "LLM Decision",     color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "llm_out" },
  llm_stream_output:     { num: "4", label: "LLM Decision",     color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "llm_out" },
  tool_call_start:       { num: "4", label: "Tool Start",       color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "tool_start" },
  tool_call_end:         { num: "4", label: "Tool End",         color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "tool_end" },
  sql_execute_start:     { num: "5", label: "SQL Executing",    color: "bg-amber-50 text-amber-800 border-amber-200",    short: "sql_start" },
  sql_execute_done:      { num: "6", label: "SQL Result",       color: "bg-orange-50 text-orange-800 border-orange-200", short: "sql_done" },
  sql_execute_error:     { num: "6", label: "SQL Error",        color: "bg-red-50 text-red-700 border-red-200",          short: "sql_error" },
  inspect_data_format:   { num: "5", label: "Data Sample",      color: "bg-amber-50 text-amber-800 border-amber-200",    short: "sample" },
  summarise_dataframe_done: { num: "5", label: "Stats",         color: "bg-amber-50 text-amber-800 border-amber-200",    short: "stats" },
  ingest_llm_prompt:     { num: "i", label: "Ingest Prompt",    color: "bg-zinc-100 text-zinc-700 border-zinc-300",       short: "ingest_p" },
  ingest_llm_response:   { num: "i", label: "Ingest Reply",     color: "bg-zinc-100 text-zinc-700 border-zinc-300",       short: "ingest_r" },
  final_answer:          { num: "✓", label: "Final Answer",     color: "bg-emerald-50 text-emerald-800 border-emerald-200", short: "answer" },

  // ── Orchestration pipeline stages ─────────────────────────────────────────
  retrieval_filtered:          { num: "2", label: "Retrieval",      color: "bg-blue-50 text-blue-800 border-blue-200",         short: "retrieval" },
  retrieval_fallback:          { num: "2", label: "Ret Fallback",   color: "bg-orange-50 text-orange-800 border-orange-200",   short: "ret_fallback" },
  resolver_pins_injected:      { num: "2", label: "Resolver Pins",  color: "bg-blue-50 text-blue-800 border-blue-200",         short: "resolver_pins" },
  prior_files_pinned:          { num: "2", label: "Prior Files",    color: "bg-blue-50 text-blue-800 border-blue-200",         short: "prior_files" },
  explicit_file_pinned:        { num: "2", label: "File Pinned",    color: "bg-blue-50 text-blue-800 border-blue-200",         short: "file_pinned" },
  catalog_hydrated:            { num: "2", label: "Hydrated",       color: "bg-blue-50 text-blue-800 border-blue-200",         short: "hydrated" },
  execution_strategy_planned:  { num: "2", label: "Exec Strategy",  color: "bg-indigo-50 text-indigo-800 border-indigo-200",   short: "exec_strat" },
  orchestration_confidence:    { num: "2", label: "Confidence",     color: "bg-teal-50 text-teal-800 border-teal-200",         short: "confidence" },
  confidence_degradation:      { num: "2", label: "Conf Degraded",  color: "bg-orange-50 text-orange-800 border-orange-200",   short: "conf_deg" },
  graph_health_issue:          { num: "2", label: "Graph Health ⚠", color: "bg-red-50 text-red-700 border-red-200",            short: "graph_health" },
  broaden_nudge_injected:      { num: "4", label: "Broaden Nudge",  color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200", short: "nudge" },
};

function pipelineSummary(ev: LogEntry): string {
  const e = ev.event as string;
  switch (e) {
    case "query_received":      return String(ev.query ?? "");
    case "catalog_loaded":      return `${ev.file_count} files in '${ev.container}' (${ev.parquet_count} parquet, ${ev.relationship_count} relationships)`;
    case "catalog_empty":       return "No files ingested yet — agent cannot answer";
    case "system_prompt_built": return `${ev.catalog_file_count} files | ${ev.parquet_file_count} parquet | ctx:${ev.has_conversation_context ? "yes" : "no"} | ${String(ev.system_prompt ?? "").length.toLocaleString()} chars`;
    case "search_catalog":      return `query="${ev.query}" → ${ev.files_found ?? (Array.isArray(ev.matched_files) ? (ev.matched_files as unknown[]).length : 0)} matches`;
    case "get_file_schema":     return `${ev.blob_path}${ev.found ? "" : " (NOT FOUND)"}`;
    case "llm_input":
    case "llm_stream_input":    return `iteration ${ev.iteration} · ${ev.message_count ?? (Array.isArray(ev.messages) ? (ev.messages as unknown[]).length : 0)} messages`;
    case "llm_output":
    case "llm_stream_output": {
      const tcs = (ev.tool_calls as unknown[]) || [];
      const toks = `${ev.prompt_tokens ?? "?"}+${ev.completion_tokens ?? "?"} tok`;
      if (tcs.length > 0) {
        const names = tcs.map((tc) => (tc as { name?: string }).name ?? "?").join(", ");
        return `iter ${ev.iteration} · ${toks} · → ${names}`;
      }
      return `iter ${ev.iteration} · ${toks} · final answer`;
    }
    case "tool_call_start":     return `${ev.tool} (iter ${ev.iteration})`;
    case "tool_call_end":       return `${ev.tool} done`;
    case "sql_execute_start":   return String(ev.sql ?? "").replace(/\s+/g, " ").slice(0, 120);
    case "sql_execute_done":    return `${ev.rows_returned}/${ev.total_rows} rows · ${ev.duration_ms} ms`;
    case "sql_execute_error":   return String(ev.error ?? "").slice(0, 120);
    case "inspect_data_format": return `${Array.isArray(ev.columns) ? (ev.columns as unknown[]).length : 0} cols · ${Array.isArray(ev.rows) ? (ev.rows as unknown[]).length : 0} sample rows`;
    case "summarise_dataframe_done": return `${ev.row_count} rows · ${ev.column_count} cols · focus=${ev.focus}`;
    case "ingest_llm_prompt":   return `${ev.filename} (~${ev.estimated_prompt_tokens} tok)`;
    case "ingest_llm_response": return `${ev.filename} · ${ev.duration_ms} ms`;
    case "final_answer":        return `${ev.tool_calls} tool calls · ${ev.row_count} rows · ${ev.total_duration_ms} ms`;

    // ── Orchestration pipeline stages ──────────────────────────────────────
    case "retrieval_filtered": {
      const topScores = (ev.top_scores as [string, number][]) || [];
      const top = topScores.length > 0 ? topScores[0][1] : "?";
      return `${ev.retrieved_files}/${ev.total_files} files · top RRF: ${top}`;
    }
    case "retrieval_fallback":
      return `fallback · ${ev.reason} · ${Array.isArray(ev.fallback_files) ? (ev.fallback_files as unknown[]).length : "?"} files`;
    case "resolver_pins_injected":
      return `pinned ${Array.isArray(ev.pinned) ? (ev.pinned as unknown[]).length : 0} files (${ev.path} path)`;
    case "prior_files_pinned":
      return `re-pinned ${Array.isArray(ev.pinned) ? (ev.pinned as unknown[]).length : 0} prior-turn files`;
    case "explicit_file_pinned":
      return `pinned: ${Array.isArray(ev.pinned) ? (ev.pinned as string[]).join(", ") : "?"}`;
    case "catalog_hydrated":
      return `shortlist ${ev.shortlist_size} · hydrated ${ev.hydrated_files} · ${ev.sample_rows_files} with sample rows`;
    case "execution_strategy_planned":
      return `mode=${ev.mode} · ${ev.clusters} cluster(s) [${Array.isArray(ev.cluster_sizes) ? (ev.cluster_sizes as number[]).join(", ") : "?"}]`;
    case "orchestration_confidence": {
      const level = ev.level as string;
      return `${level} · score ${ev.score} · graph=${ev.graph_health}`;
    }
    case "confidence_degradation":
      return `score ${ev.score} · ${Array.isArray(ev.chain) ? (ev.chain as string[]).join(" → ") : "?"}`;
    case "graph_health_issue":
      return `${ev.health_level} · coverage=${ev.edge_coverage} · p50=${ev.confidence_p50}`;
    case "broaden_nudge_injected":
      return `reason: ${ev.reason} · after ${ev.tool_call_count} tool calls`;

    default:                    return "";
  }
}

/* ─ small renderers used inside the expanded card ─ */

function PipelineCodeBlock({ children, lang }: { children: string; lang?: string }) {
  return (
    <pre className="bg-[#0d1117] border border-border rounded p-2 overflow-x-auto text-[11px] leading-relaxed text-zinc-200 font-mono whitespace-pre-wrap break-all">
      {lang && <div className="text-[9px] uppercase text-muted-foreground mb-1">{lang}</div>}
      <code>{children}</code>
    </pre>
  );
}

function PipelineRowsTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows || rows.length === 0) return <div className="text-[11px] text-muted-foreground">(no rows)</div>;
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto border border-border rounded">
      <table className="w-full text-[10px] font-mono">
        <thead className="bg-surface-raised text-muted-foreground">
          <tr>{cols.map((c) => <th key={c} className="px-2 py-1 text-left font-medium">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 20).map((r, i) => (
            <tr key={i} className="border-t border-border/50">
              {cols.map((c) => <td key={c} className="px-2 py-1 text-foreground">{String(r[c] ?? "")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 20 && (
        <div className="px-2 py-1 text-[10px] text-muted-foreground bg-surface-raised">
          … {rows.length - 20} more rows
        </div>
      )}
    </div>
  );
}

function PipelineEventDetail({ ev }: { ev: LogEntry }) {
  const e = ev.event as string;

  if (e === "query_received") {
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">User query:</div>
        <PipelineCodeBlock>{String(ev.query ?? "")}</PipelineCodeBlock>
        {ev.has_conversation_context ? (
          <>
            <div className="text-[11px] text-muted-foreground">Conversation context (preview):</div>
            <PipelineCodeBlock>{String(ev.conversation_context_preview ?? "")}</PipelineCodeBlock>
          </>
        ) : (
          <div className="text-[11px] text-muted-foreground">No conversation context</div>
        )}
      </div>
    );
  }

  if (e === "catalog_loaded") {
    const files = (ev.files as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Container:</span> <span className="text-foreground font-mono">{String(ev.container)}</span></div>
          <div><span className="text-muted-foreground">Files:</span> <span className="text-foreground font-mono">{String(ev.file_count)}</span></div>
          <div><span className="text-muted-foreground">Parquet:</span> <span className="text-foreground font-mono">{String(ev.parquet_count)}</span></div>
          <div><span className="text-muted-foreground">Relationships:</span> <span className="text-foreground font-mono">{String(ev.relationship_count)}</span></div>
        </div>
        <div className="text-[11px] text-muted-foreground">Files available to the agent:</div>
        <div className="border border-border rounded p-2 bg-[#0d1117] max-h-48 overflow-auto">
          {files.map((f, i) => (
            <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>
          ))}
        </div>
      </div>
    );
  }

  if (e === "system_prompt_built") {
    const prompt = String(ev.system_prompt ?? "");
    const chars = prompt.length;
    const words = prompt.split(/\s+/).filter(Boolean).length;
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Container:</span> <span className="text-foreground font-mono">{String(ev.container)}</span></div>
          <div><span className="text-muted-foreground">Files in catalog:</span> <span className="text-foreground font-mono">{String(ev.catalog_file_count)}</span></div>
          <div><span className="text-muted-foreground">Parquet-ready:</span> <span className="text-foreground font-mono">{String(ev.parquet_file_count)}</span></div>
          <div><span className="text-muted-foreground">Conv context:</span> <span className="text-foreground font-mono">{ev.has_conversation_context ? "yes" : "no"}</span></div>
          <div><span className="text-muted-foreground">Prompt size:</span> <span className="text-foreground font-mono">{chars.toLocaleString()} chars · ~{words.toLocaleString()} words</span></div>
        </div>
        <div className="text-[11px] text-muted-foreground">Full prompt sent to LLM:</div>
        <div className="max-h-72 overflow-auto">
          <PipelineCodeBlock>{prompt}</PipelineCodeBlock>
        </div>
      </div>
    );
  }

  if (e === "search_catalog") {
    const matched = (ev.matched_files as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="text-[11px]"><span className="text-muted-foreground">Query:</span> <span className="font-mono text-foreground">{String(ev.query)}</span></div>
        <div className="text-[11px] text-muted-foreground">Matched files (names only):</div>
        <div className="border border-border rounded p-2 bg-[#0d1117]">
          {matched.length === 0 ? <div className="text-[11px] text-muted-foreground">(no matches)</div> :
            matched.map((f, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>)}
        </div>
      </div>
    );
  }

  if (e === "get_file_schema") {
    const cols = (ev.columns as string[]) || [];
    const types = (ev.column_types as Record<string, string>) || {};
    const samples = (ev.sample_values as Record<string, unknown[]>) || {};
    return (
      <div className="space-y-2">
        <div className="text-[11px]">
          <span className="text-muted-foreground">File:</span>{" "}
          <span className="font-mono text-foreground">{String(ev.blob_path)}</span>{" "}
          <span className={cn("ml-2 px-1.5 py-0.5 rounded text-[10px] border",
            ev.found ? "bg-green-50 text-green-700 border-green-200"
                     : "bg-red-50 text-red-700 border-red-200")}>
            {ev.found ? "found" : "not found"}
          </span>
        </div>
        {cols.length > 0 && (
          <div className="border border-border rounded overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead className="bg-surface-raised text-muted-foreground">
                <tr><th className="px-2 py-1 text-left">column</th><th className="px-2 py-1 text-left">type</th><th className="px-2 py-1 text-left">sample</th></tr>
              </thead>
              <tbody>
                {cols.map((c) => (
                  <tr key={c} className="border-t border-border/50">
                    <td className="px-2 py-1 text-foreground">{c}</td>
                    <td className="px-2 py-1 text-cyan-700">{types[c] ?? ""}</td>
                    <td className="px-2 py-1 text-muted-foreground">{(samples[c] || []).slice(0,3).map(String).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }

  if (e === "llm_output" || e === "llm_stream_output") {
    const tcs = (ev.tool_calls as { name?: string; args?: Record<string, unknown> }[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Iteration:</span> <span className="font-mono">{String(ev.iteration)}</span></div>
          <div><span className="text-muted-foreground">Tokens:</span> <span className="font-mono">{String(ev.prompt_tokens)} + {String(ev.completion_tokens)}</span></div>
          <div><span className="text-muted-foreground">Duration:</span> <span className="font-mono">{String(ev.duration_ms ?? "?")} ms</span></div>
        </div>
        {tcs.length > 0 ? (
          <>
            <div className="text-[11px] text-muted-foreground">LLM decided to call {tcs.length} tool(s):</div>
            {tcs.map((tc, i) => (
              <div key={i} className="border border-border rounded p-2 bg-[#0d1117]">
                <div className="text-[11px] text-fuchsia-700 font-mono mb-1">→ {tc.name}</div>
                <PipelineCodeBlock lang="args">{JSON.stringify(tc.args ?? {}, null, 2)}</PipelineCodeBlock>
              </div>
            ))}
          </>
        ) : (
          <>
            <div className="text-[11px] text-muted-foreground">LLM decided: generate final answer</div>
            <PipelineCodeBlock>{String(ev.content ?? "")}</PipelineCodeBlock>
          </>
        )}
      </div>
    );
  }

  if (e === "llm_input" || e === "llm_stream_input") {
    const msgs = (ev.messages as { type: string; content: string; tool_calls?: unknown[] }[]) || [];
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">{msgs.length} messages going into the LLM:</div>
        <div className="max-h-72 overflow-auto space-y-1">
          {msgs.map((m, i) => (
            <div key={i} className="border border-border rounded p-2 bg-[#0d1117]">
              <div className="text-[10px] text-cyan-300 font-mono mb-1">[{i + 1}] {m.type}</div>
              <pre className="text-[11px] font-mono text-zinc-200 whitespace-pre-wrap break-all">{m.content?.slice(0, 800)}{(m.content?.length ?? 0) > 800 ? " …" : ""}</pre>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (e === "tool_call_start") {
    return (
      <div className="space-y-2">
        <div className="text-[11px]"><span className="text-muted-foreground">Tool:</span> <span className="font-mono text-fuchsia-700">{String(ev.tool)}</span></div>
        <PipelineCodeBlock lang="input">{JSON.stringify(ev.input ?? {}, null, 2)}</PipelineCodeBlock>
      </div>
    );
  }

  if (e === "tool_call_end") {
    const out = String(ev.output ?? "");
    return (
      <div className="space-y-2">
        <div className="text-[11px]"><span className="text-muted-foreground">Tool:</span> <span className="font-mono text-fuchsia-700">{String(ev.tool)}</span></div>
        <PipelineCodeBlock lang="output">{out.length > 4000 ? out.slice(0, 4000) + "\n… (truncated)" : out}</PipelineCodeBlock>
      </div>
    );
  }

  if (e === "sql_execute_start") {
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">SQL being executed:</div>
        <PipelineCodeBlock lang="sql">{String(ev.sql ?? "")}</PipelineCodeBlock>
      </div>
    );
  }

  if (e === "sql_execute_done") {
    const rows = (ev.preview_rows as Record<string, unknown>[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Returned:</span> <span className="font-mono text-emerald-700">{String(ev.rows_returned)}/{String(ev.total_rows)}</span></div>
          <div><span className="text-muted-foreground">Duration:</span> <span className="font-mono">{String(ev.duration_ms)} ms</span></div>
          <div><span className="text-muted-foreground">Columns:</span> <span className="font-mono">{Array.isArray(ev.columns) ? (ev.columns as unknown[]).length : 0}</span></div>
        </div>
        <PipelineRowsTable rows={rows} />
      </div>
    );
  }

  if (e === "sql_execute_error") {
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">SQL that failed:</div>
        <PipelineCodeBlock lang="sql">{String(ev.sql ?? "")}</PipelineCodeBlock>
        <div className="text-[11px] text-muted-foreground">Error:</div>
        <div className="border border-red-200 bg-red-50 rounded p-2 text-[11px] font-mono text-red-700 whitespace-pre-wrap break-all">
          {String(ev.error ?? "")}
        </div>
      </div>
    );
  }

  if (e === "final_answer") {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Tool calls:</span> <span className="font-mono">{String(ev.tool_calls)}</span></div>
          <div><span className="text-muted-foreground">Rows:</span> <span className="font-mono">{String(ev.row_count)}</span></div>
          <div><span className="text-muted-foreground">Total time:</span> <span className="font-mono text-emerald-700">{String(ev.total_duration_ms)} ms</span></div>
        </div>
        <div className="text-[11px] text-muted-foreground">Answer delivered to user:</div>
        <PipelineCodeBlock>{String(ev.answer ?? "")}</PipelineCodeBlock>
      </div>
    );
  }

  // ── Orchestration pipeline events ─────────────────────────────────────────

  if (e === "retrieval_filtered") {
    const topScores = (ev.top_scores as [string, number][]) || [];
    const lookups = (ev.lookup_slots_added as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Total catalog:</span> <span className="font-mono">{String(ev.total_files)}</span></div>
          <div><span className="text-muted-foreground">Shortlisted:</span> <span className="font-mono text-emerald-700">{String(ev.retrieved_files)}</span></div>
        </div>
        {topScores.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Top RRF scores:</div>
            <div className="border border-border rounded overflow-x-auto">
              <table className="w-full text-[10px] font-mono">
                <thead className="bg-surface-raised text-muted-foreground">
                  <tr><th className="px-2 py-1 text-left">file_id</th><th className="px-2 py-1 text-left">RRF score</th></tr>
                </thead>
                <tbody>
                  {topScores.map(([fid, score], i) => (
                    <tr key={i} className="border-t border-border/50">
                      <td className="px-2 py-1 text-foreground font-mono">{fid}</td>
                      <td className="px-2 py-1 text-cyan-700">{score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {lookups.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Lookup slots added ({lookups.length}):</div>
            <div className="border border-border rounded p-2 bg-[#0d1117]">
              {lookups.map((f, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>)}
            </div>
          </>
        )}
      </div>
    );
  }

  if (e === "retrieval_fallback") {
    const files = (ev.fallback_files as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Total catalog:</span> <span className="font-mono">{String(ev.total_files)}</span></div>
          <div><span className="text-muted-foreground">Fallback files:</span> <span className="font-mono text-orange-700">{String(files.length)}</span></div>
        </div>
        <div className="text-[11px]"><span className="text-muted-foreground">Reason:</span> <span className="font-mono text-orange-700">{String(ev.reason ?? "")}</span></div>
        {files.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Selected files:</div>
            <div className="border border-border rounded p-2 bg-[#0d1117] max-h-48 overflow-auto">
              {files.map((f, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>)}
            </div>
          </>
        )}
      </div>
    );
  }

  if (e === "resolver_pins_injected") {
    const pinned = (ev.pinned as string[]) || [];
    const entities = (ev.entities as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Path:</span> <span className="font-mono">{String(ev.path)}</span></div>
          <div><span className="text-muted-foreground">Files injected:</span> <span className="font-mono text-blue-700">{String(pinned.length)}</span></div>
        </div>
        <div className="text-[11px]"><span className="text-muted-foreground">Entities resolved:</span> <span className="font-mono">{entities.length > 0 ? entities.join(", ") : "—"}</span></div>
        <div className="text-[11px] text-muted-foreground">Pinned files:</div>
        <div className="border border-border rounded p-2 bg-[#0d1117]">
          {pinned.map((f, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>)}
        </div>
      </div>
    );
  }

  if (e === "prior_files_pinned") {
    const pinned = (ev.pinned as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">Files re-pinned from prior turns:</div>
        <div className="border border-border rounded p-2 bg-[#0d1117]">
          {pinned.length === 0
            ? <div className="text-[11px] text-muted-foreground">(none)</div>
            : pinned.map((f, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {f}</div>)}
        </div>
      </div>
    );
  }

  if (e === "explicit_file_pinned") {
    const pinned = (ev.pinned as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="text-[11px]"><span className="text-muted-foreground">User query:</span> <span className="font-mono text-foreground">{String(ev.query ?? "")}</span></div>
        <div className="text-[11px] text-muted-foreground">Files explicitly mentioned in query:</div>
        <div className="border border-border rounded p-2 bg-[#0d1117]">
          {pinned.map((f, i) => <div key={i} className="text-[11px] font-mono text-emerald-400">• {f}</div>)}
        </div>
      </div>
    );
  }

  if (e === "catalog_hydrated") {
    return (
      <div className="grid grid-cols-3 gap-3 text-[11px]">
        <div><span className="text-muted-foreground">Shortlist size:</span> <span className="font-mono">{String(ev.shortlist_size)}</span></div>
        <div><span className="text-muted-foreground">Hydrated:</span> <span className="font-mono text-emerald-700">{String(ev.hydrated_files)}</span></div>
        <div><span className="text-muted-foreground">With sample rows:</span> <span className="font-mono">{String(ev.sample_rows_files)} files</span></div>
      </div>
    );
  }

  if (e === "execution_strategy_planned") {
    const sizes = (ev.cluster_sizes as number[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Mode:</span> <span className="font-mono font-medium text-indigo-700">{String(ev.mode)}</span></div>
          <div><span className="text-muted-foreground">Clusters:</span> <span className="font-mono">{String(ev.clusters)}</span></div>
        </div>
        {sizes.length > 0 && (
          <div className="text-[11px]">
            <span className="text-muted-foreground">Cluster sizes:</span>{" "}
            <span className="font-mono">[{sizes.join(", ")}]</span>
          </div>
        )}
      </div>
    );
  }

  if (e === "orchestration_confidence") {
    const signals = (ev.signals as string[]) || [];
    const level = ev.level as string;
    const levelColor = level === "high" ? "text-emerald-700" : level === "medium" ? "text-amber-700" : "text-red-700";
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div>
            <span className="text-muted-foreground">Score:</span>{" "}
            <span className={cn("font-mono font-medium", levelColor)}>{String(ev.score)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Level:</span>{" "}
            <span className={cn("font-mono font-medium", levelColor)}>{level}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Graph health:</span>{" "}
            <span className="font-mono">{String(ev.graph_health)}</span>
          </div>
        </div>
        {signals.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Confidence signals:</div>
            <div className="border border-border rounded p-2 bg-[#0d1117]">
              {signals.map((s, i) => <div key={i} className="text-[11px] font-mono text-zinc-200">• {s}</div>)}
            </div>
          </>
        )}
      </div>
    );
  }

  if (e === "confidence_degradation") {
    const chain = (ev.chain as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Final score:</span> <span className="font-mono text-orange-700">{String(ev.score)}</span></div>
          <div><span className="text-muted-foreground">Avg ingestion:</span> <span className="font-mono">{String(ev.avg_ingestion)}</span></div>
        </div>
        {chain.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Degradation chain:</div>
            <div className="border border-orange-200 bg-orange-50 rounded p-2">
              {chain.map((step, i) => (
                <div key={i} className="text-[11px] font-mono text-orange-800">
                  {i > 0 && <span className="text-orange-400 mr-1">→</span>}{step}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  if (e === "graph_health_issue") {
    const flags = (ev.anomaly_flags as string[]) || [];
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-3 gap-2 text-[11px]">
          <div><span className="text-muted-foreground">Health level:</span> <span className="font-mono text-red-700">{String(ev.health_level)}</span></div>
          <div><span className="text-muted-foreground">Edge coverage:</span> <span className="font-mono">{String(ev.edge_coverage)}</span></div>
          <div><span className="text-muted-foreground">Conf p50:</span> <span className="font-mono">{String(ev.confidence_p50)}</span></div>
        </div>
        {flags.length > 0 && (
          <>
            <div className="text-[11px] text-muted-foreground">Anomaly flags:</div>
            <div className="border border-red-200 bg-red-50 rounded p-2">
              {flags.map((f, i) => <div key={i} className="text-[11px] font-mono text-red-700">• {f}</div>)}
            </div>
          </>
        )}
      </div>
    );
  }

  if (e === "broaden_nudge_injected") {
    return (
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div><span className="text-muted-foreground">Reason:</span> <span className="font-mono text-fuchsia-700">{String(ev.reason ?? "")}</span></div>
        <div><span className="text-muted-foreground">Tool calls used:</span> <span className="font-mono">{String(ev.tool_call_count)}</span></div>
      </div>
    );
  }

  // Fallback: dump the JSON
  return <PipelineCodeBlock>{JSON.stringify(ev, null, 2)}</PipelineCodeBlock>;
}

function PipelineEventRow({ ev }: { ev: LogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const eventName = (ev.event as string) || "";
  const cfg = PIPELINE_STEPS[eventName];
  const ts = formatTimestamp(ev.timestamp as string);
  const summary = pipelineSummary(ev);

  // Unknown / non-pipeline events get a muted row
  if (!cfg) {
    return (
      <div className="px-3 py-1.5 border-b border-border/40 text-[11px] font-mono text-muted-foreground">
        <span className="mr-2">{ts}</span>{eventName} <span className="opacity-60">{JSON.stringify(ev).slice(0, 120)}</span>
      </div>
    );
  }

  const isFinal = eventName === "final_answer";
  const isError = eventName === "sql_execute_error" || eventName === "catalog_empty";

  return (
    <div
      className={cn(
        "border-b border-border/50 transition-colors cursor-pointer hover:bg-surface-raised/40",
        expanded && "bg-surface-raised/30",
        isFinal && "bg-emerald-500/[0.04]",
        isError && "bg-red-500/[0.04]",
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2 px-3 py-1.5">
        {/* Time */}
        <span className="text-[10px] text-muted-foreground font-mono w-16 shrink-0">{ts}</span>

        {/* Step badge */}
        <span className={cn(
          "inline-flex items-center justify-center w-6 h-5 rounded text-[10px] font-bold border shrink-0",
          cfg.color,
        )}>
          {cfg.num}
        </span>

        {/* Step label */}
        <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-medium border shrink-0", cfg.color)}>
          {cfg.label}
        </span>

        {/* Summary */}
        <span className="text-xs text-foreground truncate font-mono">{summary}</span>

        {/* Expand */}
        <ChevronDown className={cn("w-3 h-3 text-muted-foreground transition-transform shrink-0 ml-auto", expanded && "rotate-180")} />
      </div>

      {expanded && (
        <div className="px-3 pb-3 pt-1 pl-[6.5rem]">
          <PipelineEventDetail ev={ev} />
        </div>
      )}
    </div>
  );
}

function PipelinePanel() {
  const [events, setEvents] = useState<LogEntry[]>([]);
  const [lines, setLines] = useState<number>(300);
  const [stepFilter, setStepFilter] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [totalLines, setTotalLines] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchPipeline = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/logs/ai-pipeline-events?lines=${lines}`);
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data: LogResponse = await res.json();
      setEvents(data.lines || []);
      setTotalLines(data.returned || 0);
      requestAnimationFrame(() => {
        if (containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight;
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch pipeline log");
    } finally {
      setLoading(false);
    }
  }, [lines]);

  useEffect(() => { fetchPipeline(); }, [fetchPipeline]);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchPipeline, 3000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, fetchPipeline]);

  const filtered = stepFilter === "all"
    ? events
    : events.filter((ev) => {
        const cfg = PIPELINE_STEPS[(ev.event as string) || ""];
        return cfg?.num === stepFilter;
      });

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="border-b border-border px-4 py-2 flex flex-wrap items-center gap-2 shrink-0">
        {/* Step filter chips */}
        <div className="flex items-center gap-1">
          {[
            { v: "all", label: "All" },
            { v: "1",   label: "1 · Query" },
            { v: "2",   label: "2 · Catalog" },
            { v: "3",   label: "3 · Prompt" },
            { v: "4",   label: "4 · LLM" },
            { v: "5",   label: "5 · SQL" },
            { v: "6",   label: "6 · Result" },
            { v: "✓",   label: "✓ Final" },
          ].map((s) => (
            <button
              key={s.v}
              onClick={() => setStepFilter(s.v)}
              className={cn(
                "px-2 py-1 rounded text-[11px] font-medium transition-colors",
                stepFilter === s.v
                  ? "bg-primary text-primary-foreground"
                  : "bg-surface-raised text-muted-foreground hover:text-foreground"
              )}
            >
              {s.label}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-border mx-1 hidden sm:block" />

        <select
          value={lines}
          onChange={(e) => setLines(Number(e.target.value))}
          className="px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground"
        >
          <option value={100}>100 events</option>
          <option value={300}>300 events</option>
          <option value={500}>500 events</option>
          <option value={1000}>1000 events</option>
        </select>

        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={cn(
            "px-2 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1",
            autoRefresh
              ? "bg-green-50 text-green-700 border border-green-200"
              : "bg-surface-raised text-muted-foreground hover:text-foreground"
          )}
        >
          <RefreshCw className={cn("w-3 h-3", autoRefresh && "animate-spin")} />
          {autoRefresh ? "Live" : "Auto"}
        </button>

        <button
          onClick={fetchPipeline}
          disabled={loading}
          className="p-1.5 rounded bg-surface-raised text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>

        <span className="text-[10px] text-muted-foreground ml-auto hidden sm:block">
          {filtered.length} / {totalLines} events
        </span>
      </div>

      {error && (
        <div className="mx-4 mt-2 px-3 py-2 rounded bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          {error}
        </div>
      )}

      <div ref={containerRef} className="flex-1 overflow-y-auto">
        {filtered.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
            <Activity className="w-8 h-8" />
            <p className="text-sm">No pipeline events — send a chat query to see the trace.</p>
          </div>
        )}
        {filtered.map((ev, i) => (
          <PipelineEventRow key={i} ev={ev} />
        ))}
      </div>
    </div>
  );
}

function PerformancePanel() {
  const [timings, setTimings] = useState<FileTiming[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTimings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/logs/file-timings?limit=50");
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setTimings(data.files);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch timings");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchTimings(); }, [fetchTimings]);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="border-b border-border px-4 py-2 flex items-center gap-2 shrink-0">
        <span className="text-xs text-muted-foreground">
          {timings.length} file{timings.length !== 1 && "s"} — most recent first
        </span>
        <button
          onClick={fetchTimings}
          disabled={loading}
          className="ml-auto p-1.5 rounded bg-surface-raised text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>
      </div>

      {error && (
        <div className="mx-4 mt-2 px-3 py-2 rounded bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-surface z-10">
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="px-4 py-2 font-medium">File</th>
              <th className="px-3 py-2 font-medium">Size</th>
              <th className="px-3 py-2 font-medium">Uploaded</th>
              <th className="px-3 py-2 font-medium text-center">Upload Time</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium text-center">Processing Time</th>
              <th className="px-3 py-2 font-medium text-center">Total Time</th>
            </tr>
          </thead>
          <tbody>
            {timings.map((t) => (
              <tr key={t.file_id} className="border-b border-border/50 hover:bg-surface-raised/50 transition-colors">
                <td className="px-4 py-2 text-foreground font-medium truncate max-w-[200px]" title={t.name}>
                  {t.name}
                </td>
                <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                  {formatBytes(t.size)}
                </td>
                <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                  {formatDateTime(t.uploaded_at)}
                </td>
                <td className="px-3 py-2 text-center">
                  {t.upload_secs !== null ? (
                    <span className="flex items-center justify-center gap-1 text-foreground font-mono">
                      <Upload className="w-3 h-3 text-blue-600" />
                      {formatSecs(t.upload_secs)}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <StatusBadge status={t.ingest_status} />
                  {t.parquet_status && t.parquet_status !== t.ingest_status && (
                    <span className="ml-1"><StatusBadge status={t.parquet_status} /></span>
                  )}
                </td>
                <td className="px-3 py-2 text-center">
                  {t.processing_secs !== null ? (
                    <span className="flex items-center justify-center gap-1 text-foreground font-mono">
                      <Zap className="w-3 h-3 text-yellow-600" />
                      {formatSecs(t.processing_secs)}
                      {t.ingestion_secs !== null && t.parquet_secs !== null && (
                        <span className="text-[9px] text-muted-foreground ml-0.5">
                          ({formatSecs(t.ingestion_secs)} + {formatSecs(t.parquet_secs)})
                        </span>
                      )}
                    </span>
                  ) : t.parquet_error ? (
                    <span className="text-red-700 text-[10px]" title={t.parquet_error}>Error</span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-3 py-2 text-center">
                  {t.total_secs !== null ? (
                    <span className="flex items-center justify-center gap-1 text-foreground font-mono font-semibold">
                      <Clock className="w-3 h-3 text-green-600" />
                      {formatSecs(t.total_secs)}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
              </tr>
            ))}
            {timings.length === 0 && !loading && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">
                  No files found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── download helper ──────────────────────────────────────────────────────────────── */

/* ── Ingestion panel helpers ──────────────────────────────────────────────────── */

function getIngestCfg(ev: LogEntry): { num: string; label: string; color: string } {
  const e = ev.event as string;
  if (e === "chain_start") return { num: "→", label: "Start",    color: "bg-cyan-50 text-cyan-800 border-cyan-200" };
  if (e === "chain_end")
    return ev.outcome === "error"
      ? { num: "✗", label: "Failed", color: "bg-red-50 text-red-700 border-red-200" }
      : { num: "✓", label: "Done",   color: "bg-emerald-50 text-emerald-800 border-emerald-200" };
  if (e === "chain_skip") return { num: "⊘", label: "Skipped", color: "bg-zinc-100 text-zinc-600 border-zinc-300" };
  if (e === "cleanup")    return { num: "·", label: "Cleanup",  color: "bg-zinc-100 text-zinc-600 border-zinc-300" };
  if (e === "ingest_stage") {
    const stage = (ev.stage as string) ?? "stage";
    const status = (ev.status as string) ?? "";
    const STAGE: Record<string, { num: string; label: string; color: string }> = {
      clean:          { num: "0", label: "Clean",      color: "bg-violet-50 text-violet-800 border-violet-200" },
      metadata:       { num: "1", label: "Metadata",   color: "bg-teal-50 text-teal-800 border-teal-200" },
      ai_description: { num: "2", label: "AI Desc",    color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200" },
      ontology:       { num: "3", label: "Roles",      color: "bg-cyan-50 text-cyan-800 border-cyan-200" },
      embedding:      { num: "4", label: "Embed",      color: "bg-indigo-50 text-indigo-800 border-indigo-200" },
      opensearch:     { num: "5", label: "Search",     color: "bg-blue-50 text-blue-800 border-blue-200" },
      parquet:        { num: "P", label: "Parquet",    color: "bg-orange-50 text-orange-800 border-orange-200" },
      analytics:      { num: "A", label: "Analytics",  color: "bg-amber-50 text-amber-800 border-amber-200" },
      relationships:  { num: "R", label: "Relations",  color: "bg-lime-50 text-lime-800 border-lime-200" },
      semantic_layer: { num: "S", label: "Semantic",   color: "bg-purple-50 text-purple-800 border-purple-200" },
      complete:       { num: "✓", label: "Complete",   color: "bg-emerald-50 text-emerald-800 border-emerald-200" },
    };
    const def = STAGE[stage] ?? { num: "·", label: stage, color: "bg-blue-50 text-blue-800 border-blue-200" };
    if (status === "failed")  return { ...def, color: "bg-red-50 text-red-700 border-red-200" };
    if (status === "skipped") return { ...def, color: "bg-zinc-100 text-zinc-600 border-zinc-300" };
    return def;
  }
  if (e === "ingest_stage_nonfatal_failed") return { num: "!", label: "Nonfatal", color: "bg-amber-50 text-amber-800 border-amber-200" };
  if (e === "metadata_schema_detected") return { num: "1", label: "Schema", color: "bg-teal-50 text-teal-800 border-teal-200" };
  if (e === "step") {
    const name = (ev.name as string) ?? "";
    const status = (ev.status as string) ?? "";
    const STEP: Record<string, { num: string; label: string; color: string }> = {
      probe:             { num: "0", label: "Probe",      color: "bg-zinc-100 text-zinc-700 border-zinc-300" },
      preprocess:        { num: "0", label: "Preprocess", color: "bg-violet-50 text-violet-800 border-violet-200" },
      duckdb_sample:     { num: "1", label: "DuckDB",     color: "bg-blue-50 text-blue-800 border-blue-200" },
      ai_description:    { num: "2", label: "AI Desc",    color: "bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200" },
      save_metadata:     { num: "3", label: "Metadata",   color: "bg-teal-50 text-teal-800 border-teal-200" },
      embed_metadata:    { num: "4", label: "Embed",      color: "bg-indigo-50 text-indigo-800 border-indigo-200" },
      compute_analytics: { num: "5", label: "Analytics",  color: "bg-amber-50 text-amber-800 border-amber-200" },
      parquet:           { num: "5", label: "Parquet",    color: "bg-orange-50 text-orange-800 border-orange-200" },
    };
    const def = STEP[name] ?? { num: "·", label: name, color: "bg-blue-50 text-blue-800 border-blue-200" };
    if (status === "failed")  return { ...def, color: "bg-red-50 text-red-700 border-red-200" };
    if (status === "skipped") return { ...def, color: "bg-zinc-100 text-zinc-600 border-zinc-300" };
    return def;
  }
  if (e === "preprocess")        return { num: "0", label: "Preprocess", color: "bg-violet-50 text-violet-800 border-violet-200" };
  if (e === "analytics_compute") return { num: "5", label: "Analytics",  color: "bg-amber-50 text-amber-800 border-amber-200" };
  if (e === "parquet_service" || e === "parquet_conversion") {
    const bad = (ev.status as string) === "failed" || ev.level === "error" || ev.level === "warning";
    return { num: "P", label: "Parquet", color: bad ? "bg-red-50 text-red-700 border-red-200" : "bg-orange-50 text-orange-800 border-orange-200" };
  }
  return { num: "·", label: e, color: "bg-zinc-100 text-zinc-700 border-zinc-300" };
}

function ingestSummary(ev: LogEntry): string {
  const e = ev.event as string;
  if (e === "chain_start") return `${ev.filename ?? ""} · ${ev.container ?? ""}`;
  if (e === "chain_end")
    return ev.outcome === "error"
      ? `FAILED: ${ev.filename ?? ""} — ${String(ev.error ?? "").slice(0, 100)}`
      : `Done in ${ev.total_duration_ms}ms — ${ev.filename ?? ""}`;
  if (e === "chain_skip") return `skipped: ${ev.reason ?? ""}`;
  if (e === "cleanup")    return String(ev.action ?? "");
  if (e === "ingest_stage") {
    const stage = (ev.stage as string) ?? "stage";
    const status = (ev.status as string) ?? "";
    const duration = ev.duration_ms ? ` · ${ev.duration_ms}ms` : "";
    if (status === "done") {
      if (stage === "clean" && ev.clean_rows) return `${ev.original_rows} → ${ev.clean_rows} rows${duration}`;
      if (stage === "metadata" && ev.row_count) return `${ev.columns} cols · ${ev.row_count} rows${duration}`;
      if (stage === "ontology") return `${ev.resolved ?? 0} roles · ${ev.source ?? ""}${duration}`;
      if (stage === "analytics") return `${ev.row_count ?? ""} rows${duration}`;
      if (stage === "relationships") return `${ev.relationships_created ?? 0} relationships${duration}`;
      return `done${duration}`;
    }
    return `${stage} · ${status}`;
  }
  if (e === "ingest_stage_nonfatal_failed") return `${ev.stage ?? "stage"}: ${String(ev.error ?? "").slice(0, 120)}`;
  if (e === "metadata_schema_detected") return `${ev.filename ?? ""} · ${ev.row_count ?? ""} rows`;
  if (e === "step") {
    const name = (ev.name as string) ?? "";
    const status = (ev.status as string) ?? "";
    if (name === "probe")          return `encoding=${ev.encoding} · safe=${ev.safe_for_raw_sample} · ${ev.reason ?? "ok"}`;
    if (name === "preprocess") {
      if (status.startsWith("done")) return `${ev.original_rows} → ${ev.clean_rows} rows · ${ev.duration_ms}ms`;
      if (status === "failed")       return `FAILED: ${String(ev.error ?? "").slice(0, 100)}`;
      if (status === "skipped")      return `skipped: ${ev.reason ?? ""}`;
      return `${status} (${ev.mode ?? ""})`;
    }
    if (name === "duckdb_sample")     return status === "done" ? `${ev.columns} cols · ${ev.row_count} rows` : status;
    if (name === "ai_description")    return status === "done" ? String(ev.summary ?? "").slice(0, 100) : status;
    if (name === "save_metadata")     return status === "done" ? String(ev.action ?? "") : status;
    if (name === "embed_metadata")    return status === "failed" ? `FAILED: ${ev.error}` : status === "done" ? "ok" : status;
    if (name === "compute_analytics") return status === "done" ? `${ev.row_count} rows · ${ev.duration_ms}ms` : status === "failed" ? `FAILED: ${String(ev.error ?? "").slice(0, 80)}` : status;
    if (name === "parquet")           return `${status}${ev.reason ? ` (${ev.reason})` : ""}`;
    return `${name} · ${status}`;
  }
  if (e === "preprocess")         return `${ev.status ?? ""} · ${ev.blob_path ?? ""}`;
  if (e === "analytics_compute")  return String(ev.status ?? "");
  if (e === "parquet_service")    return `${ev.step ?? ""} · ${ev.status ?? ""}`;
  if (e === "parquet_conversion") return `${ev.status ?? ""}${ev.job_id ? ` · job=${String(ev.job_id).slice(0, 8)}` : ""}`;
  return "";
}

function IngestEventRow({ ev }: { ev: LogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = getIngestCfg(ev);
  const ts = formatTimestamp(ev.timestamp as string);
  const summary = ingestSummary(ev);
  const isError = cfg.color.includes("red");
  const isDone = ev.event === "chain_end" && ev.outcome !== "error";
  const durationMs = (ev.duration_ms ?? ev.total_duration_ms) as number | undefined;
  const HIDE = new Set(["event", "level", "timestamp", "trace_id", "pipeline", "file_id"]);
  const extraKeys = Object.keys(ev).filter((k) => !HIDE.has(k) && ev[k] != null);
  return (
    <div
      className={cn(
        "border-b border-border/50 transition-colors cursor-pointer hover:bg-surface-raised/40",
        expanded && "bg-surface-raised/30",
        isError && "bg-red-500/[0.04]",
        isDone  && "bg-emerald-500/[0.03]",
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2 px-3 py-1.5">
        <span className="text-[10px] text-muted-foreground font-mono w-16 shrink-0">{ts}</span>
        <span className={cn("inline-flex items-center justify-center w-6 h-5 rounded text-[10px] font-bold border shrink-0", cfg.color)}>
          {cfg.num}
        </span>
        <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-medium border shrink-0", cfg.color)}>
          {cfg.label}
        </span>
        <span className="text-xs text-foreground truncate font-mono">{summary}</span>
        {durationMs !== undefined && (
          <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground shrink-0 ml-auto">
            <Clock className="w-3 h-3" />{formatDuration(durationMs)}
          </span>
        )}
        <ChevronDown className={cn("w-3 h-3 text-muted-foreground transition-transform shrink-0", expanded && "rotate-180")} />
      </div>
      {expanded && extraKeys.length > 0 && (
        <div className="px-3 pb-2 pt-1 pl-[6.5rem]">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
            {extraKeys.map((key) => (
              <div key={key} className="contents">
                <span className="text-muted-foreground font-mono">{key}</span>
                <span className={cn("font-mono break-all", key === "error" ? "text-red-700" : "text-foreground")}>
                  {typeof ev[key] === "object" ? JSON.stringify(ev[key]) : String(ev[key])}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function IngestionPanel() {
  const [events, setEvents] = useState<LogEntry[]>([]);
  const [lines, setLines] = useState(300);
  const [filter, setFilter] = useState<"all" | "errors" | "chain">("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [totalLines, setTotalLines] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchEvents = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await apiFetch(`/api/logs/ingest-events?lines=${lines}`);
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data: LogResponse = await res.json();
      setEvents(data.lines || []); setTotalLines(data.returned || 0);
      requestAnimationFrame(() => {
        if (containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight;
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch ingest events");
    } finally { setLoading(false); }
  }, [lines]);

  useEffect(() => { fetchEvents(); }, [fetchEvents]);
  useEffect(() => {
    if (autoRefresh) intervalRef.current = setInterval(fetchEvents, 3000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, fetchEvents]);

  const filtered = events.filter((ev) => {
    if (filter === "errors") return ev.level === "error" || ev.level === "warning" || ev.outcome === "error" || String(ev.status ?? "").includes("fail");
    if (filter === "chain")  return ev.event === "chain_start" || ev.event === "chain_end" || ev.event === "chain_skip";
    return true;
  });

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="border-b border-border px-4 py-2 flex flex-wrap items-center gap-2 shrink-0">
        <div className="flex items-center gap-1">
          {(["all", "errors", "chain"] as const).map((v) => (
            <button key={v} onClick={() => setFilter(v)}
              className={cn("px-2 py-1 rounded text-[11px] font-medium transition-colors",
                filter === v ? "bg-primary text-primary-foreground" : "bg-surface-raised text-muted-foreground hover:text-foreground")}>
              {v === "all" ? "All" : v === "errors" ? "Errors / Warnings" : "Completions"}
            </button>
          ))}
        </div>
        <div className="w-px h-5 bg-border mx-1 hidden sm:block" />
        <select value={lines} onChange={(e) => setLines(Number(e.target.value))}
          className="px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground">
          <option value={100}>100 events</option>
          <option value={300}>300 events</option>
          <option value={500}>500 events</option>
          <option value={1000}>1000 events</option>
        </select>
        <button onClick={() => setAutoRefresh(!autoRefresh)}
          className={cn("px-2 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1",
            autoRefresh ? "bg-green-50 text-green-700 border border-green-200" : "bg-surface-raised text-muted-foreground hover:text-foreground")}>
          <RefreshCw className={cn("w-3 h-3", autoRefresh && "animate-spin")} />
          {autoRefresh ? "Live" : "Auto"}
        </button>
        <button onClick={fetchEvents} disabled={loading}
          className="p-1.5 rounded bg-surface-raised text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50">
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>
        <span className="text-[10px] text-muted-foreground ml-auto hidden sm:block">
          {filtered.length} / {totalLines} events
        </span>
      </div>

      {error && (
        <div className="mx-4 mt-2 px-3 py-2 rounded bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />{error}
        </div>
      )}

      <div ref={containerRef} className="flex-1 overflow-y-auto">
        {filtered.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
            <Upload className="w-8 h-8" />
            <p className="text-sm">No ingestion events — upload a file to see the trace.</p>
          </div>
        )}
        {filtered.map((ev, i) => <IngestEventRow key={i} ev={ev} />)}
      </div>
    </div>
  );
}

function statusClass(status: number | null): string {
  if (!status) return "bg-zinc-100 text-zinc-600 border-zinc-300";
  if (status >= 500) return "bg-red-50 text-red-700 border-red-200";
  if (status >= 400) return "bg-yellow-50 text-yellow-700 border-yellow-200";
  return "bg-green-50 text-green-700 border-green-200";
}

function AuditRow({ row }: { row: AuditEntry }) {
  const [expanded, setExpanded] = useState(false);
  const req = row?.request;
  const details = row?.details || {};
  const status = req?.status_code ?? null;
  const actor = row?.actor ?? {};
  const actorLabel = actor.email || actor.user_id || "Anonymous";
  const domain = row?.domain_tag || "—";

  return (
    <div
      className={cn(
        "border-b border-border/50 transition-colors cursor-pointer hover:bg-surface-raised/40",
        expanded && "bg-surface-raised/30"
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="grid grid-cols-[84px_180px_1fr_80px_92px_24px] gap-2 items-center px-3 py-2 text-xs">
        <span className="text-[10px] text-muted-foreground font-mono">
          {formatTimestamp(row?.created_at || undefined)}
        </span>
        <div className="min-w-0">
          <p className="text-foreground truncate">{actorLabel}</p>
          <p className="text-[10px] text-muted-foreground truncate">{actor.role || "—"}</p>
        </div>
        <div className="min-w-0">
          <p className="font-mono text-foreground truncate">{row?.event}</p>
          <p className="font-mono text-[10px] text-muted-foreground truncate">{req?.path || "—"}</p>
        </div>
        <span className={cn("inline-flex justify-center px-1.5 py-0.5 rounded text-[10px] font-medium border", statusClass(status))}>
          {status ?? "—"}
        </span>
        <span className="text-[10px] text-muted-foreground truncate" title={domain}>{domain}</span>
        <ChevronDown className={cn("w-3 h-3 text-muted-foreground transition-transform", expanded && "rotate-180")} />
      </div>

      {expanded && (
        <div className="px-3 pb-3 pl-[17rem] text-[11px]">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono">
            <span className="text-muted-foreground">role</span><span className="text-foreground">{actor.role || "—"}</span>
            <span className="text-muted-foreground">method</span><span className="text-foreground">{req?.method || "—"}</span>
            <span className="text-muted-foreground">duration</span><span className="text-foreground">{formatDuration(req?.duration_ms ?? undefined)}</span>
            <span className="text-muted-foreground">ip</span><span className="text-foreground break-all">{req?.ip_address || "—"}</span>
            <span className="text-muted-foreground">route</span><span className="text-foreground break-all">{req?.route_template || "—"}</span>
            <span className="text-muted-foreground">file</span><span className="text-foreground break-all">{row?.file_name || row?.file_id || String(details.file_name ?? "") || "—"}</span>
            <span className="text-muted-foreground">folder</span><span className="text-foreground break-all">{String(details.folder_name ?? "") || String(details.folder_id ?? "") || "—"}</span>
            <span className="text-muted-foreground">container</span><span className="text-foreground break-all">{String(details.container_id ?? "") || "—"}</span>
            <span className="text-muted-foreground">target_user</span><span className="text-foreground break-all">{String(details.target_user_email ?? "") || String(details.target_user_id ?? "") || "—"}</span>
            {details.error != null && <><span className="text-muted-foreground">error</span><span className="text-red-700 break-all">{String(details.error)}</span></>}
            {Object.keys(details).filter(k => !["route_template","user_agent","folder_id","folder_name","container_id","target_user_id","target_user_email","target_user_name","file_name","error"].includes(k)).length > 0 && (
              <><span className="text-muted-foreground">details</span><span className="text-foreground break-all">{JSON.stringify(details)}</span></>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function AuditPanel() {
  const [rows, setRows] = useState<AuditEntry[]>([]);
  const [scope, setScope] = useState<AuditResponse["scope"]>("self");
  const [lines, setLines] = useState(200);
  const [userFilter, setUserFilter] = useState("");
  const [domainFilter, setDomainFilter] = useState("");
  const [actionFilter, setActionFilter] = useState("");
  const [pathFilter, setPathFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchAudit = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ lines: String(lines) });
      if (userFilter.trim()) params.set("email", userFilter.trim());
      if (domainFilter.trim()) params.set("domain", domainFilter.trim());
      if (actionFilter.trim()) params.set("event", actionFilter.trim());
      if (pathFilter.trim()) params.set("path", pathFilter.trim());
      const res = await apiFetch(`/api/logs/audit?${params.toString()}`);
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data: AuditResponse = await res.json();
      setRows(data.lines || []);
      setScope(data.scope);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch audit logs");
    } finally {
      setLoading(false);
    }
  }, [actionFilter, domainFilter, lines, pathFilter, userFilter]);

  useEffect(() => { fetchAudit(); }, [fetchAudit]);
  useEffect(() => {
    if (autoRefresh) intervalRef.current = setInterval(fetchAudit, 5000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, fetchAudit]);

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border px-4 py-2 flex flex-wrap items-center gap-2 shrink-0">
        <div className="relative min-w-[180px] max-w-xs flex-1">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <input
            value={userFilter}
            onChange={(e) => setUserFilter(e.target.value)}
            placeholder="User name or email"
            className="w-full pl-7 pr-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        <input
          value={domainFilter}
          onChange={(e) => setDomainFilter(e.target.value)}
          placeholder="Domain"
          className="w-32 px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <input
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
          placeholder="Action"
          className="w-40 px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <input
          value={pathFilter}
          onChange={(e) => setPathFilter(e.target.value)}
          placeholder="Path"
          className="w-40 px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <select value={lines} onChange={(e) => setLines(Number(e.target.value))}
          className="px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground">
          <option value={100}>100 rows</option>
          <option value={200}>200 rows</option>
          <option value={500}>500 rows</option>
          <option value={1000}>1000 rows</option>
        </select>
        <button onClick={() => setAutoRefresh(!autoRefresh)}
          className={cn("px-2 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1",
            autoRefresh ? "bg-green-50 text-green-700 border border-green-200" : "bg-surface-raised text-muted-foreground hover:text-foreground")}>
          <RefreshCw className={cn("w-3 h-3", autoRefresh && "animate-spin")} />
          {autoRefresh ? "Live" : "Auto"}
        </button>
        <button onClick={fetchAudit} disabled={loading}
          className="p-1.5 rounded bg-surface-raised text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50">
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>
        <span className="text-[10px] text-muted-foreground ml-auto hidden sm:block">
          {rows.length} rows · {scope}
        </span>
      </div>

      {error && (
        <div className="mx-4 mt-2 px-3 py-2 rounded bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />{error}
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {rows.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
            <Activity className="w-8 h-8" />
            <p className="text-sm">No audit events found</p>
          </div>
        )}
        {rows.filter((r): r is AuditEntry => r != null).map((row) => <AuditRow key={row.id} row={row} />)}
      </div>
    </div>
  );
}

const LOG_FILES = [
  { name: "llm_calls.log", label: "LLM Calls", description: "Token usage & timing" },
  { name: "costs.log",     label: "Costs",     description: "Billing events" },
];

export default function AdminLogsPage() {
  const { user } = useAuth();
  const [pageView, setPageView] = useState<PageView>("audit");
  const [activeFile, setActiveFile] = useState("llm_calls.log");
  const [lines, setLines] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [lineCount, setLineCount] = useState(200);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Non-admins can only see the Audit tab — redirect them if somehow on another tab.
    if (user && !user.is_admin && pageView !== "audit") setPageView("audit");
  }, [pageView, user]);

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let url: string;
      if (searchQuery.trim()) {
        url = `/api/logs/${activeFile}/search?q=${encodeURIComponent(searchQuery)}&lines=${lineCount}`;
      } else {
        url = `/api/logs/${activeFile}?lines=${lineCount}`;
      }
      const res = await apiFetch(url);
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body}`);
      }
      const data: LogResponse = await res.json();
      setLines(
        searchQuery.trim()
          ? (data.lines || []).map((line) => (line as { data?: LogEntry }).data || line)
          : (data.lines || [])
      );

      // Scroll to bottom
      requestAnimationFrame(() => {
        if (logContainerRef.current) {
          logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch logs");
    } finally {
      setLoading(false);
    }
  }, [activeFile, lineCount, searchQuery]);

  // Fetch on file/filter change
  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchLogs, 5000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, fetchLogs]);

  if (!user) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading logs…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border px-4 py-3 shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-foreground">Server Logs</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Activity, ingestion, LLM, and system events
          </p>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setPageView("audit")}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              pageView === "audit"
                ? "bg-primary text-primary-foreground"
                : "bg-surface-raised text-muted-foreground hover:text-foreground"
            )}
          >
            <Activity className="w-3.5 h-3.5 inline mr-1" />
            Audit
          </button>
          {user.is_admin && <button
            onClick={() => setPageView("pipeline")}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              pageView === "pipeline"
                ? "bg-primary text-primary-foreground"
                : "bg-surface-raised text-muted-foreground hover:text-foreground"
            )}
          >
            <Activity className="w-3.5 h-3.5 inline mr-1" />
            AI Pipeline
          </button>}
          {user.is_admin && <button
            onClick={() => setPageView("ingestion")}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              pageView === "ingestion"
                ? "bg-primary text-primary-foreground"
                : "bg-surface-raised text-muted-foreground hover:text-foreground"
            )}
          >
            <Upload className="w-3.5 h-3.5 inline mr-1" />
            Ingestion
          </button>}
          {user.is_admin && <button
            onClick={() => setPageView("logs")}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              pageView === "logs"
                ? "bg-primary text-primary-foreground"
                : "bg-surface-raised text-muted-foreground hover:text-foreground"
            )}
          >
            <FileText className="w-3.5 h-3.5 inline mr-1" />
            Logs
          </button>}
          {user.is_admin && <button
            onClick={() => setPageView("performance")}
            className={cn(
              "px-3 py-1.5 rounded text-xs font-medium transition-colors",
              pageView === "performance"
                ? "bg-primary text-primary-foreground"
                : "bg-surface-raised text-muted-foreground hover:text-foreground"
            )}
          >
            <Zap className="w-3.5 h-3.5 inline mr-1" />
            Performance
          </button>}
        </div>
      </div>

      {pageView === "audit" ? (
        <AuditPanel />
      ) : pageView === "pipeline" ? (
        <PipelinePanel />
      ) : pageView === "ingestion" ? (
        <IngestionPanel />
      ) : pageView === "performance" ? (
        <PerformancePanel />
      ) : (
      <>

      {/* Toolbar */}
      <div className="border-b border-border px-4 py-2 flex flex-wrap items-center gap-2 shrink-0">
        {/* Log file tabs */}
        <div className="flex gap-1">
          {LOG_FILES.map((f) => (
            <button
              key={f.name}
              onClick={() => {
                setActiveFile(f.name);
                setSearchQuery("");
              }}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition-colors",
                activeFile === f.name
                  ? "bg-primary text-primary-foreground"
                  : "bg-surface-raised text-muted-foreground hover:text-foreground"
              )}
              title={f.description}
            >
              {f.label}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-border mx-1 hidden sm:block" />

        {/* Search */}
        <div className="relative flex-1 min-w-[180px] max-w-xs">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && fetchLogs()}
            className="w-full pl-7 pr-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Lines selector */}
        <select
          value={lineCount}
          onChange={(e) => setLineCount(Number(e.target.value))}
          className="px-2 py-1 rounded bg-surface-raised border border-border text-xs text-foreground"
        >
          <option value={50}>50 lines</option>
          <option value={200}>200 lines</option>
          <option value={500}>500 lines</option>
          <option value={1000}>1000 lines</option>
        </select>

        {/* Auto refresh toggle */}
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={cn(
            "px-2 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1",
            autoRefresh
              ? "bg-green-50 text-green-700 border border-green-200"
              : "bg-surface-raised text-muted-foreground hover:text-foreground"
          )}
        >
          <RefreshCw className={cn("w-3 h-3", autoRefresh && "animate-spin")} />
          {autoRefresh ? "Live" : "Auto"}
        </button>

        {/* Manual refresh */}
        <button
          onClick={fetchLogs}
          disabled={loading}
          className="p-1.5 rounded bg-surface-raised text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
        </button>

        {/* Line count */}
        <span className="text-[10px] text-muted-foreground ml-auto hidden sm:block">
          {lines.length} rows
        </span>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mt-2 px-3 py-2 rounded bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          {error}
        </div>
      )}

      {/* Log entries */}
      <div
        ref={logContainerRef}
        className="flex-1 overflow-y-auto bg-[#0d1117] font-mono"
      >
        {lines.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground">
            <FileText className="w-8 h-8" />
            <p className="text-sm">No log entries found</p>
          </div>
        )}
        {lines.map((entry, i) => (
          <LogLine key={i} entry={entry} />
        ))}
      </div>
      </>
      )}
    </div>
  );
}

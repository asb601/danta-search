export interface ChartMeta {
  type: string;
  title?: string;
  x_column?: string;
  y_column?: string;
}

/**
 * Optional governance envelope attached to an assistant message when the
 * "visible governance" flag is ON. When absent/undefined the chat UI is
 * byte-identical — the renderer short-circuits and draws nothing.
 */
export interface Governance {
  mode: "answer" | "caveat" | "refusal";
  confidence: { level: "high" | "medium" | "low"; score: number };
  reason: string | null; // why caveated / refused
  files: { name: string; trust_state?: string }[];
  approved_joins: { from: string; to: string; on: string }[];
  feasibility: { answerable: boolean; note: string | null };
}

export interface AssistantPayload {
  answer: string;
  data: Record<string, unknown>[];
  chart: ChartMeta | null;
  row_count?: number;
  total_rows?: number;
  suggested_rephrase?: string | null;
  tool_calls?: number;
  files_used?: string[];
  retrieved_files?: number;
  total_files?: number;
  governance?: Governance;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload?: AssistantPayload;
  error?: boolean;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

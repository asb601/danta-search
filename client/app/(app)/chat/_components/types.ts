export interface ChartMeta {
  type: string;
  title?: string;
  x_column?: string;
  y_column?: string;
}

export interface AssistantPayload {
  answer: string;
  data: Record<string, unknown>[];
  result_sets?: Array<{
    title?: string;
    data: Record<string, unknown>[];
    row_count?: number;
    columns?: string[];
    files_used?: string[];
  }>;
  chart: ChartMeta | null;
  row_count?: number;
  total_rows?: number;
  suggested_rephrase?: string | null;
  tool_calls?: number;
  files_used?: string[];
  retrieved_files?: number;
  total_files?: number;
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

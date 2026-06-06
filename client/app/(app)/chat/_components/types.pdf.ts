// ─────────────────────────────────────────────────────────────────────────────
// PDF chat type surface (additive — mirrors server/pdf_chat/schemas/pdf_schemas.py
// and server/pdf_chat/models/enums.py). These types are isolated from the Excel
// chat types in types.ts so the existing analytics path is untouched.
// ─────────────────────────────────────────────────────────────────────────────

/** The active chat mode. "combined" is a dummy/disabled option (see ModeSwitcher). */
export type ChatMode = "excel" | "pdf" | "combined";

/** upload_manifest.status — document-level lifecycle (DocStatus enum). */
export type PdfDocStatus =
  | "uploaded"
  | "splitting"
  | "processing"
  | "indexed"
  | "partially_indexed"
  | "failed";

/** One row of GET /api/pdf/documents (DocumentSummary). There is no filename
 *  field — the upload_id is the label (and the same value cited as doc.id). */
export interface PdfDocument {
  upload_id: string;
  status: PdfDocStatus;
  page_count: number | null;
  mime_type: string | null;
  created_at: string | null;
}

/** One inline [N] citation pointing at a source chunk's document + page. */
export interface PdfCitation {
  n: number;
  doc_id: string;
  page: number;
}

/** POST /api/pdf/chat response (plain JSON, NOT SSE). */
export interface PdfChatResponse {
  answer: string;
  citations: PdfCitation[];
  chunks_used: number;
  cached: boolean;
}

/** POST /api/pdf/chat request body.
 *  tenant_id is optional on the backend — the route derives the trusted tenant
 *  from the JWT principal. We still pass user.organization_id (which equals the
 *  token's tenant) for explicitness; an empty/absent value tells the backend to
 *  use the principal's tenant. PDF scope is expressed via doc_ids, so we omit
 *  container_id (it defaults to the tenant server-side). */
export interface PdfChatRequestBody {
  query: string;
  tenant_id?: string;
  doc_ids?: string[];
}

/** POST /api/pdf/upload response (UploadResponse). */
export interface PdfUploadResponse {
  upload_id: string;
  status: PdfDocStatus;
  deduplicated: boolean;
}

/** GET /api/pdf/status/{upload_id} response (StatusResponse). */
export interface PdfStatusResponse {
  upload_id: string;
  status: PdfDocStatus;
  page_count: number;
  pages_succeeded: number;
  pages_failed: number;
  pages_pending: number;
  error_message: string | null;
}

/** An ephemeral PDF chat turn held only in component state (never persisted to
 *  the Excel conversation sidebar). */
export interface PdfMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: PdfCitation[];
  chunksUsed?: number;
  cached?: boolean;
  error?: boolean;
}

/** Document statuses that are usable as a retrieval scope. */
export const PDF_SELECTABLE_STATUSES: ReadonlySet<PdfDocStatus> = new Set([
  "indexed",
  "partially_indexed",
]);

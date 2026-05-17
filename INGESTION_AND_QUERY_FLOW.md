# G-CHAT Backend: Ingestion Pipeline & Query Flow

> Deep technical reference for the two core backend flows: **file ingestion** and **chat query execution**. Covers every stage, every fallback, every prompt, and how the agent decides its next step.

---

## Table of Contents

1. [Ingestion Pipeline — End to End](#1-ingestion-pipeline)
   - [1.1 Trigger](#11-trigger)
   - [1.2 Celery Task Graph (11 stages)](#12-celery-task-graph)
   - [1.3 Stage-by-Stage Detail](#13-stage-by-stage-detail)
   - [1.4 Retry / Failure Handling](#14-retry--failure-handling)
   - [1.5 What Each Stage Writes to the DB](#15-what-each-stage-writes-to-the-db)

2. [Query Pipeline — End to End](#2-query-pipeline)
   - [2.1 HTTP Entry Point](#21-http-entry-point)
   - [2.2 Pre-Flight: Cache + Concurrency Guard](#22-pre-flight-cache--concurrency-guard)
  - [2.3 Retrieval + Hydration](#23-retrieval-9-stage-pipeline--hydration)
   - [2.4 Semantic Planner Fast Path](#24-semantic-planner-fast-path)
   - [2.5 LangGraph Agent](#25-langgraph-agent)
   - [2.6 The System Prompt (Full Anatomy)](#26-the-system-prompt-full-anatomy)
   - [2.7 How the Agent Decides the Next Step](#27-how-the-agent-decides-the-next-step)
   - [2.8 All Agent Tools](#28-all-agent-tools)
   - [2.9 SQL Safety Layer](#29-sql-safety-layer)
   - [2.10 Answer Extraction & Fallbacks](#210-answer-extraction--fallbacks)
   - [2.11 Streaming Response Back to Client](#211-streaming-response-back-to-client)

3. [Data Models Referenced](#3-data-models-referenced)
4. [Supporting Statement: Why This Is the Right Production Direction](#4-supporting-statement-why-this-is-the-right-production-direction)

---

## 1. Ingestion Pipeline

### 1.1 Trigger

**API call:**  
```
POST /chat/ingest
body: { file_id: "<uuid>" }
```

The `ingest.py` route calls:
```python
run_ingest_pipeline.delay(file_id)
```

`run_ingest_pipeline` is a **Celery task** (`gchat.ingest_pipeline`) on queue `ingest_normal`. It is the only externally visible task name — all 11 stages are internal.

---

### 1.2 Celery Task Graph

```
run_ingest_pipeline(file_id)
       │
       ▼
prepare_pipeline()   ← sets file.ingest_status = "pending", returns early if already pending
       │
       ▼ (Celery chain — each task receives the previous task's Payload dict)
┌─────────────────────────────────────────────────────────────────┐
│  1. clean_file_task           gchat.ingest.clean               │
│  2. parquet_task              gchat.ingest.parquet             │
│  3. metadata_task             gchat.ingest.metadata            │
│  4. ai_description_task       gchat.ingest.ai_description      │
│  5. ontology_task             gchat.ingest.ontology            │
│  6. embedding_task            gchat.ingest.embedding           │
│  7. opensearch_index_task     gchat.ingest.opensearch_index    │
│  8. analytics_task            gchat.ingest.analytics           │
│  9. relationship_task         gchat.ingest.relationships       │
│ 10. semantic_layer_task       gchat.ingest.semantic_layer      │
│ 11. complete_ingestion_task   gchat.ingest.complete            │
└─────────────────────────────────────────────────────────────────┘
```

**All tasks share these Celery options:**
| Option | Value |
|---|---|
| `acks_late` | `True` — message not acknowledged until task completes |
| `reject_on_worker_lost` | `True` — requeued if worker dies mid-task |
| `max_retries` | 3 |
| `retry_backoff` | exponential (base 30s, max 300s) |
| `queue` | `ingest_normal` |
| `soft_time_limit` | 45 minutes |
| `time_limit` | 50 minutes |
| `worker_prefetch_multiplier` | 1 (no speculative prefetch) |

**Payload contract:**  
Each task receives the dict returned by the previous task. The `_run_stage()` helper short-circuits the entire rest of the chain if `payload["status"] == "failed"` — avoiding pointless retries downstream when a hard error has already been recorded.

```python
def _run_stage(task, stage, payload, func):
    if isinstance(payload, dict) and payload.get("status") == "failed":
        return payload   # pass-through — skip this stage
    ...
    try:
        return _run_async(func(stage_payload))
    except (SoftTimeLimitExceeded, Exception) as exc:
        if task.request.retries >= task.max_retries:
            asyncio.run(mark_ingestion_failed(file_id, stage, exc))
            return _failed_payload(file_id, stage, exc, task.request.retries)
        raise task.retry(exc=exc)
```

---

### 1.3 Stage-by-Stage Detail

#### Stage 1 — Clean (`gchat.ingest.clean`)
**File:** `ingestion_stages.py → clean_file_stage()`

What it does:
- Checks if the file extension is in `_PREPROCESS_EXTS` (CSV, TSV, TXT, etc.) and `is_preprocessed=False`.
- If yes → calls `preprocess_file()` from `data_preprocessor.py`.
  - Preprocessor applies `cleaning_config` rules from `ContainerConfig` (e.g. null fill, deduplication, column renames).
  - Writes a cleaned blob back to Azure with a `_clean` suffix.
  - Returns: `clean_blob_path`, `original_rows`, `clean_rows`, `quarantine_count`, `quarantine_sample`, `cleaning_audit`.
- Updates `file.blob_path` → `clean_blob_path`, `file.is_preprocessed = True`.
- Writes `FileAnalytics.quarantine_count`, `quarantine_sample`, `cleaning_audit`.
- If blob does not exist in Azure → marks `ingest_status = "not_ingested"`, raises `RuntimeError`.
- If file is already preprocessed or extension not supported → skips, passes original blob_path forward.

**Fallback:** Blob existence check (`_blob_exists()`) via Azure SDK before raising.

---

#### Stage 2 — Parquet (`gchat.ingest.parquet`)
**File:** `ingestion_stages.py → parquet_stage()`

What it does:
- Checks `FileAnalytics.parquet_blob_path` — if already exists, skips (idempotent).
- If file extension is not CSV-like (`_CSV_LIKE_EXTS = {.csv, .tsv, .txt}`) → skips.
- Calls `trigger_parquet_conversion()` from `analytics_service.py`.
  - Reads CSV from Azure Blob via **PyArrow streaming** in 128–256 MB chunks — zero disk usage.
  - Converts to Parquet with Snappy compression.
  - Writes Parquet blob back to Azure with `_converted.parquet` suffix.
- Writes `FileAnalytics.parquet_blob_path`, `parquet_size_bytes`.
- Passes `parquet_blob_path` in the payload for downstream stages.

**Why Parquet matters downstream:** Every SQL query at chat time uses `read_parquet(...)` instead of `read_csv_auto(...)` — 10–50× faster reads.

---

#### Stage 3 — Metadata (`gchat.ingest.metadata`)
**File:** `ingestion_stages.py → metadata_stage()`

What it does:
- Uses **DuckDB** to sample 500 rows from the clean blob (CSV or Parquet).
- Calls `sample_file()` from `duckdb_client.py` → returns column names, dtypes, sample rows.
- Extracts: `row_count`, `column_count`, `column_names`, `columns_info` (name + dtype + samples + uniques).
- If file is a schema glossary (`_is_schema_file()` heuristic: filename contains "schema", "glossary", "dictionary") → loads it with `_load_schema_glossary()` and writes `schema_glossary` to `FileMetadata`.
- Writes all extracted fields to `FileMetadata`.

**DuckDB sample query:**
```sql
SELECT * FROM read_csv_auto('az://CONTAINER/blob.csv', sample_size=500) LIMIT 500
```

---

#### Stage 4 — AI Description (`gchat.ingest.ai_description`)
**File:** `ingestion_stages.py → ai_description_stage()`

What it does:
- Calls `generate_file_description()` from `ai_client.py`.
- Sends a single `gpt-4o-mini` call with:
  - File name, column names + dtypes
  - 5–10 sample rows from `columns_info`
  - Container description (if any)
  - Existing schema glossary entries matching the file's columns
- The LLM returns a structured JSON with:
  - `ai_description` — narrative description of what the file contains
  - `good_for` — list of 3–5 analytical questions it can answer
  - `key_metrics` — columns that represent measurable quantities
  - `key_dimensions` — columns used for slicing/grouping
  - `date_range_start`, `date_range_end` — inferred from sample values
  - `domain_tag` — business domain (finance, supply_chain, hr, etc.)
- Writes all these to `FileMetadata`.

**Prompt used (inside `ai_client.py`):**
```
You are a data analyst. Given this file's name, columns, and sample rows,
generate a business description.

File: {filename}
Columns: {column list with dtypes}
Sample rows: {5 rows as JSON}
Schema glossary entries: {matching glossary terms}

Return JSON with keys:
  description, good_for, key_metrics, key_dimensions,
  date_range_start, date_range_end, domain_tag
```

---

#### Stage 5 — Ontology / Column Roles (`gchat.ingest.ontology`)
**File:** `ingestion_stages.py → ontology_stage()` → calls `column_role_resolver.py`

What it does:
- Calls `resolve_column_roles()` which makes a **single LLM call** (gpt-4o-mini) with all column names + dtypes.
- The resolver starts with per-container roles from `container_configs.semantic_config` if the tenant has configured them.
- If no configured role fits, the resolver emits a typed dynamic role. There is no shipped business-role list in code.
- Dynamic role format:
  ```text
  custom:<kind>:<short_snake_case_label>
  ```
  Supported kinds are `entity_key`, `reference_key`, `additive_measure`, `non_additive_measure`, `date`, and `attribute`.
- Example dynamic roles: `custom:entity_key:claim`, `custom:additive_measure:premium`, `custom:non_additive_measure:exchange_rate`, `custom:date:service_date`.
- Returns `(roles_dict, source_string)` where `roles_dict = {column_name: role}`.
- Writes `FileMetadata.column_semantic_roles`, `role_source`.

**Why roles matter:**
- They power the ER relationship detector (fingerprinting matching `primary_key` ↔ `foreign_key`).
- They are the input to `semantic_layer_builder.py` for cardinality inference.
- They help the agent understand which columns are metrics vs dimensions without reading all rows.

---

#### Stage 6 — Embedding (`gchat.ingest.embedding`)
**File:** `ingestion_stages.py → embedding_stage()`

What it does:
- Builds a "search text" from metadata: file name + description + good_for + key_metrics + key_dimensions + column names.
- Calls Azure OpenAI text-embedding-ada-002 (1536-dim).
- Writes the vector to `FileMetadata.description_embedding` (pgvector `Vector(1536)` column).
- Used by vector search in the retrieval pipeline.

**`build_search_text()` function:**
```python
" ".join([
    file.name,
    meta.ai_description or "",
    " ".join(meta.good_for or []),
    " ".join(meta.key_metrics or []),
    " ".join(meta.key_dimensions or []),
    " ".join(col_names[:40]),
])
```

---

#### Stage 7 — OpenSearch Index (`gchat.ingest.opensearch_index`)
**File:** `ingestion_stages.py → opensearch_stage()` → `retrieval/opensearch_indexer.py`

What it does:
- If `OPENSEARCH_URL` is not configured → logs a skip, continues.
- Otherwise: indexes the file's metadata into a **per-container OpenSearch index** (`gchat_{container_id}`).
- Document fields: `file_id`, `blob_path`, `description`, `good_for`, `key_metrics`, `key_dimensions`, `column_names`, `domain_tag`, `date_range_start`, `date_range_end`.
- The index supports BM25 + fuzzy + dense vector hybrid search at query time.

**Fallback:** If OpenSearch is not available, the entire retrieval falls back to PostgreSQL (BM25 via tsvector, fuzzy via pg_trgm, vector via pgvector). Ingestion never fails because of OpenSearch.

---

#### Stage 8 — Analytics (`gchat.ingest.analytics`)
**File:** `ingestion_stages.py → analytics_stage()` → `analytics_service.py`

What it does:
- Calls `compute_and_store_analytics()`.
- Runs DuckDB SQL on the 500-row sample (or full file if small) to compute:
  - Per-column stats: min, max, mean, std, null count, unique count.
  - Top-10 value counts per categorical column.
  - Cross-tab pairs for key dimension columns.
- Writes to `FileAnalytics.column_stats`, `value_counts`, `cross_tabs`.
- These are used by the `query_precomputed_analytics` agent tool — returns instantly without hitting Azure at chat time.

---

#### Stage 9 — Relationship Detection (`gchat.ingest.relationships`)
**File:** `ingestion_stages.py → relationship_stage()` → `relationship_detector.py`

What it does:

**Pass 1: Key registration**
- Uses `semantic_roles.py` behavior metadata to decide which columns are relationship-capable keys.
- Registers only `entity_key` and `reference_key` columns in `ColumnKeyRegistry`.
- Measure, date, and attribute roles do not become joins just because two files share a label.

**Pass 2: Fingerprint-based relationship evidence**
- Calls `register_file_key_fingerprints()` → samples policy-limited distinct values from candidate key columns.
- Normalizes each value: `normalize_key_value()` — trim, lowercase, collapse whitespace, and normalize integer identifiers with leading zeros.
- Stores normalized values as a PostgreSQL array in `ColumnKeyRegistry`.
- Finds matches using `find_fingerprint_matches()`: GIN array overlap query (`&&` operator).
- Value overlap percentage is computed and compared against `SemanticPolicy.min_value_overlap`.
- A role name alone no longer creates a relationship. It only makes a column eligible for the indexed value-overlap check.

**Writes to `FileRelationship`:**
- `file_a_id`, `file_b_id`, `shared_column`, `related_column`
- `semantic_role` — the typed semantic role (for example `custom:entity_key:customer`)
- `confidence_score` — 0.0–1.0
- `value_overlap_pct` — fraction of values shared
- `join_type` — inferred (`inner`, `left`, etc.)

**Minimum confidence threshold:** `SemanticPolicy.min_relationship_confidence` — configurable with `GCHAT_SEMANTIC_MIN_RELATIONSHIP_CONFIDENCE`.

---

#### Stage 10 — Semantic Layer (`gchat.ingest.semantic_layer`)
**File:** `ingestion_stages.py → semantic_layer_stage()` → `semantic_layer_builder.py → build_semantic_layer_for_file()`

What it does:
- Loads this file's `FileMetadata` (column roles, key_metrics, key_dimensions, confidence scores).
- Calls `infer_entity_spec()`:
  - Determines `entity_name` from the dominant `entity_key` role label.
  - Classifies columns into `metrics`, `dimensions`, `attributes` buckets using the centralized role groups in `semantic_roles.py`.
  - Infers `grain` (the level of detail one row represents, e.g. "one invoice line per vendor per date").
- Writes to `SemanticEntity`: `entity_name`, `primary_key`, `attributes`, `metrics`, `dimensions`, `grain`, `confidence_score`, `status=active`.

Then for each `FileRelationship` involving this file:
- Calls `_relationship_type()`:
  - pk→fk = `one_to_many`
  - fk→pk = `many_to_one`
  - pk→pk = `one_to_one`
  - else = `many_to_many`
- Calls `_approval_status()`:
  - **Approved** if the relationship passes the thresholds from `semantic_policy.py` and the role is not risky as a single-column join.
  - **Candidate** if: the role is risky as a single-column join, the relationship is `many_to_many`, or confidence is low.
  - Risky roles get a `risk_reason` explaining why the join may fan out incorrectly.
- Writes to `SemanticRelationship`: `from_entity`, `to_entity`, `from_column`, `to_column`, `relationship_type`, `approval_status`, `risk_reason`, `join_rule` (JSONB with the exact join predicate), `confidence_score`.

**The critical distinction:**
- `FileRelationship` = "these two files CAN be joined (technical fact)"
- `SemanticRelationship.approval_status = approved` = "this join is SAFE for business analytics"
- `SemanticRelationship.approval_status = candidate` = "this join exists but may produce wrong results — human review needed"

**Example of a risky join:**
```
invoice.currency_code  ←→  fx_rate.currency_code
```
`currency_code` can resolve to `custom:reference_key:currency`. `fx_rate` has multiple rows per currency (one per date) → `many_to_many` → financial metric fan-out. `risk_reason = "custom:reference_key:currency alone is not a safe business join key"`. Status = `candidate`.

---

#### Stage 11 — Complete (`gchat.ingest.complete`)
**File:** `ingestion_stages.py → complete_ingestion_stage()`

What it does:
- Sets `file.ingest_status = "ingested"`, `file.ingested_at = datetime.now()`.
- Sets `FileMetadata.ingestion_complete = True`.
- Writes the final payload summary to the file record.
- Calls `invalidate_catalog_cache()` (now a no-op — cache was removed, but kept for compatibility).
- Logs `ingest_complete` with total duration and all stage timings from the payload.

---

### 1.4 Retry / Failure Handling

```
Stage fails with exception
       │
       ▼
task.request.retries < max_retries?
  YES → raise task.retry(exc=exc)   ← Celery re-queues with exponential backoff
  NO  → asyncio.run(mark_ingestion_failed(file_id, stage, exc))
        return _failed_payload(...)   ← passed to all downstream stages
                                        which detect status=="failed" and skip
```

`mark_ingestion_failed()`:
- Sets `file.ingest_status = "failed"`.
- Sets `FileMetadata.ingestion_error = {"stage": stage, "error": str(exc)[:500]}`.
- Commits to DB.

Every downstream task that receives a failed payload:
```python
if isinstance(payload, dict) and payload.get("status") == "failed":
    return payload  # skip silently
```
So a failure in stage 4 (ai_description) still lets stages 5–11 run (most are idempotent) — except `_run_stage()` short-circuits the whole chain on `status=failed`, meaning only the `complete_ingestion_task` runs in degraded mode to set the final status.

---

### 1.5 What Each Stage Writes to the DB

| Stage | Table(s) Written |
|---|---|
| clean | `files.blob_path`, `files.is_preprocessed`, `file_analytics.quarantine_count/sample/audit` |
| parquet | `file_analytics.parquet_blob_path`, `parquet_size_bytes` |
| metadata | `file_metadata.row_count`, `column_count`, `columns_info`, `column_names`, `schema_glossary` |
| ai_description | `file_metadata.ai_description`, `good_for`, `key_metrics`, `key_dimensions`, `date_range_*`, `domain_tag` |
| ontology | `file_metadata.column_roles`, `column_role_source` |
| embedding | `file_metadata.description_embedding` (pgvector) |
| opensearch | OpenSearch index `gchat_{container_id}` |
| analytics | `file_analytics.column_stats`, `value_counts`, `cross_tabs` |
| relationships | `file_relationships` + `column_key_registry` |
| semantic_layer | `semantic_entities` + `semantic_relationships` |
| complete | `files.ingest_status=ingested`, `files.ingested_at`, `file_metadata.ingestion_complete=True` |

---

## 2. Query Pipeline

### 2.1 HTTP Entry Point

Two endpoints — same logic, different transport:

**SSE Streaming:**
```
POST /chat/message/stream
```
Returns `text/event-stream` SSE. Tokens arrive as the LLM generates them. This is the primary production path.

**Non-streaming (deprecated path):**
```
POST /chat/message
```
Waits for the full answer, returns JSON. Used for simple clients.

**Request body (`ChatMessageRequest`):**
```json
{
  "query": "What was total invoice amount by vendor last month?",
  "conversation_id": "<uuid or null>",
  "container_id": "<uuid or null — admin only>"
}
```

**Authentication:** Bearer JWT. `get_current_user()` dependency extracts `user_id`, `is_admin`, `allowed_domains` from the token.

**Validations:**
- `query` must not be empty, max 2000 chars.
- `conversation_id` must belong to the authenticated user (if provided).
- Archived conversations return 410.
- If `msg_count >= MAX_MESSAGES_PER_CONVERSATION` → auto-archive old conversation, create new one with summary transferred.

---

### 2.2 Pre-Flight: Cache + Concurrency Guard

**Concurrency guard:**
```python
_LLM_SEMAPHORE = asyncio.Semaphore(5)
```
At most 5 simultaneous LLM requests per worker. If all slots are taken, return `503 Service Unavailable` with `Retry-After: 10` header immediately. This prevents Azure OpenAI quota exhaustion from request queuing.

**Response cache:**
- Key: `(container_id or "default", normalize(query))`
- `normalize()` = lowercase + strip punctuation + collapse whitespace
- TTL: 10 minutes, max 500 entries (FIFO eviction), fuzzy match threshold: 0.92 (SequenceMatcher ratio).
- Fuzzy scan: last 50 entries only (bounded O(50) check).
- **Never caches hollow answers** — any answer containing phrases like `"wasn't able to find"`, `"no data found"`, `"unfortunately"` is excluded.
- If cache hit → stream the cached answer as 80-char token chunks immediately, skip all downstream work.

---

### 2.3 Retrieval (9-Stage Pipeline) + Hydration

**Entry:** `retrieval/orchestrator.py → retrieve_with_scores()`

```
retrieve_with_scores(query, user_id, is_admin, db, top_k=20, container_id=None)
```

**Stage 1: Domain restriction check**
- If `is_admin=False` → load `user.allowed_domains` from DB.
- All subsequent DB queries are filtered to `domain_tag IN allowed_domains` (if set).

**Stage 2: Temporal parsing**
- `parse_temporal(query)` — pure regex, <1ms, no LLM.
- Extracts `date_from`, `date_to` from phrases like "last month", "Q3 2024", "since January", "YTD".
- These become SQL `WHERE date_range_start <= :date_to AND date_range_end >= :date_from` clauses in DB retrieval.

**Stages 3–5: Permission + date baked into `build_base_query()`**
- Every DB retrieval query automatically applies:
  - `files.folder_id` → check if the user owns the folder or has access (admin bypasses).
  - `file_metadata.domain_tag = ANY(:allowed_domains)` if domain restriction is set.
  - Date overlap filter if `date_from/date_to` are set.

**OpenSearch fast path (when available):**
```python
can_use_opensearch = bool(container_id) and (is_admin or bool(allowed_domains))
```
If yes → calls `opensearch_retrieve_with_scores()` which runs BM25 + fuzzy + vector on the per-container OpenSearch index. If OpenSearch returns results, PostgreSQL BM25/fuzzy/vector stages are skipped, but semantic graph expansion still runs on the returned seed files.

This means semantic neighbours are not skipped just because metadata retrieval came from OpenSearch.

**Stage 4: BM25 (`bm25_search`)**
- PostgreSQL tsvector GIN index.
- Query: `to_tsvector('english', search_text) @@ plainto_tsquery('english', :query)`
- Returns up to `_STAGE_LIMIT = 50` results with BM25 rank.

**Stage 5: Fuzzy (`fuzzy_search`)**
- pg_trgm trigram similarity GIN index.
- `similarity(search_text, :query) > 0.15`
- Handles typos, partial matches, acronym expansion.

**Stage 6: Vector (`vector_search`)**
- pgvector HNSW index on `description_embedding`.
- Query embedding built from `build_search_text(query_as_fake_entry)`.
- Cosine similarity, top 50.

> **Note:** Stages 4–6 run **sequentially** on the same SQLAlchemy `AsyncSession` — async sessions share one connection and cannot run concurrent operations.

**Stage 7: Approved Semantic Graph Expansion (`graph_expand`)**
- Takes the top fused seed files from BM25 + fuzzy + vector retrieval.
- Expands one hop only through `semantic_relationships` where:
  - `status = active`
  - `approval_status = approved`
  - `confidence_score >= SemanticPolicy.graph_expand_min_confidence`
- Uses the same permission, domain, and container filters before returning neighbour metadata.
- Raw `file_relationships` candidates are not used for graph expansion.

**Stage 8: RRF Fusion (`rrf_fuse`)**
- Reciprocal Rank Fusion formula: `score(d) = Σ 1 / (k + rank_in_list)` where `k=60`.
- Merges all four ranked lists (BM25, fuzzy, vector, approved semantic graph) into a single unified score.
- Files appearing in multiple lists get combined scores (reward for multi-signal hits).

**Stage 9: Top-K**
- Returns top `top_k` (default 20) `FileMetadata` rows sorted by RRF score descending.

**Stage 10: Hydration (in `graph.py`, not orchestrator)**
- `hydrate_files(shortlist, db)` — fetches `columns_info`, `sample_rows`, `column_stats` for the top-8 files only.
- These "heavy fields" are NOT loaded in the initial catalog query (performance).
- `merge_hydrated(full_catalog, hydrated)` — merges heavy data back into catalog entries.

---

### 2.4 Semantic Planner Fast Path

Before handing off to LangGraph, `graph.py` tries `_try_planner()`:

```python
ep = await planner(query, catalog, db, timeout_seconds=2.5)
```

The semantic planner (`services/semantic_planner.py`) uses the **pre-built semantic layer** (entities + approved relationships) to generate SQL directly without an LLM agent loop.

Important production rule now implemented: deterministic planner joins must come from `semantic_relationships`, not directly from raw `file_relationships`. The old ER table is still useful because it stores technical join candidates, but the planner only treats a join as safe when the semantic layer marks it as `approval_status = "approved"`. If the only available semantic relationship is `candidate`, the planner returns a fallback reason and the query continues through the LangGraph agent path instead of generating fast-path SQL from a risky join.

**Returns an `ExecutionPlan`:**
```python
@dataclass
class ExecutionPlan:
    sql: str                 # ready-to-execute DuckDB SQL
    files: list[str]         # blob paths used
    joins: list[dict]        # join predicates from SemanticRelationship
    confidence: float        # 0.0–1.0
    fallback_reason: str     # non-empty = use agent instead
    planning_ms: float
```

**Planner fast path succeeds when:**
- `confidence >= SemanticPolicy.planner_fast_path_confidence`
- `sql` is non-empty
- `fallback_reason` is empty

**If fast path succeeds:**
1. Execute SQL via `DataFusion` or `DuckDB`.
2. If 0 rows returned → fallback to agent (data may exist under different filter).
3. If rows found → send `rows[:25]` to gpt-4o-mini with a synthesis prompt:
   ```
   "The user asked: {query}
   Query returned {N} row(s) (showing first 25):
   {data_preview}
   Write a concise, precise analytical response. Include key totals, top values, and observations."
   ```
4. Return answer with `route = "planner"`. **LangGraph agent is never started.**

**Planner falls back to agent when:**
- confidence is below `SemanticPolicy.planner_fast_path_confidence` (ambiguous query, no obvious entity match)
- `fallback_reason` is set (e.g. "no approved joins found", "query spans unknown entities")
- the relationship exists technically but is only `candidate` in the semantic layer, such as a risky single-column reference/attribute join
- timeout 2.5s exceeded
- any exception (never raises — always returns None to trigger agent)

---

### 2.5 LangGraph Agent

**Entry:** `graph.py → run_agent_query_stream()` / `run_agent_query()`

#### Initial State Construction

```python
AgentState = TypedDict("AgentState", {
    "messages":        list,           # LangChain message history
    "catalog":         list[dict],     # hydrated file catalog (visible to this user)
    "connection_string": str,          # Azure Blob connection string (decrypted)
    "container_name":  str,            # Azure container name
    "parquet_blob_path": str | None,   # single-file context (if set)
    "tool_call_count": int,            # incremented after every tool use
    "request_id":      str,            # trace ID for pipeline logging
    "broaden_nudges":  int,            # how many times agent was nudged to try different approach
    "is_first_turn":   bool,           # True on turn 1
})
MAX_TOOL_CALLS = 8
```

**State initialization in `run_agent_query_stream()`:**
1. Load full catalog via `load_catalog(db, user_id, is_admin, container_id, allowed_domains)` — DB query, no cache.
2. Build `parquet_paths` dict: `{csv_blob_path: parquet_blob_path}` from FileAnalytics.
3. Call `retrieve_with_scores()` → get top-20 retrieval hits.
4. Extract top-8 hits as "shortlist" (these get hydrated with heavy fields).
5. Hydrate shortlist → adds `columns_info`, `sample_rows`, `column_stats` for top files.
6. Call `_extract_mentioned_files(query, full_catalog)` — find any file explicitly named in the query (e.g. "on AP_invoices_all.csv"). Pin these at the TOP of the shortlist regardless of retrieval rank.
7. Build system prompt from shortlist + full catalog metadata.
8. Build all tools.
9. Create `AgentState` with `SystemMessage + HumanMessage(query)`.

#### LangGraph Graph Structure

```
START
  │
  ▼
agent_node  ────────────────────────────────────────────┐
  │                                                      │
  │ (has tool_calls?)                                    │
  ├─ YES ──► tool_node (ToolNode runs the tool)          │
  │               │                                      │
  │               ▼ (tool result added to messages)      │
  │          agent_node ◄────────────────────────────────┘
  │
  ├─ NO (or tool_call_count >= MAX_TOOL_CALLS)
  │
  ▼
END
```

**Routing logic (`_should_continue()`):**
```python
def _should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if (
        isinstance(last_message, AIMessage)
        and getattr(last_message, "tool_calls", None)
        and state.get("tool_call_count", 0) < MAX_TOOL_CALLS
    ):
        return "tools"
    return END
```

**Hard stop at 8 tool calls:**
```python
if count >= MAX_TOOL_CALLS:
    return {"messages": [AIMessage(content="I've gathered enough data. Let me summarise.")]}
```

---

### 2.6 The System Prompt (Full Anatomy)

Built by `agent/prompts/prompt_builder.py → build_system_prompt()`.

**Structure (in order):**

```
{file_override_note}   ← if user explicitly named a file ("use AP_invoices.csv")

You are a data analyst with DuckDB SQL access to files in Azure Blob Storage.

Today's date: {today_iso} ({today_human}).
Resolve every relative time expression in the user's question against THIS date.
[date calculation examples: last month, YTD, last year, last 30 days — all resolved to exact ISO dates]

Container: {container_name}

--- FILES IN SCOPE ({N} files) ---
{for each file in shortlist:}
  File: az://{container_name}/{blob_path}
  SQL:  read_parquet('az://...') or read_csv_auto('az://...')
  Description: {ai_description, neutralized — "PRIMARY source" phrases stripped}
  Good for: {good_for[:3]}
  Columns: {column_names}
  Key metrics: {key_metrics}
  Key dimensions: {key_dimensions}
  Date range: {date_range_start} → {date_range_end}
  Sample rows: {sample_rows[:3]}

{parquet_note}  ← "Parquet files available for faster queries"
{sample_note}   ← "Sample rows shown above are from ingest-time sampling"

--- TOOLS ---
1. run_sql              — Execute DuckDB SQL
2. get_file_schema      — Returns column names, types, sample values for a file
3. inspect_column       — Returns dtype, samples, suggested WHERE predicate for ONE column
4. search_catalog       — Searches FULL catalog ({total_file_count} files)
5. inspect_data_format  — Preview raw rows from a specific file
6. summarise_dataframe  — Compute stats on last SQL result

--- HOW TO WORK ---
Four principles:
1. VERIFY BEFORE YOU ACT
   Call get_file_schema before writing SQL.
   Call inspect_column for any column whose format is unclear.

2. EVIDENCE OVER ASSUMPTION
   If 0 rows: investigate first (inspect_column, MIN/MAX probe, search_catalog for another file).

3. CHANGE STRATEGY ON FAILURE
   Different file, different column, different filter. Never retry same thing with cosmetic changes.

4. search_catalog searches METADATA only (names, descriptions, column names).
   To find a row value, filter inside the file.

--- QUESTION TYPE ---
Conceptual → answer from knowledge + file descriptions. No SQL.
Data       → run SQL using the steps above.

--- OUTPUT STYLE ---
Do NOT narrate reasoning. Do NOT say "Let me start by...".
[~20 lines of strict output formatting rules]
```

**Description neutralization:**  
AI-generated descriptions often contain phrases like "This file is the PRIMARY source for..." or "Unlike similar files...". These over-anchor the LLM to one file. The prompt builder strips these patterns before injecting descriptions:
```python
_ANCHOR_PATTERNS = [
    re.compile(r"\bThis file is the PRIMARY source\b", re.IGNORECASE),
    re.compile(r"\bPRIMARY source\b"),
    re.compile(r"\bUnlike (?:other|similar) files,?\s*", re.IGNORECASE),
    ...
]
```

**`file_override_note`** (when user explicitly names a file):
```
NOTE: The user explicitly requested file '{filename}'. 
This file is pinned at the top of the shortlist. Prefer it over other files.
```

---

### 2.7 How the Agent Decides the Next Step

The agent does **not** have a hardcoded decision tree. It is a ReAct-style LLM agent. The LLM reads the system prompt (which lists all tools + 4 work principles) and the current message history (including all prior tool call inputs and outputs), then decides:

1. **What tool to call next** (or to stop and answer).
2. **What arguments to pass** to that tool.

**The message history is the agent's "memory" within one request:**
```
SystemMessage(prompt)
HumanMessage(query)
AIMessage(tool_call: get_file_schema, args={blob_path: "invoices.csv"})
ToolMessage(result: {"columns": [...]})
AIMessage(tool_call: inspect_column, args={col: "posting_date"})
ToolMessage(result: {"dtype": "VARCHAR", "suggested_predicate": "..."})
AIMessage(tool_call: run_sql, args={sql: "SELECT ..."})
ToolMessage(result: {"row_count": 15, "rows": [...]})
AIMessage(content: "The total invoice amount for last month was $4.2M...")
```

**Every tool result is appended as a `ToolMessage`** before the next LLM call. The LLM sees the full context including what it already tried and what the data actually contains.

**Model selection:**
- Always `gpt-4o-mini` (800 max_tokens, temperature=0, timeout=25s).
- `gpt-4o` is available but disabled — it has lower RPM quota which becomes the bottleneck.
- Rate limit handling: exponential backoff, up to `_MAX_LLM_RETRIES=3`. After 3 failures → return error message.

**Tool call count tracking:**
```python
# In agent_node:
count = state.get("tool_call_count", 0)
if count >= MAX_TOOL_CALLS:
    return {"messages": [AIMessage(content="I've gathered enough data. Let me summarise.")]}

# After each tool run (ToolNode):
state["tool_call_count"] += 1
```

**Streaming tool progress events:**
During streaming, every tool call emits a `thinking` SSE event to the client before the tool runs:
```json
{"event": "thinking", "tool": "run_sql"}
```
And a `tool_result` event after:
```json
{"event": "tool_result", "tool": "run_sql"}
```
This is how the frontend "AI Pipeline" panel is populated in real-time.

---

### 2.8 All Agent Tools

All tools are built at request time, bound to the specific user's connection context and catalog slice.

#### 1. `run_sql`
**File:** `agent/tools/sql.py`  
**What it does:** Executes a DuckDB SQL query against Azure Blob Storage.  
**Routing:** `QUERY_ENGINE=datafusion` → DataFusion; otherwise → DuckDB.  
**Output:**
```json
{
  "row_count": 15,
  "total_rows": 15,
  "columns": ["entity_id", "total_amount"],
  "rows": [{"entity_id": "E001", "total_amount": 450000.0}, ...]
}
```
**Caps:** Results truncated at 20 rows server-side. If `total_rows > 20` → message tells the LLM to refine with `WHERE/LIMIT/GROUP BY`.  
**Safety:** Every SQL passes through `validate_and_normalise()` before execution (see 2.9).  
**Logging:** Full SQL + all rows + duration logged to `pipeline.log`.

#### 2. `get_file_schema`
**File:** `agent/tools/catalog.py`  
**What it does:** Returns column names, dtypes, and sample values for a specific file from the catalog metadata. Does NOT hit Azure — reads from `FileMetadata.columns_info` already loaded in the catalog.  
**Use case:** Called before writing any SQL to confirm column names and types exist.

#### 3. `inspect_column`
**File:** `agent/tools/column.py`  
**What it does:** For a `(blob_path, column_name)` pair — returns:
- `dtype` (from metadata)
- Sample values (from `columns_info`)
- `suggested_predicate` — a ready-to-paste WHERE clause snippet  

**Suggested predicate logic:**
```
Identifier column (_id, _key, _code, _ref)  → "col = '<value>' -- compare as string"
Oracle DD-MON-YYYY strings                   → "strptime(col, '%d-%b-%Y') BETWEEN DATE '...' AND DATE '...'"
Float-typed year column                      → "col = 2024.0 -- column is float; match with .0"
Timestamp column                             → "col BETWEEN TIMESTAMP '...' AND TIMESTAMP '...'"
Date column                                  → "col BETWEEN DATE '...' AND DATE '...'"
Categorical (object dtype)                   → "col = '<value>'"
```
**When it falls back to DuckDB:** If the column isn't in cached metadata, runs `SELECT DISTINCT {col} LIMIT 20` against the actual file.

#### 4. `search_catalog`
**File:** `agent/tools/catalog.py`  
**What it does:** Text-matches the user's query against the FULL catalog (all files this user can see, not just the shortlist).  
**Scoring algorithm:**
```python
score = (tokens matching in description/good_for/key_metrics)
      + 2 * (tokens matching in column_names)  # column match weighted 2×
      + 1 * (tokens matching in blob_path filename)
```
**Output:** Top-10 files sorted by score, each with `blob_path`, `sql_path`, `description`, `columns`, `key_metrics`, `key_dimensions`, `good_for`, `date_range`.  
**Lookup file padding:** Lookup/master files (detected by `is_lookup_file()` — files with < 500 rows and mostly key columns) are padded to always appear, even at score=0, to prevent the agent from missing reference tables.

#### 5. `inspect_data_format`
**File:** `agent/tools/sample.py`  
**What it does:** Returns N raw rows from a specific file. First checks `sample_rows` cached in the catalog; if not available, runs `SELECT * FROM ... LIMIT N` via DuckDB/DataFusion.  
**Use case:** Understand messy raw data before writing a transformation query.

#### 6. `summarise_dataframe`
**File:** `agent/tools/stats.py`  
**What it does:** Computes statistical summary of the **last `run_sql` result** stored in `state_store["sql_results"]` — no re-query needed.  
**Output:** Per-column: dtype, null count, min/max/mean/std (numeric), or top-10 values + unique count (categorical).  
**Use case:** Agent calls `run_sql` to get data, then `summarise_dataframe` to describe it without re-querying.

#### 7. `extract_relations`
**File:** `agent/tools/relations.py`  
**What it does:** Returns known semantic join relationships for specified files from `SemanticRelationship`. If no semantic relationship exists yet, it can return raw technical relationships only as `technical_candidate` evidence with a risk warning.  
**Input:** Comma-separated blob paths (optional — if omitted returns strongest tenant-visible joins).  
**Output:**
```json
{
  "relations": [
    {
      "file_a": "facts.csv",
      "file_b": "entity_lookup.csv",
      "join_on": {"file_a_col": "entity_id", "file_b_col": "entity_id"},
      "semantic_role": "custom:entity_key:business_entity",
      "relationship_type": "many_to_one",
      "approval_status": "approved",
      "risk_reason": null,
      "confidence": 0.92,
      "value_overlap_pct": 0.87,
      "join_type": "LEFT JOIN",
      "evidence": "semantic_layer"
    }
  ],
  "context": {
    "file_a": {"description": "...", "key_metrics": [...]}
  }
}
```
**Join safety:** The agent should use `approval_status = "approved"` for SQL joins. `candidate` or `technical_candidate` rows are context only and require additional validation.
**Security:** Only returns relationships where BOTH `file_a_id` and `file_b_id` are in `allowed_file_ids` (the user's visible catalog). Cross-tenant leakage is impossible.

#### 8. `query_precomputed_analytics`
**File:** `agent/tools/analytics.py`  
**What it does:** Returns pre-computed column stats (min/max/mean/top-values) captured from the 500-row ingest sample.  
**Important warning included in the tool output:**
```
"WARNING: All numbers below are from a 500-row ingest sample.
 For accurate counts/totals, use run_sql on the parquet path."
```
**Use case:** Understanding column value distributions, categories, ranges — NOT for business aggregations.

#### 9. `build_definition_lookup_tool` / `load_schema_registry`
**File:** `agent/tools/definition_lookup.py`  
**What it does:** Looks up business term definitions from the container's schema glossary files.  
**Use case:** When a query uses domain-specific jargon ("what is NTE?", "define WBSE"), returns the definition from uploaded glossary files rather than hallucinating.

---

### 2.9 SQL Safety Layer

**File:** `agent/tools/sql_safety.py → validate_and_normalise()`

Every SQL string from the LLM passes through this before execution:

**Check 1 — DML/DDL rejection:**
```python
_FORBIDDEN = ("DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER",
              "TRUNCATE", "COPY", "ATTACH", "DETACH", "EXEC", "EXECUTE",
              "PRAGMA", "VACUUM", "CHECKPOINT", "LOAD", "INSTALL", "CALL")
```
Matched as whole-word tokens so `CREATED_AT` doesn't trip `CREATE`.

**Check 2 — Blob path allowlist:**
```python
for path in _extract_az_paths(sql):
    if path not in allowed_blob_paths:
        raise ValueError(f"Blob path '{path}' is not in the authorised file list...")
```
`allowed_blob_paths` is derived from the user's catalog shortlist at request build time. **This closes the prompt injection gap** — a malicious instruction embedded in file data cannot direct the LLM to query files outside the user's catalog.

**Check 3 — Auto LIMIT injection:**
```python
if "LIMIT" not in sql_upper:
    sql = sql.rstrip(";") + " LIMIT 10000"
```
Prevents unbounded full-table scans.

**Metrics:** Each violation increments `sql_forbidden_count` or `sql_blob_acl_denied` counters.

---

### 2.10 Answer Extraction & Fallbacks

**Happy path — `extract_answer(messages)`:**
```python
for msg in reversed(messages):
    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
        return msg.content
return ""
```
The last `AIMessage` that has content but no pending tool calls is the answer.

**Fallback — `fallback_answer(messages)` (when LLM produced no text answer):**
1. Scans all `ToolMessage` items for `"error"` keys.
2. Collects up to 3 unique error strings.
3. Returns human-readable message: `"I encountered the following issues: ..."` or generic `"I wasn't able to find relevant data..."`.

**Response cache write-back:**
- Only caches answers that:
  - Are NOT hollow (don't contain phrases like "unfortunately", "no data found", etc.)
  - Are >= 15 tokens long (not trivially short)
  - Have no tool errors
- Cache key: `(container_id, normalized_query)`

**Chart inference (`infer_chart`):**
- Triggered when `run_sql` produced rows.
- Returns `{"type": "bar"|"line"|"pie"|"table", "x_column": ..., "y_column": ..., "title": null}`.
- Logic:
  - "over time", "trend", "monthly" → `line`
  - "distribution", "proportion", "percent" → `pie`
  - `len(rows) > 50` → `table`
  - Otherwise → `bar`

---

### 2.11 Streaming Response Back to Client

**SSE event types emitted:**

| Event | When | Payload |
|---|---|---|
| `started` | Immediately | `{"conversation_id": "..."}` |
| `thinking` | Before each tool call | `{"tool": "run_sql"}` |
| `tool_result` | After each tool call | `{"tool": "run_sql"}` |
| `pipeline_step` | After retrieval | `{"step": "retrieval", "retrieved_files": 8, "total_files": 247}` |
| `token` | Each LLM output token | `{"content": "The total..."}` |
| `done` | Answer complete | Full payload (answer, data, chart, files_used, tool_calls, conversation_id) |

**Cache hit path:**
```
started → token (answer in 80-char chunks) → done (with from_cache=true)
```
No LLM calls, no tool calls.

**After `done` event (background tasks):**
- Save `Message(role=assistant)` with full payload to DB.
- `bg_title_and_summary()` — if this was turn 1, generate conversation title via mini LLM.
- If `msg_count >= WARN_MESSAGES_THRESHOLD` → emit `conversation_warn` event.

---

## 3. Data Models Referenced

| Model | Table | Key Fields |
|---|---|---|
| `File` | `files` | `id`, `blob_path`, `is_preprocessed`, `ingest_status`, `ingested_at`, `container_id`, `folder_id` |
| `FileMetadata` | `file_metadata` | `file_id`, `ai_description`, `good_for`, `key_metrics`, `key_dimensions`, `columns_info`, `column_names`, `column_semantic_roles`, `domain_tag`, `description_embedding` (pgvector) |
| `FileAnalytics` | `file_analytics` | `file_id`, `parquet_blob_path`, `column_stats`, `value_counts`, `cross_tabs`, `quarantine_count`, `cleaning_audit` |
| `FileRelationship` | `file_relationships` | `file_a_id`, `file_b_id`, `shared_column`, `related_column`, `semantic_role`, `confidence_score`, `value_overlap_pct`, `join_type` |
| `ColumnKeyRegistry` | `column_key_registry` | `file_id`, `column_name`, `semantic_role`, `key_kind`, `value_fingerprints[]` (GIN array) |
| `SemanticEntity` | `semantic_entities` | `file_id`, `entity_name`, `primary_key`, `metrics`, `dimensions`, `attributes`, `grain`, `confidence_score`, `status` |
| `SemanticRelationship` | `semantic_relationships` | `from_entity`, `to_entity`, `relationship_type`, `approval_status`, `risk_reason`, `join_rule`, `confidence_score` |
| `Conversation` | `conversations` | `id`, `user_id`, `title`, `summary`, `token_count`, `archived_at` |
| `Message` | `messages` | `conversation_id`, `role`, `content`, `token_count`, `payload` (JSONB with data/chart/files_used) |
| `ContainerConfig` | `containers` | `id`, `container_name`, `connection_string` (Fernet-encrypted), `cleaning_config`, `semantic_config` |

---

## 4. Supporting Statement: Why This Is the Right Production Direction

I am not claiming this is **100% mathematically perfect**. That would be the wrong claim for uploaded business files, because CSV/XLSX/Parquet files usually do not carry real database constraints, primary keys, foreign keys, or trustworthy cardinality metadata.

The correct production statement is:

> This is the safest and most extensible architecture currently available in this codebase because it separates **technical relationship discovery** from **business-approved semantic joins**, assigns confidence and cardinality, and falls back instead of executing risky deterministic SQL when the semantic evidence is not strong enough.

The two improvements from the pictures map to these concrete backend layers:

1. **ER / Relationship graph improvement**  
   The backend does not assume joins only from column names. It builds technical join candidates from semantic role matching plus normalized value overlap fingerprints. This answers: "Can file A and file B possibly join?"

2. **Semantic layer improvement**  
   The backend now builds `semantic_entities` and `semantic_relationships` after relationship detection. This answers: "Is this join business-safe? What is the grain? Is it one-to-many, many-to-one, one-to-one, or risky many-to-many?"

**Why role keys are not a shipped business enum**

The backend code does not ship vendor/customer/invoice/etc. as a fixed ontology anymore.

`server/app/services/semantic_roles.py` now defines behavior kinds only: `entity_key`, `reference_key`, `additive_measure`, `non_additive_measure`, `date`, and `attribute`.

Domains can add roles in `container_configs.semantic_config` or allow the resolver to emit typed dynamic roles such as:

```text
custom:entity_key:claim
custom:additive_measure:premium
custom:non_additive_measure:exchange_rate
custom:date:service_date
```

Downstream code reasons from the role kind, not from an assumption that every possible business role was known at development time.

Operational thresholds and scoring weights are centralized separately in `server/app/services/semantic_policy.py`. The detector, fingerprint index, semantic builder, planner, graph expansion, and planner gate read policy values instead of embedding semantic magic numbers locally. Each value can be overridden with a `GCHAT_SEMANTIC_*` environment variable.

For example, source-specific columns may resolve to `custom:entity_key:business_entity` when the data/glossary supports that meaning. A transaction system may resolve to `custom:entity_key:transaction`; a numeric business fact may resolve to `custom:additive_measure:business_amount`.

This is necessary because downstream systems need predictable behavior:

```text
relationship detector needs to know whether a role is join-capable
semantic layer needs to know whether a role is metric/dimension/date/attribute
planner needs to know whether a metric can be summed or only averaged/min/maxed
```

Without stable behavior kinds, every stage would receive arbitrary labels from the LLM and the planner could not reliably reason over them.

So the design is:

```text
dynamic input columns -> LLM maps them once at ingestion -> tenant-configured or typed dynamic semantic roles
```

Not:

```text
fixed column names -> hardcoded behavior
```

**Will it work for any file?**

It will work for many business/ERP-style structured files, but the honest answer is: **not perfectly for every possible file without fallback or extension**.

It works best when the file has recognizable business structure:

```text
IDs / keys
amounts / quantities
dates
status/category fields
descriptions / names
```

It still handles unfamiliar column names because the role resolver looks at:

```text
column name
data type
sample values
optional schema glossary
filename/context
```

So source-specific columns such as supplier IDs, account IDs, or custom codes can map to a typed role like `custom:entity_key:supplier` when the file context supports that meaning.

Where it still will not be fully automatic:

```text
files with no meaningful headers
files with only free-text/unstructured notes
files where sample values are too sparse to identify roles
domain-specific entities with unclear headers/samples that cannot be typed safely
joins that require composite business rules not yet modeled
```

In those cases the system should not pretend. The expected behavior is:

```text
low role coverage -> lower confidence
unsafe relationship -> candidate, not approved
planner cannot prove safe path -> fallback to LangGraph agent
```

The role registry is therefore a controlled behavior system, not a limitation to fixed files. If a new domain appears, the extension is configuration or typed dynamic roles. Only when a new behavior kind is needed should code change.

In code terms:

```text
base role specs + typed dynamic roles = controlled semantic behavior
hardcoded file behavior = bad and should be avoided
```

The backend should never hardcode client-specific columns into relationship rules. Those belong on the dynamic input side and are mapped into typed roles during ingestion.

The most important implementation detail is now in the query path:

```text
Raw FileRelationship edge
  -> may be a technical candidate only

SemanticRelationship approval_status = approved
  -> allowed for deterministic semantic planner SQL

SemanticRelationship approval_status = candidate
  -> planner refuses fast-path SQL and falls back to the agent path
```

So the system does **not** say:

```text
same tenant + same values = safe join
```

It says:

```text
same tenant + same values = candidate technical relationship
candidate relationship + safe cardinality + non-risky role + sufficient confidence = approved semantic join
```

That is why this is better than a plain ER diagram. A plain ER edge can still produce wrong business numbers. The semantic layer adds the missing controls: `relationship_type`, `approval_status`, `risk_reason`, `confidence_score`, and `join_rule`.

**Example: currency rate join**

```text
invoice.currency_code -> fx_rate.currency_code
```

The ER graph can detect this as a possible join because values overlap. But the semantic layer marks it as `candidate`, because a single currency reference alone is not safe. FX rates usually have multiple rows per currency across dates, so joining only on currency duplicates rows and inflates financial metrics.

Correct business-safe rule requires a composite condition:

```sql
invoice.currency_code = fx_rate.currency_code
AND invoice.invoice_date BETWEEN fx_rate.valid_from AND fx_rate.valid_to
```

Until that composite rule exists as an approved semantic relationship, the deterministic planner must not use it.

**What was implemented as the production hardening step**

The production hardening moved static-looking operational behavior out of scattered code paths and into runtime semantic infrastructure:

1. `semantic_policy.py` centralizes relationship confidence, fingerprint thresholds, semantic approval thresholds, planner gates, graph expansion gates, and planner confidence weights.
2. `relationship_detector.py`, `relationship_index.py`, `semantic_layer_builder.py`, `semantic_planner.py`, and `graph.py` now read policy values instead of embedding semantic magic numbers locally.
3. `retrieval/orchestrator.py` now calls `graph_expand()` and fuses approved semantic neighbours through RRF.
4. `retrieval/graph_expand.py` only expands through active approved `SemanticRelationship` rows.
5. `agent/tools/relations.py` exposes `approval_status`, `relationship_type`, `risk_reason`, `join_rule`, and semantic confidence to the agent.
6. `core/llm_tasks.py` no longer asks the LLM to call a file the primary or best source unless metadata explicitly proves that.
7. `semantic_roles.py` is now a behavior-aware registry with base roles plus typed dynamic roles, so the system is not capped by a fixed list.
8. `container_configs.semantic_config` stores per-container role extensions without a code deploy.
9. `relationship_detector.py` no longer creates relationships from role labels alone; it requires indexed value-overlap evidence from `column_key_registry`.

The planner now reads from `semantic_relationships` for join planning and only uses relationships where:

```text
status = active
approval_status = approved
confidence_score >= SemanticPolicy.planner_join_min_confidence
```

If the available semantic relationships are only candidates, the planner returns a fallback reason like:

```text
candidate_semantic_relationship: custom:reference_key:currency alone is not a safe business join key
```

Then the existing LangGraph agent path runs instead. This is intentionally conservative: fast deterministic SQL is allowed only when the semantic layer says the join is safe.

**Next step after this**

The next production improvement is to support **composite semantic join rules**. Today the semantic layer can reject risky single-column joins, but it does not yet automatically promote a composite rule like `currency_code + effective_date_range` to approved. The next implementation should add:

```text
semantic_relationships.join_rule.conditions[]
semantic_relationships.requires_columns[]
semantic_relationships.grain_check_sql
semantic_relationships.validity_window = true/false
```

That would let the planner approve joins such as FX rates, price lists, slowly changing dimensions, and status history tables only when the full business key is present.

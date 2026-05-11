# G-CHAT R&D Implementation Plan
# Written by Copilot as a reference for future sessions.
# This document captures every improvement discussed, in the exact order to implement them.
# DO NOT skip phases. Each phase is a prerequisite for the next.

---

## Context for Future Copilot Sessions

This plan was developed through an R&D discussion covering:
- Why DuckDB fails under concurrent load (serialization, Azure HTTP queuing)
- Why DataFusion is the correct query engine replacement
- How the catalog (file schema lookup) must move from in-memory dict to PostgreSQL + Redis
- How RBAC (role-based access control) must be enforced at the query worker level
- How the system scales from 30 users to 1000+ without code changes

The system design reference document is at: gchat_system_design.md (rated 8.5/10, RBAC gap identified)

Current stack: FastAPI + LangGraph + DuckDB + Azure Blob + PostgreSQL + Next.js

---

## The 3 Core Problems Being Fixed

1. DuckDB serializes concurrent queries → 2s query becomes 20s under 10 users
2. Every query reads Parquet from Azure Blob over HTTP with no predicate pushdown → Azure throttles
3. Module-level Python dict catalog diverges across server processes → files "disappear" per process

---

## PHASE 1 — Fix the Catalog and Parquet Quality (No engine change)
### Timeline: 2-3 weeks
### Goal: Fix the module-level dict bug. Improve Parquet files for future DataFusion migration.

### 1A. Replace module-level catalog dict with PostgreSQL + Redis

**What to change:**
- `server/app/agent/catalog_cache.py` — replace module-level dict with 3-tier lookup:
  - L0: process-level LRU dict (bounded, max 1000 entries, 5-min TTL)
  - L1: Redis (shared across all processes, 1-hour TTL, key = "file_columns:{file_id}")
  - L2: PostgreSQL file_metadata table (source of truth, always authoritative)

**Read pattern (cache-aside):**
```
get_file_schema(file_id):
  1. Check L0 dict → hit? return. miss? continue.
  2. Check Redis["file_columns:{file_id}"] → hit? write to L0, return. miss? continue.
  3. Query PostgreSQL file_metadata → write to L1 Redis (TTL=1h), write to L0, return.
```

**Write pattern (on ingestion complete):**
```
After Parquet written and file_metadata updated in PostgreSQL:
  → DELETE Redis["file_columns:{file_id}"]  (invalidate, not update)
  → Clear from L0 if present
  → Next read rebuilds from PostgreSQL
```

**Why:** Module-level dict = different answers per process. Redis = shared truth.
**Risk:** Requires Redis deployed. Use Azure Cache for Redis (managed, no ops overhead).

---

### 1B. Fix Parquet write settings in parquet_service.py

**Current state:** DuckDB COPY command writes Parquet with basic settings.

**What to change in `server/app/services/parquet_service.py`:**

Switch from DuckDB COPY → PyArrow writer with explicit settings:

```python
import pyarrow as pa
import pyarrow.parquet as pq

pq.write_table(
    table,
    output_path,
    compression='zstd',               # ZSTD level 3 — better ratio than SNAPPY, fast decode
    compression_level=3,
    row_group_size=1_000_000,         # 1M rows per row group (or 128MB, whichever smaller)
    write_statistics=True,            # MANDATORY — enables row group skipping in DataFusion
    write_page_index=True,            # Page-level statistics — finer than row group
    bloom_filter_columns=<low_cardinality_cols>,  # detect during profiling
    data_page_size=1_048_576,         # 1MB pages — enables page-level pruning
    use_dictionary=True,              # dictionary encoding for string columns
)
```

**For files > 2GB:** Split into multiple Parquet files of ~512MB each.
**For date columns detected during profiling:** Sort by date before writing — multiplies row group skip effectiveness for time-range queries.
**For files < 50MB:** Single row group, no sorting overhead needed.

**Why:** Without write_statistics=True, DataFusion cannot skip row groups. The predicate pushdown advantage (reading 40MB vs 400MB) ONLY works if the Parquet was written correctly.

---

### 1C. Extend file_metadata / file_columns in PostgreSQL

**Current state:** file_metadata exists but is not fully used as catalog source of truth.

**What to add/verify exists:**
```sql
-- Ensure these columns exist on file_metadata (or a new file_columns table):
-- column_name, arrow_type, semantic_type (date/id/category/measure/text),
-- null_count, distinct_estimate, min_value, max_value, top_values (JSONB),
-- bloom_columns (bool)

-- Index required:
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_id ON file_metadata(file_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_org_id ON file_metadata(org_id);  -- if org_id exists
```

**What the ingestion worker must write after Parquet conversion:**
- All column names and inferred Arrow types
- Semantic type per column (is it a date? a number? a category?)
- min, max, null_count, top-K distinct values
- Which columns have bloom filters written

**Why:** This is what the AI agent reads to write SQL. Quality of column profiles = quality of AI SQL generation. A rich profile means the AI writes correct SQL on the first try.

---

### Phase 1 Success Criteria
- [ ] Any 2 FastAPI processes return identical file schemas for the same file_id
- [ ] Server restart does not lose catalog (data survives, Redis rebuilds from PostgreSQL)
- [ ] New files appear in catalog within seconds of ingestion completing
- [ ] All new Parquet files have write_statistics=True (verify with pyarrow.parquet.read_metadata)

---

## PHASE 2 — Introduce DataFusion Query Worker (Feature-Flagged)
### Timeline: 4-6 weeks
### Goal: Replace DuckDB with DataFusion behind a feature flag. Validate correctness first.

### 2A. Build datafusion_client.py

**Replace:** `server/app/core/duckdb_client.py`
**New file:** `server/app/core/datafusion_client.py`

**Core pattern:**
```python
import datafusion
from datafusion import SessionContext

async def execute_query(
    sql: str,
    file_ids: list[str],
    allowed_file_ids: list[str],   # RBAC boundary — passed from FastAPI
    connection_string: str,
    timeout_seconds: int = 30,
    max_rows: int = 1000,
) -> tuple[list[dict], int]:

    # RBAC check — every file_id must be in allowed_file_ids
    for fid in file_ids:
        if fid not in allowed_file_ids:
            raise PermissionError(f"Access denied to file {fid}")

    # Resolve blob URIs from catalog (PostgreSQL/Redis — NOT from agent)
    blob_uris = await catalog.get_blob_uris(file_ids)

    # Per-request isolated context — zero shared state
    ctx = SessionContext()
    ctx.register_object_store("az", azure_object_store)  # Azure credentials from env

    for i, (file_id, uri) in enumerate(zip(file_ids, blob_uris)):
        await ctx.register_parquet(f"t{i}", uri)

    # Execute with timeout
    result = await asyncio.wait_for(
        ctx.sql(sql).collect(),
        timeout=timeout_seconds
    )

    # ctx goes out of scope here — garbage collected — nothing left behind
    return _to_dicts(result, max_rows)
```

**Key rule:** The agent never passes blob URIs. It passes file_ids. The worker resolves URIs internally after RBAC check.

---

### 2B. SQL safety layer (parse before execute)

**Before any SQL reaches DataFusion, validate it:**
```python
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE", "COPY", "ATTACH"]

def validate_sql(sql: str) -> None:
    sql_upper = sql.upper().strip()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_upper:
            raise ValueError(f"SQL contains forbidden operation: {kw}")
    # Auto-inject LIMIT if missing
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + " LIMIT 10000"
    return sql
```

---

### 2C. Feature flag routing

**Add to config.py:**
```python
QUERY_ENGINE: str = "duckdb"   # "duckdb" | "datafusion"
```

**In sql.py tool and column.py tool:**
```python
if settings.QUERY_ENGINE == "datafusion":
    rows, total = await datafusion_client.execute_query(sql, file_ids, allowed_file_ids, ...)
else:
    rows, total = execute_query_sync(sql, connection_string, ...)  # existing DuckDB
```

**Validation process:**
1. Deploy with QUERY_ENGINE="duckdb" (no change)
2. Shadow test: run both engines on same queries, compare results
3. Once results match and latency is equal or better → flip to QUERY_ENGINE="datafusion"
4. Keep DuckDB fallback for 2 weeks → then remove

---

### 2D. Update agent tools to pass file_ids instead of connection_string

**Current:** `run_sql(sql, connection_string, parquet_blob_path)`
**New:** `run_sql(sql, file_ids, allowed_file_ids)`

The blob path is never given to the agent. The worker resolves it.

---

### Phase 2 Success Criteria
- [ ] DataFusion produces identical results to DuckDB on all existing test queries
- [ ] 40 concurrent requests complete without queue buildup (measure p50, p95, p99 latency)
- [ ] RBAC: a user cannot query a file_id not in their allowed_file_ids (tested explicitly)
- [ ] No blob URIs appear in agent prompts or tool inputs

---

## PHASE 3 — gRPC Separation (Query Worker as Independent Service)
### Timeline: 3-4 weeks (only needed at 100+ concurrent users)
### Goal: Separate the query worker from the FastAPI process. Independent scaling.

### 3A. Why separate

Running DataFusion inside FastAPI means:
- A runaway query OOMs the API process → kills all in-flight chat sessions
- You can't scale query workers independently from API workers
- You can't hot-deploy agent changes without restarting query workers

### 3B. The gRPC contract

**New service:** `query_worker/` — separate Python service, separate deployment

```protobuf
service QueryWorker {
  rpc Execute(ExecuteRequest) returns (stream RecordBatch);
}

message ExecuteRequest {
  string sql = 1;
  repeated string file_ids = 2;
  repeated string allowed_file_ids = 3;  // RBAC boundary
  string org_id = 4;
  string request_id = 5;
  int32 timeout_seconds = 6;
  int32 max_rows = 7;
}
```

FastAPI sends ExecuteRequest. Query worker validates RBAC, resolves URIs, runs DataFusion, streams Arrow RecordBatches back.

### 3C. Consistent hashing router

**In front of query workers:**
```python
worker_id = hash(primary_file_id) % len(workers)
route_to(workers[worker_id])
```

Same file always goes to same worker → local NVMe cache stays warm → Azure reads drop after warm-up period.

**Failure handling:** If target worker is down, route to next worker in ring. Cache miss on first request, rebuilds automatically.

---

### Phase 3 Success Criteria
- [ ] FastAPI OOM does not kill in-flight queries (queries continue on query worker)
- [ ] Query workers can be restarted independently of FastAPI
- [ ] Cache hit rate on query workers > 70% for hot files after 30 minutes of traffic

---

## PHASE 4 — Observability (Non-Negotiable Before Production Scale)
### Timeline: 1 week (parallel with Phase 3)
### Goal: You cannot operate at 300+ users blind.

### What to instrument

**Per-query trace (OpenTelemetry):**
- request_id → user_id → org_id → file_ids → sql → execution time → rows scanned → bytes read from Azure vs cache

**Key metrics to track:**
- `query_duration_p50`, `query_duration_p95`, `query_duration_p99` — per org
- `azure_blob_bytes_read` — if this spikes, cache isn't working
- `azure_blob_429_rate` — if this rises, Azure is throttling
- `catalog_redis_hit_rate` — if this drops, Redis TTL is too short or Redis is undersized
- `worker_cache_hit_rate` — per query worker, per file
- `llm_sql_failure_rate` — SQL generation errors per org. If this rises, column profiles degraded.
- `query_queue_depth` — how many requests waiting for a worker. Alert at > 5.

**Why llm_sql_failure_rate matters:** When the AI generates bad SQL, it's almost always because the column profile in the catalog is poor quality (wrong types, missing sample values). This metric tells you when profiles need re-ingestion.

---

## PHASE 5 — Ballista (Only at 1000+ Users)
### Timeline: 4-6 weeks (do NOT start until you actually need it)
### Goal: Split single heavy queries across multiple machines.

**Trigger:** You need Ballista when you see queries that:
- Scan many files simultaneously (cross-org aggregates, folder-wide summaries)
- Take > 30s even on a well-provisioned single worker
- Require scanning > 50GB of data per query

**What changes:** Only the query worker service. FastAPI, LangGraph, agents, catalog — unchanged.

**How it works:**
- Ballista Scheduler replaces the single DataFusion SessionContext
- Scheduler splits query plan into stages
- Each Ballista executor runs one stage (still using DataFusion internally)
- Results merge at scheduler and return to FastAPI

**What doesn't change:** SQL syntax, gRPC contract, RBAC enforcement, catalog lookups.

---

## RBAC Rules (Non-Negotiable — Must Be Enforced in Every Phase)

These rules must hold from Phase 1 onwards:

1. **Organization boundary:** A user can never access files from a different org_id. Enforced in FastAPI resolve_chat_scope (already exists).

2. **Domain boundary:** A user with allowed_domains=["finance"] cannot access files in folders tagged "hr" or "legal". Enforced in FastAPI domain filtering (already exists).

3. **allowed_file_ids[]:** FastAPI computes the list of accessible file_ids for every request. This list is passed to the query worker. The query worker rejects any file_id not in the list.

4. **Blob URI isolation:** The AI agent NEVER sees Azure blob paths. Only file_ids. The query worker resolves file_id → blob URI internally after RBAC check.

5. **SQL validation:** Every SQL query is parsed and validated before execution. No DML (INSERT/UPDATE/DELETE/DROP). Auto-inject LIMIT if missing.

---

## Scale Reference

| Users | Infrastructure | Code changes |
|---|---|---|
| 30-40 | 2 API pods, 3 DataFusion workers, 1 PostgreSQL, 1 Redis, Azure Blob | Phase 1+2 |
| 100-200 | 4 API pods, 6 workers, PgBouncer, Redis cluster | None — add pods |
| 300-400 | 8 API pods, 12 workers, PostgreSQL read replica, Redis 3-shard | None — add pods |
| 1000+ | 20+ pods, Ballista cluster, partitioned PostgreSQL | Phase 5 — Ballista only |

---

## Files That Change in Each Phase

### Phase 1:
- `server/app/agent/catalog_cache.py` — replace dict with 3-tier Redis/PostgreSQL
- `server/app/services/parquet_service.py` — PyArrow writer with statistics/bloom filters
- `server/app/core/config.py` — add REDIS_URL setting

### Phase 2:
- `server/app/core/datafusion_client.py` — NEW FILE (replaces duckdb_client.py)
- `server/app/agent/tools/sql.py` — route to datafusion_client when flag set
- `server/app/agent/tools/column.py` — same routing
- `server/app/core/config.py` — add QUERY_ENGINE flag

### Phase 3:
- `query_worker/` — NEW SERVICE (separate deployment)
- `server/app/core/query_router.py` — NEW FILE (consistent hashing router)
- `server/app/agent/tools/sql.py` — call gRPC instead of local function

### Phase 4:
- `server/app/core/telemetry.py` — NEW FILE (OpenTelemetry setup)
- Instrument duckdb_client → datafusion_client (already has some logging)

### Phase 5:
- `query_worker/ballista_executor.py` — replace SessionContext with Ballista scheduler

### Files that NEVER change regardless of scale:
- `server/app/api/` — all FastAPI routes
- `server/app/agent/graph/` — LangGraph workflow
- `server/app/agent/prompts/` — AI prompts
- `client/` — entire frontend
- `server/app/models/` — PostgreSQL models
- `server/app/core/security.py` — JWT auth

---

## Key Decisions Made (Don't Revisit Without Strong Reason)

1. **DataFusion over ClickHouse** — ClickHouse requires predefined schemas. G-CHAT has 1M files each with different schemas. DataFusion registers schema at query time. This is non-negotiable for the product's core use case.

2. **No local SSD as primary storage** — Breaks with multiple VMs. Azure Blob is the only source of truth for files. Local disk is only acceptable as a cache (write-through, ETag-invalidated).

3. **PostgreSQL as catalog source of truth** — Not Redis, not in-memory. Redis is a cache with TTL. PostgreSQL is the ground truth.

4. **Agent never sees blob URIs** — Security boundary. Prevents prompt injection attacks where a user crafts a question that makes the AI query files they don't own.

5. **Ballista deferred until 1000+ users** — Don't over-engineer. Single-node DataFusion handles 300-400 users with horizontal scaling. Ballista complexity is not justified before that.

---

## Open Questions (Not Yet Resolved)

1. **Re-ingestion / schema evolution:** When a user re-uploads the same file with a new schema, current approach is new file_id = new version. UI shows versions, agent picks latest. This needs UI work.

2. **Azure OpenAI PTU:** At 200+ concurrent users, the LLM API rate limit becomes the first bottleneck. Need dedicated PTU capacity before hitting that scale. Cost discussion deferred.

3. **parquet_service.py DuckDB dependency:** Currently uses DuckDB's COPY command for CSV→Parquet conversion. Phase 1 switches this to PyArrow directly. The conversion itself is not the concurrency problem — it's the query execution. But the Parquet quality improvement requires this change.

4. **Row group stats in PostgreSQL:** System design suggests storing row_group_stats in PostgreSQL for cross-file pruning at 1M+ files. Not in Phase 1 or 2 — revisit when file count exceeds 100k.

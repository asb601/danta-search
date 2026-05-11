# G-CHAT System Design

A complete architecture for an AI-powered data chat product that handles arbitrary user-uploaded schemas, runs 30–40 concurrent users today, and scales to 1000+ without an architectural rewrite.

---

## 1. Executive summary

**Pick DataFusion as the query engine.** It is the only candidate of the three you listed that simultaneously satisfies four hard constraints: (a) arbitrary per-file schemas registered at query time, (b) true intra- and inter-query parallelism without an in-process global lock, (c) native object-store readers with predicate and projection pushdown on Parquet, and (d) a credible distributed-execution path (Ballista) when you outgrow a single node.

The system has five layers, each independently scalable:

1. **Edge / API** — FastAPI behind a load balancer, stateless, autoscaled.
2. **Agent runtime** — LangGraph workers; one logical agent per chat request. Stateless.
3. **Query plane** — a pool of DataFusion "query workers" with a per-request `SessionContext`, fronted by a small router. This is the only layer that needs careful capacity planning.
4. **Catalog plane** — Postgres as the system of record + Redis as a hot cache. The module-level dict goes away.
5. **Storage plane** — Azure Blob (cold/warm Parquet) + a per-VM NVMe scratch cache (hot Parquet) + a thin object-store byte-range cache for Parquet footers.

The two architectural ideas that do most of the work are: **statelessness everywhere except the catalog**, and **physically separating the "where is the data" question (catalog) from the "execute SQL" question (query engine)**. Once those are clean, scaling is a sizing exercise, not a redesign.

---

## 2. Query engine: DataFusion, and why

### Why not DuckDB (your current pain)

DuckDB is an in-process database. Even with thread-local connections, two issues bite under concurrency:

- **Internal serialization on shared state.** Connections share buffer manager, catalog, and file handle structures. Concurrent queries compete for these and effectively serialize on contended paths — which is exactly the "2s → 20s under load" pattern you're seeing.
- **No true horizontal story.** DuckDB is in-process and file-locked; you cannot ship a query to another node. Every scaling option becomes "make the box bigger."
- **Python GIL amplification.** Each LangGraph step is Python; under concurrency the GIL plus DuckDB's own contention compound.

DuckDB remains excellent for single-user analytics and embedded use. It is the wrong shape for a multi-tenant query service.

### Why not ClickHouse

You called this correctly. ClickHouse's strength is concurrent analytical scans over **predefined schemas**. With 1M files each carrying a different schema, you'd be running 1M `CREATE TABLE` / `ATTACH PARTITION` operations and managing them forever. ClickHouse's metadata layer was not designed for that cardinality. You'd also pay for a stateful storage tier you don't need — your data already lives in Parquet on Blob.

If you ever want to offer "shared organizational datasets" (a small number of large, well-known tables shared across users), ClickHouse or StarRocks make sense as a **second** engine alongside DataFusion. Not as the primary.

### Why not local-SSD-only cache

It's a cache, not a query engine. Fine as a tier, but it doesn't solve concurrency, doesn't solve catalog, and breaks multi-node. Keep it — but as a sub-component, not as "the answer."

### Why DataFusion fits

| Constraint | How DataFusion satisfies it |
|---|---|
| Arbitrary schema per query | Register Parquet at query time on a per-request `SessionContext` via `ctx.register_parquet("t", "az://...")`. No global catalog of user schemas required in the engine. |
| Intra-query parallelism | Tokio task scheduler distributes plan execution across CPU cores; row-group-level parallelism for Parquet scans ([DataFusion async execution model](https://datafusion.apache.org/blog/2025/06/30/cancellation/)). |
| Inter-query parallelism | Sessions are independent and share no mutable state. N concurrent sessions truly run in parallel, bounded by CPU and the object-store reader. |
| Object-store reads | First-class `ObjectStore` abstraction with an Azure implementation; range-GETs only the footer plus the column chunks needed, with row-group skipping via min/max stats and optional page-level pruning ([Efficient Filter Pushdown in Parquet — DataFusion blog](https://datafusion.apache.org/blog/2025/03/21/parquet-pushdown/)). |
| External catalogs / indexes | Custom `TableProvider` + `ParquetAccessPlan` let you bring your own pruning (your Postgres-stored stats can drive row-group pruning before the engine reads metadata) ([Using External Indexes — DataFusion blog](https://datafusion.apache.org/blog/2025/08/15/external-parquet-indexes/)). |
| Horizontal scale | Ballista provides scheduler + executor topology with Arrow IPC shuffle for true distributed execution when you outgrow a node ([Ballista Architecture](https://datafusion.apache.org/ballista/contributors-guide/architecture.html)). |
| Python integration | `datafusion-python` 46.x adds async record-batch iteration that composes naturally with FastAPI's event loop ([DataFusion Python 46.0.0 release](https://datafusion.apache.org/blog/output/2025/03/30/datafusion-python-46.0.0/)). |
| Production validation | Bauplan migrated their lakehouse from DuckDB to DataFusion for precisely your reasons (ephemeral per-request engines, S3-native, easier custom planning) ([Bauplan engineering blog](https://www.bauplanlabs.com/post/duck-hunt-moving-bauplan-from-duckdb-to-datafusion)). |

### The non-obvious tradeoff

DataFusion's SQL dialect is narrower than DuckDB's. Some quality-of-life functions (e.g., some date arithmetic, some pivots, `STRUCT` ergonomics) are weaker. For an LLM that generates SQL, this is mostly fine **if** your prompt scaffolding restricts the agent to a curated subset (which you should do anyway for safety). Plan for a small "supported function" allowlist in the agent prompt.

A second tradeoff: the Python bindings are thinner than DuckDB's. You will write more glue. Budget a week.

---

## 3. Data flow: upload → query

### 3.1 Upload & ingestion

```
Client ──upload──► FastAPI ──► Azure Blob (raw/)        (multipart, resumable)
                       │
                       └──► enqueue ingestion job (Postgres outbox or Azure Service Bus)
                                          │
                                          ▼
                              Ingestion Worker (separate pool)
                                          │
            ┌─────────────────────────────┼──────────────────────────────┐
            ▼                             ▼                              ▼
   Sniff & profile          Convert to Parquet            Update catalog (Postgres)
   (Arrow CSV reader,       (PyArrow / Polars,            + warm Redis
   schema inference,        tuned row groups,             + emit stats
   sampling for stats)      ZSTD, statistics on)
```

**Ingestion is asynchronous and decoupled from the API.** Two reasons:

1. The user-perceived upload completes fast (just blob PUT + job enqueue).
2. Ingestion is CPU-heavy and bursty; running it in the API process steals capacity from chat requests.

Run ingestion workers as a separate Kubernetes deployment with its own HPA. They consume from a durable queue (Postgres `SKIP LOCKED` is enough at your scale; move to Service Bus / Kafka past ~50 ingestion workers).

### 3.2 Parquet write configuration (this matters a lot)

These choices materially change query latency and Blob cost. Pick once, document, and don't drift.

| Setting | Value | Why |
|---|---|---|
| Compression | `ZSTD` level 3 | Best balance of ratio vs. decode CPU. SNAPPY is faster to decode but ~25% larger, increasing Blob egress. |
| Row group size | **128 MB uncompressed** (or 1M rows, whichever smaller) | Large enough that footer overhead is tiny; small enough to give row-group skipping real selectivity. Community consensus around 1M rows / 128 MB is the right default ([Stack Overflow discussion on row group sizing](https://stackoverflow.com/questions/76782018/what-is-actually-meant-when-referring-to-parquet-row-group-size)). |
| Page size | 1 MB | Enables page-level statistics pruning, which DataFusion uses. |
| Statistics | Enabled at row-group **and** page level | Mandatory. Without page stats you lose late-materialization pushdown. |
| Bloom filters | Enabled for low-cardinality string columns and any column the profiler tags as a likely filter key | Cheap probabilistic pruning on equality predicates. |
| Dictionary encoding | Enabled (default) | Huge win for string columns. |
| File size target | 256 MB – 1 GB per Parquet file | Avoid the small-files problem; for very large user CSVs, split into multiple Parquet files with a deterministic naming scheme. |
| Sort order | If the profiler detects a clear date/time column, sort by it during write | Multiplies row-group skipping effectiveness on time-range filters (the most common analytical predicate). |

For files **under ~50 MB**, write a single row group and skip sorting — the overhead isn't worth it.

For files **over ~2 GB**, split into N Parquet files of ~512 MB each. This unlocks per-file parallelism in DataFusion and bounds memory for any single scan.

### 3.3 Schema profiling at ingestion

The ingestion worker doesn't just convert; it **profiles**. Store profile output in Postgres (see §4). What to capture per column:

- Inferred type (with confidence; CSVs lie)
- Cardinality estimate (HyperLogLog sketch is fine)
- Min, max, null count, sample of distinct values (top-K)
- Whether the column "looks like" a date, id, category, or measure
- For text columns: average length, whether it's likely free text vs. categorical

This profile is what the LLM agent sees as the table description. **It is the single biggest lever on SQL quality.** A great profile means a small, cheap model can still write correct SQL; a bad profile means even GPT-4o hallucinates columns.

### 3.4 Query execution

```
Chat request ──► FastAPI ──► LangGraph agent
                                  │
                                  │  (1) Catalog lookup: get table profile(s) from Redis/Postgres
                                  │  (2) LLM generates SQL conditioned on profile
                                  │  (3) SQL safety check (parse, allowlist, row limit, timeout)
                                  ▼
                            Query Plane (gRPC)
                                  │
                                  ▼
                       DataFusion worker (one of N)
                          │           │
                          │           └── Local NVMe Parquet cache? ──yes──► read local
                          │                                     │
                          │                                    no
                          │                                     ▼
                          └──────────────────────────► Azure Blob (range GETs)
                                  │
                                  ▼
                       Arrow RecordBatch stream ──► back to agent ──► SSE to client
```

Key properties:

- **Per-request `SessionContext`.** Each query creates a fresh context, registers only the file(s) it needs, executes, drops it. Zero shared mutable state between queries.
- **Result streaming.** Don't materialize the full result in the worker; stream Arrow record batches back. The agent can summarize/format the first few batches for the LLM and start streaming the answer immediately.
- **Hard limits.** Every query gets a wall-clock timeout (default 30 s), a row-count cap (default 100k for raw results; aggregations exempt), and a memory budget. DataFusion supports spill-to-disk for sorts; enable it.

---

## 4. Catalog management at 1M files

The module-level dict is your single biggest correctness bug right now — it diverges across workers and dies on restart. Replace it with a tiered catalog.

### 4.1 Three-tier catalog

| Tier | Store | Latency | Holds | Eviction |
|---|---|---|---|---|
| L0 — process | Python dict, per-process, **bounded LRU** | µs | Last ~10k accessed file metadata blobs | LRU, capped at e.g. 256 MB heap |
| L1 — shared cache | Redis cluster | ~1 ms | Hot file metadata, table profiles, recent query plans | TTL 1 h, write-through |
| L2 — system of record | PostgreSQL | ~5 ms | Everything; source of truth | Never |

L0 is fine **as long as it's bounded and read-through**. The current bug is that it's unbounded and write-cached without invalidation.

### 4.2 Postgres schema (essentials)

```sql
-- One row per logical "file" (user upload). Immutable after ingestion.
CREATE TABLE files (
  file_id        UUID PRIMARY KEY,
  org_id         UUID NOT NULL,
  folder_id      UUID,
  owner_user_id  UUID NOT NULL,
  original_name  TEXT NOT NULL,
  blob_uri_raw   TEXT NOT NULL,        -- az://raw/...
  size_bytes     BIGINT NOT NULL,
  status         TEXT NOT NULL,        -- uploaded | ingesting | ready | failed
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One or more Parquet "parts" per file (large files split).
CREATE TABLE file_parts (
  part_id        UUID PRIMARY KEY,
  file_id        UUID NOT NULL REFERENCES files(file_id),
  blob_uri       TEXT NOT NULL,        -- az://parquet/<org>/<file>/part-000.parquet
  size_bytes     BIGINT NOT NULL,
  row_count      BIGINT NOT NULL,
  row_group_count INT NOT NULL,
  part_index     INT NOT NULL,
  UNIQUE (file_id, part_index)
);
CREATE INDEX ON file_parts (file_id);

-- One row per column per file. This is what the LLM agent reads.
CREATE TABLE file_columns (
  file_id          UUID NOT NULL REFERENCES files(file_id),
  ordinal          INT  NOT NULL,
  name             TEXT NOT NULL,
  arrow_type       TEXT NOT NULL,       -- e.g. "Int64", "Utf8", "Timestamp(ns)"
  semantic_type    TEXT,                -- date | id | category | measure | text
  null_count       BIGINT,
  distinct_estimate BIGINT,
  min_value        TEXT,
  max_value        TEXT,
  top_values       JSONB,
  bloom_columns    BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (file_id, ordinal)
);

-- Per-row-group stats, for engine-side or app-side pruning.
CREATE TABLE row_group_stats (
  part_id      UUID NOT NULL REFERENCES file_parts(part_id),
  row_group_idx INT NOT NULL,
  column_name  TEXT NOT NULL,
  min_value    BYTEA,
  max_value    BYTEA,
  null_count   BIGINT,
  PRIMARY KEY (part_id, row_group_idx, column_name)
);
```

### 4.3 Why store row-group stats in Postgres

Parquet already has them in the footer — so why duplicate? Two reasons that matter at your scale:

1. **Avoid downloading 1M footers to plan a query.** If a user asks "what's in folder X across all files," you can prune at the catalog level before touching Blob. This is exactly the external-index pattern DataFusion supports via custom `TableProvider` ([DataFusion external indexes blog](https://datafusion.apache.org/blog/2025/08/15/external-parquet-indexes/)).
2. **Cross-file pruning.** Parquet's own stats are per-file; your stats can answer "which 5 files out of 1000 might contain `customer_id = 42`" without opening any of them.

At 1M files with ~10 row groups each and ~20 columns, that's 200M rows in `row_group_stats`. That's fine in Postgres with proper indexing, but it's the row that needs partitioning eventually — partition by `org_id` once you exceed ~50M rows.

### 4.4 Catalog access pattern (agent side)

When a chat turn starts:

1. Agent resolves `org_id` + `folder_id` + selected files from the conversation context.
2. Agent fetches column profiles for selected files (Redis hit usually; Postgres fallback).
3. Agent builds the prompt: "You have these tables with these columns and these properties."
4. Agent generates SQL.
5. Worker takes `file_id`s, resolves to Parquet blob URIs from `file_parts`, registers them on a fresh `SessionContext`, executes.

The agent **never** sees blob URIs. The worker **never** sees prose. That separation is what lets you change either independently.

---

## 5. Concurrency model

Three layers of concurrency, each with a different scaling primitive.

### 5.1 API layer (FastAPI)

- **Single async event loop per process.** Don't fight Python on this — run multiple processes (gunicorn/uvicorn workers) and let the OS schedule.
- **Process count = vCPU × 1 to 2.** Don't oversubscribe; the agent does network I/O, not CPU.
- **Per-request context** carries `org_id`, `user_id`, `request_id`, and a `correlation_id` propagated as gRPC metadata downstream.

### 5.2 Agent layer (LangGraph)

LangGraph is mostly I/O bound — it's waiting on Azure OpenAI. Async all the way. The only CPU-heavy bit is JSON manipulation; keep results small.

**Critical rule:** the agent must not block on the query worker. Use an async gRPC client and stream record batches back as they arrive. If the agent waits for full result materialization, your tail latency explodes.

### 5.3 Query worker layer (DataFusion)

This is where the design pays for itself.

- **Each worker process** runs one Tokio runtime with a thread pool sized to `num_cpus`.
- **Each incoming query** gets its own `SessionContext`. No shared catalog, no shared cache (beyond the object-store byte cache, which is thread-safe).
- **Concurrency cap per worker:** start with `max_concurrent_queries = num_cpus` (one query × one core). Lower if queries are memory-heavy. This is your single most important capacity knob.
- **Admission control:** if a worker is at capacity, the router routes to another worker; if all are saturated, the request waits in a bounded queue with a max wait time, then 503s back to the agent. The agent can then either retry or surface a "system is busy" message.

This is the opposite of DuckDB's "one connection serializes everything." Here, N workers × M cores = N×M independent query slots, each truly parallel.

### 5.4 Why a separate query-worker pool (instead of in-process DataFusion)

You *could* run DataFusion inside the FastAPI process. Don't, for three reasons:

1. **Blast radius.** A runaway query OOMs the API process, killing in-flight chat sessions.
2. **Independent scaling.** API capacity ≠ query capacity. They scale on different signals.
3. **Hot deploys.** You'll iterate on the agent far more often than on the engine. Decoupling lets you ship agent changes without restarting query workers (and losing their in-memory caches).

Communication: gRPC with Arrow Flight semantics. Either roll your own thin gRPC service that exposes `Execute(sql, file_refs) → stream<RecordBatch>`, or adopt **Arrow Flight SQL** outright — it gives you a standardized protocol, columnar transport, and parallel result-partition fetching ([Arrow Flight SQL spec](https://arrow.apache.org/docs/format/FlightSql.html)). Flight SQL has documented 20–160× speedups over JDBC-style transport for analytical results ([StarRocks Flight SQL benchmarks](https://docs.starrocks.io/docs/unloading/arrow_flight/)). I'd start with a hand-rolled gRPC and adopt Flight SQL once you have a second consumer (e.g., a notebook integration).

---

## 6. Horizontal scaling story

The whole point of getting layers 1–4 right is that scaling becomes additive, not architectural.

### 6.1 Scale targets and what changes

| Stage | Users | What you run | What changes |
|---|---|---|---|
| Today | 30–40 | 2 API pods, 2 ingestion pods, 3 query-worker pods, 1 Postgres, 1 Redis, Azure Blob | Nothing — this design handles it on day one. |
| Medium | 300–400 | 8 API, 6 ingestion, 12 query workers, Postgres read replica, Redis cluster (3 shards) | Add NVMe-backed VM SKUs for query workers; turn on the local Parquet cache (§7). |
| Long | 1000+ | 20+ API, 15+ ingestion, 40+ query workers **or** Ballista cluster, partitioned Postgres, Redis cluster (6+ shards) | Introduce Ballista for queries that scan many files in parallel; partition `row_group_stats` by `org_id`. |

### 6.2 Ballista — when and why

Single-node DataFusion scales beautifully **up to one node's worth of CPU, RAM, and NIC**. Past that, you need to distribute the *plan*, not just add more single-node workers. That's Ballista:

- Scheduler accepts a logical plan, splits into stages, assigns to executors.
- Executors exchange data via Arrow IPC shuffle.
- Same DataFusion engine on every executor, so your SQL surface doesn't change ([Ballista Architecture](https://datafusion.apache.org/ballista/contributors-guide/architecture.html)).

**You don't need Ballista at 300–400 users.** A pool of independent single-node DataFusion workers, each handling whole queries, gets you very far — most user queries hit one file or a handful of files, and that fits on one node. Adopt Ballista only when you start seeing queries that legitimately need many machines for a single execution (cross-organization aggregates, large joins across many files).

### 6.3 What doesn't horizontally scale by default

- **Postgres writes.** Catalog write throughput at 1M files isn't a problem (it's bursty during ingestion); past 10M files, consider sharding by `org_id`.
- **Azure Blob throughput.** Each storage account has a bandwidth ceiling (~50 Gbps ingress per account in many regions). At 1000+ users with hot files, you'll hit it. Mitigations: (a) spread files across multiple storage accounts keyed by `org_id` hash, (b) the local Parquet cache (§7) absorbs most reads anyway.
- **Azure OpenAI tokens-per-minute.** This is often the *real* scaling bottleneck, not the query plane. Provision dedicated capacity (PTUs) before you hit 200+ concurrent users.

---

## 7. Storage tiering & caching

Three tiers, each absorbing a different access pattern.

### 7.1 Tier 1 — Azure Blob (cold/warm)

Authoritative store. Hot path on first access; warm on infrequent access. Your current setup. Two improvements:

- **Lifecycle rules.** Move parts unread for 90 days to Cool tier; 365 days to Archive (org-policy gated). Cuts storage cost meaningfully at 1M files.
- **Separate raw and parquet containers.** Different lifecycle policies; the raw CSVs can go Cool after 7 days because you only need them for re-ingestion.

### 7.2 Tier 2 — Per-worker NVMe Parquet cache

Each query worker VM has 200–500 GB local NVMe. Use it.

- **What gets cached:** entire Parquet parts, on first read, with an LRU eviction policy. Not row groups — full parts (simpler, and Parquet readers love sequential local access).
- **Cache key:** `blob_uri + etag`. ETag from Blob's `If-Match` semantics guarantees invalidation on re-ingest.
- **What this fixes:** the "Azure gets throttled under concurrency" symptom. After warm-up, 80%+ of reads hit local NVMe at 3–5 GB/s instead of Blob at 100–500 MB/s.
- **What this does NOT fix on its own:** multi-VM coherence. That's exactly why you rejected this as a primary solution. The fix: treat it as a *cache*, not the source of truth, and accept that cache misses go to Blob. With consistent hashing of `file_id → worker` in the router (§7.4), the same file tends to land on the same worker, so the cache hit rate is high.

### 7.3 Tier 3 — Object-store byte-range cache (in-process)

Even on a cache miss to Blob, you don't want to re-download the Parquet footer every query. DataFusion's `ObjectStore` layer caches range reads in memory; size this generously (a few GB per worker). The `fsspec.parquet` work demonstrated 85%+ throughput gains from format-aware footer caching on remote object stores ([NVIDIA developer blog on fsspec.parquet](https://developer.nvidia.com/blog/optimizing-access-to-parquet-data-with-fsspec/)) — DataFusion gets this effect natively.

### 7.4 Routing: locality-aware

The router in front of the query workers should do **consistent hashing on `file_id`** (or on the primary file in a multi-file query) to bias each file toward the same worker. This is what makes the per-worker NVMe cache effective. On worker failure, the ring rebalances and the cache rebuilds on the next access — acceptable.

For multi-file queries, pick the largest file as the routing key; smaller files re-fetched from Blob are cheap relative to the dominant scan.

---

## 8. The agent ↔ engine contract

This is the easiest place to make the system either great or fragile. Five rules.

1. **The agent only ever sees the column profile.** Never raw rows, never the Parquet path. This prevents prompt-injection style leakage and keeps the agent prompt small.
2. **SQL is parsed and validated before execution.** Use `sqlparser-rs` (which DataFusion uses anyway). Reject `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ATTACH`, `COPY`, anything with side effects. Reject queries without a `LIMIT` (or auto-inject `LIMIT 10000`).
3. **All queries time out.** Default 30 s wall clock, 60 s for explicit "compute-heavy" turns the user asked for.
4. **All queries are budgeted.** Per-org row-scan budget per minute. A runaway agent on one org cannot starve another org.
5. **Results to the LLM are pre-summarized.** Don't feed the LLM 100k rows. Feed it: schema of the result, top-N rows, summary stats. The LLM composes the natural-language answer from this digest. This is also what makes the system responsive — the digest is tiny.

---

## 9. Observability (the component you didn't list, and need)

You will not run this at 300+ users without it.

- **Per-query trace:** request_id → agent steps → SQL → query plan → row groups read → bytes read from Blob vs. cache → wall time. OpenTelemetry, end to end.
- **Top-N expensive queries per org, per day.** Drives both billing and the "this user's prompt is making the agent do dumb things" investigation.
- **Cache hit rates per worker.** If NVMe hit rate drops below ~70%, your router's hashing is wrong or your working set is too large.
- **Azure Blob 429s.** Direct signal that the cache tier isn't doing its job, or the storage account needs sharding.
- **LLM-generated-SQL failure rate.** When this rises, your column profile quality has degraded. Single most important agent-quality metric.

---

## 10. What you didn't ask about but will need

- **Row-level security.** With 1M files across many orgs in one Blob namespace, the catalog must enforce "this user can only register these `file_id`s." Bake this into the gRPC contract — workers receive `(sql, allowed_file_ids[])` and reject any table reference outside the allowlist. Don't rely on the agent to be careful.
- **Cost attribution.** Track bytes scanned and CPU-seconds per `org_id`. You'll want this for pricing and for finding pathological tenants long before you "want" it.
- **Re-ingestion / schema evolution.** When a user re-uploads the same logical file with a new schema, you need a versioning story. Simplest: each upload is a new `file_id`; the UI shows "versions"; the agent picks the latest unless told otherwise.
- **Backpressure to the LLM.** If the query plane is saturated, the agent should know and degrade gracefully (queue, retry, or tell the user "still working").
- **A "system" dataset.** Reserve one path for engineering-controlled reference data (date dimensions, geography). The agent can join user data against these.

---

## 11. Concrete migration path from today

Don't rewrite in one shot. Three phases.

**Phase 1 (2–3 weeks) — fix the bleeding without changing engines.**

- Replace the module-level dict with Postgres + Redis catalog. Same DuckDB underneath.
- Add the ingestion-side Parquet tuning (row groups, statistics, ZSTD). Re-ingest existing files in the background.
- Add per-VM NVMe cache for Parquet, even with DuckDB. Concurrency still bad, but I/O pressure on Blob drops immediately.

This alone takes you from "20s under load" to "~5s under load" and unblocks the rest.

**Phase 2 (4–6 weeks) — introduce DataFusion behind a feature flag.**

- Build the gRPC query-worker service with DataFusion.
- Route a small percentage of traffic through it. Compare correctness against DuckDB on shadow traffic.
- Once correctness ≥ DuckDB and latency ≤ DuckDB, flip default to DataFusion. Keep DuckDB available for SQL features DataFusion lacks; if the agent emits one of those, route to DuckDB. Over time the dialect gap closes and you remove DuckDB.

**Phase 3 (ongoing) — scale knobs.**

- Add workers as concurrency demands.
- Partition Postgres tables when row counts justify it.
- Adopt Ballista only when you have a query shape that demonstrably needs distributed execution. Not before.

---

## 12. The one-paragraph summary you can give a stakeholder

G-CHAT moves from an in-process DuckDB query model to a stateless, horizontally scalable query plane built on Apache DataFusion. Each user query gets its own ephemeral query session that registers only the Parquet files it needs, executes in parallel across CPU cores, and reads from Azure Blob with column- and row-group-level pruning — cached on local NVMe for hot files. A Postgres + Redis catalog replaces the in-memory dict, providing a coherent system of record across many API and worker processes. The agent and the engine are physically separated by a gRPC contract that prevents the agent from ever seeing storage paths or raw rows. The result handles today's 30–40 concurrent users on a small footprint, scales to 300–400 users by adding worker pods, and extends to 1000+ users via Ballista without changing the SQL surface or the agent.

---

## Sources

- DataFusion async execution model — [Using Rust async for Query Execution and Cancelling Long-Running Queries](https://datafusion.apache.org/blog/2025/06/30/cancellation/)
- DataFusion Parquet filter pushdown internals — [Efficient Filter Pushdown in Parquet](https://datafusion.apache.org/blog/2025/03/21/parquet-pushdown/)
- DataFusion external indexes / custom TableProvider — [Using External Indexes, Metadata Stores, Catalogs and Caches](https://datafusion.apache.org/blog/2025/08/15/external-parquet-indexes/)
- DataFusion Python async record batches — [Apache DataFusion Python 46.0.0 Released](https://datafusion.apache.org/blog/output/2025/03/30/datafusion-python-46.0.0/)
- Production migration from DuckDB to DataFusion — [Duck Hunt: moving Bauplan from DuckDB to DataFusion](https://www.bauplanlabs.com/post/duck-hunt-moving-bauplan-from-duckdb-to-datafusion)
- Ballista distributed architecture — [Ballista Architecture](https://datafusion.apache.org/ballista/contributors-guide/architecture.html) and [Ballista GitHub](https://github.com/apache/datafusion-ballista)
- Arrow Flight SQL protocol — [Apache Arrow Flight SQL](https://arrow.apache.org/docs/format/FlightSql.html)
- Flight SQL transport performance — [StarRocks Arrow Flight SQL benchmarks](https://docs.starrocks.io/docs/unloading/arrow_flight/)
- Row group sizing guidance — [Stack Overflow: what is meant by Parquet row-group size](https://stackoverflow.com/questions/76782018/what-is-actually-meant-when-referring-to-parquet-row-group-size) and [Parquet file anatomy](https://dev.to/databro/apache-parquet-file-anatomy-row-groups-column-chunks-pages-and-metadata-explained-4ebg)
- Iceberg metadata at scale (cautionary tale for catalog design) — [Tackling Apache Iceberg Metadata at Massive Scale](https://www.e6data.com/blog/apache-iceberg-million-files-metadata)
- Object-store Parquet caching — [Optimizing Access to Parquet Data with fsspec](https://developer.nvidia.com/blog/optimizing-access-to-parquet-data-with-fsspec/)
- DuckDB concurrency model — [Real Python on DuckDB Python integration](https://realpython.com/lessons/python-functions-inside-duckdb-queries/)

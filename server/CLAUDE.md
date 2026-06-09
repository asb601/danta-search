# CLAUDE.md — server/ (Deep Dive)

## HOW TO WORK WITH ME (Claude Instructions)

### Token Efficiency Rules
- ONLY read files I explicitly mention
- NEVER scan the full codebase unless I say "scan all"
- ALWAYS ask clarifying questions BEFORE writing code
- STOP and confirm if a change affects more than 2 files

### Project Stack (reference only, don't read all files)
- Backend: Python/FastAPI → server/
- Frontend: Client → client/
- DB: SQLAlchemy ORM + Alembic migrations
- LLM: Anthropic Claude via llm_tasks.py

### Off-limits unless I ask
- testing/
- .github/
- *.txt, *.log files
- __pycache__

This document gives Claude the complete picture of the `server/` directory — every layer, module, data flow, design decision, and pitfall. Read this before modifying any server code.

---

## What This Server Is

The `server/` is a **FastAPI** async Python backend named **danta-search**. It powers:

1. **File management** — upload, organize in folders/containers, manage lifecycle
2. **Ingestion pipeline** — clean → Parquet → embed → index in OpenSearch → build ontology
3. **Semantic chat** — user sends natural language query → retrieval → semantic planning → DataFusion execution → LLM synthesis → SSE streaming response
4. **Admin/RBAC** — organizations, users, roles, domain access control, access request flows
5. **Observability** — structured logs (structlog), in-process metrics, audit trail
6. **Dashboard generation** — natural-language prompt → fan-out of agent queries → recommended visualizations → persisted, render-ready dashboard config (see "Dashboard Generation Layer" below)

Python version: **3.12**. Package manager: **uv**. All code is async-first.

---

## Directory Map

```
server/
├── app/
│   ├── main.py                   ← FastAPI app factory, lifespan, middleware, routers
│   ├── dependencies.py           ← FastAPI dependency injection (DB session, current user)
│   │
│   ├── api/v1/                   ← Route handlers (thin — delegate to services)
│   │   ├── auth.py               ← JWT login, Google OAuth, token refresh
│   │   ├── users.py              ← User CRUD, profile
│   │   ├── organizations.py      ← Org/tenant management
│   │   ├── containers.py         ← Data container CRUD + config
│   │   ├── folders.py            ← Folder hierarchy
│   │   ├── files.py              ← Upload, list, delete, ingest trigger
│   │   ├── chat.py               ← Chat endpoint (delegates to chat_stream/chat_message)
│   │   ├── chat_stream.py        ← SSE streaming chat response
│   │   ├── chat_message.py       ← Non-streaming chat response
│   │   ├── chat_common.py        ← Shared chat logic (retrieval, agent invocation)
│   │   ├── conversations.py      ← Conversation/message history CRUD
│   │   ├── ingest.py             ← Manual ingest trigger endpoint
│   │   ├── admin.py              ← Admin-only operations
│   │   ├── logs.py               ← Log query endpoint
│   │   ├── access.py             ← Access request flow
│   │   └── dashboards.py         ← Dashboard CRUD + folders + NL /generate route
│   │
│   ├── agent/                    ← LangGraph agent + all agent utilities
│   │   ├── state.py              ← AgentState TypedDict
│   │   ├── llm.py                ← LLM factory (gpt-4o vs gpt-4o-mini)
│   │   ├── catalog_cache.py      ← Per-container file catalog cache
│   │   ├── catalog_hydration.py  ← Enriches catalog with metadata
│   │   ├── search_normalization.py ← Query tokenization
│   │   ├── response_helpers.py   ← Extract answer/blob paths/chart hints from agent output
│   │   ├── graph/
│   │   │   ├── graph.py          ← PUBLIC ENTRY POINT — full pipeline orchestration
│   │   │   └── graph_builder.py  ← Builds LangGraph from nodes/edges
│   │   ├── prompts/
│   │   │   └── prompt_builder.py ← System prompt construction
│   │   └── tools/
│   │       ├── sql.py            ← SQL execution tools (DataFusion / DuckDB)
│   │       ├── catalog.py        ← File catalog inspection tools
│   │       ├── column.py         ← Column schema inspection tool
│   │       ├── sample.py         ← Data sampling tool
│   │       ├── stats.py          ← Column statistics tool
│   │       ├── relations.py      ← Relationship graph inspection tool
│   │       └── definition_lookup.py ← Schema dictionary lookup tool
│   │
│   ├── core/                     ← Infrastructure: config, DB, AI clients, logging
│   │   ├── config.py             ← Pydantic Settings (all env vars)
│   │   ├── database.py           ← SQLAlchemy async engine + session factory
│   │   ├── ai_client.py          ← Azure OpenAI async client factory
│   │   ├── openai_client.py      ← OpenAI client utilities
│   │   ├── datafusion_client.py  ← DataFusion session pool + Parquet execution engine
│   │   ├── duckdb_client.py      ← DuckDB fallback execution engine
│   │   ├── opensearch_client.py  ← OpenSearch async client factory
│   │   ├── response_cache.py     ← In-process response cache (Redis-backed in production)
│   │   ├── logger.py             ← structlog setup, named loggers
│   │   ├── db_logger.py          ← DB-persisted log writer
│   │   ├── metrics.py            ← In-process metrics counters/histograms
│   │   ├── orchestration_trace.py ← Per-request trace object
│   │   ├── cost_tracker.py       ← LLM token cost tracking
│   │   ├── token_counter.py      ← Token counting utilities
│   │   ├── llm_tasks.py          ← Async LLM task helpers
│   │   ├── security.py           ← Password hashing, JWT encoding/decoding
│   │   ├── crypto.py             ← Fernet encryption for Azure connection strings
│   │   └── email.py              ← SMTP email sending
│   │
│   ├── models/                   ← SQLAlchemy ORM models (one file per table group)
│   │   ├── user.py               ← User
│   │   ├── organization.py       ← Organization
│   │   ├── container.py          ← ContainerConfig
│   │   ├── folder.py             ← Folder
│   │   ├── file.py               ← File
│   │   ├── file_metadata.py      ← FileMetadata (embeddings, schema, semantic roles)
│   │   ├── file_relationship.py  ← FileRelationship (approved/candidate joins)
│   │   ├── file_analytics.py     ← FileAnalytics (row/col counts, quality scores)
│   │   ├── column_key_registry.py ← ColumnKeyRegistry
│   │   ├── semantic_layer.py     ← SemanticEntity, SemanticMetric, SemanticJoin
│   │   ├── conversation.py       ← Conversation + Message
│   │   ├── conversation_memory.py ← ConversationMemory (compressed history)
│   │   ├── background_job.py     ← BackgroundJob (ingestion job tracking)
│   │   ├── access_request.py     ← AccessRequest
│   │   ├── schema_dictionary.py  ← SchemaDictionary (column business definitions)
│   │   ├── server_log.py         ← ServerLog (audit + request logs)
│   │   └── dashboard.py          ← Dashboard + DashboardFolder (metadata-driven dashboards)
│   │
│   ├── schemas/                  ← Pydantic request/response schemas
│   │
│   ├── services/                 ← Business logic (bulk of the intelligence)
│   │   ├── ingestion_service.py  ← Orchestrates full file ingestion pipeline
│   │   ├── ingestion_stages.py   ← Individual pipeline stage implementations
│   │   ├── ingestion_policy.py   ← Configurable ingestion behavior policy
│   │   ├── ingestion_config.py   ← Ingestion configuration loading
│   │   ├── ingestion_audit.py    ← Ingestion audit trail
│   │   ├── ingestion_confidence.py ← Ingestion quality scoring
│   │   ├── parquet_service.py    ← Parquet conversion + upload to Blob
│   │   ├── data_preprocessor.py  ← CSV/XLSX cleaning, normalization
│   │   ├── preprocessor/         ← Preprocessing sub-modules
│   │   │  (semantic_planner.py removed — was dead code; planning currently lives in the LangGraph agent)
│   │   ├── semantic_roles.py     ← Semantic role classification for columns/files
│   │   ├── semantic_enrichment.py ← Enriches file metadata with semantic context
│   │   ├── semantic_expansion.py ← Workflow domain expansion + continuity notes
│   │   ├── semantic_layer_builder.py ← Builds semantic layer (entities, metrics, joins)
│   │   ├── semantic_rebuild.py   ← Triggers semantic layer rebuild
│   │   ├── semantic_policy.py    ← Semantic governance rules
│   │   ├── workflow_capability_resolver.py ← Domain activation + semantic closure
│   │   ├── workflow_topology.py  ← Workflow graph topology + bridge file detection
│   │   ├── entity_resolver.py    ← Entity extraction and resolution from query
│   │   ├── business_intent_planner.py ← High-level business intent classification
│   │   ├── execution_strategy.py ← Chooses execution path (planner vs agent fallback)
│   │   ├── execution_guards.py   ← Safety guards on generated SQL
│   │   ├── sql_context_builder.py ← Builds SQL context (table registry, column map)
│   │   ├── sql_ast.py            ← SQL AST parsing utilities
│   │   ├── sql_ast_validator.py  ← SQL structural validation (sqlglot AST)
│   │   ├── sql_repair.py         ← Auto-repair malformed SQL
│   │   ├── sql_plan_signature.py ← SQL plan deduplication signatures
│   │   ├── logical_sql.py        ← Logical SQL IR generation
│   │   ├── relationship_detector.py ← Detects join relationships between files
│   │   ├── relationship_index.py ← Tenant-scoped relationship graph index
│   │   ├── column_role_resolver.py ← Column semantic role assignment
│   │   ├── context_service.py    ← Conversation context management
│   │   ├── analytics_service.py  ← Analytics computation entry point
│   │   ├── analytics_computer.py ← Core analytics computations
│   │   ├── calibration_manifest.py ← Model calibration data
│   │   ├── query_confidence.py   ← Query-level confidence scoring
│   │   ├── graph_health.py       ← Semantic graph health scoring
│   │   ├── trust_propagation.py  ← Ingestion trust score propagation
│   │   ├── file_identity.py      ← File identity normalization
│   │   ├── audit_log.py          ← Request audit logging
│   │   └── ingestion_policy.py   ← Ingestion behavior policy
│   │
│   ├── retrieval/                ← Multi-stage retrieval pipeline
│   │   ├── orchestrator.py       ← PUBLIC ENTRY POINT — 9-stage retrieval
│   │   ├── bm25.py               ← PostgreSQL tsvector keyword search
│   │   ├── fuzzy.py              ← pg_trgm trigram similarity search
│   │   ├── embeddings.py         ← Embedding generation
│   │   ├── embeddings_search.py  ← pgvector HNSW cosine similarity search
│   │   ├── opensearch_search.py  ← OpenSearch BM25 + vector hybrid retrieval
│   │   ├── opensearch_indexer.py ← OpenSearch document indexing
│   │   ├── graph_expand.py       ← One-hop semantic graph expansion
│   │   ├── rrf.py                ← Reciprocal Rank Fusion
│   │   ├── temporal.py           ← Temporal date bound extraction from queries
│   │   ├── filters.py            ← Permission + domain filters
│   │   └── semantic_recovery.py  ← Fallback recovery when retrieval is insufficient
│   │
│   ├── migrations/               ← Runtime migration scripts (NOT Alembic)
│   │   ├── retrieval_schema_upgrade.py   ← pgvector + pg_trgm columns
│   │   ├── domain_schema_upgrade.py      ← domain_tag + allowed_domains
│   │   ├── schema_dictionary_upgrade.py  ← SchemaDictionary new columns
│   │   ├── cleaning_config_upgrade.py    ← Cleaning config + quarantine columns
│   │   ├── ontology_schema_upgrade.py    ← column_semantic_roles + GIN index
│   │   ├── semantic_config_upgrade.py    ← Per-container semantic role extensions
│   │   ├── relationship_index_upgrade.py ← Relationship fingerprint index
│   │   ├── semantic_layer_upgrade.py     ← Semantic layer tables
│   │   ├── ingestion_trust_upgrade.py    ← Ingestion trustworthiness columns
│   │   ├── drop_audit_logs.py            ← Remove legacy audit_logs table
│   │   ├── backfill_embeddings.py        ← One-time embedding backfill
│   │   ├── chat_memory_upgrade.py        ← Chat memory schema additions
│   │   └── audit_log_schema_upgrade.py   ← Audit log schema evolution
│   │
│   ├── worker/                   ← Celery async workers
│   │   ├── celery_app.py         ← Celery app configuration
│   │   └── ingest_tasks.py       ← Celery ingestion task definitions
│   │
│   └── policies/                 ← Access control policies
│       └── retrieval_policy.py   ← Retrieval permission policy
│
├── config/
│   └── ingestion_policy.json     ← Default ingestion policy configuration
├── logs/                         ← Local log files
├── uploads/                      ← Temporary upload staging area
├── testing/                      ← Server-side tests
├── pyproject.toml                ← Python project + dependencies
├── .python-version               ← Python 3.12
└── .env                          ← Environment variables (not committed)
```

---

## Application Bootstrap (main.py)

The FastAPI `lifespan` context manager runs at startup and handles:

1. **SQLAlchemy `create_all`** — creates any missing tables
2. **Column migrations** — additive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for every post-initial-schema column
3. **Sequential runtime migrations** — each migration module is imported and called; failures are logged as warnings (non-fatal)
4. **DataFusion context pool warm-up** — pre-registers UDFs in a pool of `SessionContext` instances so the first N concurrent queries don't pay the 150-UDF registration cost

Middleware stack (in order):
1. `SessionMiddleware` — required by authlib OAuth (session cookie)
2. `CORSMiddleware` — allows `FRONTEND_URL` and `localhost:3000`
3. `log_requests` — structured request logging + audit trail (replaces uvicorn access log)

**Important:** Uvicorn's access log is explicitly silenced because it would log raw OAuth codes and tokens.

---

## Configuration (core/config.py)

All settings are in `class Settings(BaseSettings)`. Loaded from `.env` via pydantic-settings. Key settings:

| Setting | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL async URL (`postgresql+asyncpg://...`) |
| `AZURE_OPENAI_ENDPOINT` / `KEY` | Azure OpenAI base + key |
| `AZURE_OPENAI_DEPLOYMENT` | gpt-4o deployment (turn 1 queries) |
| `AZURE_OPENAI_DEPLOYMENT_MINI` | gpt-4o-mini deployment (turn 2+ queries) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment |
| `QUERY_ENGINE` | `"duckdb"` (default) or `"datafusion"` |
| `SQL_VALIDATOR_AST_MODE` | `"primary"` (AST authoritative) / `"shadow"` / `"disabled"` |
| `OPENSEARCH_URL` | Empty = OpenSearch disabled, PostgreSQL retrieval used |
| `REDIS_URL` | Redis for Celery broker (db=0) |
| `REDIS_URL_RESULTS` | Redis for Celery results (db=1) |
| `STORAGE_ENCRYPTION_KEY` | Fernet key for encrypting Azure connection strings at rest |
| `FRONTEND_URL` | Allowed CORS origin |

Ingestion-related settings are proxied through `ingestion_policy.py` — the `__getattr__` override transparently routes `INGEST_*` and policy proxy names to the policy object.

---

## Database Layer (core/database.py + models/)

### Engine
- Async SQLAlchemy 2.0 with `asyncpg`
- Connection pool: configure before deploying multiple workers (200 total connections with 4 workers × 50 each is safe on Azure Postgres Flexible Server)

### ORM Models

| Model | Table | Key columns |
|---|---|---|
| `User` | `users` | id, email, hashed_password, role, is_admin, organization_id |
| `Organization` | `organizations` | id, name, plan |
| `ContainerConfig` | `container_configs` | id, organization_id, name, azure_connection_string (encrypted) |
| `Folder` | `folders` | id, container_id, parent_id, name, domain_tag |
| `File` | `files` | id, container_id, folder_id, filename, blob_path, ingest_status, is_preprocessed |
| `FileMetadata` | `file_metadata` | file_id, container_id, display_name, description, embedding (pgvector), semantic_roles, schema_json, parquet_blob_path, domain_tag, ingestion_confidence |
| `FileRelationship` | `file_relationships` | source_file_id, target_file_id, join_columns, relationship_type (approved/candidate), confidence |
| `FileAnalytics` | `file_analytics` | file_id, row_count, col_count, quality_score |
| `SemanticEntity` | `semantic_entities` | container_id, entity_name, aliases, resolution_examples |
| `SemanticMetric` | `semantic_metrics` | container_id, metric_name, formula, related_entities |
| `SemanticJoin` | `semantic_joins` | container_id, source_entity, target_entity, join_path, confidence |
| `Conversation` | `conversations` | id, user_id, container_id, title |
| `Message` | `messages` | id, conversation_id, role, content, files_used, sql_used |
| `BackgroundJob` | `background_jobs` | id, container_id, file_id, job_type, status, progress |
| `SchemaDictionary` | `schema_dictionary` | id, container_id, column_name, business_definition, parquet_blob_path |
| `ServerLog` | `server_logs` | id, level, logger_name, message, metadata_json, created_at |
| `Dashboard` | `dashboards` | id, container_id, folder_id, owner_id, title, config (JSONB), prompt_history, source_file_ids, is_pinned, status |
| `DashboardFolder` | `dashboard_folders` | id, container_id, owner_id, parent_id, name |

### Migration Strategy
There is **no Alembic**. Migrations are runtime-applied Python scripts in `app/migrations/`. Each runs at startup in the lifespan. They are:
- Idempotent (check column/index existence before adding)
- Non-fatal (wrapped in try/except; failures log a warning)
- Additive only (no destructive changes except `drop_audit_logs` which removes a legacy table)

---

## Authentication (api/v1/auth.py + core/security.py)

- **Local auth**: Email + bcrypt-hashed password → JWT access token
- **Google OAuth**: authlib OIDC flow; session cookie stores OAuth state
- JWT expiry: 7 days (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- All protected endpoints use the `get_current_user` dependency from `dependencies.py`
- Admin-only endpoints also check `user.is_admin` or `user.role == "admin"`

---

## Multi-Tenancy and RBAC

### Tenant isolation
- Every user belongs to an `Organization`
- Every dataset belongs to a `ContainerConfig` (scoped to an organization)
- All data queries filter on `container_id` — this is the primary tenant boundary
- Users can only access containers they have been explicitly granted access to

### Domain access control
- Folders have a `domain_tag` field (e.g., `"finance"`, `"hr"`, `"operations"`)
- Users have an `allowed_domains` list
- The retrieval pipeline applies domain filtering at query time
- Domain filtering runs as an ES query filter on `domain_tag` (not a Python folder walk — that approach is eliminated)

### RBAC roles
- `"user"` — standard user, container-scoped access
- `"admin"` — full access, can manage users/orgs/containers
- Access requests flow: users request access → admins approve → `AccessRequest` table tracks state

---

## File Ingestion Pipeline

### Trigger
Files are ingested via:
1. Manual trigger: `POST /api/ingest/{file_id}` (calls Celery task)
2. Bulk re-ingest: admin endpoint
3. Upload-time auto-ingest: optionally triggered on upload

### Pipeline stages (sequential, Celery-managed)

```
1. Download file from Azure Blob (streaming, no full-memory load)
2. Detect encoding + format (clevercsv, charset-normalizer)
3. Clean / normalize (data_preprocessor.py):
   - strip BOM, normalize line endings
   - standardize column names (snake_case)
   - infer + cast types
   - detect and quarantine outliers
4. Convert to Parquet (parquet_service.py):
   - write via PyArrow
   - upload to Azure Blob alongside original
5. Extract metadata (ingestion_stages.py):
   - row/col counts, quality score, schema JSON
6. Generate AI description (LLM call — gpt-4o-mini):
   - business-friendly description of what the file contains
7. Generate embeddings (Azure OpenAI text-embedding-3-small, 1536-dim):
   - embed the description + schema summary
8. Index in OpenSearch (opensearch_indexer.py):
   - BM25 text fields: display_name, description, column_names
   - vector field: embedding
9. Detect semantic roles (semantic_roles.py):
   - classify each column into semantic roles (vendor, amount, date, etc.)
10. Build relationship graph (relationship_detector.py):
    - detect join keys between this file and existing files in the container
    - write FileRelationship rows
11. Update ontology layer (semantic_layer_builder.py):
    - update SemanticEntity, SemanticMetric, SemanticJoin records
```

### Priority queues
Three Celery queue priorities:
- **high**: clean + Parquet conversion (user is waiting to see file ready)
- **normal**: description generation + embedding
- **low**: relationship detection + ontology updates

### Ingestion policy
Configurable via `config/ingestion_policy.json` or `INGESTION_POLICY_JSON` env var. Controls:
- `REINGEST_BATCH_SIZE`, `REINGEST_BATCH_DELAY_SECONDS`
- `PARQUET_CONVERSION_CONCURRENCY`
- `CELERY_WORKER_CONCURRENCY`, `CELERY_WORKER_PREFETCH_MULTIPLIER`

---

## Retrieval Pipeline (retrieval/orchestrator.py)

The retrieval orchestrator is the single entry point: `retrieve_with_scores(query, user_id, is_admin, db, top_k=20)`.

### 9 stages

| Stage | Module | Method |
|---|---|---|
| 1 | `temporal.py` | Extract date bounds from query text (pure regex, <1ms) |
| 2 | *(implicit)* | Permission clause applied inside every DB query via `build_base_query()` |
| 3 | *(implicit)* | Date overlap filter baked into `build_base_query()` |
| 4 | `bm25.py` | PostgreSQL `tsvector` keyword search (GIN index) |
| 5 | `fuzzy.py` | `pg_trgm` trigram similarity (GIN index) |
| 6 | `embeddings_search.py` | pgvector HNSW cosine similarity |
| 7 | `graph_expand.py` | One-hop expansion through approved semantic joins |
| 8 | `rrf.py` | Reciprocal Rank Fusion across all rank lists |
| 9 | *(orchestrator)* | Return top-K `FileMetadata` rows |

**When OpenSearch is configured** (OPENSEARCH_URL is set), stages 4–6 are replaced by a single OpenSearch hybrid query (BM25 + vector with native RRF). The Python-side BM25/fuzzy/RRF modules become fallback-only.

### After retrieval (graph.py)
The graph pipeline further enriches the shortlist:
1. `resolve_workflow_requirements()` — activates workflow domains via semantic closure
2. `decide_expansion()` — breadth-first domain expansion if workflow is partial
3. `semantic_recovery_retrieve()` — bounded recovery aggregation if retrieval is insufficient
4. `build_workflow_topology()` — builds workflow graph topology + bridge file detection

---

## LangGraph Agent (agent/graph/graph.py)

This is the main public entry point for query execution. Called from `chat_common.py`.

### Full pipeline in `graph.py`

```python
# 1. Normalize query
tokens = tokenize_search_query(query)

# 2. Retrieve candidate files
shortlist, scores = await retrieve_with_scores(query, ...)

# 3. Semantic workflow closure
workflow = await resolve_workflow_requirements(shortlist, query, ...)
expansion = await decide_expansion(workflow, ...)

# 4. Recovery if shortlist is insufficient
if workflow.coverage_state in ("partial", "activation_failed"):
    recovered = await semantic_recovery_retrieve(...)

# 5. Build topology + continuity note
topology = await build_workflow_topology(shortlist + expansion_files, ...)
continuity_note = render_workflow_continuity_note(workflow, topology)

# 6. Load catalog + hydrate
catalog = await load_catalog(container_id, db)
hydrated = await hydrate_files(shortlist, db)

# 7. Build SQL context
sql_context = await build_sql_context(hydrated, ...)

# 8. Classify business intent
intent: BusinessIntentPlan = await build_business_intent_plan(query, catalog, ...)

# 9. Resolve entities
entities: list[EntityCandidate] = await resolve_entities(query, catalog, ...)

# 10. Choose execution strategy
strategy = await plan_execution_strategy(query, intent, entities, sql_context, ...)

# 11. Build system prompt
system_prompt = build_system_prompt(catalog, sql_context, topology, continuity_note, ...)

# 12. Run LangGraph
state = AgentState(messages=[...], catalog=catalog, ...)
result = await graph.ainvoke(state)

# 13. Extract response
answer = extract_answer(result)
blob_paths = extract_blob_paths(result)
chart = infer_chart(result, query)
```

### AgentState fields
```python
class AgentState(TypedDict):
    messages: list          # LangChain messages (add_messages reducer)
    catalog: list[dict]     # File catalog for the container
    connection_string: str  # Azure Blob connection string
    container_name: str     # Azure Blob container name
    parquet_blob_path: str  # Preferred Parquet path (or None)
    tool_call_count: int    # Safety counter
    request_id: str         # Per-request correlation ID
    broaden_nudges: int     # How many "search wider" retries have been issued
    is_first_turn: bool     # Turn 1 vs turn 2+ (kept for state shape compat)
```

### MAX_TOOL_CALLS = 8
The agent is limited to 8 tool calls per query to prevent runaway execution.

### LLM selection
- Turn 1: `gpt-4o` (full deployment — `AZURE_OPENAI_DEPLOYMENT`)
- Turn 2+: `gpt-4o-mini` (mini deployment — `AZURE_OPENAI_DEPLOYMENT_MINI`)

---

## Agent Tools

| Tool | Module | What it does |
|---|---|---|
| SQL execution | `tools/sql.py` | Executes SQL against Parquet via DataFusion or DuckDB |
| File catalog | `tools/catalog.py` | Lists available files + their descriptions/schemas |
| Column inspection | `tools/column.py` | Shows column names, types, sample values |
| Data sampling | `tools/sample.py` | Returns sample rows from a file |
| Column statistics | `tools/stats.py` | min, max, mean, null%, cardinality |
| Relationship graph | `tools/relations.py` | Shows approved join relationships between files |
| Definition lookup | `tools/definition_lookup.py` | Looks up business definitions from SchemaDictionary |

---

## Query Execution Engines

### DataFusion (core/datafusion_client.py) — PREFERRED
- One `SessionContext` per request → zero shared mutable state → true concurrency
- SQL format: AI writes `read_parquet('az://CONTAINER/file.parquet')` → client rewrites to register as table `t0, t1, ...`
- **Context pool**: Pre-warmed at startup. Clean base contexts are reused; contexts with registered tables are discarded.
- Enable via `QUERY_ENGINE=datafusion`

### DuckDB (core/duckdb_client.py) — DEFAULT FALLBACK
- Simpler setup; serializes under concurrent load (thread-local connection sharing)
- Default until DataFusion shadow testing confirms correctness

### Switching
Set `QUERY_ENGINE=datafusion` in `.env` to use DataFusion. No code changes needed.

---

## Semantic Planning Layer (services/semantic_planner.py removed — was dead code; planning currently lives in the LangGraph agent)

The semantic planner sits between retrieval and execution:

1. **Entity resolution** (`entity_resolver.py`) — maps query tokens to known entities (vendors, customers, time periods) using the ontology
2. **Ontology matching** — maps resolved entities to `SemanticEntity` records
3. **Relationship planning** — identifies valid join paths from `SemanticJoin` records
4. **Join-path selection** — selects minimum required join path (never excessive)
5. **Metric resolution** — maps requested metrics to `SemanticMetric` formulas

**High confidence** → deterministic execution plan, bypasses LLM SQL generation entirely.
**Low confidence** → falls back to LangGraph agent path (LLM generates SQL).

The fallback rate is the primary quality metric — log every fallback with the unresolved tokens to drive ontology coverage.

---

## Workflow Capability Resolver (services/workflow_capability_resolver.py)

Performs **semantic workflow closure** to activate workflow domains beyond simple query-token overlap.

### Coverage states
```
complete          → all required workflow domains are covered
partial           → some domains covered, expansion possible
activation_failed → no domains could be activated (returns completeness=0.0)
unknown           → insufficient context to determine
```

### Closure mechanism
Uses bounded semantic closure — not unconstrained graph traversal:
- Safety bounds: `_MAX_CLOSURE_ROUNDS = 2`, `_MAX_CLOSURE_DOMAINS = 16`, `_MAX_CLOSURE_FILE_FANOUT = 40`
- Signals: entity token overlap, original shortlist semantic roles, retrieval vector/opensearch evidence, approved graph edges
- Graph-only activations cannot recursively seed role continuity
- Expansion-added files are visible context but cannot recursively broaden closure

### Example workflow continuity
```
invoice → vendor, payment, receipt, purchase_order
po_lifecycle → vendor, payment, invoice, receipt
payment_reconciliation → receipt, vendor, purchase_order
delivery_status → carrier, receipt
```

---

## OpenSearch Integration (retrieval/opensearch_*)

### Index structure
- One index per container (named `{OPENSEARCH_INDEX_PREFIX}-{container_id}`)
- Created at container creation time
- Deleted entirely at container offboarding (clean tenant removal)
- An alias covers all indices for admin-scope queries

### Document fields
```json
{
  "display_name": "...",      // BM25 text
  "description": "...",       // BM25 text
  "column_names": [...],      // BM25 text
  "domain_tag": "finance",    // filter field
  "container_id": "...",      // scoping
  "embedding": [...]          // dense_vector for HNSW
}
```

### Retrieval
`opensearch_retrieve_with_scores()` executes a single hybrid BM25 + vector query with native RRF (OpenSearch 2.x / Elasticsearch 8.x). The Python-side BM25/RRF modules are fallback only.

---

## Response Streaming (api/v1/chat_stream.py)

Chat responses are streamed via Server-Sent Events (SSE):

```
POST /api/chat/stream
→ EventSourceResponse
→ yields tokens as: data: {"token": "..."}\n\n
→ final message: data: {"done": true, "files_used": [...], "sql": "..."}\n\n
```

The streaming path uses `AsyncIterator` from `graph.py`'s streaming entry point.

---

## Structured Logging (core/logger.py)

All logging uses **structlog** with JSON output. Named loggers:

| Logger | Purpose |
|---|---|
| `upload_logger` | File upload events |
| `folder_logger` | Folder operations |
| `container_logger` | Container operations |
| `auth_logger` | Authentication events |
| `chat_logger` | Chat pipeline events |
| `pipeline_logger` | Ingestion pipeline events |

Every log event is also persisted to the `server_logs` table via `db_logger.py`.

**Never log OAuth codes, tokens, or credentials.** The `log_requests` middleware replaces uvicorn's access log specifically to avoid this.

---

## Metrics (core/metrics.py)

In-process metrics exposed at `GET /api/metrics`:
- Query latency percentiles
- LLM error rates
- Rate limit rejection counts
- Blob bytes transferred
- Celery queue depth
- Parquet conversion failure rate
- PostgreSQL connection pool state

---

## SQL Validation (services/sql_ast_validator.py)

Two-mode SQL validation:
- **AST mode** (`SQL_VALIDATOR_AST_MODE=primary`): sqlglot parses SQL into an AST, validates structure, column references, join safety — authoritative in production
- **Regex mode**: legacy pattern-based validation, used as shadow or fallback
- **Disabled mode**: no validation (emergency bypass only)

### SQL safety guards (services/execution_guards.py)
- Blob allowlist: only whitelisted Azure Blob paths can be queried (per-request)
- No `DROP`, `DELETE`, `INSERT`, `UPDATE` — read-only enforcement
- No arbitrary schema exploration

---

## Celery Workers (worker/)

### celery_app.py
```python
app = Celery(
    broker=settings.REDIS_URL,        # db=0
    backend=settings.REDIS_URL_RESULTS # db=1
)
```

### ingest_tasks.py
Defines Celery tasks:
- `ingest_file_task` — full ingestion pipeline for a single file
- `reingest_container_task` — bulk re-ingest all files in a container
- `backfill_embeddings_task` — one-time embedding backfill

### Running workers
```bash
cd server
uv run celery -A app.worker.celery_app worker --loglevel=info -Q high,normal,low
```

---

## Key Design Rules (Do Not Violate)

1. **Never query raw CSV/XLSX in analytical paths** — always use Parquet
2. **Never do runtime schema discovery** — all schema intelligence is pre-computed at ingestion
3. **Never generate arbitrary LLM joins** — joins must be relationship-validated and ontology-backed
4. **Never hydrate full datasets into memory** — stream files, use lazy execution
5. **Never run ingestion in-process** — use Celery workers
6. **Never use pandas at scale** — use Polars + PyArrow for large datasets
7. **pgvector is a fallback only** — OpenSearch is the production retrieval engine
8. **The response cache must be Redis-backed in multi-worker deployment** — the in-process `OrderedDict` is per-worker and invisible across workers; don't cache answers shorter than 50 tokens or containing fallback phrasing

---

## Common Development Tasks

### Run the server
```bash
cd server
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### Run Celery worker
```bash
cd server
uv run celery -A app.worker.celery_app worker --loglevel=info
```

### Add a new API endpoint
1. Create/update route file in `app/api/v1/`
2. Add business logic to `app/services/`
3. Add/update Pydantic schemas in `app/schemas/`
4. Mount router in `app/main.py` if new file
5. Update ORM models in `app/models/` if new table needed
6. Write migration in `app/migrations/` if schema change
7. Call migration from `app/main.py` lifespan

### Add a new ingestion stage
1. Implement in `app/services/ingestion_stages.py`
2. Wire into `app/services/ingestion_service.py`
3. Wrap as Celery task in `app/worker/ingest_tasks.py` if it needs separate queue priority

### Add a new LangGraph tool
1. Create module in `app/agent/tools/`
2. Build tool factory function returning a LangChain `Tool`
3. Register in `app/agent/graph/graph.py` tool build section

### Debug a chat query
1. Check `server_logs` table — filter by `trace_id`
2. Check `GET /api/metrics` for error rate spike
3. Set `SQL_VALIDATOR_AST_MODE=shadow` to disable AST validator if SQL is being incorrectly rejected
4. Set `QUERY_ENGINE=duckdb` to fall back from DataFusion if execution errors

---

## Dashboard Generation Layer (services/dashboard/ + api/v1/dashboards.py)

A thin orchestration layer ABOVE the existing query runtime. It turns ONE
natural-language dashboard prompt into MANY analytical datasets, recommends a
visualization per dataset, and assembles a persisted, render-ready config.
Canonical design lives in the repo-root `response.txt`.

### Hard reuse rule
The dashboard layer contains NO query logic. It does not write SQL, resolve
joins, or discover schema. **One widget == one `run_agent_query()` call.** All
retrieval/planning/execution/grounding is delegated to the existing agent.

### Pipeline (api/v1/dashboards.py :: generate_dashboard)
```
prompt
 → resolve_chat_scope()                     # same RBAC as chat (container + domains)
 → data_catalog.build_catalog()             # read projection over FileMetadata/FileAnalytics
 → query_engine.decompose_prompt()          # LLM → list[WidgetIntent] (capped, viz-aware)
 → for each intent (SEQUENTIAL):            # async DB session is NOT concurrency-safe
     query_engine.run_widget()              #   → run_agent_query() (REUSE)
     query_engine.profile_dataset()         #   → DatasetShape (cardinality/intent/roles)
     recommendation_engine.recommend()      #   → ResolvedWidget (component + bound config)
 → assembly_engine.assemble()               # order + 12-col grid → DashboardConfig
 → persist Dashboard.config (JSONB) + prompt_history; status="ready"
```

### Modules
| Module | Role |
|---|---|
| `component_catalog.py` | Metadata registry of components. Visualization logic is DATA (`visualization_rules`, `config_schema`, `rendering_metadata`). 9 seed types; add components as registry rows. |
| `data_catalog.py` | READ-ONLY projection over `FileMetadata` + `FileAnalytics` + semantic roles → `DataCatalogTable` DTOs. No new storage. Scoped by `container_id` + `allowed_domains`. |
| `query_engine.py` | `decompose_prompt()` (LLM, never raises → deterministic single-widget fallback), `run_widget()` (wraps `run_agent_query`), `profile_dataset()` (pure profiling → `DatasetShape`). |
| `recommendation_engine.py` | `recommend()` — explicit user viz wins; else rule-score every component vs shape; fallback to table. Pure + explainable (a future ML ranker slots behind the same signature). |
| `assembly_engine.py` | `assemble()` — order by info hierarchy, pack into 12-col grid, emit versioned `DashboardConfig`. |

### Persistence
- `models/dashboard.py`: `Dashboard` (config JSONB, prompt_history, source_file_ids,
  is_pinned, status) + `DashboardFolder`. Tenant isolation via `container_id`.
- `migrations/dashboard_upgrade.py`: additive, non-fatal, runs in the lifespan
  alongside the other runtime migrations.
- The dataset snapshot is embedded in `config.widgets[].data` so users return
  later and render instantly WITHOUT regenerating.

### Routes (`/api/dashboards`)
CRUD + folders + `/duplicate` + `/generate` + `/catalog/components` +
`/catalog/data`. Ownership is enforced on every dashboard (owner_id == user).

### Pitfalls
- Do NOT parallelize widget generation over the request's `db` session — the
  async session is not concurrency-safe. Widgets run sequentially (cap ≤ 8).
- `decompose_prompt` and `run_widget` never raise; a failed widget degrades to
  an empty "no data" table tile so the dashboard always returns.

---

## Things That Are Intentionally Not Built Yet

| Not Built | Reason |
|---|---|
| Kubernetes | Reach 1000 DAU on a single VM first |
| gRPC between services | REST is fast enough; gRPC adds protobuf maintenance burden |
| Kafka for ingestion | Batch processing is correct; Kafka adds ops overhead |
| Fine-tuned LLM | Prompt engineering still has 90% of headroom untapped |
| GraphQL API | REST + SSE is the right transport for streaming |
| LLM output validation guards | SQL safety layer + blob allowlist are sufficient |
| Custom vector store | OpenSearch HNSW handles 100M+ vectors in production |

# CLAUDE.md — G-CHAT- / danta-search

This file gives Claude (and any AI assistant) the essential context needed to work effectively in this repository. Read this before touching any code.

---

## What This Project Is

**danta-search** is a production-grade enterprise analytics AI platform. It is **not** a simple RAG chatbot or a dataframe assistant.

Users upload business datasets (CSV, XLSX, Parquet). The system ingests, cleans, converts to Parquet, embeds, and indexes them. Users then chat with the system in natural language. The system semantically plans, retrieves relevant files, executes deterministic analytical SQL via DataFusion, and synthesizes a grounded response.

The core design principle: **intelligence lives in ingestion, not at query time.** The runtime only retrieves, plans, executes, and synthesizes — it never discovers schema or business meaning dynamically.

---

## Repository Layout

```
G-CHAT-/
├── server/          # FastAPI Python backend (the primary codebase)
│   ├── app/
│   │   ├── main.py              # FastAPI app, lifespan, middleware, router mounts
│   │   ├── api/v1/              # REST API route handlers
│   │   ├── agent/               # LangGraph agent, state, tools, prompts
│   │   ├── core/                # Config, DB engine, AI clients, logging, metrics
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── services/            # Business logic layer
│   │   ├── retrieval/           # Retrieval pipeline (BM25, embeddings, OpenSearch, RRF)
│   │   ├── migrations/          # Runtime migration scripts (not Alembic)
│   │   ├── worker/              # Celery app + ingestion tasks
│   │   └── policies/            # Data access / RBAC policies
│   ├── pyproject.toml           # Python deps (uv)
│   └── .python-version          # Python 3.12
│
├── client/          # Next.js 14 frontend (TypeScript)
│   ├── app/                     # Next.js App Router pages
│   ├── components/              # React components
│   ├── hooks/                   # Custom React hooks
│   ├── lib/                     # Utility functions
│   └── package.json
│
├── testing/         # Integration / end-to-end tests
├── ARCHITECTURE_DEEP_DIVE.txt   # Canonical architecture decisions
├── IMPLEMENTATION_DETAILS.md    # Latest semantic workflow implementation notes
└── DATABASE_TABLES.txt          # Database schema reference
```

---

## Tech Stack

### Backend (server/)
| Layer | Technology |
|---|---|
| Web framework | FastAPI (async) + uvicorn |
| ORM | SQLAlchemy 2.0 async + asyncpg |
| Database | PostgreSQL (users, conversations, metadata, jobs, RBAC) |
| Analytical execution | DataFusion (Parquet scans, joins, aggregations) |
| File storage | Azure Blob Storage (adlfs + fsspec) |
| Analytical format | Apache Parquet (PyArrow) |
| Retrieval | OpenSearch (BM25 + vector hybrid with native RRF) |
| Embeddings | Azure OpenAI (text-embedding-3-large) |
| LLM | Azure OpenAI (gpt-4o / gpt-4o-mini) |
| Agent orchestration | LangGraph |
| Background jobs | Celery + Redis |
| Cache | Redis |
| Auth | JWT + OAuth2 (authlib) |
| Structured logging | structlog |
| Dataframe ops | Polars + PyArrow (NOT pandas at scale) |

### Frontend (client/)
| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Deployment | Vercel |

---

## Architecture: The 4-Layer Runtime

```
User Query
    ↓
[1] Retrieval Service      ← OpenSearch hybrid (BM25 + vector + RRF)
    ↓
[2] Semantic Planner       ← entity resolution, ontology matching, join-path selection
    ↓
[3] DataFusion Executor    ← Parquet scans, predicate pushdown, lazy execution
    ↓
[4] Response Synthesizer   ← LLM answer generation with grounded evidence
```

The planner sits between retrieval and execution. High-confidence plans bypass LLM SQL generation entirely. Low-confidence plans fall back to the LangGraph agent path.

---

## Ingestion Pipeline

Files go through a strict sequential pipeline. Each step must complete before the next:

```
Upload → Cleaning/Normalization → Column Standardization → Type Inference
       → Partitioning → Parquet Conversion → Metadata Extraction
       → Description Generation (LLM) → Embedding → OpenSearch Indexing
       → Relationship Detection → Ontology Mapping
```

**Critical rules:**
- Never query raw CSV/XLSX in analytical paths — always use Parquet
- Ingest asynchronously through Celery workers (not in-process asyncio tasks)
- Stream large files; never hydrate entire datasets into memory

---

## Key Architectural Decisions (Do Not Change Without Reason)

| Decision | Rationale |
|---|---|
| Azure Blob + Parquet | Immutable analytical storage, column pruning, predicate pushdown |
| PostgreSQL for metadata | Transactional, relational, RBAC — NOT for vector retrieval at scale |
| OpenSearch for retrieval | Per-tenant index, HNSW vector + BM25, native RRF, fast deletes |
| DataFusion for analytics | Lazy execution, UDF support, Arrow-native, no full-memory hydration |
| Celery for ingestion | Multi-worker parallelism, retry policies, priority queues |
| Redis for caching | Cross-worker response cache; per-container key prefix |

### What NOT to use
- Do not query CSV or XLSX directly in analytical execution paths
- Do not use pandas for large-scale runtime analytics (use Polars + Arrow)
- Do not perform runtime schema discovery — all schema intelligence is pre-computed at ingestion
- Do not generate arbitrary LLM joins — joins must be relationship-validated and ontology-backed
- Do not use pgvector for enterprise-scale retrieval — use OpenSearch

---

## Multi-Tenancy Model

Isolation is enforced at every layer using `container_id`:

| Layer | Isolation mechanism |
|---|---|
| PostgreSQL | Row-level `container_id` scoping on all tenant tables |
| Azure Blob | Separate storage account per enterprise client |
| OpenSearch | One index per container (created at container creation, deleted at offboarding) |
| Redis | Key prefix per `container_id` |
| DataFusion | Blob allowlist enforced per request |

---

## Security Notes

- JWT tokens are used for API authentication
- OAuth codes/tokens are NOT logged (uvicorn access log is suppressed; structured middleware handles request logging)
- RBAC is enforced through the policies layer
- Domain access control: users have `allowed_domains`; files have `domain_tag`
- Multi-tenant data is fully isolated — a leaked key for one tenant cannot reach another tenant's data

---

## Development Setup

### Backend
```bash
cd server
uv sync                    # install deps
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd client
npm install
npm run dev
```

### Celery Worker
```bash
cd server
uv run celery -A app.worker.celery_app worker --loglevel=info
```

### Environment
Copy `.env.example` to `.env` in both `server/` and `client/` and fill in:
- `DATABASE_URL` — PostgreSQL async URL
- `AZURE_STORAGE_*` — Blob storage credentials
- `AZURE_OPENAI_*` — LLM + embedding endpoints/keys
- `OPENSEARCH_*` — OpenSearch connection
- `REDIS_URL` — Redis connection
- `SECRET_KEY` — Session/JWT secret

---

## API Structure

All API routes are mounted under `/api`:

| Router | Prefix | Purpose |
|---|---|---|
| auth | `/api/auth` | Login, OAuth, JWT refresh |
| users | `/api/users` | User management |
| organizations | `/api/organizations` | Org/tenant management |
| containers | `/api/containers` | Data container management |
| folders | `/api/folders` | Folder hierarchy |
| files | `/api/files` | File upload, ingest, status |
| chat | `/api/chat` | Chat SSE streaming |
| admin | `/api/admin` | Admin operations |
| logs | `/api/logs` | Structured log access |
| access | `/api/access` | Access request flow |
| dashboards | `/api/dashboards` | Metadata-driven dashboard generation + CRUD |

Health: `GET /api/health` — Metrics: `GET /api/metrics`

---

## Dashboard Generation Layer (metadata-driven)

A thin orchestration layer **above** the existing query runtime that turns a
natural-language prompt into a persisted, render-ready dashboard. It does **not**
introduce a second query brain — it REUSES retrieval → planner → DataFusion →
agent (`run_agent_query`). Full design: `response.txt`.

```
Dashboard prompt
   ↓
[A] Query Engine      — decompose prompt → N widget intents (LLM), then call
    (services/dashboard/query_engine.py)   run_agent_query() PER intent (reuse),
                                           then profile each dataset → DatasetShape
   ↓
[B] Recommendation    — score every catalog component vs DatasetShape + intent +
    (recommendation_engine.py)             explicit viz request → bind columns
   ↓
[C] Assembly          — order (KPIs→trends→comparisons→tables) + 12-col grid
    (assembly_engine.py)                   → DashboardConfig (versioned JSON)
   ↓
[D] Persist           — Dashboard.config JSONB (dataset snapshot embedded so
    (models/dashboard.py)                  users return later without regenerating)
```

**Component Catalog** (`services/dashboard/component_catalog.py`): a metadata
registry of reusable components (KPI card, metric tile, table, line/bar/pie/area
chart, heatmap, funnel). Visualization logic is DATA (`visualization_rules`,
`config_schema`, `rendering_metadata`) — adding a component is a registry change,
not a renderer change. Supports hundreds of components without hardcoding.

**Data Catalog** (`services/dashboard/data_catalog.py`): a READ-ONLY projection
over existing `FileMetadata` + `FileAnalytics` + semantic roles. No new storage —
the catalog already exists in normalized form from ingestion.

**New tables**: `dashboards`, `dashboard_folders` (models/dashboard.py;
migrations/dashboard_upgrade.py). Tenant-isolated via `container_id`.

**Frontend**: sidebar "Dashboards" entry → `/dashboards` workspace (folders, pin,
search, rename, duplicate, delete) → `/dashboards/[id]` (chat-style prompt +
`DashboardRenderer`). The renderer is a pure function of `DashboardConfig`;
catalog components live in `client/components/analytics-catalog/` (pure SVG/CSS,
zero chart dependency, driven by the `--chart-*` OKLch tokens).

**Key reuse rule**: the dashboard layer NEVER writes SQL, resolves joins, or
discovers schema. One widget == one `run_agent_query` call. The only
orchestration change is a "fan-out coordinator" (one prompt → many agent calls).
Widgets run sequentially because the async DB session is not concurrency-safe.

---

## What To Build Next (Priority Order)

1. **Semantic Planner** — entity resolution + ontology-backed join path selection (biggest runtime risk reducer)
2. **Celery worker graph** — proper priority queues for ingestion (eliminates in-process asyncio task management)
3. **OpenSearch migration** — replace pgvector with per-tenant OS indices
4. **DataFusion context pool** — pre-warmed session contexts to eliminate UDF registration overhead per query
5. **Redis cross-worker cache** — replace in-process `OrderedDict` response cache

Do NOT build yet: Kubernetes, gRPC, GraphQL, fine-tuned LLM, Kafka.

---

## Important Files to Know

| File | Why it matters |
|---|---|
| `server/app/main.py` | App bootstrap, all lifespan migrations, router mounts |
| `server/app/core/config.py` | All settings via pydantic-settings |
| `server/app/agent/graph/graph.py` | LangGraph agent — central query execution path |
| `server/app/services/semantic_planner.py` | Semantic planning layer |
| `server/app/services/workflow_capability_resolver.py` | Workflow domain activation + semantic closure |
| `server/app/services/dashboard/` | Dashboard generation engines (catalog, query, recommendation, assembly) |
| `server/app/api/v1/dashboards.py` | Dashboard CRUD + the `/generate` route |
| `client/components/analytics-catalog/` | Metadata-driven chart components + `DashboardRenderer` |
| `server/app/retrieval/orchestrator.py` | Retrieval pipeline orchestration |
| `server/app/core/datafusion_client.py` | DataFusion session pool + Parquet execution |
| `server/app/worker/ingest_tasks.py` | Celery ingestion task definitions |
| `ARCHITECTURE_DEEP_DIVE.txt` | Canonical architecture decisions — read before redesigning anything |
| `IMPLEMENTATION_DETAILS.md` | Latest semantic workflow assembly implementation notes |

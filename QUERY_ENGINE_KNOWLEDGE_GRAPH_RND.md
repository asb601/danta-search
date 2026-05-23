# G-CHAT Query Engine and Knowledge Graph R&D Notes

Date: 2026-05-23

Audience: client developers, backend developers, and technical stakeholders who need to understand what the system does today, why the latest SAP query failed, how ingestion and querying work, and what the ideal knowledge graph/query engine should become.

Scope: documentation only. This file explains current code, current runtime evidence, and recommended architecture direction. It does not describe an application code change.

---

## 1. Executive Summary

The system already has the correct foundation:

- Uploaded files live in Azure Blob Storage.
- Ingestion extracts metadata, sample rows, schema, descriptions, semantic roles, embeddings, analytics, technical relationships, and semantic entities/relationships.
- Postgres stores lightweight knowledge about the files, not full source datasets.
- Chat-time retrieval works over metadata first.
- SQL execution reads actual data remotely from Azure Blob through DuckDB or DataFusion.
- The LLM sees file metadata, tool outputs, and result previews, not full files.

The current weakness is query-time planning. Today, complex queries can still behave like this:

```text
retrieve a few files -> let the LLM inspect schemas -> let the LLM write SQL -> react to errors
```

That is acceptable for simple single-file questions. It is not enough for enterprise ERP questions with many joins, optional enrichments, missing fields, and business constraints.

The target architecture should be:

```text
user query
  -> business intent decomposition
  -> canonical concept resolution
  -> physical file/column binding
  -> dependency graph expansion
  -> join contract validation
  -> deterministic SQL plan
  -> SQL execution
  -> answer with provenance and limitations
```

The LLM should not guess ten joins from a short prompt. The knowledge graph and planner should produce the join plan; the LLM should help interpret language and explain results.

---

## 2. What We Are Building

G-CHAT is a metadata-first query engine over client-uploaded business files.

It is not intended to become a local data warehouse on the VM. The VM coordinates ingestion, metadata extraction, retrieval, SQL execution, and LLM calls. The full data remains in object storage.

The important split is:

| Layer | Stores | Purpose |
|---|---|---|
| Azure Blob Storage | source CSV/TXT/Excel files and generated Parquet | full data source of truth |
| Postgres | metadata, samples, roles, embeddings, analytics, relationships, semantic graph | retrieval and planning knowledge |
| Optional OpenSearch | per-container metadata index | faster BM25/fuzzy/vector metadata retrieval |
| DuckDB/DataFusion | query execution engine | reads remote `az://` files and returns bounded results |
| LLM | selected metadata, tool outputs, small result previews | language understanding and answer synthesis |

At query time, the system should read from Azure Blob only when SQL execution is required. It should not send full datasets to the LLM.

---

## 3. How We Avoid Pushing Full Data Load Onto The VM

The current design is network-backed and metadata-first.

### 3.1 Every External Call Is A Network/REST Call

Nothing in the runtime stack reads from local disk. Every data exchange leaves the VM over the network:

| Call | Protocol | What goes over the wire | Direction |
|---|---|---|---|
| Postgres metadata read | TCP / PostgreSQL wire protocol | SQL query; result rows (metadata, embeddings, samples) | VM -> Postgres instance |
| OpenSearch metadata query | HTTPS REST `POST /_search` | JSON query body; JSON result hits | VM -> OpenSearch endpoint |
| Azure OpenAI LLM call | HTTPS REST `POST /chat/completions` | JSON payload with shortlisted metadata, tool outputs; JSON streaming SSE response | VM -> Azure OpenAI |
| Azure OpenAI embedding call | HTTPS REST `POST /embeddings` | JSON with text chunk; JSON with float vector | VM -> Azure OpenAI |
| Azure Blob Storage — Parquet read | HTTPS REST `GET` with `Range` header | byte range request for Parquet row group; byte response | VM -> Azure Blob |
| Azure Blob Storage — file upload | HTTPS REST `PUT` | CSV/Excel/Parquet bytes | client -> Azure Blob (proxied via server) |
| Azure Blob Storage — Parquet write | HTTPS REST `PUT` | Parquet byte stream | VM -> Azure Blob |
| Celery task dispatch | AMQP / Redis TCP | task message JSON | VM -> message broker |
| SSE response to browser | HTTP chunked transfer-encoding | JSON event tokens | VM -> browser |

There is no `open('/data/myfile.csv')` call for query execution. DuckDB and DataFusion issue `Range` HTTP requests to Azure Blob to read only the Parquet row groups they need.

### 3.2 How DuckDB/DataFusion Read Remote Files

DuckDB uses the Azure extension:

```python
conn.execute("INSTALL azure; LOAD azure;")
conn.execute(f"SET azure_storage_connection_string = '{connection_string}';")
result = conn.execute("SELECT ... FROM read_parquet('az://container/path.parquet')")
```

Internally DuckDB translates `az://` into HTTPS `GET` + `Range` requests against Azure Blob REST API. Only the Parquet row groups needed by the predicate are fetched. A 500 MB file can often be answered with 2-5 MB of network traffic if filters are selective.

DataFusion does the same with its Azure object store client:

```python
azure_store = MicrosoftAzure(account_name=..., access_key=...)
ctx = SessionContext()
ctx.register_object_store("az://container/", azure_store)
ctx.register_parquet("t0", "az://container/path.parquet")
ctx.sql("SELECT ... FROM t0")
```

DataFusion's per-request `SessionContext` means 40 concurrent queries can read from Azure Blob in truly parallel HTTPS threads, bounded only by CPU and blob bandwidth. There is no shared mutable connection state.

### 3.3 Full Network Call Sequence For A Chat Query

```text
Browser
  |-- HTTPS POST /message/stream ---------> Server VM
                                               |
                                               |-- TCP SQL (SELECT metadata) ----------> Postgres
                                               |<- metadata rows -------------------------------|
                                               |
                                               |-- HTTPS POST /embeddings ------------> Azure OpenAI
                                               |<- embedding vector -----------------------------|
                                               |
                                               |-- TCP SQL (vector search, BM25 etc.) -> Postgres / OpenSearch
                                               |<- ranked file list ----------------------------|
                                               |
                                               |-- HTTPS POST /chat/completions (stream) -> Azure OpenAI
                                               |   [sends: shortlisted metadata + history]
                                               |   LLM decides: call tool run_sql
                                               |
                                               |-- HTTPS GET Range request -----------> Azure Blob (Parquet)
                                               |<- Parquet row group bytes -------------------|
                                               |   DuckDB/DataFusion executes SQL locally
                                               |   Result rows (capped) buffered in RAM
                                               |
                                               |-- HTTPS POST /chat/completions (stream) -> Azure OpenAI
                                               |   [sends: tool output (result preview)]
                                               |<- final answer tokens (SSE stream) ----------|
                                               |
  |<-- SSE chunks (answer tokens) --------- Server VM
```

At no point does the full dataset leave Azure Blob. The VM holds only:
- small Postgres metadata read results
- the bounded SQL result preview (capped rows)
- the LLM streaming buffer for one request

### 3.4 Load Bounds

| Feature | How it limits VM load |
|---|---|
| Parquet format | columnar, compressed; row group predicate pushdown; much less data fetched vs raw CSV |
| Metadata-first retrieval | zero Azure Blob calls if the answer is cached or fully metadata-based |
| SQL result cap in `run_sql` | LLM never receives millions of rows; result is bounded |
| Response cache | repeated non-hollow answers served from in-process cache; no LLM or Blob call |
| `asyncio.Semaphore(5)` in chat endpoint | max 5 concurrent LLM calls; prevents token-rate burnout |
| DataFusion per-request `SessionContext` | concurrent queries do not block each other |
| Celery worker pool | ingestion runs in separate worker processes; does not compete with chat VM RAM |

The honest production statement: this is not zero VM load; it is bounded and predictable VM load. The VM coordinates and executes against remote storage instead of hosting the datasets locally.

---

## 4. Ingestion Pipeline From Current Code

### 4.1 API Trigger

The ingestion route is `server/app/api/v1/ingest.py`.

Flow:

```text
POST /ingest
  -> validate file ids and extensions
  -> run_ingest_pipeline.delay(file_id)
  -> return Celery task ids immediately
```

Ingestion runs in Celery workers, not inside the API request.

### 4.2 Effective Stage Order

The effective order comes from `INGEST_STAGE_SPECS` in `server/app/services/ingestion_config.py`:

```text
clean
metadata
ai_description
ontology
embedding
opensearch
parquet
analytics
relationships
semantic_layer
complete
```

Some comments in older files mention `clean -> parquet -> metadata`; the code's source of truth is `INGEST_STAGE_SPECS`.

### 4.3 Stage Responsibilities

| Stage | Main code | What it does | Main outputs |
|---|---|---|---|
| `clean` | `ingestion_stages.clean_file_stage` | preprocess text/Excel, cleaning audit, quarantine samples | cleaned blob path, preprocessing flags, analytics audit |
| `metadata` | `ingestion_stages.metadata_stage`, `duckdb_client.sample_file` | sample through DuckDB, infer schema/types/sample values | `FileMetadata.columns_info`, `sample_rows`, `row_count` |
| `ai_description` | `core.ai_client.generate_file_description` | one LLM call per file for summary, metrics, dimensions, date range | `ai_description`, `good_for`, `key_metrics`, `key_dimensions`, `date_range_*` |
| `ontology` | `column_role_resolver.resolve_column_roles` | one LLM call per file to assign typed semantic roles | `column_semantic_roles`, `role_source` |
| `embedding` | `retrieval.embeddings.embed_text` | build search text and vector | `search_text`, `description_embedding` |
| `opensearch` | `retrieval.opensearch_indexer` | optional metadata indexing | OpenSearch document |
| `parquet` | `analytics_service.trigger_parquet_conversion` | convert supported text sources to Parquet in Azure Blob | `FileAnalytics.parquet_blob_path` |
| `analytics` | `analytics_service.compute_and_store_analytics` | stats/value counts/crosstabs | `FileAnalytics.column_stats`, `value_counts`, `cross_tabs` |
| `relationships` | `relationship_detector.detect_relationships` | register key fingerprints and create technical relationship candidates | `ColumnKeyRegistry`, `FileRelationship` |
| `semantic_layer` | `semantic_layer_builder.build_semantic_layer_for_file` | build business entities and semantic relationships | `SemanticEntity`, `SemanticRelationship` |
| `complete` | `complete_ingestion_stage` | mark ingested and invalidate catalog cache | final status |

### 4.4 Failure Behavior

`server/app/worker/ingest_tasks.py` treats these later stages as non-fatal:

```text
ontology, embedding, opensearch, parquet, analytics, relationships, semantic_layer
```

If one fails, ingestion can continue with a warning. This is good for availability, but it means a file may be usable while the knowledge graph is incomplete. Query quality depends on those stages being healthy and complete.

---

## 5. Knowledge Graph Today

### 5.1 Semantic Roles

`server/app/services/semantic_roles.py` defines behavior kinds, not a closed business enum:

```text
entity_key
reference_key
additive_measure
non_additive_measure
date
attribute
```

Concrete roles can be dynamic:

```text
custom:entity_key:vendor
custom:reference_key:cost_center
custom:additive_measure:amount
custom:date:posting_date
custom:attribute:country
```

This is the right direction. Client columns vary by source system, so code should not hardcode every possible business noun. The resolver maps source columns once at ingestion.

### 5.2 Technical Relationship Graph

`server/app/services/relationship_index.py` implements a DB-backed fingerprint registry:

```text
column value -> normalize -> hash fingerprint -> store in Postgres array
```

Then `relationship_detector.py` finds tenant-scoped overlaps using Postgres GIN array overlap. This avoids scanning every file against every other file.

`FileRelationship` means:

```text
these columns have technical evidence that they may join
```

It is not automatically business-safe.

### 5.3 Semantic Layer

`SemanticRelationship` adds business-level metadata:

```text
relationship_type
approval_status
risk_reason
join_rule
confidence_score
```

This distinction is critical:

```text
FileRelationship = possible technical join
SemanticRelationship = business relationship with safety metadata
```

The current semantic layer is the correct place to prevent dangerous joins, but it needs stronger evidence and richer join contracts.

### 5.4 What The Knowledge Graph Actually Looks Like In Postgres

This section shows what real rows in the DB look like for the SAP GL example. Every field listed maps directly to a SQLAlchemy model column.

---

**Table: `file_metadata`**

One row per uploaded file. Stores the file's knowledge profile.

```json
{
  "id": "fm-uuid-001",
  "file_id": "dac47968-bseg",
  "blob_path": "az://tenant-container/dac47968_BSEG.csv",
  "container_id": "ctr-uuid-tenant-01",
  "row_count": 1250400,
  "ai_description": "SAP FI line item table. Each row is one accounting document line. Stores debit/credit amounts, cost center, profit center, GL account, customer, vendor, WBS, internal order. Does not store posting date — joins to BKPF by BUKRS+BELNR+GJAHR.",
  "good_for": ["GL line item analysis", "cost center drill-down", "vendor payment analysis"],
  "key_metrics": ["DMBTR", "WRBTR"],
  "key_dimensions": ["BUKRS", "HKONT", "KOSTL", "PRCTR", "LIFNR", "KUNNR"],
  "date_range_start": null,
  "date_range_end": null,
  "column_semantic_roles": {
    "BUKRS":  "custom:reference_key:company_code",
    "BELNR":  "custom:entity_key:document_number",
    "GJAHR":  "custom:reference_key:fiscal_year",
    "BUZEI":  "custom:attribute:line_item_number",
    "SHKZG":  "custom:attribute:debit_credit_indicator",
    "HKONT":  "custom:reference_key:gl_account",
    "DMBTR":  "custom:additive_measure:amount_local_currency",
    "WRBTR":  "custom:additive_measure:amount_document_currency",
    "KOSTL":  "custom:reference_key:cost_center",
    "PRCTR":  "custom:reference_key:profit_center",
    "AUFNR":  "custom:reference_key:internal_order",
    "PROJK":  "custom:reference_key:wbs_element",
    "LIFNR":  "custom:reference_key:vendor",
    "KUNNR":  "custom:reference_key:customer",
    "AUGBL":  "custom:attribute:clearing_document",
    "AUGDT":  "custom:date:clearing_date"
  },
  "role_source": "llm",
  "search_text": "SAP BSEG FI accounting line items GL cost center profit center vendor customer internal order WBS debit credit",
  "description_embedding": [0.0123, -0.0456, 0.0789, "... 1536 floats ..."],
  "sample_rows": [
    {"BUKRS": "1000", "BELNR": "1400000100", "GJAHR": "2026", "BUZEI": "001",
     "SHKZG": "S", "HKONT": "0051000000", "DMBTR": 12345.50, "WRBTR": 12345.50,
     "KOSTL": "US100", "PRCTR": "US-CORP", "LIFNR": "V-00123", "AUGBL": null}
  ]
}
```

Key things to note:
- `column_semantic_roles` is a JSONB object mapping source column names to `custom:<kind>:<label>` strings. This is the result of one LLM call at ingestion time.
- `description_embedding` is a 1536-float pgvector column enabling cosine-similarity metadata search.
- `date_range_start`/`date_range_end` being null for BSEG is correct because line items do not carry a posting date directly.

---

**Table: `column_key_registry`**

One row per join-key column per file. Stores the fingerprint bloom-filter for key matching.

```json
{
  "id": "ckr-uuid-001",
  "container_id": "ctr-uuid-tenant-01",
  "file_id": "dac47968-bseg",
  "blob_path": "az://tenant-container/dac47968_BSEG.csv",
  "column_name": "BELNR",
  "semantic_role": "custom:entity_key:document_number",
  "key_kind": "entity_key",
  "cardinality": 84320,
  "sample_size": 10000,
  "unique_rate": 0.843,
  "null_rate": 0.001,
  "value_fingerprints": ["a1b2c3d4", "e5f6a7b8", "c9d0e1f2", "... up to N hashes ..."]
}
```

Key things to note:
- `value_fingerprints` is a Postgres `TEXT[]` (ARRAY) column. It stores normalized hashed values of the actual key column values. This allows GIN array overlap queries to find potential join partners without scanning rows.
- When `BKPF` is ingested, its `BELNR` column also gets a `column_key_registry` row. If the fingerprint arrays overlap, a `file_relationships` row is created.

---

**Table: `file_relationships`**

One row per detected technical join candidate between two files.

```json
{
  "id": "fr-uuid-001",
  "file_a_id": "dac47968-bseg",
  "file_b_id": "501c7960-bkpf",
  "file_a_path": "az://tenant-container/dac47968_BSEG.csv",
  "file_b_path": "az://tenant-container/501c7960_BKPF.csv",
  "shared_column": "BELNR",
  "related_column": "BELNR",
  "semantic_role": "custom:entity_key:document_number",
  "role_source": "llm",
  "confidence_score": 0.84,
  "value_overlap_pct": 0.81,
  "join_type": "LEFT JOIN"
}
```

Note: this is a technical candidate. It knows the columns share fingerprints and role. It does NOT yet know whether the join is business-safe, what the grain is, or whether extra composite columns are required.

---

**Table: `semantic_entities`**

One row per file representing its business entity profile.

```json
{
  "id": "se-uuid-001",
  "container_id": "ctr-uuid-tenant-01",
  "file_id": "dac47968-bseg",
  "entity_name": "bseg",
  "primary_key": "BELNR",
  "attributes": ["BUKRS", "BUZEI", "SHKZG", "HKONT", "KOSTL", "PRCTR"],
  "metrics": ["DMBTR", "WRBTR"],
  "dimensions": ["AUFNR", "PROJK", "LIFNR", "KUNNR"],
  "grain": "document_number",
  "confidence_score": 0.87,
  "source": "ingestion",
  "status": "active"
}
```

Note: `grain` here is based on the resolved primary key role. The weakness is that the correct SAP grain for BSEG is actually `BUKRS + BELNR + GJAHR + BUZEI`, not just `BELNR`. This is one of the known gaps.

---

**Table: `semantic_relationships`**

One row per business-level relationship edge between two entities.

```json
{
  "id": "sr-uuid-001",
  "container_id": "ctr-uuid-tenant-01",
  "source_relationship_id": "fr-uuid-001",
  "file_a_id": "dac47968-bseg",
  "file_b_id": "501c7960-bkpf",
  "from_entity": "bseg",
  "to_entity": "bkpf",
  "from_column": "BELNR",
  "to_column": "BELNR",
  "relationship_type": "many_to_one",
  "join_rule": {
    "left_column": "BELNR",
    "right_column": "BELNR",
    "join_type": "LEFT JOIN"
  },
  "approval_status": "approved",
  "risk_reason": null,
  "confidence_score": 0.84,
  "status": "active"
}
```

Note: the `join_rule` currently stores a single-column rule. It does not yet store the full composite `BUKRS + BELNR + GJAHR` that SAP requires. This is a known gap described in Section 6.

---

**What Is Not In The DB (stays in Azure Blob only)**

```text
The full CSV/Excel source rows: NEVER in Postgres. Always in Azure Blob.
The full Parquet files: NEVER in Postgres. Always in Azure Blob.
Binary file contents: NEVER in Postgres. Always in Azure Blob.
```

The only rows ever in Postgres are the metadata rows above.

---

## 6. Why The Current DB Graph Is Not Yet Great Enough

The DB graph is useful infrastructure, but it is not yet the final business brain.

Observed weaknesses:

1. **Value overlap can approve nonsense.** Numeric IDs from unrelated systems can overlap in samples. This can create high-confidence-looking edges between unrelated business objects.

2. **Correct ERP joins can be under-scored.** In the AP investigation, `AP_INVOICES_ALL.VENDOR_ID -> AP_SUPPLIERS.VENDOR_ID` existed in the DB but was only a candidate because sample value overlap was low. A canonical OEBS graph knows it is the correct join.

3. **Some roles are wrong.** In the SAP logs, `BSEG.BSCHL` appeared as a date-like field even though it is a posting key. Role assignments need validation with glossary and samples.

4. **Composite keys are not strong enough.** SAP joins often require `BUKRS + BELNR + GJAHR`, not just `BELNR`.

5. **The query path does not force graph use before SQL.** The prompt says to call `extract_relations`, but the LLM can still write joins without it.

6. **`extract_relations` is scoped to visible catalog entries.** If retrieval misses a relevant dependency, the relation tool may not expose the needed path.

7. **The semantic planner falls back on complex questions.** The planner is conservative and currently not enough for 10-join ERP questions.

So the current DB graph is a foundation, not the finished beast.

---

## 7. Query Pipeline From Current Code

### 7.1 Streaming Chat Flow

Main route: `server/app/api/v1/chat_stream.py`.

```text
validate query
resolve conversation
build conversation context
resolve tenant/container scope
check response cache
acquire LLM semaphore
run_agent_query_stream
persist assistant answer
cache non-hollow answer
stream SSE events to UI
```

The endpoint enforces query length, conversation ownership, container scoping, audit logging, and LLM concurrency backpressure.

### 7.2 Retrieval

`server/app/retrieval/orchestrator.py` runs:

```text
parse_temporal
BM25 metadata search
fuzzy metadata search
vector metadata search
approved semantic graph expansion
RRF fusion
top-K metadata rows
```

One issue: temporal filtering is applied at retrieval time. For ERP questions, date filters should apply to fact data after planning. They should not hide master/reference files before dependency expansion.

### 7.3 Agent Context

`server/app/agent/graph/graph.py`:

```text
load full catalog
retrieve top files
reserve lookup/master slots
pin explicitly mentioned files
hydrate heavy fields only for shortlist
bind tools to full catalog and allowed blob paths
build system prompt
try semantic planner
fallback to LangGraph agent
```

Important current constants:

```text
_SHORTLIST_TOP_K = 7
_LOOKUP_RESERVED_SLOTS = 3
```

The full catalog remains reachable by `search_catalog`, but the LLM initially sees only a small shortlist.

### 7.4 SQL Execution

`server/app/agent/tools/sql.py` validates SQL and routes it to DuckDB or DataFusion.

The SQL tool:

- enforces blob-path allowlisting
- rejects dangerous DDL/DML through SQL safety
- auto-limits unbounded queries
- caps returned rows
- logs SQL and preview rows for the AI Pipeline UI

DuckDB reads remote `az://` paths through the Azure extension. DataFusion registers Azure object-store paths as query tables and is better for concurrent execution.

---

## 8. What The Latest `ai_pipeline (4).log` Shows

The latest SAP query asked for open GL line items in the last 90 days, filtered by cost centers/profit centers, excluding deleted vendors and fully depreciated assets, enriched with internal order, WBS, and customer name.

The catalog contained 21 SAP-related files, including:

```text
BSEG, BKPF, BSIS, BSAS, FBL3N, CSKS, CEPC, LFA1, ANLA, AUFK, PRPS, KNA1
```

The agent inspected:

```text
AUFK, KNA1, BSIS, BSEG
```

It did not correctly plan across the full dependency set.

### 8.1 Evidence The Agent Saw

The tools returned:

```text
Column 'BUDAT' not found in dac47968_BSEG.csv.
Column 'LIFNR' not found in c0825bc3_BSIS.csv.
```

BSEG had useful line item columns:

```text
BUKRS, BELNR, GJAHR, BUZEI, SHKZG, HKONT, DMBTR, WRBTR,
KOSTL, PRCTR, AUFNR, PROJK, LIFNR, KUNNR, ZUONR, AUGBL, AUGDT
```

But not `BUDAT`.

### 8.2 Bad SQL Path

Despite seeing that `BSEG.BUDAT` did not exist, the LLM generated:

```sql
WHERE B.BUDAT >= DATE '2026-02-22'
```

DuckDB failed correctly:

```text
Table "B" does not have a column named "BUDAT"
```

The agent then substituted:

```sql
WHERE B.AUGDT >= DATE '2026-02-22'
```

That is semantically wrong. `AUGDT` is clearing date, not posting date.

It also used `BSIS` as if it were vendor information even after the tool showed `BSIS` has no `LIFNR`.

It did not call `extract_relations` before multi-file SQL.

### 8.3 Why The Final Answer Was A False Negative

The answer said no matching open GL line items were found. That is not reliable because:

- the SQL used the wrong date field
- it referenced missing columns
- it did not inspect all required files
- it did not validate joins through the graph
- it did not report missing required filters
- broaden/search happened after the first final answer, not before plan validation

This was an orchestration/planning failure, not proof that the data had no results.

### 8.4 Correct Business Reasoning Shape

The planner should have decomposed the query like this:

| Requirement | Likely source candidates | Expected behavior |
|---|---|---|
| open GL line items | `FBL3N`, `BSIS`, or `BSEG + BKPF` | choose fact source by coverage |
| posting date | `FBL3N.BUDAT`, `BKPF.BUDAT`, `BSIS.BUDAT` | never use `BSEG.BUDAT` if absent |
| clearing document null/missing | `AUGBL` or open-item flag | validate null/blank semantics |
| cost center | `KOSTL`, `CSKS` | validate source and join |
| profit center starts with US | `PRCTR`, `CEPC`, maybe `CSKS` if available | inspect actual fields |
| vendor deletion flag | `LFA1` or vendor master | report limitation if absent |
| fully depreciated asset | `ANLA` or asset book/depreciation file | report limitation if absent |
| internal order | `AUFNR -> AUFK` | optional enrichment |
| WBS element | `PROJK -> PRPS` | optional enrichment |
| customer name | `KUNNR -> KNA1` | optional enrichment |

If a requested filter field is absent, the engine should say so before SQL. It should not invent or substitute.

---

## 9. OEBS CSV Knowledge Graph vs Current DB Knowledge Graph

The attached `OEBS_SQL_Knowledge_Graph_for_VectorDB.csv` is better than the current DB graph for Oracle EBS AP business relationships because it carries canonical relationship evidence from reporting SQL.

Parsed evidence from that CSV:

```text
389 nodes
991 edges
174 JOINS_TO edges
174 COLUMN_JOINS_TO edges
136 INFERRED_FK edges
242 USES_TABLE edges
```

For AP, it explicitly knows relationships such as:

```text
AP_SUPPLIERS.vendor_id = AP_INVOICES_ALL.vendor_id
AP_SUPPLIER_SITES_ALL.vendor_site_id = AP_INVOICES_ALL.vendor_site_id
AP_INVOICE_LINES_ALL.invoice_id = AP_INVOICES_ALL.invoice_id
AP_INVOICE_DISTRIBUTIONS_ALL.invoice_id = AP_INVOICES_ALL.invoice_id
GL_CODE_COMBINATIONS.code_combination_id = AP_INVOICE_DISTRIBUTIONS_ALL.dist_code_combination_id
```

That is why it corrected the AP reasoning better than the current DB graph.

But the CSV is not enough to execute tenant queries. It does not know:

- actual uploaded file IDs
- blob/parquet paths
- actual columns present in this tenant
- permissions
- data quality
- actual value coverage
- runtime date ranges

The app DB knows those things.

Verdict:

```text
OEBS CSV = stronger canonical business knowledge
App DB = stronger tenant/runtime execution knowledge
Ideal = reference KG imported into DB + mapped to physical files + validated live
```

---

## 10. What The Ideal Knowledge Graph Should Look Like

The ideal graph is layered.

### 10.1 Canonical Reference Graph

Stores source-system knowledge independent of any upload:

```text
system: SAP / Oracle EBS / custom
module: FI / CO / AP / AR / GL / FA / MM / SD
canonical_table
canonical_column
canonical_relationship
canonical_query_pattern
```

Example:

```json
{
  "system": "SAP",
  "from": "BSEG",
  "to": "BKPF",
  "conditions": [
    {"left": "BUKRS", "right": "BUKRS"},
    {"left": "BELNR", "right": "BELNR"},
    {"left": "GJAHR", "right": "GJAHR"}
  ],
  "relationship_type": "many_to_one",
  "grain_note": "line item to document header",
  "source": "curated_or_reporting_sql",
  "approval_status": "approved"
}
```

### 10.2 Physical File Mapping

Maps canonical tables to uploaded tenant files:

```text
BSEG -> dac47968_BSEG.csv
BKPF -> 501c7960_BKPF.csv
AP_INVOICES_ALL -> dba1285e_AP_INVOICES_ALL.csv
AP_SUPPLIERS -> ed7fe37e_AP_SUPPLIERS.csv
```

Evidence should include filename, columns, schema glossary, embeddings, samples, and folder/domain.

### 10.3 Column Binding

Maps canonical columns to physical columns:

```text
BKPF.BUDAT -> 501c7960_BKPF.BUDAT
BSEG.LIFNR -> dac47968_BSEG.LIFNR
KNA1.NAME1 -> 40ee5ac8_KNA1.NAME1
```

Each binding should store data type, sample values, null rate, role, glossary definition, confidence, and source evidence.

### 10.4 Join Contracts

A join contract should include:

```text
left file/table
right file/table
all join conditions
relationship type
expected cardinality
expected grain
fanout risk
coverage statistics
approval status
source evidence
validator SQL
last validated timestamp
```

Example:

```json
{
  "left_file": "dac47968_BSEG.csv",
  "right_file": "501c7960_BKPF.csv",
  "conditions": [
    {"left": "BUKRS", "op": "=", "right": "BUKRS"},
    {"left": "BELNR", "op": "=", "right": "BELNR"},
    {"left": "GJAHR", "op": "=", "right": "GJAHR"}
  ],
  "relationship_type": "many_to_one",
  "grain": "line item to document header",
  "approval_status": "approved",
  "risk_reason": null,
  "coverage_pct": 0.97,
  "source": ["canonical_sap_kg", "live_validation"]
}
```

### 10.5 Business Requirement Registry

The graph should know how business terms bind to fields:

```text
posting date -> BKPF.BUDAT or FBL3N.BUDAT
open item -> AUGBL is null/blank or open-item flag depending source
vendor deletion flag -> LFA1 deletion/block field if present
fully depreciated asset -> asset book/depreciation field if present
customer name -> KNA1.NAME1
WBS element -> PRPS.POSID or PRPS.PSPNR mapping
```

This must be evidence-backed and extensible, not prompt hardcoding.

---

## 11. Recommended Improvements

1. **Import reference KGs into DB.** Store canonical systems, tables, columns, relationships, query patterns, and provenance. The OEBS CSV should become evidence rows, not prompt text.

2. **Add canonical-to-physical mapping.** Map canonical SAP/OEBS tables to actual uploaded files and columns per tenant.

3. **Strengthen relationship approval.** Value overlap should create candidates. Approval should require compatible roles, domain/module compatibility, composite key completeness, cardinality validation, and/or canonical KG evidence.

4. **Support composite join rules.** Extend `SemanticRelationship.join_rule` to represent multiple conditions, validity windows, grain checks, and fanout limits.

5. **Build a real business planner.** The planner should handle detail queries, multi-hop joins, optional enrichments, missing-field reporting, fact-source selection, and join path search.

6. **Make relation validation mandatory before multi-file SQL.** Do not rely on the LLM to remember to call `extract_relations`.

7. **Move broaden/search before final empty answers.** A zero-row SQL should not finalize until the system validates that the plan was correct.

8. **Do not date-filter away structural files before planning.** Date filters should apply to chosen fact sources in SQL, not to master/dependency discovery too early.

9. **Add plan-level regression tests.** Test expected files, joins, columns, limitations, and rejected substitutions, not just final answer wording.

10. **Expose the selected plan in logs/UI.** Developers should see chosen fact source, selected joins, missing fields, rejected joins, confidence, and fallback reason.

---

## 12. What Production-Grade Behavior Should Look Like

For the SAP GL query, the ideal engine should produce a trace like:

```text
Intent:
  detail query over open GL line items
  date filter: posting date between 2026-02-22 and 2026-05-23
  filter: profit center starts with US
  filter: clearing document null/missing
  optional enrichments: internal order, WBS, customer name
  requested exclusions: deleted vendors, fully depreciated assets

Candidate fact sources:
  FBL3N: has BUDAT, AUGBL/open fields, KOSTL, PRCTR, AUFNR, PROJK, LIFNR, KUNNR
  BSEG + BKPF: BSEG has line details, BKPF has posting date
  BSIS: has open GL basics and BUDAT but lacks enrichment keys

Missing/limited fields:
  vendor deletion flag not found in available LFA1 columns
  fully depreciated asset flag not found in available ANLA columns

Selected plan:
  choose best fact source by coverage
  apply only verified filters
  use approved joins for enrichments
  report unsupported exclusions explicitly
```

The response should not say simply "no rows" unless the validated plan returns no rows.

---

## 13. Client-Facing Explanation

What we can tell the client developer:

```text
The application stores full uploaded data in Azure Blob Storage. During ingestion,
it extracts metadata, schema samples, semantic roles, embeddings, analytics, and
relationship evidence into Postgres. Chat queries first search that metadata and
only execute SQL against remote Parquet/CSV files when needed. The LLM receives
metadata and result previews, not full datasets.

The next evolution is to turn the semantic graph into a true business planning
layer. That means importing source-system knowledge graphs, mapping them to
tenant files, validating relationships live, and generating SQL from approved
join contracts rather than having the LLM guess joins.
```

This is the distinction:

```text
Current system: metadata retrieval + LLM tool loop
Target system: evidence graph + deterministic planner + LLM explanation
```

---

## 14. Short Verdict

The current app DB knowledge graph is a strong foundation, but it is not yet the final business brain.

The OEBS CSV-style knowledge graph is better for canonical ERP relationships, but it cannot execute tenant queries by itself.

The ideal architecture is:

```text
curated/reference KG
  -> imported into DB
  -> mapped to tenant files
  -> validated against live data
  -> used by deterministic planner
  -> executed through remote Parquet SQL
  -> explained by LLM with provenance
```

That is how the system can support real business-specific, multi-join questions without static prompt hints or query-specific hacks.

---

## 15. Developer Checklist

Near-term:

- Verify ingestion stage order and update stale comments/docs.
- Add regression tests for the SAP GL query and OEBS AP vendor spend query.
- Import the OEBS KG CSV into DB reference graph tables.
- Add canonical-to-physical file/table mapping.
- Strengthen semantic relationship approval so overlap alone cannot approve unrelated IDs.
- Add composite join rule support.
- Require planner/relation validation before multi-file SQL.
- Change zero-row handling so false negatives cannot finalize before validation.
- Log selected plan, missing fields, and rejected joins in the AI Pipeline UI.

Longer-term:

- Business intent parser.
- Requirement binder.
- Dependency closure engine.
- Join path search over approved contracts.
- SQL compiler from structured plans.
- Runtime validator for fanout, null coverage, and missing fields.
- Evaluation suite with expected files, joins, columns, limitations, and answer behavior.

---

## 16. Important Code References

Ingestion:

- `server/app/api/v1/ingest.py`
- `server/app/worker/ingest_tasks.py`
- `server/app/services/ingestion_config.py`
- `server/app/services/ingestion_stages.py`
- `server/app/services/column_role_resolver.py`
- `server/app/services/relationship_index.py`
- `server/app/services/relationship_detector.py`
- `server/app/services/semantic_layer_builder.py`
- `server/app/services/semantic_rebuild.py`

Querying:

- `server/app/api/v1/chat_stream.py`
- `server/app/core/response_cache.py`
- `server/app/retrieval/orchestrator.py`
- `server/app/retrieval/graph_expand.py`
- `server/app/agent/graph/graph.py`
- `server/app/agent/graph/graph_builder.py`
- `server/app/agent/tools/catalog.py`
- `server/app/agent/tools/column.py`
- `server/app/agent/tools/relations.py`
- `server/app/agent/tools/sql.py`
- `server/app/core/duckdb_client.py`
- `server/app/core/datafusion_client.py`

Models:

- `server/app/models/file_metadata.py`
- `server/app/models/file_relationship.py`
- `server/app/models/column_key_registry.py`
- `server/app/models/semantic_layer.py`

Evidence files:

- `server/logs/ai_pipeline (4).log`
- `server/logs/logs_need_care.log`
- `server/logs/OEBS_SQL_Knowledge_Graph_for_VectorDB.csv`

## Plan: Industry-Grade Excel Analytics

The server already has a serious analytics orchestration spine: scoped catalog loading, business intent planning, entity resolution, hybrid retrieval, semantic recovery, workflow cognition, an execution retrieval gate, SQL context building, logical SQL canonicalization, and guarded SQL execution. The next improvement should not copy RagFlow wholesale. It should borrow RagFlow's production retrieval patterns and combine them with a stronger Excel/table semantic catalog so business-relevant files reliably reach the LLM while SQL remains constrained to a deterministic execution envelope.

**Core finding**
- Current server query path: `load_catalog` -> `build_business_intent_plan` -> `BrainService.resolve` -> `resolve_entities` -> `retrieve_with_scores` -> resolver/prior pins -> lookup injection -> semantic recovery fallback -> explicit file pinning -> workflow requirements -> adaptive semantic expansion -> workflow cognition -> `narrow_catalog_for_execution` -> hydration -> `build_sql_context` -> execution strategy -> Plan IR -> scoped prompt/tools -> LangGraph agent.
- `response.txt` describes real implemented code, especially `server/app/services/execution_retrieval_gate.py`, `server/app/agent/graph/graph.py`, `server/app/core/metrics.py`, `server/app/services/ingestion_stages.py`, and `server/app/worker/ingest_tasks.py`.
- The main remaining issue is not that orchestration is absent. The issue is contract and recall: once the execution gate narrows the catalog, the normal tools are scoped to that final envelope, so a missed related file can no longer be discovered by the LLM. This is good for execution safety, but it needs a controlled discovery/expansion path.
- There is a current contract mismatch: `server/app/agent/prompts/prompt_builder.py` tells the model that `search_catalog` searches the full catalog, but `server/app/agent/graph/graph.py` binds `build_catalog_tools` to the gated `catalog`. `server/app/agent/tools/sample.py` comments also still describe full-catalog binding even though graph binding is scoped.
- RagFlow patterns worth borrowing: configurable hybrid retrieval weights and thresholds, optional reranking, metadata filters, query rewriting/keyword extraction, token-budgeted context assembly, explicit reference metadata, retrieval APIs for testing, table-aware Excel parsing, and ingestion-time keyword/question/metadata enrichment with concurrency controls and caching.

**Steps**

1. Phase 1 — Align the Current Runtime Contract
   - Update prompt/tool wording so the LLM is told the truth: `search_catalog`, `get_file_schema`, `inspect_column`, `inspect_data_format`, `extract_relations`, and `run_sql` operate inside the current execution envelope unless a separate discovery path exists.
   - Rename internal comments in `sample.py` and prompt sections from “FULL catalog” to “execution envelope” or “authorized discovery catalog” depending on the chosen implementation.
   - Add trace fields that explain why each candidate did or did not reach the LLM: retrieval channel, rank, resolver pin, workflow expansion, gate authority class, suppression reason, cap reason, and final prompt visibility.
   - Add a small test asserting that prompt text and actual tool scope agree.
   - Dependency: none. This should be the first implementation step because it removes LLM confusion before broader changes.

2. Phase 2 — Introduce Two Envelopes Instead of One
   - Keep the existing execution envelope as the only SQL-authorized scope.
   - Add a separate authorized discovery envelope that can search metadata for a larger candidate pool without allowing SQL or schema execution against every file.
   - Implement a controlled expansion mechanism. Recommended design: a request-local `ScopeState` object shared by catalog/schema/sample/relation/SQL tools. It contains `discovery_catalog`, `execution_catalog`, `allowed_file_ids`, `allowed_blob_paths`, hydrated records, and expansion audit events.
   - Add a tool such as `request_scope_expansion(file_refs, reason)` or an internal graph node that can promote a discovered candidate into execution scope only after rerunning `narrow_catalog_for_execution` plus file identity and authorization checks.
   - Update broaden-nudge logic so 0-row SQL, missing columns, or failed joins can trigger discovery and gated expansion instead of ending early.
   - Dependency: Phase 1. This directly addresses “related files are not getting till the LLM” without loosening SQL safety.

3. Phase 3 — Add RagFlow-Style Retrieval Reranking and Query Expansion
   - Add a first-pass query normalization layer similar to RagFlow’s `full_question`/`keyword_extraction`: rewrite follow-up questions, extract business terms, and expand synonyms from uploaded dictionaries and observed metadata.
   - Keep query-agnostic behavior. Do not hardcode ERP table names or tenant-specific phrases. Build terms from catalog metadata, data dictionaries, semantic roles, and user-approved glossary entries.
   - Add an optional cross-encoder/reranker stage after BM25/fuzzy/vector/OpenSearch/graph candidates and before final RRF/gate selection. Cap candidates tightly, similar to RagFlow’s rerank limit approach.
   - Make `RetrievalPolicy` deployment-configurable: candidate caps, vector/lexical weights, score floors, reranker enablement, and shortlist sizes should be runtime policy values, not scattered constants.
   - Emit metrics: retrieval recall proxy, candidates before/after rerank, reranker latency, retrieval miss count by channel, gate suppression count by authority class.
   - Dependency: can run partly parallel with Phase 2, but final behavior should be validated after two-envelope scope exists.

4. Phase 4 — Make Excel a First-Class Analytical Source
   - Move from file-level catalog entries to workbook/sheet/table/range-level analytical assets.
   - Add an Excel/table extraction service inspired by RagFlow’s `rag/app/table.py`: detect multiple sheets, merged headers, multi-row headers, hidden sheets, empty rows, named ranges, formula columns, pivots, images/comments when useful, and table boundaries inside a sheet.
   - Produce one logical analytical table per detected sheet/range, with lineage back to workbook, sheet name, cell range, source blob, and parquet output.
   - Profile each table: row count, data types, null rates, distinct counts, sample values, date ranges, candidate keys, likely dimensions, measures, units, currencies, formula lineage, and grain.
   - Store table-level metadata and index it into retrieval. Suggested new model area: `server/app/models/analytics_catalog.py` plus a migration; or extend existing `FileMetadata` only if a smaller first slice is preferred.
   - Dependency: can start after Phase 1. It is the largest product-quality unlock for Excel business analytics.

5. Phase 5 — Strengthen the Business Semantic Layer
   - Persist behavior-aware roles rather than relying only on query-time derived labels. Suggested roles: metric, dimension, entity key, reference key, date, status, amount, rate, identifier, hierarchy, scenario/version, currency, unit, formula-derived measure.
   - Keep roles extensible and evidence-backed. Store evidence from column names, data values, uploaded dictionaries, formulas, relation overlap, user corrections, and LLM resolver output.
   - Add authority/readiness metadata: raw source, cleaned source, source-like extract, curated report, staging extract, derived summary, data quality score, and freshness.
   - Add user-reviewable relationship governance for joins: approved, candidate, rejected, confidence, overlap, cardinality, risk reason.
   - Dependency: Phase 4 makes this much more valuable because roles become table/range specific instead of only file specific.

6. Phase 6 — Upgrade Planning and Execution for Business Analytics
   - Make Plan IR the central execution contract: selected tables, metric definitions, dimensions, filters, joins, grain, expected output shape, and fallback strategy.
   - Before calling the LLM, build a compact “business execution brief” from the final envelope: selected tables, why selected, metrics/dimensions available, validated joins, and risks.
   - Keep deterministic fast path for common analytics: single-table aggregation, top-N, time trend, group-by, filtered detail rows, simple approved joins.
   - Add safe Python/DataFrame analysis only for cases SQL is bad at: Excel formulas, pivot-style reshaping, statistical summaries, correlation, forecasting, and generated workbooks. Use a sandbox/guard and explicit artifacts.
   - Dependency: Phase 2 for scope expansion and Phase 4 for table-level metadata.

7. Phase 7 — Build Retrieval and Analytics Evaluation
   - Create a golden query suite with Excel/workbook fixtures and expected source tables. Measure whether the right files/tables reached retrieval, gate, prompt, SQL, and final answer.
   - Add metrics: recall@k, selected-source accuracy, SQL success rate, answer correctness, 0-row false negatives, reranker nDCG, prompt token use, latency by stage.
   - Add tests for: explicit file mention, related lookup/master inclusion, multi-table workflows, transformed/reporting allowance, unknown table recovery, stale conversation follow-up, and cross-sheet Excel queries.
   - Use these tests to tune policy values; avoid query-specific patches.
   - Dependency: can begin immediately with current fixtures, then expand after Phase 4.

8. Phase 8 — Productize Observability and UX
   - Add an admin/debug view that shows: query, candidate files, retrieval channels, rerank scores, gate decisions, prompt-visible tables, tools called, SQL, files used, and reasons for missed candidates.
   - In final answers, show sources and filters clearly, but avoid exposing internal blob paths.
   - Add user/admin correction loops: “this table is authoritative for X,” “these two columns join,” “this report is derived from these sources.” Persist corrections as evidence, not as prompt hacks.
   - Dependency: Phases 1, 2, and 7.

**Relevant files**
- `server/app/agent/graph/graph.py` — `_build_agent_context`, execution gate insertion, tool binding, prompt construction, `ScopeState` location if implemented.
- `server/app/services/execution_retrieval_gate.py` — `narrow_catalog_for_execution`, `render_execution_gate_note`, authority classification, cap behavior.
- `server/app/agent/prompts/prompt_builder.py` — tool contract text, catalog prompt budget, execution context injection.
- `server/app/agent/tools/catalog.py` — `search_catalog`, `get_file_schema`, future discovery-vs-execution behavior.
- `server/app/agent/tools/sample.py` — `inspect_data_format`, comments and scope behavior.
- `server/app/retrieval/orchestrator.py` — BM25/fuzzy/vector/OpenSearch/graph/RRF retrieval pipeline.
- `server/app/retrieval/opensearch_search.py` — OpenSearch lexical/vector retrieval and future rerank candidate support.
- `server/app/retrieval/semantic_recovery.py` — fallback recovery when standard retrieval returns empty.
- `server/app/services/business_intent_planner.py` — query behavior/entity extraction and future query rewrite signal source.
- `server/app/services/entity_resolver.py` — where business entities map into files/tables.
- `server/app/services/workflow_cognition.py` — authority/workflow primitives used by the gate.
- `server/app/services/sql_context_builder.py` — approved joins and semantic role constraints scoped to shortlist.
- `server/app/services/semantic_planner.py` — deterministic SQL fast path.
- `server/app/models/file_metadata.py` — current file-level metadata; may need table-level complement.
- `server/app/models/file_relationship.py` and `server/app/models/semantic_layer.py` — relationship graph and approved joins.
- `server/app/services/ingestion_stages.py` — staged ingestion and best insertion point for Excel/table extraction.
- `server/app/services/parquet_service.py` — parquet conversion and likely output path for per-table parquet assets.
- `server/app/services/relationship_detector.py` and `server/app/services/relationship_index.py` — relation detection/indexing.
- `server/testing/_execution_gate_check.py` — current focused gate regression check.
- `ragflow/rag/nlp/search.py` — RagFlow hybrid retrieval, rerank, thresholding, and page/window strategy reference.
- `ragflow/rag/prompts/generator.py` — query rewriting, context fitting, `kb_prompt`, citation/reference formatting.
- `ragflow/agent/tools/retrieval.py` — agent retrieval tool pattern with metadata filters, rerank, KG option, child retrieval.
- `ragflow/rag/app/table.py` — table/Excel parser reference for multi-row headers and workbook structure.
- `ragflow/rag/svr/task_executor_refactor/chunk_builder.py`, `chunk_post_processor.py`, and `embedding_service.py` — ingestion parser, metadata enrichment, and embedding batching patterns.

**Verification**
1. Contract checks
   - Add a unit/static test that fails if prompt text says full-catalog search while tools are bound to execution scope.
   - Validate `search_catalog`, `get_file_schema`, `inspect_column`, `inspect_data_format`, and `run_sql` all agree about scope.

2. Existing regressions
   - Run from `server`: `PYTHONPATH=. python3 -m testing._execution_gate_check`.
   - Run from `server`: `PYTHONPATH=. python3 -m testing._workflow_cognition_check`.
   - Run compile checks over touched modules with `python3 -m compileall`.

3. New retrieval evaluations
   - Add `testing/_retrieval_scope_check.py` for related-file recall, gated suppression, explicit expansion, lookup/master injection, and zero-row recovery.
   - Add `testing/_excel_table_catalog_check.py` for multi-sheet/multi-table Excel extraction and logical table identity.
   - Add a small fixture set of Excel files with known expected sources and expected joins.

4. Manual validation
   - Upload multi-sheet Excel files with lookup/master/detail/report sheets.
   - Ask operational, reporting, follow-up, and cross-sheet business questions.
   - Confirm trace shows: candidates found, why selected, why suppressed, what reached prompt, what SQL executed, and which tables were used.

**Decisions**
- Do not loosen SQL to full catalog. Full-catalog execution is the old failure mode.
- Do not hardcode source-system table names or query-specific terms. Use metadata, glossary, dictionaries, and role evidence.
- Do not copy RagFlow directly. Borrow patterns: hybrid retrieval, rerank, metadata filters, prompt fitting, Excel parser ideas, and retrieval APIs.
- Keep the modular monolith until metrics prove a real scaling boundary. The current bottleneck is evidence, retrieval, and Excel semantics, not necessarily service decomposition.
- Create the requested human-readable `.txt` artifact during implementation handoff, for example `BUSINESS_ANALYTICS_PLATFORM_BLUEPRINT.txt` at the workspace root, because Plan mode should only persist the plan here.

**Further Considerations**
1. Recommended first implementation slice: Phase 1 + a minimal Phase 2 discovery envelope. This gives immediate relief for “related files do not reach the LLM” without changing ingestion.
2. Recommended second implementation slice: Excel table extraction and table-level logical identities. This is the biggest strategic improvement for business analytics over Excel.
3. Recommended third implementation slice: reranker/evaluation harness. This lets improvements become measurable instead of prompt-driven.
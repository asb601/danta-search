# Agentic Graph RAG for PDFs — Target Architecture & Roadmap

**Date:** 2026-06-03 (v3: 2026-06-04)
**Status:** DRAFT v3 (reviewer + SME + architect board hardened; pending user approval)
**Module:** `server/pdf_chat/`
**Scope (locked):** Full target architecture · Full GraphRAG · Cross-domain = PDFs **+** structured/CSV semantic layer · **Cross-enterprise comprehension ("superhuman memory")**
**Boards run:** reviewer board (architect, data-science, business-analyst, static-code-sentinel) + AI-architect + knowledge-representation SME + RAGFlow extraction evaluator.

---

## 0. Guiding Principles

1. **Intelligence lives in ingestion, not query time.** GraphRAG + ontology + glossary are computed once at ingest and amortized. The runtime only retrieves, plans, traverses, synthesizes.
2. **A confident wrong answer is worse than an honest failure.** Every edge, report, join, glossary entry, and "no data" claim is grounded in cited evidence or refused.
3. **Information type changes, intent does not** (the cross-enterprise law). A small **fixed INTENT layer (code)** sits over a **per-tenant learned SEMANTIC layer (data)**. No code path names any customer's domain. This mirrors the structured side exactly: `semantic_roles.py` defines fixed `RoleKind` behavior-kinds with an empty `_BASE_ROLE_SPECS`, and the LLM mints concrete roles per tenant as `custom:<kind>:<slug>` with confidence + value evidence (`column_role_resolver.py`, `relationship_index.py`).

---

## 1. Current State

`server/pdf_chat/` is a ~5,700 LoC enterprise **scaffold** that does not yet run end-to-end. Page
extraction is a `NotImplementedError` (`ingestion/tasks.py:224`); reranker/cache/synthesis/audit are
framework-only; the Neo4j "graph" is a decorative vector store (`neo4j_writer.py:88-98` writes only
Document/Page/Chunk; `state.entity` is never populated). The owner's operational design lives in
`project_memory/enterprise-pdf-4.md` — it is the substrate (control plane, manifests, state machine,
unified element schema) this upgrade builds on.

### Reconciliation with `enterprise-pdf-4.md`
That doc is the **how/library reference**; this spec is the **what-must-be-true (invariants)**. Two corrections override it:
- **No gpt-4o** — its Stage 6/9 `gpt-4o` / `gpt-4o vision` / Claude 3.5 usages are superseded by the tiered model router in §5 (bulk = gpt-4o-mini; strong tier configurable, gated, capped).
- **No magic literals** — its `<10 chars = scanned` and `image_entropy > 0.85` are exactly the literals we forbid; keep the *signals*, move thresholds to `pdf_graphrag_tunables`.
Adopt as the Phase-1 implementation reference: manifest DDL, document state machine, crash-recovery reconciler, unified element schema (+ mandatory `bbox` and `extract_confidence`).

---

## 2. Target Architecture (layers)

```
                          INGESTION (one-time, amortized)
  Upload → Preflight → Dedup → Page extract (PyMuPDF + ONNX layout/table + OCR; bbox + confidence)
        → Chunk → Batch-embed → KG construct (grounded entities/relations/communities)
        → ONTOLOGY + GLOSSARY build (per-tenant comprehension artifact) → Neo4j + Postgres
                                          │  value-evidenced bridge (NOT name match)
                                          ▼
                       [ existing CSV/DataFusion semantic layer + relationship_index ]
  ────────────────────────────────────────────────────────────────────────
                          QUERY RUNTIME (per request) — STABLE INTENT LAYER
  Query → Planner/Router (local | global | cross_domain | definitional) → Agent tool-loop:
        vector_search · graph_traverse · community_report_lookup · structured_query · get_entity_neighbors · glossary_lookup
   → RRF + adaptive rerank + ACL → assemble (token-budgeted) → grounded synthesis (+citations/bbox + provenance labels)
   → negative-claim + conflict gate → cache → audit
  ────────────────────────────────────────────────────────────────────────
                          ONBOARDING SURFACE (read-only projections)
  Topic map (community reports) · Entity browse · Glossary ("what does X mean here") · Doc taxonomy
```

### Layer 1 — Ingestion
**1a. Make it run:** PyMuPDF (digital) + OCR (Azure Document Intelligence) for scanned; **vendor RAGFlow DeepDoc micro-components** (ONNX multi-column/reading-order, table-structure recognition for spanning cells, table-rotation) as optional enhancers gated on hard pages — NOT the whole DeepDoc monolith. bbox + `extract_confidence` retained and propagated. Data-driven scanned/complex routing (per-page text-coverage ratio, tunable, logged). Batch embeddings. Wire reranker/cache/synthesis/audit; finalization task + state machine.

**1b. Knowledge graph (grounded, full GraphRAG):** entity + relation extraction per chunk via the model router (§5); prompt-cached; adaptive capped gleaning; idempotent on `chunk_fingerprint + prompt/model version`; **open-vocabulary entity types** (LLM-proposed + confidence + span; no closed type list). **Grounding gate**: every edge persists `src_chunk` + verbatim span + confidence; reject edges absent from the cited span. Entity resolution: embedding + type agreement + co-occurrence, **unmerged-by-default for ambiguous pairs**, bands derived per-container (not literals). Neo4j schema with `tenant_id` on every node/edge; per-hop tenant isolation. Leiden communities (networkx) + **cited** community reports; PageRank over grounded edges.

### Layer 2 — Cross-domain bridge (PDF ↔ CSV) ⚠️ highest risk
Value-evidenced only: PDF `Entity` → `SemanticEntity` via `relationship_index.fingerprint_value` to a real master key; **no name/embedding join**; sub-threshold ⇒ refuse. `pdf_entity_bridge` table. `structured_query` delegates to `run_agent_query` (passes `container_id`/`allowed_domains`/`user_id`, runs **strictly sequentially**, inherits CSV feasibility + negative-claim gates). Grain alignment + numeric reconciliation.

### Layer 3 — Agentic runtime (the stable INTENT layer)
LangGraph agent. Planner emits typed intent — `local` / `global` / `cross_domain` / **`definitional`** — with named tunable confidence thresholds + typed `fallback_reason` logged; simple/cached queries bypass. Entity-linking step populates `state.entity`. Tools (capped loop, monotonic-progress guard, mirror `MAX_TOOL_CALLS`): `vector_search`, `graph_traverse` (per-hop tenant), `community_report_lookup` (cited), `structured_query` (Phase 4 seam), `get_entity_neighbors`, **`glossary_lookup`**. Query decomposition with output-completeness check. Grounded synthesis with citations + bbox + **provenance labels** (§4). Negative-claim + conflict gate.

### Layer 4 — Comprehension layer (the "superhuman memory" — NEW)
Converts grounded-QA into *company comprehension* so a domain-naive engineer is productive. All learned per tenant, all grounded.
- **Materialized Tenant Ontology artifact** (Postgres-backed, versioned — mirrors `semantic_layer_builder`): registries for entities, relationships, **document taxonomy** (learned doc-classes, open-vocab), **temporal coverage** (per entity/topic date ranges + density), and **key metrics**. Not just Neo4j traversal substrate — a browsable, queryable object.
- **Corpus-learned glossary**: acronym/codename/jargon mining at ingest from three grounded signals — explicit definitions (LLM-confirmed with span), distributional anomaly (corpus-internal vs background frequency → `definition: inferred`), and co-reference variants. Powers `glossary_lookup` ("what does X mean here", with *stated* vs *inferred* provenance) and transparent, tenant-scoped query expansion.
- **Onboarding surface**: read-only projections — topic map (community reports as the company's table-of-contents), entity browse, glossary, doc taxonomy. Built on Phase-2 data; the new work is the projection, not a new brain.

### Layer 5 — Tiered model router & token reduction (cross-cutting)
**Model router (all model ids + thresholds are `pdf_graphrag_tunables`, no literals):**
- **Bulk** (extraction, synthesis, glossary, community reports): `gpt-4o-mini`.
- **Strong tier** (escalation only): data-driven gate — escalate a chunk/page/query when extraction confidence is low, content is figure/formula/diagram-heavy, the query is cross-domain or `definitional`, AND a **hard per-tenant escalation budget** is not exhausted. Default strong model = **Claude Sonnet 4.6**; switchable to Azure GPT-4-class (single-provider). **Opus is query-time-only, off by default** (never bulk ingestion — ~90× cost).
- **Embeddings**: `text-embedding-3-small` (1536), configurable.

**Token reduction (Full GraphRAG is the costly path, so savings are architectural):** prompt caching on extraction + synthesis prompts (biggest lever); batch embeddings; idempotent extraction (chunk_fingerprint); amortized community reports + ontology + glossary (built once); query-embedding cache; planner bypass + versioned response cache; adaptive rerank; context token budget; **strict, capped escalation** (the escalation fraction is the dominant cost driver — keep ≤~3–5% of pages).

---

## 3. Correctness & Generality Invariants (hold every phase)

1. **Grounding** — no edge / report / glossary entry / synthesized claim without a cited source span; else reject/refuse.
2. **Honest absence** — "no data / not found" requires coverage proof + diagnosis (retrieval-empty ≠ absent). Ports `negative_claim_gate`.
3. **Tenant isolation on every Neo4j hop**, every tool, every cross-domain delegation.
4. **No magic literals** — every threshold (resolution bands, gleaning passes, planner-bypass, rerank-skip, escalation gate, budget caps, token budget, parser routing) resolves from `pdf_graphrag_tunables`, defaulted/overridable, logged with its score. Entity/relationship/domain **types are never hardcoded dictionaries**.
5. **Cross-domain joins are value-evidenced master keys only** — never name/embedding equality.
6. **Stable intent over learned semantics** — the INTENT layer (tools, planner, kinds, invariants) is fixed code identical for every tenant; all domain meaning (entities, types, edges, communities, glossary, ontology) is learned per tenant as data. Adding a new industry is learned data, never a code `if`.
7. **Linkage is discovered; siloing is valid** — no code path may require an inter-document/entity edge to exist. A disconnected (siloed) graph is a correct, queryable state. Relationship-absence answers go through the negative-claim gate; relationships are three-state: **asserted / not-stated / conflicting** — conflicts are surfaced with provenance + recency, never silently resolved.

---

## 4. Faithfulness for a non-expert (a newcomer can't catch a subtle error)

- **`definitional` intent** has a higher bar: a single authoritative verbatim span, no paraphrase-only synthesis.
- **Provenance labels instead of raw confidence numbers**: *stated in docs* / *inferred from usage* / *conflicting sources* / *not found*. (Rides on data already persisted.)
- **Staleness annotation** from temporal coverage ("most recent mention is 2025-09; may be outdated").
- **Org/ownership/process claims** emitted only if a grounded edge backs them; else "the documents don't state…".

---

## 5. Phased Roadmap (the TODO) — each phase ends at a board gate

- **Phase 0 — Token guards + model router:** `pdf_chat/tunables.py` (config source + score-logging) · batch embeddings · prompt caching · query-embedding cache · context budget · response cache · **model router scaffold** (bulk/strong selection, escalation gate, per-tenant budget — all tunables).
- **Phase 1 — Make it run + eval:** wire page extraction (PyMuPDF + OCR + tables + bbox + confidence) · **vendor DeepDoc micro-components** (multi-column, table-structure, rotation) gated on hard pages · data-driven routing · reranker · cache · synthesis · audit · finalization/state-machine (from enterprise-pdf-4) · **gold-question eval baseline**.
- **Phase 2 — Knowledge graph (grounded):** extraction (router-driven, cached, idempotent, open-vocab) · grounding gate · entity resolution · Neo4j schema (per-node tenant) · Leiden communities · cited reports · PageRank. *Exit: per-hop tenant isolation verified + faithfulness eval (edge precision, merge P/R, report groundedness) passes, **including on a held-out tenant the system has never seen** (the only honest cross-enterprise test).*
- **Phase 3 — Agentic runtime:** LangGraph planner (incl. `definitional`) · entity linking · tool loop w/ caps + monotonic-progress · decomposition w/ output-completeness · negative-claim + conflict gate · provenance labels + staleness · adaptive rerank · citation-density floor.
- **Phase 4 — Cross-domain bridge:** `pdf_entity_bridge` (value-evidenced) · sequential `structured_query` · grain alignment + numeric reconciliation · cross-domain cache invalidation.
- **Phase 5 — Comprehension layer:** materialized Tenant Ontology artifact · corpus-learned glossary (`glossary_lookup` + query expansion) · doc taxonomy + temporal coverage + metrics registries · onboarding surface (topic map / entity browse / glossary). *Exit: a domain-naive engineer can browse the topic map, ask "what does X mean here", and get cited company-specific answers.*
- **Phase 6 — Hardening:** per-tenant cost/observability (incl. escalation budget tracking) · cascading delete · rate-limit backoff · eval expansion (graph/global/cross-domain/negative/definitional) + CI gate.

---

## 6. RAGFlow disposition
Adopt (vendor as optional ONNX enhancers): multi-column/reading-order, table-structure recognition (spanning cells), table-rotation. **Reject** wholesale DeepDoc (monolith, Python 3.13+, ~100 deps) and its type-specific chunkers (hardcoded heuristics — violates invariant 4). License is Apache-2.0 (clean).

## 7. Non-Goals (YAGNI)
RAGFlow Canvas DSL · second query brain · new datastore · Opus at ingestion scale · fine-tuning · K8s/gRPC/Kafka · hardcoded entity/relationship/domain dictionaries · name-equality cross-domain joins.

## 8. Cost model (client-facing dial)
One-time ingestion ≈ `N_docs × pages/doc × ($0.002 mini-baseline) × (1 + escalation% × (strong_multiplier − 1))`. ~$40k per 1M docs at mini-only; escalation fraction is the dominant lever (≤3–5% with Sonnet ≈ ~1.6× total; Opus or 10%+ balloons it). Quote clients by setting the per-tenant escalation %. Embeddings ~$0.00004/page (3-small). Chat cost is per-query, far below ingestion volume.

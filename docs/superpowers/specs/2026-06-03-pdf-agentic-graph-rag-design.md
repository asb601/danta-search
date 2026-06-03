# Agentic Graph RAG for PDFs — Target Architecture & Roadmap

**Date:** 2026-06-03
**Status:** DRAFT v2 (reviewer-board hardened; pending user approval)
**Module:** `server/pdf_chat/`
**Scope decisions (locked):** Full target architecture · Full GraphRAG · Cross-domain = PDFs **+** structured/CSV semantic layer
**Review board:** AI-architect, data-science, business-analyst, static-code-sentinel — all REWORK→incorporated below (delivery-manager synthesis in §7).

---

## 0. Guiding Principle

Mirror the main system's law: **intelligence lives in ingestion, not at query time.**
GraphRAG construction (entity/relation extraction, communities, reports) is an *ingestion*
concern, computed once and amortized. The query runtime only **retrieves, plans, traverses,
and synthesizes** — it never discovers schema, entities, or business meaning dynamically.

This is also how we reconcile **Full GraphRAG** with **lower token usage**: the expensive LLM
work happens once per chunk at ingest (idempotent, prompt-cached), never per query.

**Second law (from the board):** a confident wrong answer is worse than an honest failure.
Every edge, community report, cross-domain join, and "no data" claim must be **grounded in
evidence** or refused. Nothing in this system is allowed to assert what it cannot cite.

---

## 1. Current State (what we're building on)

`server/pdf_chat/` is a ~5,700 LoC enterprise **scaffold** that does not yet run end-to-end.

| Status | Components |
|---|---|
| ✅ Done (pure, tested) | preflight, SHA-256 dedup, retry/DLQ, RRF fusion, ACL filter, state machine, chunker, manifests, API skeleton |
| ⚠️ Critical stub | **page extraction** (`ingestion/tasks.py:224` → `NotImplementedError`) — no PDF is parsed today |
| ⚠️ Framework only | reranker, Redis cache, LLM synthesis call, audit writes |
| ❌ Missing | entity extraction, relationships, communities (graph is a decorative vector store), agentic behavior, cross-domain |

The graph leg is **dead**: `neo4j_writer.py:88-98` only creates `(:Document)-[:CONTAINS]->(:Page)
-[:CONTAINS]->(:Chunk)` — no `:Entity` nodes or `RELATED_TO` edges. The searcher's graph Cypher
(`neo4j_searcher.py:51-59`) queries a shape nothing writes, and `state.entity` is never populated.

---

## 2. Target Architecture (4 layers)

```
                          INGESTION (one-time, amortized)
  Upload → Preflight → Dedup → Page extract (PyMuPDF/OCR/tables + bbox + extract-confidence)
        → Chunk → Batch-embed → KG construct (GROUNDED entities/relations/communities) → Neo4j
                                          │
                                          ▼ value-evidenced bridge (NOT name match)
                       [ existing CSV/DataFusion semantic layer + relationship_index ]
                                          │
  ────────────────────────────────────────────────────────────────────────
                          QUERY RUNTIME (per request)
  Query → Planner/Router → Agent tool-loop ──┬─ vector_search (Neo4j HNSW)
              │ (typed intent + confidence;    ├─ graph_traverse (entity multi-hop, per-hop tenant)
              │  bypass for simple/cached)      ├─ community_report_lookup (global; cited)
              │  HARD caps: tool-calls,         ├─ structured_query (CSV layer; SEQUENTIAL)
              ▼  depth, monotonic-progress      └─ get_entity_neighbors
   RRF + rerank + ACL → assemble (token-budgeted) → grounded synthesis (+citations/bbox)
                                          → NEGATIVE-CLAIM GATE + verify → cache → audit
```

### Layer 1 — Ingestion (make it real + add the brain)

**1a. Make it work (replaces the stubs):**
- Page extraction: PyMuPDF for digital; OCR (Azure Document Intelligence or Tesseract) for scanned;
  table extraction; **bbox retained** on every element for click-to-highlight citations.
- **Digital-vs-scanned routing is data-driven**: route on measured per-page extractable-text
  coverage ratio (configurable threshold, logged), per-page not per-document — never `if text=="" `.
- **Layout classification** uses a learned model / OCR-native region typing with confidence — no
  font-size/whitespace rule literals; any rule fallback is config-driven + logged.
- **Batch embeddings** (config-sized batches) instead of per-chunk.
- **Extraction confidence propagates** to chunks → answers (low-confidence OCR/table cells flagged, not asserted).
- Wire the framework-only backends: reranker, Redis cache, LLM synthesis, audit writes; finalization task ("document ready" only after all pages settle).

**1b. Knowledge-graph construction (Full GraphRAG, grounded + cost-controlled):**
- **Entity + relation extraction** per chunk via `gpt-4o-mini` (no gpt-4o), with:
  - **Prompt caching** on the extraction system prompt; **capped gleaning** (adaptive — stop when
    marginal new-entity yield < configurable delta; pass count is config, not a literal `2`).
  - **Idempotent** on `chunk_fingerprint + prompt/model version` — never re-extract unchanged content.
  - **Open-vocabulary entity types** (LLM-proposed + confidence, persisted with supporting span).
    The prompt does **not** enumerate a closed type set. Phase-2 gate flags if >X% of entities
    collapse to a handful of types (prompt-steering smell).
- **GROUNDING GATE (blocking, from data-science B1 / business B4):** every `RELATED_TO` edge MUST
  persist its source `chunk_id` + verbatim span + confidence. An edge whose subject/object/predicate
  does not appear in the cited span is **rejected before write**. Mirrors `edge_provenance` and the
  value-overlap gate in the main system's `relationship_detector.py`.
- **Entity resolution:** merge across chunks/docs via embedding similarity **+ type agreement +
  co-occurrence evidence**, with an **unmerged-by-default** stance for ambiguous pairs. Bands
  (auto-merge / tie-break / hold) are **per-container, derived from the score distribution**, not
  `.py` constants; every merge decision + evidence is persisted and auditable. No transitive
  auto-merge without a confidence floor.
- **Neo4j graph model:** `(:Entity)-[:RELATED_TO {desc,weight,confidence,evidence_count,src_chunk}]->(:Entity)`,
  `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)`, `(:Entity)-[:IN_COMMUNITY]->(:Community)`.
  **Every node carries `tenant_id`.** The searcher Cypher is rewritten to this schema (resolves architect B2).
- **Community detection:** Leiden (resolution / min-size are config; community count + size distribution logged).
- **Community reports:** `gpt-4o-mini` summarizes each community once at ingest. **Every report claim
  carries drill-down citations to underlying chunks/bbox.** A report whose findings can't be traced to
  ≥N grounded edges is suppressed. A report may **route** a query but is **never the evidence of record** —
  synthesis cites source pages, not the report.
- **PageRank** computed only over grounded edges, weighted by edge confidence.

### Layer 2 — Value-evidenced bridge to structured data (cross-domain) ⚠️ highest-risk layer

All four reviewers flagged name/embedding-based linking as the top risk (reproduces the documented
`erp_flat` master-key failure). Hardened contract:

- **No name-equality or embedding-cosine join.** A PDF entity links to the CSV side only when its
  literal value reconciles against the existing **`relationship_index` value-overlap / fingerprint
  registry** — i.e. it resolves to a real reconciling **master key** (Vendor_ID/Region/Plant), not a
  display name. Reuse `relationship_index.fingerprint_value`.
- **Explicit confidence-gated bridge table** (resolves open-Q #4): `pdf_entity_bridge` maps PDF
  `Entity` → `SemanticEntity` with value-overlap %, confidence, and evidence. **Sub-threshold ⇒
  refuse + say so**, never silently pick the top match.
- **Grain alignment (business B3):** cross-domain synthesis must reconcile the grain of the PDF-derived
  fact (e.g. contract = agreement grain) with the CSV aggregate (invoice/line grain) — align period &
  unit before comparing. Where possible, attach a **deterministic numeric reconciliation check**
  (e.g. contract rate × volume vs invoiced total within tolerance) as automatic proof the join was correct.

### Layer 3 — Agentic query runtime

LangGraph agent (no RAGFlow Canvas — repo standardizes on LangGraph):

- **Planner/Router** (`gpt-4o-mini`): typed intent — *local* / *global* / *cross-domain* — with
  **named, configurable confidence thresholds and typed fallback reasons logged**, mirroring
  `semantic_planner.py` (`planner_fast_path_confidence`, typed `fallback_reason`). High-confidence
  simple/cached queries **bypass** the loop (threshold is config + logged, not a literal).
- **Entity linking step (resolves architect B2):** a defined runtime step populates `state.entity`
  from the query before `graph_traverse` is reachable.
- **Tools** (capped loop): `vector_search`, `graph_traverse`, `community_report_lookup`,
  `structured_query`, `get_entity_neighbors`.
  - **`graph_traverse` / `get_entity_neighbors` enforce tenant isolation on EVERY path hop**
    (`ALL(n IN nodes(path) WHERE n.tenant_id=$tenant_id)`) — not just endpoints (architect B1; security, not hardening).
  - **`structured_query` runs strictly sequentially**, never concurrent with another DB-touching tool
    (the async session is not concurrency-safe), and passes `container_id` + `allowed_domains` +
    `user_id` into `run_agent_query` so it inherits the CSV side's feasibility + negative-claim gates (architect B3, business I2).
- **Loop/cost ceilings (architect I3):** hard total tool-call cap (mirror `MAX_TOOL_CALLS`),
  per-tool-type cap, decomposition depth ≤ N, and a **monotonic-progress guard** (abort if a round
  adds no new accessible chunks). All caps are config + log what was dropped/truncated.
- **Query decomposition** sufficiency check verifies **all requested output components** are present
  (not just that *an* answer formed) — prevents dropping the 3rd part of a multi-part ask.
- RRF + ACL reused; rerank backend wired; **adaptive rerank** skip threshold is config + logged.
- **Grounded synthesis** (`gpt-4o-mini`) with citations incl. bbox. **Citation-density floor:** refuse
  to emit a claim with zero supporting chunk/bbox citation.

### Layer 4 — Token-reduction (cross-cutting)

Savings from architecture, not feature cuts: (1) prompt caching, (2) batch embeddings,
(3) query-embedding cache, (4) idempotent extraction, (5) amortized community reports,
(6) planner bypass + response cache, (7) adaptive rerank + gpt-4o-mini everywhere,
(8) context token budget. **Cache invalidation (architect I5):** the response-cache key must include
the CSV semantic-layer version whenever a `structured_query` tool was used, plus graph-extraction
version — otherwise cross-domain answers go stale silently.

---

## 3. Correctness & Safety Invariants (must hold every phase)

1. **Grounding:** no edge / community-report claim / synthesized claim without a citation to source span. Ungrounded ⇒ rejected or refused.
2. **Honest absence (business B1, architect I4):** a "no data / not found" claim requires proof of
   coverage (the relevant pages/sections were actually in-context, via bbox/page coverage) + diagnosis.
   Retrieval-empty ≠ absent. Port `negative_claim_gate` semantics to the PDF/graph world.
3. **Tenant isolation on every Neo4j hop**, every tool, every cross-domain delegation.
4. **No magic literals (static-code-sentinel):** every threshold (resolution bands, gleaning passes,
   community resolution, planner-bypass, rerank-skip, token budget, max_rounds, depth, parser-routing)
   resolves from a single per-container tunables source (config or `pdf_graphrag_tunables`), defaulted
   but overridable, and every gate/cap/skip/merge decision is **logged with its score**. No
   score-comparison literal in a `.py` file. Entity/relationship/domain **types are never hardcoded dictionaries**.
5. **Cross-domain joins are value-evidenced master keys only** — never name/embedding equality.

---

## 4. Approach decision (agentic runtime style)

| Option | Trade-off |
|---|---|
| **LangGraph agent (recommended)** | Repo already uses LangGraph; reuse patterns + the existing gate wiring; no new runtime dependency. |
| RAGFlow Canvas DAG | Second orchestration brain — violates "one query brain". Borrow concepts only. |
| Status-quo pipeline | No planning/graph/cross-domain. Rejected. |

---

## 5. Phased Roadmap (the TODO) — each phase ends at a reviewer-board gate

- **Phase 0 — Token guards (cheap):** batch embeddings · prompt caching · query-embedding cache ·
  context token budget · response-cache wiring · **`pdf_graphrag_tunables` config source + score-logging harness**.
- **Phase 1 — Make it run end-to-end + eval:** wire page extraction (PyMuPDF + OCR + tables + bbox +
  extract-confidence) · data-driven parser routing & layout classification · reranker · Redis cache ·
  LLM synthesis · audit · finalization · **gold-question eval set live (moved up from Phase 5)**.
  *Exit: a PDF ingests and answers with citations; eval baseline recorded.*
- **Phase 2 — Knowledge graph (grounded):** extraction (cached, idempotent, open-vocab types) ·
  **grounding gate** · entity resolution (derived bands, audited) · Neo4j schema w/ per-node tenant ·
  Leiden communities · **cited** community reports · PageRank over grounded edges.
  *Exit gate: per-hop tenant isolation verified + faithfulness eval (edge precision vs source, merge
  precision/recall, report groundedness) passes — not merely "graph is non-empty".*
- **Phase 3 — Agentic runtime:** LangGraph planner/router (typed reasons) · entity linking · tool loop
  w/ hard caps + monotonic-progress guard · decomposition w/ output-completeness check · negative-claim
  gate port · adaptive rerank · citation-density floor.
- **Phase 4 — Cross-domain bridge:** *entry criteria:* `structured_query` session-safety contract +
  value-overlap bridge design signed off. Then `pdf_entity_bridge` (value-evidenced master keys) ·
  grain alignment + numeric reconciliation · cross-domain cache invalidation.
  *Exit: one question spans a contract PDF and the vendor CSV, with a correct value-evidenced join.*
- **Phase 5 — Hardening:** per-tenant cost/observability · cascading delete · rate-limit backoff ·
  eval expansion.

---

## 6. Non-Goals (YAGNI)

No RAGFlow Canvas import · no second query brain · no new datastore · no gpt-4o · no fine-tuning ·
no Kubernetes/gRPC/Kafka · no hardcoded entity/relationship/domain dictionaries · no name-equality cross-domain joins.

---

## 7. Reviewer-board disposition (delivery-manager)

| Reviewer | Verdict | Top blocker | Where addressed |
|---|---|---|---|
| AI-architect | REWORK runtime | per-hop tenant isolation; planner gating; structured_query concurrency | §2 L3, §3.2/3.3 |
| Data-science | REWORK | ungrounded edges/reports; unsound canonical-ID linking | §2 L1b grounding gate, §2 L2 |
| Business-analyst | REWORK | weak "no data" proof; ungated cross-domain join; eval too late | §3.2, §2 L2, §5 Phase 1 |
| Static-code-sentinel | REWORK | ~10 magic thresholds; layout/routing rules | §3.4 invariant |

**Convergent finding:** all four independently flagged the **cross-domain PDF→CSV join** as the
single highest risk. It is now gated to value-evidenced master keys via the existing
`relationship_index`, with a confidence-gated bridge table and refusal on sub-threshold.

---

## 8. Resolved open questions

1. **Leiden:** start with networkx in-worker (no GDS license/infra); revisit GDS at scale. Config-driven.
2. **OCR:** Azure Document Intelligence (managed, better tables) primary; Tesseract fallback. Routing data-driven.
3. **Entity-resolution thresholds:** derived per-container from score distribution, unmerged-by-default for ambiguous. Not literals.
4. **Cross-domain mapping:** **bridge table** (`pdf_entity_bridge`) with value-overlap evidence — DECIDED (not shared-canonical-ID-by-name).
5. **Prompt caching:** verify Azure OpenAI deployment support in Phase 0 before relying on it for token math.
```

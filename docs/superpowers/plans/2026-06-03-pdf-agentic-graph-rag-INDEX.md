# PDF Agentic Graph RAG — Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Execute phases **strictly in order**; each ends at a reviewer-board gate. **This index is authoritative — where a phase file disagrees with the Cross-Phase Contract below, this index wins.**

**Spec:** `docs/superpowers/specs/2026-06-03-pdf-agentic-graph-rag-design.md`
**Goal:** Upgrade `server/pdf_chat/` from a non-running scaffold into an agentic, grounded Graph-RAG system with value-evidenced cross-domain (PDF + CSV) reasoning, while cutting per-document token cost.

---

## Phase plans (execute in this order)

| # | Plan file | Tasks | Exit gate |
|---|---|---|---|
| 0+1 | `2026-06-03-pdf-phase0-1-foundations.md` | 18 | A PDF ingests end-to-end and answers with citations; token guards live; eval baseline recorded |
| 2 | `2026-06-03-pdf-phase2-knowledge-graph.md` | 9 | Per-hop tenant isolation verified + faithfulness eval (edge precision, merge P/R, report groundedness) passes |
| 3 | `2026-06-03-pdf-phase3-agentic-runtime.md` | 12 | Complex multi-hop answered; simple/cached bypass; negative-claim gate blocks unproven absence |
| 4 | `2026-06-03-pdf-phase4-cross-domain-bridge.md` | 10 | One question spans a contract PDF + vendor CSV via a correct **value-evidenced** join |
| 5 | `2026-06-03-pdf-phase5-hardening.md` | 9 | Per-tenant cost/observability, safe cascading delete, backoff, expanded CI eval gate |

---

## Canonical Cross-Phase Contract (single source of truth)

**C0 — Test command.** Run tests with `uv run --with pytest --with pytest-asyncio pytest <path> -q` from `server/` (pytest is NOT a project dependency). Pure tests run with no infra; infra tests are gated behind markers (`@pytest.mark.infra` / `neo4j` / `llm`) and excluded by default. Verified baseline: `pdf_chat/testing/test_ingestion.py` → 45 passed.

**C1 — Tunables & logging (Phase 0 owns; all phases consume).** `server/pdf_chat/tunables.py` exposes `get_tunable(container_id, key, default)` (config + `pdf_graphrag_tunables` table) and `log_gate_decision(name, *, score, threshold, outcome, **ctx)`. **No score-comparison literal may appear in any `.py` file**; every gate/skip/cap/merge/route threshold resolves via `get_tunable` and logs via `log_gate_decision`. Entity/relationship/domain **types are never hardcoded dictionaries**.

**C2 — Neo4j searcher methods are owned by Phase 2.** `Neo4jSearcher` in `retrieval/neo4j_searcher.py` is rewritten in **Phase 2** to the Entity schema and defines `graph_traversal(...)`, `entity_neighbors(...)`, and `community_report_lookup(...)`, each enforcing per-hop tenant isolation (`ALL(n IN nodes(path) WHERE n.tenant_id=$tenant_id)`). **Phase 3 CONSUMES these — it does not redefine them.** If a Phase-3 task adds a searcher method that Phase 2 already created, treat it as a no-op/skip. Tool name `get_entity_neighbors` wraps searcher `entity_neighbors`.

**C3 — Tool interface & registry (Phase 3 owns).** `agent/tools.py` defines `Tool` Protocol (`name: str; async def run(self, state, deps, **kwargs) -> list[dict]`), `TOOL_REGISTRY: dict[str, Tool]`, and `register_tool(tool)`. **Every tool — including Phase 4's `structured_query` — must implement this `Tool` Protocol** and register via `register_tool`. Phase 4's underlying `structured_query(deps, query) -> dict` is correct, but the object placed in `TOOL_REGISTRY` must be a `Tool`, not a raw LangChain `StructuredTool` (wrap it). Phase 3 does **not** implement `structured_query`.

**C4 — Agent entry point (Phase 3 exposes).** `agent/graph.py` exposes `run_pdf_query(query, *, tenant_id, container_id, ...)` returning a result with `.answer` and `.citations`. Phase 5's eval harness and the API route depend on this exact signature.

**C5 — Cross-domain bridge is value-evidenced only (Phase 4).** PDF `Entity` → `SemanticEntity` links ONLY via literal value reconciliation through `relationship_index.fingerprint_value` to a real master key. **No name-equality, no embedding-cosine join.** Sub-threshold ⇒ refuse + say so. `structured_query` delegates to `run_agent_query` passing `container_id`/`allowed_domains`/`user_id`, runs **strictly sequentially** (async session not concurrency-safe), and inherits the CSV-side feasibility + negative-claim gates.

**C6 — Grounding & honesty invariants (all phases).** No edge / community-report claim / synthesized claim without a source-span citation (else reject/refuse). A "no data / not found" claim requires proof of coverage + diagnosis — retrieval-empty ≠ absent. gpt-4o-mini only.

---

## Token-reduction scorecard (the user's explicit goal)

Tracked because Full GraphRAG is the costly option; savings are architectural, not feature cuts:

| Lever | Phase | Entry point |
|---|---|---|
| Prompt caching on extraction + synthesis system prompts | 0, 2 | biggest lever at ingest scale |
| Batch embeddings | 0 | `embed_texts_batched` |
| Query-embedding cache | 0 | `QueryEmbedder` + Redis |
| Context token budget | 0 | `assemble_context` |
| Idempotent extraction (chunk_fingerprint + prompt/model version) | 2 | never re-extract unchanged |
| Amortized, cited community reports | 2 | built once at ingest |
| Planner bypass + versioned response cache | 3, 4 | simple/cached queries skip the loop |
| Adaptive rerank | 3 | skip when vector-confident |

---

## Reviewer-board gates (per phase)

Before advancing past any phase, route the diff through the board (ai-architect-review, data-science-review, business-analyst-review, static-code-sentinel) under the delivery-manager. Phase 2 additionally requires the faithfulness eval to pass; Phase 4 additionally requires the bridge refuse-on-sub-threshold tests to pass.

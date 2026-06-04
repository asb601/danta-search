# PDF Agentic Graph RAG — Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Execute phases **in order**; each ends at a reviewer-board gate. **This index is authoritative — where a phase file disagrees with the Cross-Phase Contract below, this index wins.**

**Spec:** `docs/superpowers/specs/2026-06-03-pdf-agentic-graph-rag-design.md` (v3)
**Goal:** Turn `server/pdf_chat/` into an agentic, grounded Graph-RAG **comprehension** system — a per-tenant "superhuman memory" that works for any enterprise (information type varies, intent is fixed), with value-evidenced cross-domain (PDF+CSV) reasoning, at a tunable token cost.

---

## Phase plans (execute in this order)

| Roadmap phase | Plan file | Notes |
|---|---|---|
| 0+1 Foundations / make-it-run | `2026-06-03-pdf-phase0-1-foundations.md` | + see addendum for model router & DeepDoc |
| (cross-cut) Model router & DeepDoc | `2026-06-04-pdf-addendum-model-router-and-deepdoc.md` | extends Phase 0 (router) + Phase 1 (DeepDoc ONNX micro-components) |
| 2 Knowledge graph | `2026-06-05-pdf-phase2-knowledge-graph.md` (CURRENT) | section-level extraction + NER/value-overlap backbone + multi-representation index (chunk/section-card/doc-card, RRF-fused) + grounding gate + escalation OFF for bulk. Old `2026-06-03-pdf-phase2-knowledge-graph.md` is superseded/historical. |
| 3 Agentic runtime | `2026-06-03-pdf-phase3-agentic-runtime.md` | + `definitional` intent, conflict gate, provenance labels, `glossary_lookup` tool seam. **HARD ENTRY GATE (from Phase-2 data-science review): synthesis MUST route tag-/card-derived claims through `grounding_gate.tag_as_answer(tag, supporting_chunks)` and drop unsupported ones — tags are a retrieval signal, never an answer. Demote card hits to pull their `src_chunk_ids` into context rather than being quotable evidence.** Also: Phase-2 `community_report_lookup` now reads persisted+embedded reports — Phase 3 must populate the Neo4j vector indexes (`section_card_vector_index`, `doc_card_vector_index`, `community_report_vector_index`) at infra bootstrap. |
| 4 Cross-domain bridge | `2026-06-03-pdf-phase4-cross-domain-bridge.md` | unchanged |
| 5 Comprehension layer | `2026-06-04-pdf-phase5-comprehension.md` | **NEW — the superhuman-memory payload** |
| 6 Hardening | `2026-06-03-pdf-phase5-hardening.md` | (file named "phase5-hardening" = roadmap Phase 6; + escalation-budget tracking) |

---

## Canonical Cross-Phase Contract (single source of truth)

**C0 — Test command.** `uv run --with pytest --with pytest-asyncio pytest <path> -q` from `server/` (pytest is not a project dep). Infra tests behind markers, excluded by default.

**C1 — Tunables & logging (Phase 0 owns).** `pdf_chat/tunables.py`: `get_tunable(container_id, key, default)` + `log_gate_decision(name, *, score, threshold, outcome, **ctx)`. **No score-comparison literal in any `.py`.** Types are never hardcoded dictionaries.

**C2 — Neo4j searcher methods owned by Phase 2.** `Neo4jSearcher.graph_traversal/entity_neighbors/community_report_lookup`, per-hop tenant isolation (`ALL(n IN nodes(path) WHERE n.tenant_id=$tenant_id)`). Phase 3 consumes, does not redefine. Tool `get_entity_neighbors` wraps searcher `entity_neighbors`.

**C3 — Tool interface & registry (Phase 3 owns).** `agent/tools.py`: `Tool` Protocol (`name; async run(state,deps,**kw)->list[dict]`), `TOOL_REGISTRY`, `register_tool`. Every tool — `structured_query` (Phase 4) and `glossary_lookup` (Phase 5) — implements `Tool` and registers via `register_tool` (wrap any LangChain tool, never store a raw `StructuredTool`).

**C4 — Agent entry point (Phase 3 exposes).** `agent/graph.py :: run_pdf_query(query, *, tenant_id, container_id, ...)` → result with `.answer`, `.citations`. Phase 5/6 + the API route depend on it.

**C5 — Cross-domain bridge is value-evidenced only (Phase 4).** Via `relationship_index.fingerprint_value` to a real master key; no name/embedding join; sub-threshold ⇒ refuse. `structured_query` is sequential, passes `container_id`/`allowed_domains`/`user_id`, inherits CSV gates.

**C6 — Grounding & honesty (all phases).** No edge/report/glossary/claim without a cited span. Honest absence (coverage + diagnosis). gpt-4o-mini bulk; per-hop tenant isolation. Relationships three-state (asserted/not-stated/conflicting).

**C7 — Model router (Phase 0 owns; all phases consume).** `pdf_chat/model_router.py`: `select_model(*, task, container_id, signals) -> ModelChoice` and `escalation_allowed(container_id, signals) -> bool` (budget-capped). **Bulk = gpt-4o-mini.** Strong tier (tunable id, default `claude-sonnet-4-6`) only when the data-driven escalation gate fires AND the per-tenant escalation budget is not exhausted. **Opus is query-time-only, off by default; never bulk ingestion.** Embeddings = `text-embedding-3-small`. All model ids, gate thresholds, and budget caps are tunables — no literals.

**C8 — Comprehension artifact (Phase 5 owns).** A versioned, Postgres-backed **Tenant Ontology** (entities, relationships, doc-taxonomy, temporal-coverage, metrics registries) + **glossary** (term → expansion/definition/variants/provenance `stated|inferred`). Read-only projections power the onboarding surface. Mirrors `server/app/services/semantic_layer_builder`. Learned per tenant; the INTENT layer (tools/planner/kinds) stays fixed code (spec invariant 6).

---

## Generality invariants (spec §3.6, §3.7)
- **Stable intent over learned semantics**: domain meaning is per-tenant learned data, never code. New industry = learned data, not a code `if`.
- **Linkage discovered, siloing valid**: no code requires an edge; disconnected graphs are correct; conflicts surfaced, never silently resolved.

## Cost dial (spec §8)
One-time ingestion ≈ `N_docs × pages/doc × $0.002 × (1 + escalation% × (strong_mult − 1))`. ~$40k/1M docs mini-only; keep escalation ≤3–5% (Sonnet ~20×, Opus ~90×). Set per-tenant escalation % to quote a client.

## Reviewer-board gates
Each phase routes its diff through the board (architect, data-science, business-analyst, static-code-sentinel) under the delivery-manager, plus SME for the comprehension layer. Phase 2 also gates on faithfulness eval **including a held-out unseen tenant**. Phase 4 gates on bridge refuse-on-sub-threshold tests.

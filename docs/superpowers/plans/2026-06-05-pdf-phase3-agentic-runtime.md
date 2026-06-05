# PDF Phase 3 — Agentic LangGraph Query Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Implement task-by-task: write the failing test → run it (RED) → implement → run it (GREEN) → leave the diff for the manager to commit. **Agents do NOT `git commit`** — the delivery-manager commits after the reviewer board signs off. Keep the **352-test baseline green** at every step.

**Date:** 2026-06-05 (regenerates the stale `2026-06-03-pdf-phase3-agentic-runtime.md`, which predates Phase 2's real deliverables).

**Goal:** Replace `server/pdf_chat/`'s fixed 10-stage state machine (`agent/graph.py`) with a LangGraph agent that (1) plans a typed intent `local|global|cross_domain|definitional` via `model_router.select_model(task=QUERY_PLANNING)` with named tunable confidence thresholds + typed `fallback_reason`, bypassing the loop for high-confidence simple/cached queries; (2) links entities into `state.entity` before graph tools are reachable; (3) drives a `Tool`-Protocol registry (`TOOL_REGISTRY`/`register_tool`, contract C3) wrapping the Phase-2 searcher with `multi_vector_search` as the **primary** retrieval; (4) runs a capped tool loop (total + per-tool caps, decomposition depth ≤ N, monotonic-progress guard); (5) decomposes multi-part asks with an output-completeness sufficiency check; (6) synthesizes grounded answers with bbox citations + a **citation-density floor**, routing every tag-/card-derived claim through `grounding_gate.tag_as_answer` and **dropping** unsupported ones (HARD ENTRY GATE); (7) gates unproven "no data" claims via a ported negative-claim + conflict gate; (8) emits **provenance labels** (`stated|inferred|conflicting|not_found`) + a staleness hook; and (9) exposes `run_pdf_query(query, *, tenant_id, container_id, ...)` (contract C4).

**Tech Stack:** Python 3.12, LangGraph (already a repo dep — guarded import, mirrors `pdf_chat/agent/graph.py:494`), `gpt-4o-mini` bulk via the router, Neo4j (per-hop tenant isolation, owned by Phase 2). Tests: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q -p no:cacheprovider`. Deterministic seams only — mock LLM/searcher/tools with in-memory fakes (follow `pdf_chat/testing/test_agent.py:48-148`).

**Invariants (spec §3):** Grounding (1), honest absence (2), per-hop tenant isolation (3), **no score-comparison literal in any `.py`** — every threshold via `get_tunable(container_id, key, default)` and logged via `log_gate_decision(name, *, score, threshold, outcome, **ctx)` (4), value-evidenced joins only (5, Phase 4), stable intent over learned semantics (6), linkage discovered / siloing valid / three-state relationships (7).

---

## Entry Points (Phase 1/2 surfaces consumed)

All signatures verified at the cited path:line. Phase 3 **consumes** these; it does not redefine C2 searcher methods.

### `pdf_chat/retrieval/neo4j_searcher.py` (Phase-2 rewrite — contract C2)
```python
# :203  ANN over chunk HNSW index, tenant-scoped. Returns list[dict] of
#       {chunk_id,text,doc_id,page_num,element_type,acl(dict),score} desc.
def vector_search(self, query_vec: list[float], tenant_id: str,
                  top_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]

# :384  PRIMARY RETRIEVAL. Runs chunk + section-card + doc-card legs, RRF-fused
#       inside (via retrieval.rrf.rrf). Each hit dict has a dict `acl`.
def multi_vector_search(self, query_vec: list[float], tenant_id: str,
                        top_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]

# :241  1–2 hop entity→RELATED_TO→(:Chunk)-[:MENTIONS] walk to related chunks.
#       PER-HOP tenant isolation. Returns chunk dicts (acl deserialized).
def graph_traversal(self, entity: str, tenant_id: str,
                    limit: int | None = None, doc_ids: list[str] | None = None) -> list[dict]

# :276  Related-entity neighbourhood of an anchor (PER-HOP tenant). The
#       get_entity_neighbors TOOL wraps this. Returns [{name,etype,normalized_value}].
def entity_neighbors(self, entity: str, tenant_id: str,
                     limit: int | None = None, doc_ids: list[str] | None = None) -> list[dict]

# :317  ANN over the CITED community-report vector space (reads PERSISTED reports).
#       Returns [{community_id,report,citations,score}] desc.
def community_report_lookup(self, query_vec: list[float], tenant_id: str,
                            limit: int | None = None) -> list[dict]

# :445  vector + optional graph leg, RRF-fused (current state-machine path).
def hybrid_search(self, query_vector: list[float], tenant_id: str,
                  doc_ids=None, vector_top_k=None, graph_top_k=None, entity: str | None = None) -> list[dict]
```
Vector index names bootstrapped at infra (Phase-3 NOTE, NOT implemented here):
`chunk_vector_index` (:48), `section_card_vector_index` (:52), `doc_card_vector_index` (:53), `community_report_vector_index` (:54).

### `pdf_chat/ingestion/grounding_gate.py`
```python
# :256  HARD ENTRY GATE. Returns tag.label ONLY if ≥1 supporting chunk's text
#       actually contains the label (verified via _present); else None (suppress).
def tag_as_answer(tag, supporting_chunks, *, container_id: str = "") -> str | None

# :81   GroundedTag(label, scope, confidence, span, src_chunk_id)  — duck-typed on `.label`
# :106  _present(claim, haystack_norm, *, word_boundary_max_len=0) -> bool
```

### `pdf_chat/model_router.py` (contract C7)
```python
# :216
def select_model(*, task, container_id: str, signals: dict, store=None) -> ModelChoice
# :45   TaskClass.{QUERY_SYNTHESIS,QUERY_PLANNING,EXTRACTION,SYNTHESIS}
# :59   ModelChoice(provider: str, model_id: str, is_strong: bool)
# :166  escalation_allowed(container_id, signals, *, store=None) -> bool
# signals keys read: extract_confidence, figure_ratio, cross_domain, definitional
```

### `pdf_chat/retrieval/llm.py`
```python
# :53  PdfLlm.generate(self, system, user, *, container_id="", signals: dict | None = None) -> str
#      already routes via select_model(task=QUERY_SYNTHESIS); fail-safes to gpt-4o-mini.
```

### Pure retrieval helpers
```python
# retrieval/rrf.py:18      rrf(results_lists: list[list[str]], k: int = 60) -> list[str]
# retrieval/acl.py:34      filter_by_acl(chunks, user_id, user_groups, tenant_id) -> tuple[list, list[str]]
# retrieval/acl.py:82      insufficient_context(accessible, min_required) -> bool
# retrieval/reranker.py:52 rerank(query, candidates, top_n=None, *, container_id="") -> list  (adaptive skip tunable)
```

### `pdf_chat/agent/graph.py` + `agent/state.py` (CURRENT — replaced/extended)
```python
# state.py:27   @dataclass PdfChatState(query, tenant_id, user_id="", groups=[], doc_ids=None,
#               top_k=None, entity=None, acl_version="0", query_vector=None, candidates=[],
#               reranked=[], accessible_chunks=[], denied_ids=[], context="", answer="",
#               citations=[], cached=False, cache_key=None, error=None); .chunks_used()
# graph.py:105  @dataclass Deps(embedder, searcher, reranker, cache, extractor, llm, audit_repo)
# graph.py:456  async def run_pdf_chat(state, deps) -> state   (fixed 10-stage runner; kept)
# graph.py:487  def build_graph(deps)                          (guarded langgraph compile)
# graph.py:525  def build_default_deps() -> Deps               (late/guarded real adapters)
# graph.py:239  assemble_context — token budget already exists (reuse, do not reimplement)
```

### Main-system mirror (patterns to copy, NOT import)
```python
# app/services/semantic_planner.py:178  @dataclass ExecutionPlan(... intent, confidence,
#   fallback_reason: str|None, planning_ms)  — typed fallback_reason strings (:261-386):
#   "no_candidate_files","complex_query","low_confidence:<c>","planner_timeout","planner_error:<E>"
# app/agent/state.py: MAX_TOOL_CALLS=8 ; app/agent/graph/graph.py:1322 state seeds tool_call_count=0,
#   :1428/:1835 _gate_negative_claim wraps the final answer on BOTH stream + non-stream paths.
# app/services/erp/negative_claim_gate.py:103 evaluate_negative_claim(*, answer, store,
#   file_identities=None) -> NegativeClaimVerdict(is_negative_claim, proven, coverage_complete,
#   diagnosed, missing_diagnostics, signals); :133 honest_rewrite(verdict, scanned_tables=None) -> str
```

---

## File Structure

| Path | Responsibility |
|---|---|
| `pdf_chat/agent/state.py` (modify) | Add agentic fields to `PdfChatState`: `intent`, `planner_confidence`, `fallback_reason`, `bypass`, `sub_queries`, `tool_calls`, `per_tool_calls`, `seen_chunk_ids`, `decomp_depth`, `router_signals`, `provenance`, `conflicts`. All defaulted (partial states valid). |
| `pdf_chat/agent/planner.py` (create) | `QueryIntent`, `PlannerResult`, `plan_query()` — `select_model(QUERY_PLANNING)` intent classify, named thresholds, typed `fallback_reason`, bypass decision. |
| `pdf_chat/agent/entity_linker.py` (create) | `link_entities()` — populate `state.entity` before graph tools. |
| `pdf_chat/agent/tools.py` (create) | `Tool` Protocol + `TOOL_REGISTRY` + `register_tool` (C3); 5 read tools; Phase-4/5 registration seam (no impl). |
| `pdf_chat/agent/decompose.py` (create) | `decompose_query()` + `sufficiency_check()` (all output components covered). |
| `pdf_chat/agent/loop.py` (create) | `LoopBudget` + `run_tool_loop()` — total/per-tool caps, depth ≤ N, monotonic-progress guard, drop logging. |
| `pdf_chat/agent/synthesis.py` (create) | `synthesize()` — grounded bbox citations, citation-density floor, `tag_as_answer` routing + card demotion, provenance labels + staleness hook. |
| `pdf_chat/agent/negative_claim_gate.py` (create) | PDF/graph port of the ERP gate: coverage = relevant pages/sections in-context; three-state conflict surfacing. |
| `pdf_chat/agent/runtime.py` (create) | `run_pdf_query(...)` + `PdfQueryResult` — the C4 entry; assembles planner→loop→synthesis→gate over a guarded LangGraph (`AgentDeps`). |
| `pdf_chat/testing/test_phase3_planner.py` etc. (create) | One test file per task (matches existing one-concern layout). |

---

## NEW Public Types & Signatures (locked — later tasks match exactly)

```python
# agent/state.py — extend PdfChatState (additive; every field defaulted)
intent: str = "local"                       # local|global|cross_domain|definitional
planner_confidence: float = 0.0
fallback_reason: str | None = None
bypass: bool = False
sub_queries: list[str] = field(default_factory=list)
decomp_depth: int = 0
tool_calls: int = 0
per_tool_calls: dict[str, int] = field(default_factory=dict)
seen_chunk_ids: set[str] = field(default_factory=set)
router_signals: dict = field(default_factory=dict)   # passed to select_model
provenance: dict[int, str] = field(default_factory=dict)  # citation n -> label
conflicts: list[dict] = field(default_factory=list)

# agent/planner.py
QueryIntent = Literal["local", "global", "cross_domain", "definitional"]
@dataclass
class PlannerResult:
    intent: QueryIntent = "local"
    confidence: float = 0.0
    bypass: bool = False                # high-confidence simple/cached → skip the loop
    fallback_reason: str | None = None  # typed: "low_confidence:<c>"|"planner_error:<E>"|"ambiguous_intent"
    signals: dict = field(default_factory=dict)  # {cross_domain, definitional} for the router
async def plan_query(query: str, *, container_id: str, llm, cached: bool = False) -> PlannerResult

# agent/tools.py  (contract C3 — confirmed)
class Tool(Protocol):
    name: str
    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]: ...
TOOL_REGISTRY: dict[str, Tool] = {}
def register_tool(tool: Tool) -> Tool: ...        # wrap any LangChain tool, never store a raw StructuredTool
# Phase-3 tools (names): "vector_search","multi_vector_search","graph_traverse",
#   "community_report_lookup","get_entity_neighbors". SEAM ONLY for "structured_query"(P4),"glossary_lookup"(P5).

# agent/loop.py
@dataclass
class LoopBudget:
    max_total_calls: int                 # mirrors MAX_TOOL_CALLS; tunable "agent.max_tool_calls"
    max_per_tool: int                    # tunable "agent.max_per_tool_calls"
    max_decomp_depth: int                # tunable "agent.max_decomp_depth"
async def run_tool_loop(state, deps, budget: LoopBudget) -> PdfChatState  # aborts on no-new-chunk round

# agent/synthesis.py
@dataclass
class SynthesisResult:
    answer: str
    citations: list[dict]                # {n, doc_id, page, bbox}
    provenance: dict[int, str]           # n -> stated|inferred|conflicting|not_found
async def synthesize(state, deps, *, container_id: str) -> SynthesisResult

# agent/negative_claim_gate.py
@dataclass
class PdfNegativeVerdict:
    is_negative_claim: bool = False
    proven: bool = False
    coverage_complete: bool = False      # relevant pages/sections were in-context
    diagnosed: bool = False
    conflicts: list[dict] = field(default_factory=list)
def evaluate_pdf_negative_claim(*, answer, accessible_chunks, query_pages=None,
                                container_id: str = "") -> PdfNegativeVerdict
def pdf_honest_rewrite(verdict) -> str

# agent/runtime.py  (contract C4 — the public entry)
@dataclass
class PdfQueryResult:
    answer: str
    citations: list[dict]
    intent: str
    provenance: dict[int, str]
    conflicts: list[dict]
async def run_pdf_query(query: str, *, tenant_id: str, container_id: str,
                        user_id: str = "", groups: list[str] | None = None,
                        doc_ids: list[str] | None = None, deps=None) -> PdfQueryResult
```

---

## TDD Tasks

Each task: **failing test → run (RED) → implement → run (GREEN)** → leave for manager. Test command prefix (abbreviated `PYT` below):
`cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest`

### Task 1 — State extension
- **RED:** `pdf_chat/testing/test_phase3_state.py` asserts a fresh `PdfChatState(query="q", tenant_id="t")` exposes `intent=="local"`, `tool_calls==0`, `per_tool_calls=={}`, `seen_chunk_ids==set()`, `bypass is False`, `provenance=={}`, `conflicts==[]`, `decomp_depth==0`.
- Run: `PYT pdf_chat/testing/test_phase3_state.py -q -p no:cacheprovider`
- **Impl:** add the locked fields to `PdfChatState` (all defaulted; keep existing fields + `chunks_used()`).
- Run: GREEN. Re-run full suite to confirm 352→still green.

### Task 2 — Planner (intent + bypass + typed fallback)
- **RED:** `test_phase3_planner.py` with a `FakeLlm` that returns a canned intent JSON. Assert: a confident simple query yields `bypass is True`; a malformed LLM reply yields `fallback_reason=="planner_error:..."` and `bypass is False`; a `definitional`/`cross_domain` classification sets `signals["definitional"]`/`["cross_domain"]` True; the confidence threshold is read via `get_tunable(container_id, "agent.planner_bypass_confidence", ...)` and the decision is logged (assert the returned `log_gate_decision` record `outcome`).
- Run: RED.
- **Impl:** `plan_query` calls `select_model(task=TaskClass.QUERY_PLANNING, container_id=..., signals=...)` to pick the planning model, prompts it, parses intent+confidence, compares confidence to the tunable via `log_gate_decision("agent.planner_bypass", score=conf, threshold=floor, outcome="bypass"|"loop", container_id=...)`. Typed `fallback_reason` mirrors `semantic_planner.py:386` style (`low_confidence:<c>`). Never raises.
- Run: GREEN.

### Task 3 — Entity linking
- **RED:** `test_phase3_entity_linker.py`: given a query naming an entity present in a `FakeSearcher.entity_neighbors`-style resolver, `link_entities` sets `state.entity`; an unrecognized query leaves `state.entity is None` (graph tools then unreachable). No literal — any confidence floor via `get_tunable("agent.entity_link_min_confidence")`, logged.
- Run: RED → **Impl** `link_entities(state, deps)` → GREEN.

### Task 4 — Tool registry + Protocol (C3)
- **RED:** `test_phase3_tools.py`: `register_tool` adds to `TOOL_REGISTRY` keyed by `.name` and returns the tool; re-registering the same name raises; a Phase-4 `structured_query` / Phase-5 `glossary_lookup` name is **absent** (seam only). Assert each Phase-3 tool satisfies `Tool` (has `name`, awaitable `run`).
- Run: RED.
- **Impl:** `Tool` Protocol, `TOOL_REGISTRY`, `register_tool`; register 5 tools wrapping the searcher: `multi_vector_search` (primary), `vector_search`, `graph_traverse`→`searcher.graph_traversal`, `community_report_lookup`, `get_entity_neighbors`→`searcher.entity_neighbors`. Each `run` threads `tenant_id`/`doc_ids` (per-hop tenant via the searcher Cypher). Leave a commented seam + test asserting the seam keys are reservable but unimplemented.
- Run: GREEN.

### Task 5 — `multi_vector_search` is the PRIMARY retrieval
- **RED:** `test_phase3_retrieval.py`: a `FakeSearcher` records which method the loop calls first for a `local` intent. Assert `multi_vector_search` is invoked (NOT `vector_search` / `hybrid_search`) and graph results are merged only when `state.entity` is set; assert ACL filter runs before rerank, and `rerank` is called with `container_id` (adaptive skip tunable). Tenant id threaded to every leg (assert recorded kwarg).
- Run: RED → **Impl** the retrieval node in `loop.py` (multi_vector_search → +graph_traverse when entity → `filter_by_acl` → `rerank`) → GREEN.

### Task 6 — Decomposition + sufficiency
- **RED:** `test_phase3_decompose.py`: a 3-part query (`FakeLlm`) yields 3 `sub_queries`; `sufficiency_check` returns False until all output components have ≥1 grounded chunk, then True. Depth capped via `get_tunable("agent.max_decomp_depth")`, logged on drop.
- Run: RED → **Impl** → GREEN.

### Task 7 — Capped tool loop + monotonic-progress guard
- **RED:** `test_phase3_loop.py`: (a) loop aborts at `max_total_calls` (tunable `agent.max_tool_calls`, default mirrors MAX_TOOL_CALLS=8); (b) per-tool cap enforced (`agent.max_per_tool_calls`); (c) **monotonic-progress** — a round that adds zero new `chunk_id` to `state.seen_chunk_ids` aborts the loop; (d) every cap/abort logs via `log_gate_decision` with the count as `score`. Use a `FakeSearcher` returning a fixed set so the second round adds nothing.
- Run: RED → **Impl** `LoopBudget` + `run_tool_loop` → GREEN.

### Task 8 — Synthesis: citation-density floor + tag_as_answer HARD GATE
- **RED:** `test_phase3_synthesis.py`:
  - **8a citation floor:** a claim with zero citations is rejected; `get_tunable("agent.min_citations_per_claim")`, logged.
  - **8b tag_as_answer called + unsupported dropped:** monkeypatch/spy `grounding_gate.tag_as_answer`; assert it is **actually called** for each tag-/card-derived claim, and a tag whose label appears in NO supporting chunk is **dropped** from the answer (returns `None`).
  - **8c card demotion:** a card hit (section/doc) contributes only its `src_chunk_ids` to context and is never emitted as a quotable citation.
  - **8d bbox:** citations carry `bbox` + `page`.
- Run: RED.
- **Impl:** `synthesize` builds context from accessible chunks (reuse the token-budget logic from `graph.py:239`), demotes card hits to their `src_chunk_ids`, routes each tag/card claim through `tag_as_answer(tag, supporting_chunks, container_id=...)` and drops `None`, enforces the citation floor via `log_gate_decision`, calls `PdfLlm.generate(task QUERY_SYNTHESIS via signals)`, attaches `bbox`+provenance.
- Run: GREEN.

### Task 9 — Provenance labels + staleness hook
- **RED:** `test_phase3_provenance.py`: a stated-span claim → `"stated"`; a usage-inferred claim → `"inferred"`; conflicting sources → `"conflicting"`; absent → `"not_found"`. A `staleness_annotation(latest_date)` hook returns a "most recent mention is YYYY-MM; may be outdated" string (hook only; no temporal store wired).
- Run: RED → **Impl** in `synthesis.py` → GREEN.

### Task 10 — Negative-claim + conflict gate (port)
- **RED:** `test_phase3_negative_gate.py`: a "not found" answer with NO relevant pages in-context → `proven False`, `pdf_honest_rewrite` returned (coverage not proven); a "not found" with relevant pages in-context + diagnosis → `proven True`, answer kept; a three-state relationship conflict is surfaced in `verdict.conflicts` (never silently resolved). Mirror `evaluate_negative_claim`'s never-raise contract.
- Run: RED → **Impl** `evaluate_pdf_negative_claim` + `pdf_honest_rewrite` (coverage = query pages/sections present in `accessible_chunks`) → GREEN.

### Task 11 — `run_pdf_query` public entry (C4) + LangGraph wiring
- **RED:** `test_phase3_runtime.py`: `run_pdf_query("q", tenant_id="t", container_id="c", deps=<fakes>)` returns a `PdfQueryResult` with `.answer`, `.citations`, `.intent`; a cached/bypass query skips the loop (assert `FakeSearcher` not called); ACL-empty returns the deterministic insufficient-context refusal (reuse `INSUFFICIENT_CONTEXT_MESSAGE`); the negative-claim gate wraps the final answer (mirror `graph.py:1428`). LangGraph import is guarded (mirror `graph.py:494`) — the plain runner path is exercised when langgraph is absent.
- Run: RED.
- **Impl:** `run_pdf_query` builds `PdfChatState`, runs `plan_query` → (bypass? cache/short retrieval : `link_entities`→`decompose_query`→`run_tool_loop`)→`synthesize`→`evaluate_pdf_negative_claim`/rewrite; expose `build_agent_graph(deps)` (guarded langgraph) running the SAME nodes; `deps=None` → `build_default_deps()`.
- Run: GREEN.

### Task 12 — Full-suite regression + infra-bootstrap NOTE
- Run the entire suite: `PYT pdf_chat/testing/ -q -p no:cacheprovider` — confirm **352 baseline + new tests all green**.
- **NOTE (do NOT implement):** add a docstring/TODO seam in `runtime.py` documenting the infra-bootstrap requirement to create Neo4j vector indexes `section_card_vector_index`, `doc_card_vector_index`, `community_report_vector_index` (consumed by `multi_vector_search`/`community_report_lookup`). No DDL here.

---

## Cross-Phase Contracts Exposed

- **C3 (Phase 3 owns):** `agent/tools.py :: Tool` Protocol + `TOOL_REGISTRY` + `register_tool`. **Phase-4 seam:** register `structured_query` (value-evidenced CSV/DataFusion bridge, sequential, passes `container_id`/`allowed_domains`/`user_id`) via `register_tool` — no loop change. **Phase-5 seam:** register `glossary_lookup` likewise. Both reserved by name, unimplemented here.
- **C4 (Phase 3 exposes):** `agent/runtime.py :: run_pdf_query(query, *, tenant_id, container_id, ...) -> PdfQueryResult(.answer, .citations, ...)` — Phase 5/6 + the API route depend on it.
- **Infra-bootstrap seam (Phase 6 / ops):** create the three Neo4j vector indexes named above.

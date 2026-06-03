# Implementation Plan — PDF Agentic Graph RAG · Phase 2: Grounded Knowledge-Graph Construction

**Date:** 2026-06-03
**Spec:** `docs/superpowers/specs/2026-06-03-pdf-agentic-graph-rag-design.md` (§2 Layer 1b, §3 invariants 1/3/4/5, §5 Phase 2)
**Module:** `server/pdf_chat/`
**Status:** DRAFT — planning author only; this file is a plan, not an implementation.
**Author role:** planning author. Do NOT implement here.

---

## Goal

Make the PDF graph leg *real and grounded*. Today the writer only persists
`(:Document)-[:CONTAINS]->(:Page)-[:CONTAINS]->(:Chunk)` (`server/pdf_chat/ingestion/neo4j_writer.py:84-99`),
the searcher queries an `(:Entity)-[:RELATED_TO]` shape that nothing writes
(`server/pdf_chat/retrieval/neo4j_searcher.py:51-59`), and `state.entity` is never populated.

Phase 2 adds, per the spec §2 Layer 1b:
1. Open-vocabulary entity + relation extraction per chunk via `gpt-4o-mini`
   (prompt-cached, adaptive capped gleaning, idempotent on `chunk_fingerprint + prompt/model version`).
2. A **blocking grounding gate** — every `RELATED_TO` edge persists `src_chunk` + verbatim span + confidence;
   any edge whose subject/object/predicate is absent from the cited span is rejected before write.
   This mirrors the value-overlap gate + `edge_provenance` in
   `server/app/services/relationship_detector.py:151-211` and the fingerprint evidence in
   `server/app/services/relationship_index.py:223-276`.
3. Entity resolution by embedding similarity + type agreement + co-occurrence, **unmerged-by-default**
   for ambiguous pairs, with merge bands **derived per-container from the score distribution** (no literals),
   persisting the merge decision + evidence, no transitive auto-merge without a confidence floor.
4. The Neo4j schema `(:Entity)-[:RELATED_TO {desc,weight,confidence,evidence_count,src_chunk}]->(:Entity)`,
   `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)`, `(:Entity)-[:IN_COMMUNITY]->(:Community)`,
   `tenant_id` on every node/edge, and the **rewritten searcher Cypher** to this schema.
5. Leiden communities (networkx in-worker; resolution/min-size tunable + logged) and cited community reports
   (`gpt-4o-mini`, drill-down citations to chunks/bbox, suppressed unless traceable to ≥N grounded edges,
   route-only never evidence-of-record).
6. PageRank over grounded edges weighted by confidence.
7. A Phase-2 **exit gate**: per-hop tenant isolation verified + faithfulness eval
   (edge precision vs source, merge precision/recall, report groundedness).

---

## Shared Conventions (follow exactly)

- **Tunables.** Reuse `server/pdf_chat/tunables.py` from Phase 0: `get_tunable(container_id, key, default)`
  and `log_gate_decision(container_id, gate, decision, score, **ctx)`. Assume it EXISTS — do not redefine it.
  Every threshold (gleaning delta, merge bands, community resolution / min-size, report-groundedness floor `N`,
  PageRank damping, type-collapse alarm %) resolves through `get_tunable`. **No score-comparison literal in any `.py` file**
  (spec §3 invariant 4). Every gate/cap/skip/merge decision is logged via `log_gate_decision`.
- **LLM.** `gpt-4o-mini` ONLY. Reuse the guarded Azure client pattern in `server/pdf_chat/ingestion/embeddings.py:18-32`
  and the configured `chat_model`/`embedding_model` in `server/pdf_chat/config.py:42-49`. Prompt caching ON
  (system prompt is a stable cached prefix).
- **Tenant isolation.** `tenant_id` on every node and every edge. Every Cypher filters `tenant_id` on ALL path
  elements including variable-length paths: `ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)` (spec §3 invariant 3).
- **Tests.** `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/`. Match the existing layout
  (`server/pdf_chat/testing/test_ingestion.py`, `test_retrieval.py`): pure logic runs with zero infra; infra tests
  behind `@pytest.mark.neo4j` / `@pytest.mark.llm` markers. For LLM-dependent behavior, test against deterministic
  seams (mock the LLM client; assert the grounding gate rejects an ungrounded edge; assert idempotency short-circuits;
  assert the tenant filter is present in the Cypher string) — never against model output text.
- **Commits.** Conventional commits, one per task step group, ending with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit frequently; never push without approval.

---

## Cross-Phase Contracts

### Depended on (from Phase 0 / Phase 1)
- **Phase 0** `pdf_chat/tunables.py`: `get_tunable(container_id, key, default)`, `log_gate_decision(...)`.
  Prompt caching verified on the Azure deployment (spec §8 open-Q #5).
- **Phase 0/1** chat client: a `gpt-4o-mini` chat-completions seam reusing `embeddings.py`'s guarded `AzureOpenAI`
  pattern. Phase 2 introduces `pdf_chat/llm/extraction_client.py` (a thin guarded wrapper) if Phase 1 has not.
- **Phase 1** populated `(:Chunk)` nodes with `chunk_id`, `text`, `doc_id`, `page_num`, `element_type`, `tenant_id`,
  `acl`, `embedding`, `reading_order`, plus a per-chunk `bbox` (spec §2 Layer 1a: bbox retained). `Chunk` dataclass:
  `server/pdf_chat/ingestion/ton_schema.py:63-90`.
- **Phase 1** `chunk_fingerprint`: reuse `compute_sha256` (`server/pdf_chat/ingestion/fingerprint.py:11`) over chunk text.

### Exposed (to Phase 3 / Phase 4)
- **To Phase 3 (agentic runtime):** the rewritten `Neo4jSearcher` schema (`MENTIONS`, `RELATED_TO`, `IN_COMMUNITY`,
  per-hop tenant filter), `Neo4jSearcher.entity_neighbors(...)` and `Neo4jSearcher.community_report_lookup(...)`,
  and `Entity.name`/`Entity.entity_id` so the runtime entity-linking step can populate `state.entity`.
- **To Phase 4 (cross-domain bridge):** `Entity` nodes carry a `normalized_value` (via `relationship_index.fingerprint_value`
  reuse) so the Phase-4 `pdf_entity_bridge` can value-reconcile PDF entities against the CSV `relationship_index`
  master keys WITHOUT name/embedding equality (spec §2 Layer 2).

---

## NEW Public Types / Function Signatures Introduced

```python
# pdf_chat/ingestion/graph_schema.py — pure dataclasses (mirror ton_schema.py style)
@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str            # OPEN-VOCAB — LLM-proposed, never a closed enum
    confidence: float
    supporting_span: str        # verbatim substring of the source chunk text
    chunk_id: str
    tenant_id: str

@dataclass(frozen=True)
class ExtractedRelation:
    subject: str                # entity name
    predicate: str
    object: str                 # entity name
    description: str
    confidence: float
    supporting_span: str        # verbatim substring of the source chunk text
    chunk_id: str
    tenant_id: str

@dataclass(frozen=True)
class ChunkExtraction:
    chunk_id: str
    chunk_fingerprint: str
    prompt_version: str
    model_version: str
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    gleaning_passes: int

@dataclass(frozen=True)
class GroundingVerdict:
    accepted: bool
    reason: str                 # typed: "ok" | "subject_absent" | "object_absent" | "predicate_absent" | "span_empty"

@dataclass(frozen=True)
class MergeDecision:
    canonical_id: str
    member_ids: list[str]
    band: str                   # "auto_merge" | "tie_break" | "hold"
    score: float
    evidence: dict              # {embedding_sim, type_agreement, cooccurrence}

@dataclass(frozen=True)
class CommunityReport:
    community_id: int
    title: str
    summary: str
    citations: list[dict]       # [{chunk_id, bbox, src_chunk}]
    grounded_edge_count: int
    suppressed: bool

# pdf_chat/ingestion/entity_extractor.py
PROMPT_VERSION: str
def build_extraction_prompt() -> str: ...                     # cached system prefix; NO closed type list
def extract_chunk(chunk: dict, container_id: str, *, client=None) -> ChunkExtraction: ...   # adaptive capped gleaning + idempotent
def should_continue_gleaning(new_entities: int, total_entities: int, container_id: str) -> bool: ...

# pdf_chat/ingestion/grounding_gate.py — PURE, zero-infra
def span_grounds_relation(rel: ExtractedRelation) -> GroundingVerdict: ...   # reject if subj/obj/pred absent from span
def filter_grounded(rels: list[ExtractedRelation], container_id: str) -> list[ExtractedRelation]: ...

# pdf_chat/ingestion/entity_resolver_graph.py — bands DERIVED, not literal
def derive_merge_bands(scores: list[float], container_id: str) -> dict[str, float]: ...    # percentile-based
def score_pair(a: ExtractedEntity, b: ExtractedEntity, embed_sim: float, cooccurrence: int, container_id: str) -> float: ...
def resolve_entities(entities: list[ExtractedEntity], embeddings: dict, cooccurrence: dict, container_id: str) -> list[MergeDecision]: ...

# pdf_chat/ingestion/communities.py — networkx in-worker
def detect_communities(edges: list[tuple[str, str, float]], container_id: str) -> dict[str, int]: ...   # Leiden
def pagerank_grounded(edges: list[tuple[str, str, float]], container_id: str) -> dict[str, float]: ...   # confidence-weighted
def build_community_report(community_id, member_edges, container_id, *, client=None) -> CommunityReport: ...

# pdf_chat/ingestion/neo4j_writer.py — NEW methods on existing Neo4jWriter
def write_entities(self, entities, decisions: list[MergeDecision]) -> int: ...
def write_relations(self, relations: list[ExtractedRelation]) -> int: ...     # only grounded ones reach here
def write_mentions(self, mentions: list[tuple[str, str]]) -> int: ...          # (chunk_id, entity_id)
def write_next_chunk(self, ordered_chunk_ids: list[str], tenant_id: str) -> int: ...
def write_communities(self, membership: dict[str, int], reports: list[CommunityReport]) -> int: ...
def set_pagerank(self, scores: dict[str, float], tenant_id: str) -> int: ...

# pdf_chat/retrieval/neo4j_searcher.py — REWRITTEN signatures (schema-aligned)
def graph_traversal(self, entity: str, tenant_id: str, limit=None, doc_ids=None) -> list[dict]: ...   # MENTIONS+RELATED_TO; per-hop tenant
def entity_neighbors(self, entity: str, tenant_id: str, hops=None, doc_ids=None) -> list[dict]: ...    # NEW; ALL(n IN nodes(path) WHERE n.tenant_id=$tenant_id)
def community_report_lookup(self, query_vector: list[float], tenant_id: str, top_k=None) -> list[dict]: ...   # NEW; route-only

# pdf_chat/ingestion/graph_construct.py — orchestration entry (called by Phase-1 finalization task)
def construct_knowledge_graph(doc_id: str, container_id: str, tenant_id: str, chunks: list[dict], *, writer=None, client=None) -> dict: ...
```

---

## File Structure

```
server/pdf_chat/
├── ingestion/
│   ├── graph_schema.py            NEW — ExtractedEntity/Relation/ChunkExtraction/GroundingVerdict/MergeDecision/CommunityReport
│   ├── entity_extractor.py        NEW — gpt-4o-mini extraction, prompt-cached, adaptive gleaning, idempotent
│   ├── grounding_gate.py          NEW — PURE blocking gate (span grounds subject/object/predicate)
│   ├── entity_resolver_graph.py   NEW — derived bands, score_pair, unmerged-by-default resolution
│   ├── communities.py             NEW — Leiden (networkx), confidence-weighted PageRank, cited reports
│   ├── graph_construct.py         NEW — orchestrator wiring extractor→gate→resolver→writer→communities
│   ├── neo4j_writer.py            EDIT — add Entity/RELATED_TO/MENTIONS/NEXT_CHUNK/Community writes + indexes
│   └── __init__.py                EDIT — export new public surface
├── retrieval/
│   └── neo4j_searcher.py          EDIT — rewrite Cypher to new schema; per-hop tenant; entity_neighbors + community_report_lookup
├── llm/
│   └── extraction_client.py       NEW (if Phase 1 absent) — guarded gpt-4o-mini chat seam, prompt-cache enabled
└── testing/
    ├── test_graph_extraction.py   NEW — extractor seams, gleaning, idempotency (mock LLM)
    ├── test_grounding_gate.py     NEW — PURE gate accept/reject matrix
    ├── test_entity_resolution.py  NEW — derived bands, unmerged-by-default, no transitive merge
    ├── test_communities.py        NEW — Leiden determinism, PageRank weighting, report suppression
    ├── test_neo4j_writer_graph.py NEW — Cypher strings: tenant on every node/edge (string assertions)
    └── test_neo4j_searcher_v2.py  NEW — rewritten Cypher: per-hop tenant filter present (string assertions)
```

---

## Tasks

### Task 1 — Graph schema dataclasses (`graph_schema.py`)

**Files:** `pdf_chat/ingestion/graph_schema.py`, `pdf_chat/testing/test_grounding_gate.py` (placeholder import), `pdf_chat/ingestion/__init__.py`

**Steps**

1. **Failing test.** In `test_graph_extraction.py` add:
   ```python
   def test_extracted_relation_carries_provenance():
       from pdf_chat.ingestion.graph_schema import ExtractedRelation
       r = ExtractedRelation(subject="Acme", predicate="supplies", object="Globex",
                             description="Acme supplies Globex.", confidence=0.9,
                             supporting_span="Acme supplies Globex.", chunk_id="c1", tenant_id="t1")
       assert r.chunk_id == "c1" and r.supporting_span and r.tenant_id == "t1"
   ```
2. **Fails:** `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_graph_extraction.py -q` → ImportError.
3. **Impl.** Create `graph_schema.py` with the six frozen dataclasses from the signatures block above. Pure stdlib only
   (mirror `ton_schema.py:20-60`). No infra imports.
4. **Passes:** re-run the test.
5. **Commit:** `feat(pdf-graph): add grounded KG schema dataclasses`.

---

### Task 2 — Grounding gate (PURE, blocking) (`grounding_gate.py`)

This is invariant 1 (spec §3) — the heart of Phase 2. Mirror the value-overlap gate logic in
`server/app/services/relationship_detector.py:151-162`: evidence required before an edge is created.

**Files:** `pdf_chat/ingestion/grounding_gate.py`, `pdf_chat/testing/test_grounding_gate.py`

**Steps**

1. **Failing test** — the rejection matrix (deterministic, no LLM):
   ```python
   import pytest
   from pdf_chat.ingestion.graph_schema import ExtractedRelation, GroundingVerdict
   from pdf_chat.ingestion.grounding_gate import span_grounds_relation, filter_grounded

   def _rel(subj, pred, obj, span):
       return ExtractedRelation(subj, pred, obj, f"{subj} {pred} {obj}", 0.9, span, "c1", "t1")

   def test_accepts_when_all_three_present_in_span():
       r = _rel("Acme", "supplies", "Globex", "Acme supplies Globex parts.")
       assert span_grounds_relation(r).accepted is True

   def test_rejects_when_object_absent_from_span():
       r = _rel("Acme", "supplies", "Globex", "Acme is a supplier.")
       v = span_grounds_relation(r)
       assert v.accepted is False and v.reason == "object_absent"

   def test_rejects_empty_span():
       assert span_grounds_relation(_rel("A", "x", "B", "")).reason == "span_empty"

   def test_filter_grounded_drops_ungrounded(monkeypatch):
       rels = [_rel("Acme", "supplies", "Globex", "Acme supplies Globex."),
               _rel("Acme", "owns", "Initech", "unrelated text")]
       kept = filter_grounded(rels, container_id="cont1")
       assert len(kept) == 1 and kept[0].object == "Globex"
   ```
2. **Fails:** `uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_grounding_gate.py -q` → ImportError.
3. **Impl.** `span_grounds_relation`: normalize span + subject/object/predicate with a shared casefold+whitespace-collapse
   (reuse the normalization spirit of `relationship_index.normalize_key_value` at `relationship_index.py:53-68`, but inline-pure,
   no DB). Return `GroundingVerdict(accepted=False, reason="span_empty")` if span blank; else check subject, object, predicate
   membership IN the normalized span; first missing → typed reason. `filter_grounded` keeps only accepted rels and calls
   `log_gate_decision(container_id, "grounding", verdict.reason, rel.confidence, chunk_id=rel.chunk_id)` for every rel.
   **No literal thresholds** — this gate is membership-based, not score-based.
4. **Passes:** re-run.
5. **Commit:** `feat(pdf-graph): blocking grounding gate rejecting ungrounded edges`.

---

### Task 3 — Entity + relation extractor (`entity_extractor.py`)

Open-vocabulary, prompt-cached, adaptive capped gleaning, idempotent. Test against the LLM seam only.

**Files:** `pdf_chat/llm/extraction_client.py` (if absent), `pdf_chat/ingestion/entity_extractor.py`, `pdf_chat/testing/test_graph_extraction.py`

**Steps**

1. **Failing tests** (mock the client — never assert model text):
   ```python
   def _fake_client(payloads):
       # payloads: list of dicts the parser will receive per gleaning pass
       calls = {"n": 0}
       class C:
           def extract(self, system, user):
               i = min(calls["n"], len(payloads) - 1); calls["n"] += 1
               return payloads[i]
       return C(), calls

   def test_prompt_has_no_closed_type_list():
       from pdf_chat.ingestion.entity_extractor import build_extraction_prompt
       p = build_extraction_prompt().lower()
       assert "open" in p and "person, organization, location" not in p  # no enumerated closed set

   def test_gleaning_stops_when_marginal_yield_below_delta(monkeypatch):
       from pdf_chat.ingestion import entity_extractor as ex
       monkeypatch.setattr(ex, "get_tunable",
           lambda c, k, d: {"graph_gleaning_delta": 0.3, "graph_gleaning_max_passes": 5}.get(k, d))
       # pass1 yields 10 new, pass2 yields 1 new (marginal 0.1 < 0.3) → stop after pass2
       client, calls = _fake_client([
           {"entities": [{"name": f"E{i}", "entity_type": "thing", "confidence": 0.9,
                          "supporting_span": "..."} for i in range(10)], "relations": []},
           {"entities": [{"name": "E10", "entity_type": "thing", "confidence": 0.9,
                          "supporting_span": "..."}], "relations": []},
       ])
       chunk = {"chunk_id": "c1", "text": "x", "tenant_id": "t1"}
       out = ex.extract_chunk(chunk, "cont1", client=client)
       assert out.gleaning_passes == 2

   def test_extraction_idempotent_on_fingerprint(monkeypatch):
       # same fingerprint + prompt/model version → cached, client NOT called twice
       ...  # assert client.extract call count == 1 across two extract_chunk calls
   ```
2. **Fails:** ImportError / assertion.
3. **Impl.**
   - `extraction_client.py`: guarded `AzureOpenAI` chat wrapper copying the import-guard pattern of
     `embeddings.py:18-32`; `extract(system, user)` sends the **cached** system prompt + chunk user message at
     `temperature=0`, parses JSON. `chat_model` from `config.py:48` (`gpt-4o-mini`).
   - `entity_extractor.py`: `PROMPT_VERSION = "p2-extract-v1"`. `build_extraction_prompt()` returns a system prompt that
     asks the model to PROPOSE entity types (open-vocab) with confidence + the verbatim supporting span, and to emit relations
     with a verbatim supporting span — **no enumerated type list**. `should_continue_gleaning` reads
     `get_tunable(container_id, "graph_gleaning_delta", ...)` and `"graph_gleaning_max_passes"`; returns False when
     `new/total < delta` or passes hit the cap; logs via `log_gate_decision(container_id, "gleaning", ...)`.
     `extract_chunk` computes `chunk_fingerprint = compute_sha256(chunk["text"].encode())`, builds an idempotency key
     `f"{chunk_fingerprint}:{PROMPT_VERSION}:{model_version}"`, short-circuits if already extracted (in-worker memo +
     Neo4j existence check seam), loops gleaning passes accumulating de-duped entities until `should_continue_gleaning`
     is False, and returns `ChunkExtraction`. Tag every `ExtractedEntity`/`ExtractedRelation` with `chunk_id` + `tenant_id`.
4. **Passes:** re-run `test_graph_extraction.py`.
5. **Commit:** `feat(pdf-graph): open-vocab idempotent extractor with adaptive gleaning`.

---

### Task 4 — Entity resolution with derived bands (`entity_resolver_graph.py`)

Embedding similarity + type agreement + co-occurrence; bands derived per-container from the score distribution
(NOT literals); unmerged-by-default; no transitive auto-merge without a confidence floor; persist decision + evidence.

**Files:** `pdf_chat/ingestion/entity_resolver_graph.py`, `pdf_chat/testing/test_entity_resolution.py`

**Steps**

1. **Failing tests:**
   ```python
   def test_bands_derived_from_distribution_not_literal():
       from pdf_chat.ingestion.entity_resolver_graph import derive_merge_bands
       bands = derive_merge_bands([0.1, 0.4, 0.6, 0.85, 0.95], container_id="cont1")
       assert bands["auto_merge"] > bands["tie_break"] > bands["hold"]  # ordered, data-derived

   def test_ambiguous_pair_unmerged_by_default():
       # a tie_break-band pair must NOT auto-merge
       ...  # assert decision.band == "tie_break" and len(decision.member_ids) == 1

   def test_no_transitive_merge_below_floor():
       # A~B auto, B~C auto, but A~C score below floor → A,C not co-merged transitively
       ...
   ```
2. **Fails.**
3. **Impl.**
   - `derive_merge_bands(scores, container_id)`: compute percentiles of the observed pair-score distribution; map to
     `auto_merge`/`tie_break`/`hold` cut points using `get_tunable(container_id, "merge_auto_percentile", ...)` etc.
     (the *percentile knobs* are tunables; the cut *values* are data-derived). Log via `log_gate_decision`.
   - `score_pair(a, b, embed_sim, cooccurrence, container_id)`: blend embedding similarity + type agreement
     (`a.entity_type == b.entity_type`) + log-scaled co-occurrence, mirroring the cardinality-weighted blend in
     `relationship_detector.join_confidence` (`relationship_detector.py:51-65`). Weights from `get_tunable`.
   - `resolve_entities(...)`: score candidate pairs, derive bands from the score list, union-find only pairs in the
     `auto_merge` band, **gate the transitive closure by a `merge_confidence_floor` tunable** (no merge across an edge
     below the floor), leave `tie_break`/`hold` pairs UNMERGED. Return `MergeDecision`s with `evidence`
     `{embedding_sim, type_agreement, cooccurrence}`. Log every decision.
4. **Passes.**
5. **Commit:** `feat(pdf-graph): entity resolution with derived bands, unmerged-by-default`.

---

### Task 5 — Communities + PageRank + cited reports (`communities.py`)

Leiden via networkx in-worker; resolution/min-size tunable + logged; PageRank over grounded edges weighted by confidence;
reports cite chunks/bbox, suppressed unless traceable to ≥N grounded edges, route-only.

**Files:** `pdf_chat/ingestion/communities.py`, `pdf_chat/testing/test_communities.py`

**Steps**

1. **Failing tests:**
   ```python
   def test_pagerank_weights_by_confidence():
       from pdf_chat.ingestion.communities import pagerank_grounded
       # high-confidence hub outranks a low-confidence node
       scores = pagerank_grounded([("A","B",0.95),("A","C",0.95),("D","E",0.1)], "cont1")
       assert scores["A"] > scores["D"]

   def test_report_suppressed_below_min_grounded_edges(monkeypatch):
       from pdf_chat.ingestion import communities as cm
       monkeypatch.setattr(cm, "get_tunable",
           lambda c,k,d: {"report_min_grounded_edges": 3}.get(k, d))
       rep = cm.build_community_report(0, member_edges=[("A","B",0.9)], container_id="cont1",
                                       client=_fake_report_client())
       assert rep.suppressed is True   # only 1 grounded edge < 3

   def test_community_detection_deterministic():
       # same seed/edges → same membership
       ...
   ```
2. **Fails.**
3. **Impl.**
   - `detect_communities`: build a networkx graph; run Leiden (networkx `community` / `python-louvain`-style with the
     Leiden refinement; resolution + `community_min_size` from `get_tunable`); log community count + size distribution
     via `log_gate_decision(container_id, "communities", ...)`. Drop communities below `community_min_size`.
   - `pagerank_grounded`: networkx `pagerank` with `weight` = edge confidence, damping from `get_tunable`.
   - `build_community_report`: summarize via the `gpt-4o-mini` extraction-client seam; attach drill-down citations
     `{chunk_id, bbox, src_chunk}` from member edges' `src_chunk`; set `suppressed=True` when
     `grounded_edge_count < get_tunable(container_id, "report_min_grounded_edges", ...)`. Reports route only — they are
     NEVER evidence-of-record (enforced downstream in Phase 3 synthesis; documented here).
4. **Passes.**
5. **Commit:** `feat(pdf-graph): Leiden communities, confidence-weighted PageRank, cited+suppressible reports`.

---

### Task 6 — Neo4j writer: new graph schema + indexes (`neo4j_writer.py`)

Add Entity/RELATED_TO/MENTIONS/NEXT_CHUNK/Community writes. Every node and edge carries `tenant_id`.

**Files:** `pdf_chat/ingestion/neo4j_writer.py`, `pdf_chat/testing/test_neo4j_writer_graph.py`

**Steps**

1. **Failing tests** (assert on the Cypher STRINGS — pure, no infra; mirror the helper-method style at
   `neo4j_writer.py:73-99`):
   ```python
   def test_relation_cypher_sets_provenance_and_tenant():
       from pdf_chat.ingestion.neo4j_writer import Neo4jWriter
       c = Neo4jWriter._write_relation_cypher()
       for prop in ("desc", "weight", "confidence", "evidence_count", "src_chunk"):
           assert prop in c
       assert "tenant_id: $tenant_id" in c

   def test_entity_cypher_carries_tenant_on_node():
       assert "tenant_id: $tenant_id" in Neo4jWriter._write_entity_cypher()

   def test_mentions_and_next_chunk_cypher_exist():
       assert ":MENTIONS" in Neo4jWriter._write_mentions_cypher()
       assert ":NEXT_CHUNK" in Neo4jWriter._write_next_chunk_cypher()
   ```
2. **Fails.**
3. **Impl.** Add static Cypher helpers + public methods from the signatures block:
   - `_write_entity_cypher()`: `MERGE (e:Entity {entity_id: $entity_id, tenant_id: $tenant_id}) SET e.name=$name, e.entity_type=$entity_type, e.normalized_value=$normalized_value, e.confidence=$confidence, e.acl=$acl`.
     `normalized_value` reuses `relationship_index.fingerprint_value`-style normalization (Phase-4 hook).
   - `_write_relation_cypher()`: MATCH both tenant-scoped entities, then
     `MERGE (a)-[r:RELATED_TO {src_chunk: $src_chunk, tenant_id: $tenant_id}]->(b) SET r.desc=$desc, r.weight=$weight, r.confidence=$confidence, r.evidence_count=$evidence_count`.
   - `_write_mentions_cypher()`, `_write_next_chunk_cypher()`, `_write_community_cypher()`, `_set_pagerank_cypher()`
     — all tenant-scoped on every node.
   - Public methods batch via `session.run` per the existing `write_chunks` loop pattern (`neo4j_writer.py:108-123`).
     `write_relations` accepts ONLY already-grounded relations (the gate runs upstream in `graph_construct`).
4. **Passes.**
5. **Commit:** `feat(pdf-graph): Neo4j writer for entities/relations/mentions/communities`.

---

### Task 7 — Rewrite searcher Cypher to the new schema + per-hop tenant (`neo4j_searcher.py`)

Resolves spec architect B2/B1: the searcher currently queries `(:Entity)-[:RELATED_TO*1..2]-(:Chunk)`
(`neo4j_searcher.py:51-59`) — wrong shape (Entity relates to Entity; Chunk MENTIONS Entity) and endpoint-only tenant filter.

**Files:** `pdf_chat/retrieval/neo4j_searcher.py`, `pdf_chat/testing/test_neo4j_searcher_v2.py`

**Steps**

1. **Failing tests** (assert on Cypher strings + behavior with a fake driver; no infra):
   ```python
   from pdf_chat.retrieval import neo4j_searcher as ns

   def test_graph_cypher_uses_mentions_and_related_to():
       c = ns._GRAPH_CYPHER
       assert ":MENTIONS" in c and ":RELATED_TO" in c

   def test_graph_cypher_filters_tenant_on_every_path_hop():
       c = ns._GRAPH_CYPHER
       assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in c

   def test_entity_neighbors_cypher_per_hop_tenant():
       assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in ns._ENTITY_NEIGHBORS_CYPHER

   def test_community_report_lookup_filters_tenant():
       assert "tenant_id = $tenant_id" in ns._COMMUNITY_REPORT_CYPHER
   ```
2. **Fails.**
3. **Impl.** Rewrite `_GRAPH_CYPHER` to anchor on `(e:Entity {name:$entity, tenant_id:$tenant_id})`, walk
   `MATCH path = (e)-[:RELATED_TO*1..2]->(:Entity)<-[:MENTIONS]-(c:Chunk)` and gate
   `WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id) AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)`.
   Add `_ENTITY_NEIGHBORS_CYPHER` (variable-length neighbor walk, same per-hop `ALL(...)` guard) and `entity_neighbors(...)`.
   Add `_COMMUNITY_REPORT_CYPHER` (vector ANN over `(:Community)` report embeddings, `WHERE c.tenant_id = $tenant_id`)
   and `community_report_lookup(...)`. Keep `vector_search`/`hybrid_search` as-is. Hop limit/`hops` default from
   `get_tunable`. Reuse `deserialize_acl` (`neo4j_searcher.py:62-84`).
4. **Passes.**
5. **Commit:** `refactor(pdf-graph): rewrite searcher Cypher to Entity/MENTIONS schema with per-hop tenant`.

---

### Task 8 — Construction orchestrator (`graph_construct.py`)

Wires extractor → grounding gate → resolver → writer → communities/PageRank/reports. Called by the Phase-1 finalization task.

**Files:** `pdf_chat/ingestion/graph_construct.py`, `pdf_chat/ingestion/__init__.py`, `pdf_chat/testing/test_graph_extraction.py`

**Steps**

1. **Failing test** (mock writer + client; assert the gate is applied and idempotency holds):
   ```python
   def test_construct_only_writes_grounded_relations():
       # extractor yields 1 grounded + 1 ungrounded relation; writer.write_relations receives 1
       ...
   def test_construct_writes_next_chunk_in_reading_order():
       ...
   ```
2. **Fails.**
3. **Impl.** `construct_knowledge_graph(doc_id, container_id, tenant_id, chunks, writer=None, client=None)`:
   for each chunk call `extract_chunk` → collect entities/relations → `filter_grounded(relations, container_id)` (BLOCKING)
   → `resolve_entities(...)` (embeddings via `embed_texts` from `embeddings.py:35`) → `writer.write_entities`,
   `writer.write_relations` (grounded only), `writer.write_mentions`, `writer.write_next_chunk(ordered ids, tenant_id)`
   → `detect_communities` + `pagerank_grounded` + `build_community_report` per community →
   `writer.write_communities` + `writer.set_pagerank`. Return a summary dict
   `{entities, grounded_edges, rejected_edges, merges, communities, suppressed_reports}` and log it. Never raise on a
   single bad chunk (degrade + log, mirroring the dashboard "never raise" rule).
4. **Passes.**
5. **Commit:** `feat(pdf-graph): KG construction orchestrator with blocking grounding gate`.

---

### Task 9 — Phase-2 exit gate: tenant-isolation + faithfulness eval

Spec §5 Phase 2 exit: per-hop tenant isolation verified + faithfulness eval (edge precision vs source, merge
precision/recall, report groundedness). Pure + marker-gated infra parts.

**Files:** `pdf_chat/testing/test_neo4j_searcher_v2.py` (extend), `pdf_chat/testing/test_phase2_exit_gate.py` (NEW)

**Steps**

1. **Failing tests:**
   - PURE: assert EVERY Cypher constant in `neo4j_searcher.py` and `neo4j_writer.py` that contains a variable-length path
     (`*1..`) also contains `ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)` (scan the module source for `*1..`).
   - PURE: `edge_precision(predicted, gold)`, `merge_precision_recall(predicted, gold)`,
     `report_groundedness(reports)` helpers compute against a small fixture set and assert they pass thresholds read from
     `get_tunable` (no literal in the test or impl).
   - `@pytest.mark.neo4j`: cross-tenant traversal returns zero rows (real driver, skipped without infra).
2. **Fails.**
3. **Impl.** Add `pdf_chat/ingestion/graph_eval.py` with the pure metric helpers; wire the thresholds through `get_tunable`;
   the marker-gated infra test seeds two tenants and asserts isolation.
4. **Passes:** `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -q` (pure) and `... -m neo4j` (infra, when available).
5. **Commit:** `test(pdf-graph): phase-2 exit gate — tenant isolation + faithfulness eval`.

---

## Verification (run before claiming complete)

```bash
cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -q                 # all pure tests green
cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -m "not neo4j and not llm" -q
grep -rn "0\.\|>=\|<=\| < \| > " server/pdf_chat/ingestion/grounding_gate.py \
  server/pdf_chat/ingestion/entity_resolver_graph.py server/pdf_chat/ingestion/communities.py
# ^ inspect: every comparison must be against a get_tunable value, never a literal (invariant 4)
```

Confirm: no score-comparison literal in any new `.py`; every gate logs via `log_gate_decision`; every Cypher with a
variable-length path carries `ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)`.

# Implementation Plan — PDF Agentic Graph RAG · Phase 2: Section-level, Grounded, Multi-Representation Knowledge Graph

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (or `superpowers:executing-plans`). Execute tasks **in order**; each is a TDD cycle (write failing test → run → minimal impl → run → commit). **Do NOT `git commit` from a worker — the delivery manager commits after the reviewer-board gate.** Keep the existing 216 `pdf_chat/testing/` tests green. The INDEX (`2026-06-03-pdf-agentic-graph-rag-INDEX.md`) is authoritative where this plan disagrees.

**Date:** 2026-06-05 (regenerates the SUPERSEDED 2026-06-03 per-chunk plan)
**Spec:** `docs/superpowers/specs/2026-06-03-pdf-agentic-graph-rag-design.md` v3 — §1b (granularity dial, SECTION default; NER/value-overlap no-LLM backbone; multi-representation index; grounded tags as retrieval signal; misleading-tag safeguard), §3 invariants 1/3/4/5/6/7, §5 router (escalation OFF for bulk), §8 cost.
**Module:** `server/pdf_chat/`
**Status:** DRAFT — planning author only. This file is a plan, not an implementation.

---

## Goal

Make the PDF graph leg real, grounded, and cost-driven. Today the writer persists only `(:Document)-[:CONTAINS]->(:Page)-[:CONTAINS]->(:Chunk)` (`server/pdf_chat/ingestion/neo4j_writer.py:84-111`); the searcher queries a `(:Entity)-[:RELATED_TO]` shape nothing writes (`server/pdf_chat/retrieval/neo4j_searcher.py:51-59`, the DEAD `_GRAPH_CYPHER`); `select_model(task=TaskClass.EXTRACTION, ...)` (`server/pdf_chat/model_router.py:216`) has no call site.

Phase 2 builds, per spec §1b:
1. **Sectionizer** — group a doc's chunks into SECTIONS (layout/reading-order, tunable; degrade to page-grouping). A section is the LLM extraction unit.
2. **Section-level extraction** (the `TaskClass.EXTRACTION` call site) — one structured `gpt-4o-mini` call per section emits open-vocab entities, relations, and grounded tags (1 doc-level relational tag + small set of section topic tags), each with confidence + supporting span. Prompt-cached, idempotent on `section_fingerprint + prompt/model version`, adaptive capped gleaning, **escalation OFF for bulk** (`select_model` is called with `signals={}` so the bulk id is always returned; this is asserted by test).
3. **No-LLM backbone** — guarded spaCy NER proposes entity candidates (degrade gracefully when absent); value-overlap/co-reference (reusing the `fingerprint_value` concept from `server/app/services/relationship_index.py:71`) proposes links. The LLM only confirms/names/relates.
4. **Grounding gate (blocking)** — every edge AND every tag persists `src_chunk_id` + verbatim span + confidence; reject any whose subject/object/predicate (or tag claim) is absent from the cited span. Mirrors the value-overlap + `edge_provenance` gate in `server/app/services/relationship_detector.py:173-211`.
5. **Entity resolution** — embedding + type agreement + co-occurrence; unmerged-by-default for ambiguous; bands derived per-container from the score distribution (NOT literals); persist merge decision + evidence; no transitive auto-merge below a confidence floor. Open-vocab types mirror `custom:<kind>:<slug>` (`server/app/services/semantic_roles.py:82`).
6. **Neo4j schema** — every node carries `tenant_id`; per-hop isolation `ALL(n IN nodes(path) WHERE n.tenant_id=$tenant_id)`. `(:Entity)-[:RELATED_TO {desc,weight,confidence,evidence_count,src_chunk}]->(:Entity)`, `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:Section)-[:HAS_CHUNK]->(:Chunk)`, `(:Entity)-[:IN_COMMUNITY]->(:Community)`, tag nodes/props, `Entity.normalized_value` (Phase-4 bridge).
7. **Multi-representation index** — embed (via `embed_texts_batched` + `embedding_model`) chunks AND section-cards (summary+tags) AND doc-cards (doc tag/summary); store as vector-indexed nodes.
8. **Searcher rewrite** (`retrieval/neo4j_searcher.py`, owns C2) — `graph_traversal`, `entity_neighbors`, `community_report_lookup`, `multi_vector_search` (chunk + section-card + doc-card spaces, RRF-fused via `retrieval/rrf.py`). All per-hop tenant-isolated.
9. **Leiden communities** (networkx; resolution/min-size tunable+logged) + **cited** community reports (`gpt-4o-mini`; suppress reports not traceable to ≥N grounded edges) + **PageRank** over grounded edges weighted by confidence.
10. **Misleading-tag safeguard** — tags are a retrieval signal with confidence; a tag never becomes an answer without a grounded supporting chunk (asserted by test).
11. **Phase-2 EXIT GATE** — per-hop tenant isolation verified + faithfulness eval (edge precision vs span, merge P/R, report groundedness, tag groundedness) passing INCLUDING on a held-out tenant the system has never seen.

---

## Architecture

```
chunks (Phase 1, in Neo4j as Chunk nodes)
  │
  ▼ [T2] Sectionizer  ── layout/reading-order grouping (tunable) → Section{section_id, chunk_ids, fingerprint}
  │                       degrade: page-grouping when no headings
  ▼ [T3] NER backbone ── spaCy guarded → EntityCandidate[]   (degrades to empty list, no crash)
  ▼ [T4] value-overlap ── fingerprint_value co-occurrence → LinkCandidate[]
  ▼ [T5] SectionExtractor.extract(section) ── select_model(EXTRACTION, signals={}) → gpt-4o-mini
  │        one structured JSON call → ExtractedEntity[], ExtractedRelation[], ExtractedTag[]  (each: confidence + span)
  │        idempotent on section_fingerprint + prompt/model version; adaptive capped gleaning
  ▼ [T6] GroundingGate.admit(edge|tag, cited_chunk_text) ── verbatim-span check → admit | reject
  ▼ [T7] EntityResolver.resolve(...) ── embed + type + co-occur; per-container bands; unmerged-by-default
  ▼ [T8] Neo4jKGWriter ── Entity/Section/RELATED_TO/MENTIONS/HAS_CHUNK/Tag (tenant_id everywhere)
  ▼ [T9] CardBuilder ── SectionCard + DocCard text → embed_texts_batched → vector-indexed nodes
  ▼ [T10] Communities (Leiden) + [T11] cited reports + [T12] PageRank(confidence-weighted)
  ▼ [T1] Neo4jSearcher rewrite: graph_traversal / entity_neighbors / community_report_lookup / multi_vector_search(RRF 3 spaces)
  ▼ [T13] EXIT GATE: tenant-isolation cypher audit + faithfulness eval (held-out tenant)
```

## Tech Stack
Python 3.12 · `uv` · pytest/pytest-asyncio (not a project dep — `--with`) · Neo4j (guarded import, pure-testable seams) · networkx (Leiden via `networkx.community` / `python-louvain` fallback, guarded) · spaCy (guarded) · Azure OpenAI `gpt-4o-mini` (bulk) + `text-embedding-3-small`. **No score-comparison literal in any `.py`** — all via `get_tunable` + `log_gate_decision`. Reuse: `tunables.py`, `model_router.py`, `retrieval/embeddings.py`, `retrieval/rrf.py`, `ingestion/neo4j_writer.py`, `ingestion/ton_schema.py`.

---

## File Structure (new / modified)

```
server/pdf_chat/
  ingestion/
    sectionizer.py          NEW  Section dataclass + sectionize()
    kg_extraction.py        NEW  ExtractedEntity/Relation/Tag + SectionExtractor + section_fingerprint
    ner_backbone.py         NEW  EntityCandidate + propose_entities() (guarded spaCy) + propose_links()
    grounding_gate.py       NEW  GroundedEdge/GroundedTag + GroundingGate.admit()
    entity_resolution.py    NEW  ResolvedEntity + MergeDecision + EntityResolver
    kg_writer.py            NEW  Neo4jKGWriter (Entity/Section/RELATED_TO/MENTIONS/HAS_CHUNK/Tag/Card)
    card_builder.py         NEW  SectionCard/DocCard + build_section_card/build_doc_card
    communities.py          NEW  detect_communities (Leiden) + pagerank_confidence + CommunityReporter
  retrieval/
    neo4j_searcher.py       MOD  rewrite graph cypher; add entity_neighbors/community_report_lookup/multi_vector_search
  testing/
    test_sectionizer.py            NEW
    test_kg_extraction.py          NEW
    test_ner_backbone.py           NEW
    test_grounding_gate.py         NEW
    test_entity_resolution.py      NEW
    test_kg_writer.py              NEW
    test_card_builder.py           NEW
    test_communities.py            NEW
    test_kg_searcher.py            NEW
    test_phase2_exit_gate.py       NEW
```

New tunable keys added to `TUNABLE_DEFAULTS` in `tunables.py` (single source, no inline literal): `kg.extraction.granularity` (`"section"`), `kg.gleaning.max_passes` (2), `kg.gleaning.new_entity_floor` (1), `kg.resolution.merge_band_quantile` (0.85), `kg.resolution.merge_floor` (0.60), `kg.community.resolution` (1.0), `kg.community.min_size` (3), `kg.report.min_grounded_edges` (2), `kg.tag.min_confidence` (0.50), `kg.multivec.top_k` (12).

---

## TDD Tasks

> Test command (run from `server/` for every task):
> `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q -p no:cacheprovider`

---

### Task 1 — Searcher rewrite: kill the dead graph cypher, add per-hop tenant isolation (C2)

**1a. Failing test** — `pdf_chat/testing/test_kg_searcher.py`:
```python
import re
from pdf_chat.retrieval import neo4j_searcher as S

def test_graph_cypher_uses_per_hop_tenant_isolation():
    cy = S._GRAPH_CYPHER
    # every node on the matched path must be tenant-filtered (spec inv 3)
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cy
    assert ":MENTIONS" in cy and ":RELATED_TO" in cy

def test_no_score_literal_in_searcher_source():
    import inspect
    src = inspect.getsource(S)
    # no bare float comparison literal — thresholds come from get_tunable
    assert not re.search(r"score\s*[<>]=?\s*0\.\d", src)
```
Run: `... pytest pdf_chat/testing/test_kg_searcher.py -q -p no:cacheprovider` → fails (`ALL(n IN nodes(path)...` not present).

**1b. Minimal impl** — rewrite `_GRAPH_CYPHER` in `neo4j_searcher.py`:
```python
_GRAPH_CYPHER = """
MATCH path = (e:Entity {name: $entity})-[:RELATED_TO*1..2]-(other:Entity)
WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)
MATCH (c:Chunk)-[:MENTIONS]->(other)
WHERE c.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)
RETURN c.chunk_id AS chunk_id, c.text AS text, c.doc_id AS doc_id,
       c.page_num AS page_num, c.element_type AS element_type, c.acl AS acl
LIMIT $limit
"""
```
Run → passes.

**1c.** Commit message (manager applies): `pdf phase2: rewrite graph cypher to per-hop tenant isolation`.

---

### Task 2 — Sectionizer

**2a. Failing test** — `test_sectionizer.py`: build 3 `Chunk`s with `reading_order` + a heading marker; assert `sectionize(chunks, container_id="t1")` returns ≥1 `Section`, each `section_id` deterministic, `chunk_ids` non-empty; assert a no-heading input degrades to page-grouping (one section per `page_num`); assert `kg.extraction.granularity` read via `get_tunable`.

**2b. Impl** — `sectionizer.py`:
```python
@dataclass(frozen=True)
class Section:
    section_id: str          # f"{doc_id}::s{ordinal}"
    doc_id: str
    tenant_id: str
    chunk_ids: list[str]
    text: str                # concatenated chunk text (the LLM unit)
    fingerprint: str         # sha256(model-stable section text)
    page_span: tuple[int, int]

def sectionize(chunks: list[Chunk], *, container_id: str) -> list[Section]: ...
```
Group by detected heading boundaries (reading-order discontinuity / element_type signal), degrade to page-grouping; `fingerprint = sha256("\n".join(texts))[:16]`; log grouping via `log_gate_decision("kg.sectionize", ...)`.
**2c.** Commit: `pdf phase2: sectionizer (section = LLM extraction unit)`.

---

### Task 3 — NER backbone (guarded, degrades gracefully)

**3a. Failing test** — `test_ner_backbone.py`: monkeypatch `_HAS_SPACY=False`; assert `propose_entities("Acme signed with Globex", container_id="t1") == []` (no crash); with a fake nlp injected, assert candidates carry `text` + `label`.

**3b. Impl** — `ner_backbone.py`:
```python
@dataclass(frozen=True)
class EntityCandidate:
    text: str
    label: str          # spaCy ent label OR ""
    source: str         # "ner" | "value_overlap"

def propose_entities(text: str, *, container_id: str, nlp=None) -> list[EntityCandidate]: ...
def propose_links(sections: list[Section], *, container_id: str) -> list[tuple[str, str, str]]:
    # value-overlap via fingerprint_value concept → (entity_a, entity_b, evidence)
```
Guard: `try: import spacy; _HAS_SPACY=True except ImportError: _HAS_SPACY=False`. When absent return `[]`.
**3c.** Commit: `pdf phase2: NER + value-overlap backbone (guarded)`.

---

### Task 4 — Extraction dataclasses + section_fingerprint idempotency

**4a. Failing test** — `test_kg_extraction.py`: assert `section_fingerprint(section, prompt_version, model_id)` is stable across calls and changes when `model_id` changes; assert `ExtractedEntity/Relation/Tag` carry `confidence: float` and `span: str`.

**4b. Impl** — `kg_extraction.py`:
```python
@dataclass(frozen=True)
class ExtractedEntity: name: str; etype: str; confidence: float; span: str; src_chunk_id: str
@dataclass(frozen=True)
class ExtractedRelation: subject: str; predicate: str; obj: str; confidence: float; span: str; src_chunk_id: str
@dataclass(frozen=True)
class ExtractedTag: label: str; scope: str; confidence: float; span: str; src_chunk_id: str  # scope: "doc"|"section"

def section_fingerprint(section: Section, prompt_version: str, model_id: str) -> str:
    return hashlib.sha256(f"{section.fingerprint}|{prompt_version}|{model_id}".encode()).hexdigest()[:24]
```
**4c.** Commit: `pdf phase2: extraction dataclasses + fingerprint idempotency key`.

---

### Task 5 — SectionExtractor (the EXTRACTION call site; escalation OFF for bulk)

**5a. Failing test** — `test_kg_extraction.py`: inject a fake LLM returning a fixed JSON payload AND a spy `select_model`. Assert:
- `extract` calls `select_model(task=TaskClass.EXTRACTION, container_id=..., signals={})` and uses the returned (bulk) `model_id` — **assert escalation is NOT invoked** (the spy records `is_strong is False`).
- default granularity is `"section"` (the extractor consumes `Section`, one LLM call per section, NOT per chunk).
- a second `extract` with the same `section_fingerprint` returns the cached result WITHOUT a second LLM call (idempotency).
- gleaning stops at `kg.gleaning.max_passes`.

**5b. Impl** — `kg_extraction.py`:
```python
class SectionExtractor:
    PROMPT_VERSION = "p2.v1"
    def __init__(self, llm, *, cache=None): self._llm=llm; self._cache=cache
    def extract(self, section: Section, *, container_id: str
               ) -> tuple[list[ExtractedEntity], list[ExtractedRelation], list[ExtractedTag]]:
        choice = select_model(task=TaskClass.EXTRACTION, container_id=container_id, signals={})  # bulk-only
        fp = section_fingerprint(section, self.PROMPT_VERSION, choice.model_id)
        if self._cache and (hit := self._cache.get(fp)) is not None:
            log_gate_decision("kg.extract.cache", score=1, threshold=1, outcome="hit", container_id=container_id)
            return hit
        ... # adaptive gleaning loop capped at get_tunable(container_id,"kg.gleaning.max_passes")
```
`signals={}` guarantees `escalation_allowed` is False → bulk `gpt-4o-mini` always. Prompt is prompt-cached (system block stable).
**5c.** Commit: `pdf phase2: section-level extractor (bulk gpt-4o-mini, idempotent, capped gleaning)`.

---

### Task 6 — Grounding gate (blocking; rejects ungrounded edge AND ungrounded tag)

**6a. Failing test** — `test_grounding_gate.py`:
```python
def test_rejects_edge_absent_from_span():
    g = GroundingGate()
    rel = ExtractedRelation("Acme","acquired","Globex",0.9,"Acme bought Beta Corp in 2025","c1")
    assert g.admit_edge(rel, cited_text="Acme bought Beta Corp in 2025", container_id="t1") is None  # "Globex" not in span
def test_admits_grounded_edge():
    rel = ExtractedRelation("Acme","acquired","Globex",0.9,"Acme acquired Globex","c1")
    assert GroundingGate().admit_edge(rel, cited_text="Acme acquired Globex", container_id="t1") is not None
def test_rejects_tag_absent_from_span():
    tag = ExtractedTag("describes Product Z","doc",0.8,"discusses Product Y","c1")
    assert GroundingGate().admit_tag(tag, cited_text="discusses Product Y", container_id="t1") is None
```

**6b. Impl** — `grounding_gate.py`:
```python
@dataclass(frozen=True)
class GroundedEdge: subject:str; predicate:str; obj:str; confidence:float; span:str; src_chunk_id:str; evidence_count:int
@dataclass(frozen=True)
class GroundedTag: label:str; scope:str; confidence:float; span:str; src_chunk_id:str

class GroundingGate:
    def admit_edge(self, rel, *, cited_text, container_id) -> GroundedEdge | None:
        ok = all(_norm(t) in _norm(cited_text) for t in (rel.subject, rel.obj))  # predicate-claim presence
        log_gate_decision("kg.ground.edge", score=1.0 if ok else 0.0, threshold=1.0,
                          outcome="admit" if ok else "reject", container_id=container_id)
        return GroundedEdge(...) if ok else None
    def admit_tag(self, tag, *, cited_text, container_id) -> GroundedTag | None: ...  # tag claim must be in span
```
`_norm` = lowercase + whitespace-collapse. The `score>=threshold` test runs through `log_gate_decision` (no bare literal).
**6c.** Commit: `pdf phase2: blocking grounding gate (edges + tags)`.

---

### Task 7 — Entity resolution (per-container bands, unmerged-by-default)

**7a. Failing test** — `test_entity_resolution.py`: feed candidate pairs with similarity scores; assert merge band is derived from the score distribution (`kg.resolution.merge_band_quantile`) NOT a literal; assert an ambiguous pair below the per-container band stays unmerged; assert no transitive auto-merge below `kg.resolution.merge_floor`; assert each `MergeDecision` carries `evidence`.

**7b. Impl** — `entity_resolution.py`:
```python
@dataclass(frozen=True)
class MergeDecision: kept: str; merged: str; score: float; band: float; evidence: dict; merged_now: bool
@dataclass(frozen=True)
class ResolvedEntity: name: str; etype: str; normalized_value: str | None; aliases: list[str]

class EntityResolver:
    def resolve(self, entities, *, embed_fn, container_id) -> tuple[list[ResolvedEntity], list[MergeDecision]]:
        # band = quantile(score_dist, get_tunable(container_id,"kg.resolution.merge_band_quantile"))
        # merge only if score>=band AND type agreement; ambiguous→unmerged; no transitive below floor
```
`normalized_value` reuses `fingerprint_value` concept for the Phase-4 bridge.
**7c.** Commit: `pdf phase2: entity resolution (per-container bands, unmerged-by-default)`.

---

### Task 8 — Neo4j KG writer (schema; tenant_id on every node/edge)

**8a. Failing test** — `test_kg_writer.py`: with a fake driver/session capturing cypher, assert the Entity-write cypher MERGEs on `(name, tenant_id)`, sets `Entity.normalized_value`, and that `RELATED_TO` carries `desc,weight,confidence,evidence_count,src_chunk`; assert `(:Section)-[:HAS_CHUNK]->(:Chunk)` and `(:Chunk)-[:MENTIONS]->(:Entity)` and tag write all bind `$tenant_id`.

**8b. Impl** — `kg_writer.py` (`Neo4jKGWriter`, same guarded-driver pattern as `neo4j_writer.py:48-58`): helper cypher methods `_entity_cypher`, `_related_to_cypher`, `_mentions_cypher`, `_section_cypher`, `_tag_cypher` — each MERGE keyed on `(<key>, tenant_id)`, each setting grounding props from `GroundedEdge`/`GroundedTag`.
**8c.** Commit: `pdf phase2: Neo4j KG writer (entities/relations/mentions/sections/tags, tenant-scoped)`.

---

### Task 9 — Multi-representation cards + multi_vector_search (RRF over 3 spaces)

**9a. Failing test** — `test_card_builder.py`: assert `build_section_card(section, tags)` text includes the section summary + tag labels; `build_doc_card(doc_id, doc_tags)` includes doc-level tag; assert cards embed via `embed_texts_batched(..., container_id=...)`.
`test_kg_searcher.py`: with a fake searcher whose 3 vector legs return fixed id-lists, assert `multi_vector_search(query_vec, tenant_id)` fuses chunk + section-card + doc-card rankings via `rrf` (mock `rrf` and assert it is called with **3** lists).

**9b. Impl** — `card_builder.py`:
```python
@dataclass(frozen=True)
class SectionCard: card_id:str; section_id:str; tenant_id:str; text:str; embedding:list[float]|None=None
@dataclass(frozen=True)
class DocCard: card_id:str; doc_id:str; tenant_id:str; text:str; embedding:list[float]|None=None
def build_section_card(section, tags, *, container_id) -> SectionCard: ...
def build_doc_card(doc_id, doc_tags, *, tenant_id, container_id) -> DocCard: ...
```
`neo4j_searcher.py` add (per-hop tenant in every leg):
```python
def entity_neighbors(self, entity, tenant_id, limit=None, doc_ids=None) -> list[dict]: ...
def community_report_lookup(self, query_vec, tenant_id, limit=None) -> list[dict]: ...   # cited reports only
def multi_vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None) -> list[dict]:
    chunk = self.vector_search(query_vec, tenant_id, top_k, doc_ids)        # SectionCard + DocCard indices
    sec   = self._card_vector_search(query_vec, tenant_id, "section_card_vector_index", top_k, doc_ids)
    doc   = self._card_vector_search(query_vec, tenant_id, "doc_card_vector_index", top_k, doc_ids)
    fused = rrf([_ids(chunk), _ids(sec), _ids(doc)], k=get_pdf_settings().rrf_k)
    ...
```
`_card_vector_search` cypher: `... WHERE node.tenant_id = $tenant_id ...`.
**9c.** Commit: `pdf phase2: multi-representation cards + multi_vector_search (RRF over 3 spaces)`.

---

### Task 10 — Communities (Leiden) + PageRank (confidence-weighted)

**10a. Failing test** — `test_communities.py`: build a small grounded-edge list; assert `detect_communities(edges, container_id="t1")` reads `kg.community.resolution`+`kg.community.min_size` via `get_tunable` and drops communities below min_size; assert `pagerank_confidence(edges)` weights by edge `confidence` (a high-confidence hub ranks above a low-confidence one); guard: networkx absent → returns `[]` (no crash).

**10b. Impl** — `communities.py`:
```python
def detect_communities(edges: list[GroundedEdge], *, container_id: str) -> list[Community]: ...
def pagerank_confidence(edges: list[GroundedEdge]) -> dict[str, float]: ...   # weight=confidence
```
Guarded `import networkx`; Leiden via `networkx.community.greedy_modularity_communities` (or `python-louvain` fallback), resolution/min-size logged via `log_gate_decision`.
**10c.** Commit: `pdf phase2: Leiden communities + confidence-weighted PageRank`.

---

### Task 11 — Cited community reports + misleading-tag safeguard

**11a. Failing test** — `test_communities.py`: inject fake LLM; assert `CommunityReporter.report(community, edges, container_id)` is SUPPRESSED (returns None) when the community traces to fewer than `kg.report.min_grounded_edges` grounded edges; assert a produced report carries `citations` (chunk_ids). `test_grounding_gate.py`: assert a `GroundedTag` alone (no supporting grounded chunk) does NOT surface as an answer claim — `tag_as_answer(tag, supporting_chunks=[])` returns None; with a supporting chunk it returns the claim (the misleading-tag safeguard).

**11b. Impl** — `CommunityReporter` in `communities.py` (`select_model(task=TaskClass.SYNTHESIS, signals={})` → bulk); `tag_as_answer(...)` helper in `grounding_gate.py`.
**11c.** Commit: `pdf phase2: cited community reports + misleading-tag safeguard`.

---

### Task 12 — Phase-2 EXIT GATE (tenant-isolation audit + faithfulness eval incl. held-out tenant)

**12a. Failing test** — `test_phase2_exit_gate.py`:
- `test_every_traversal_cypher_is_per_hop_tenant_isolated`: introspect all cypher constants in `neo4j_searcher.py` + `kg_writer.py`; assert each MATCH binds `$tenant_id` (path-level `ALL(n IN nodes(path)...)` for multi-hop).
- `test_faithfulness_eval_held_out_tenant`: run the gate over a fixture corpus for tenant `T_seen` AND a never-seen tenant `T_holdout`; assert edge-precision-vs-span, merge P/R, report-groundedness, tag-groundedness all ≥ their tunable floors for BOTH tenants (no per-tenant code path).

**12b. Impl** — `testing/test_phase2_exit_gate.py` only (no new prod module): a deterministic fixture corpus + assertions wiring T2–T11 seams together with mocked LLM/NER/Neo4j.
**12c.** Commit: `pdf phase2: exit gate — tenant-isolation audit + faithfulness eval (held-out tenant)`.

---

## Reviewer-board gate
Route the full diff through the board (architect, data-science, business-analyst, static-code-sentinel) under the delivery-manager. Phase-2 gate additionally requires the held-out-tenant faithfulness eval (Task 12) green. **Manager commits after sign-off; workers do not commit.**

# PDF Phase 5 — Comprehension Layer ("Superhuman Memory") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is the **superhuman-memory payload** (roadmap Phase 5). It depends on Phases 0–3 (and consumes, but does not require, Phase 4). Where this file disagrees with the INDEX Cross-Phase Contract, **the INDEX wins**.

**Goal:** Turn the grounded Phase-2 Neo4j graph into a per-tenant, browsable, versioned **comprehension artifact** so a domain-naive engineer is productive: (1) a Postgres-backed **Tenant Ontology** (entities, relationships, open-vocab document taxonomy, temporal coverage, key metrics) mirroring `app/services/semantic_layer_builder.py`; (2) a **corpus-learned glossary** mined at ingest from three grounded signals (LLM-confirmed explicit definitions, distributional anomaly, co-reference variants), each entry carrying `provenance: stated|inferred` + evidence spans; (3) a `glossary_lookup` **Tool** ("what does X mean here" → expansion + definition + citation + provenance) plus a transparent tenant-scoped query-expansion helper; (4) **faithfulness** surfacing (provenance labels, staleness, conflict-both-sides); (5) a read-only **onboarding surface** (topic map, entity browse, glossary browse, doc taxonomy). EXIT: browse the topic map, ask "what does X mean here", get cited company-specific answers; ontology version is queryable.

**Architecture:** Mirror the structured side exactly (spec §0 principle 3, invariant 6). The **INTENT layer is fixed code**; all domain meaning is **learned per tenant as data**. The glossary mirrors `app/services/column_role_resolver.py` (LLM-only classification; "glossary" is a *signal*, not a heuristic dictionary — `column_role_resolver.py:37,65`) and `app/services/semantic_roles.py` (open-vocabulary `custom:<kind>:<slug>`; empty `_BASE_ROLE_SPECS` at `semantic_roles.py:45`). The ontology mirrors `app/services/semantic_layer_builder.py:1-66` (build registries from a graph substrate as a queryable object, versioned). Doc-taxonomy classes are **open-vocab learned** (never an enumerated list), exactly as roles are minted dynamically. The artifact is built **once at ingest finalization** (intelligence-at-ingest, spec §0 principle 1) and re-versioned on re-ingestion. Every glossary entry, ontology relationship row, and synthesized definition is **grounded or refused** (spec §3 invariant 1/6; faithfulness §4). Conflicts are **three-state** (asserted/not-stated/conflicting) and surfaced with both sides + recency, never silently resolved (invariant 7).

**Tech Stack:** Python 3.12, async SQLAlchemy 2.0 on the app's shared `Base` (same convention as `pdf_chat/models/manifests.py:19,32`), tenant-scoped by `container_id`/`tenant_id`. LLM via the Phase-0 model router (`pdf_chat/model_router.py: select_model`, bulk = `gpt-4o-mini`). All thresholds from `pdf_chat/tunables.py` (`get_tunable`/`log_gate_decision`) — **no score-comparison literal in any `.py`** (invariant 4). Neo4j read-only (per-hop tenant isolation; consume Phase-2 `Neo4jSearcher`, do not redefine). FastAPI router self-prefixed (mirror `pdf_chat/api/routes.py`). Tests: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q`, in-memory fakes (mock LLM/graph/searcher), matching `pdf_chat/testing/test_agent.py`.

---

## Assumptions & Cross-Phase Contracts (depend on signatures abstractly)

**Phase 0 (assume exists) — C1/C7:**
- `pdf_chat/tunables.py`: `get_tunable(name: str, container_id: str | None = None) -> float | int`; `log_gate_decision(gate: str, *, score: float, threshold: float, passed: bool, **fields) -> None`.
- `pdf_chat/model_router.py`: `select_model(*, task: str, container_id: str, signals: dict) -> ModelChoice` (`.model_id`, `.is_strong`); bulk task ⇒ `gpt-4o-mini`. We pass `task="glossary"` / `task="ontology"` (bulk). We never request a strong tier for bulk mining.

**Phase 2 (assume exists, abstract signatures) — C2/C6:**
- Neo4j graph: `(:Entity {name, normalized_value, type, tenant_id, pagerank?})`, `(:Entity)-[:RELATED_TO {desc, weight, confidence, evidence_count, src_chunk}]->(:Entity)`, `(:Chunk {chunk_id, text, doc_id, page_num, bbox})-[:MENTIONS]->(:Entity)`, `(:Entity)-[:IN_COMMUNITY]->(:Community {community_id, tenant_id, report, level})`, `(:Document {doc_id, tenant_id, title?, created_at?, doc_date?})`.
- A read interface we depend on abstractly as `GraphReader` (implemented by Phase-2 `Neo4jSearcher`): `async iter_entities(tenant_id)`, `async iter_relationships(tenant_id)`, `async iter_communities(tenant_id)`, `async iter_documents(tenant_id)`, `async iter_chunks(tenant_id)`, `async entity_chunks(tenant_id, entity_name)`. We **inject** a `GraphReader` (Protocol) so tests pass a fake; production wires the searcher. Per-hop tenant isolation is the searcher's responsibility (C2).

**Phase 3 (assume exists, abstract) — C3/C4:**
- `agent/tools.py`: `Tool` Protocol (`name: str`; `async run(state, deps, **kw) -> list[dict]`), `TOOL_REGISTRY`, `register_tool(tool)`. `glossary_lookup` implements `Tool` and registers via `register_tool` (never stores a raw LangChain tool).
- `agent/graph.py :: run_pdf_query(query, *, tenant_id, container_id, ...)` → result with `.answer`, `.citations`. The onboarding API and (optionally) a definitional path depend on it.
- The `definitional` intent exists (Phase 3); its **verbatim-span** requirement is satisfied by our glossary `evidence_spans`. We expose `glossary_lookup` as the tool the definitional planner branch calls; we **do not** redefine the intent.

**Phase 4 (optional) — C5:** not required. The onboarding surface and glossary are PDF-corpus learned; no cross-domain dependency.

---

## File Structure (NEW unless noted)

```
server/pdf_chat/
├── models/
│   └── comprehension.py          # NEW ORM: TenantOntology, OntologyEntity, OntologyRelationship,
│                                 #          DocTaxonomyClass, TemporalCoverage, KeyMetric, GlossaryEntry
├── migrations/
│   └── comprehension_upgrade.py  # NEW additive, idempotent, non-fatal (mirror control_plane_upgrade.py)
├── comprehension/
│   ├── __init__.py               # NEW
│   ├── provenance.py             # NEW Provenance enum + label mapping (stated|inferred|conflicting|not_found)
│   ├── glossary_miner.py         # NEW mine_glossary(): 3 grounded signals → GlossaryEntry rows
│   ├── ontology_builder.py       # NEW build_tenant_ontology(): GraphReader → versioned artifact
│   ├── temporal.py               # NEW compute_temporal_coverage() + staleness_annotation()
│   └── reader.py                 # NEW GraphReader Protocol + read-only ontology/glossary query helpers
├── agent/
│   └── tools_glossary.py         # NEW glossary_lookup Tool + expand_query() helper (registers via register_tool)
├── api/
│   └── onboarding.py             # NEW read-only projection router (self-prefixed /api/pdf/onboarding)
└── testing/
    ├── test_comprehension_ontology.py   # NEW
    ├── test_comprehension_glossary.py   # NEW
    ├── test_comprehension_tool.py       # NEW
    └── test_comprehension_onboarding.py # NEW
```

Background-frequency table for distributional anomaly ships as **data** (not a `.py` dict): `pdf_chat/comprehension/background_freq.json` (a generic English/business unigram log-frequency table). Loaded at runtime; absence ⇒ signal degrades gracefully (anomaly disabled, logged), never a hardcoded jargon list.

---

## TDD Tasks

Each task: **failing test → run (fails) → minimal impl → run (passes) → commit.** All commands run from `server/`. Conventional commits end with the Co-Authored-By trailer.

### Task 1 — Provenance vocabulary (faithfulness labels, no literals)

- [ ] **Write failing test** `pdf_chat/testing/test_comprehension_glossary.py::test_provenance_labels`:
```python
import pytest
from pdf_chat.comprehension.provenance import Provenance, label_for

def test_provenance_labels():
    assert Provenance.STATED.value == "stated"
    assert Provenance.INFERRED.value == "inferred"
    # Human-facing labels, NOT raw confidence numbers (spec §4).
    assert label_for(Provenance.STATED) == "stated in docs"
    assert label_for(Provenance.INFERRED) == "inferred from usage"
    assert label_for(Provenance.CONFLICTING) == "conflicting sources"
    assert label_for(Provenance.NOT_FOUND) == "not found"
```
- [ ] **Run (fails):** `uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_comprehension_glossary.py -q`
- [ ] **Minimal impl** `pdf_chat/comprehension/provenance.py`:
```python
from __future__ import annotations
from enum import Enum

class Provenance(str, Enum):
    STATED = "stated"          # explicit definition confirmed in a cited span
    INFERRED = "inferred"      # distributional/co-reference signal only
    CONFLICTING = "conflicting"  # ≥2 incompatible definitions, both surfaced
    NOT_FOUND = "not_found"

_LABELS = {
    Provenance.STATED: "stated in docs",
    Provenance.INFERRED: "inferred from usage",
    Provenance.CONFLICTING: "conflicting sources",
    Provenance.NOT_FOUND: "not found",
}

def label_for(p: Provenance) -> str:
    return _LABELS[p]
```
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): provenance vocabulary for faithfulness labels`

### Task 2 — ORM models (versioned ontology + glossary, tenant-scoped)

- [ ] **Write failing test** `test_comprehension_ontology.py::test_models_importable_and_tenant_scoped`: import every model, assert each maps a `container_id`/`tenant_id` column, assert `TenantOntology` has a `version` int and a unique `(tenant_id, version)`, assert `GlossaryEntry` carries `provenance`, `evidence_spans` (JSONB), `variants` (ARRAY), `first_seen`, `confidence`, and a `(tenant_id, term)` index. Assert `DocTaxonomyClass.doc_class` is a free-text `Text` column (open-vocab, NOT a SQLAlchemy `Enum`).
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/models/comprehension.py` (mirror `manifests.py:13-51`): `TenantOntology(ontology_id PK, tenant_id idx, container_id idx, version int, built_at, source_graph_signature str, status)`; `OntologyEntity(id, ontology_id FK CASCADE, tenant_id, name, normalized_value, entity_type Text, pagerank Float, mention_count Int, evidence_chunk_ids JSONB)`; `OntologyRelationship(id, ontology_id FK, tenant_id, src_name, dst_name, relation Text, state Text /* asserted|not_stated|conflicting */, confidence Float, evidence JSONB)`; `DocTaxonomyClass(id, ontology_id FK, tenant_id, doc_class Text, confidence Float, member_doc_ids JSONB)`; `TemporalCoverage(id, ontology_id FK, tenant_id, subject_kind Text, subject Text, min_date, max_date, density Float, last_mention_date)`; `KeyMetric(id, ontology_id FK, tenant_id, metric Text, definition Text, evidence JSONB, confidence Float)`; `GlossaryEntry(id, tenant_id idx, container_id, ontology_version int, term Text, expansion Text|None, definition Text|None, provenance Text, confidence Float, variants ARRAY(Text), evidence_spans JSONB, first_seen, created_at)` with `UniqueConstraint(tenant_id, term, ontology_version)`. Use `JSONB`/`ARRAY` from `sqlalchemy.dialects.postgresql`. **No `Enum` columns** — store provenance/state/doc_class as `Text` to keep the vocabulary open (invariant 6).
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): versioned tenant-ontology + glossary ORM models`

### Task 3 — Migration (additive, idempotent, non-fatal)

- [ ] **Write failing test** `test_comprehension_ontology.py::test_migration_idempotent`: call `apply_comprehension_migration(fake_engine)` twice against a stub engine that records `CREATE TABLE IF NOT EXISTS`/`CREATE INDEX IF NOT EXISTS` statements; assert every statement contains `IF NOT EXISTS` and a second call raises nothing.
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/migrations/comprehension_upgrade.py` mirroring `control_plane_upgrade.py`: an `async def apply_comprehension_migration(engine)` that runs `Base.metadata.create_all` for the new tables plus explicit `CREATE INDEX IF NOT EXISTS` for `(tenant_id, term)` and `(tenant_id, version)`; wrap in try/except logging a warning (non-fatal), to be called from the app lifespan (note in docstring — do NOT edit `main.py` here).
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): comprehension runtime migration (idempotent)`

### Task 4 — Glossary miner: explicit definitions (LLM-confirmed → `stated`)

- [ ] **Write failing test** `test_comprehension_glossary.py::test_explicit_definition_is_stated_with_span`: feed a fake chunk `"The Customer Acquisition Cost (CAC) measures spend per new customer."` to `mine_glossary` with a fake graph + a fake LLM that **confirms** the candidate; assert the `CAC` entry has `provenance == Provenance.STATED`, `expansion == "Customer Acquisition Cost"`, a non-empty `evidence_spans` carrying the verbatim sentence + `chunk_id`, and `confidence` resolved via `get_tunable` (not a literal). Add `::test_unconfirmed_candidate_is_dropped`: when the fake LLM declines, no entry is produced (grounding gate, invariant 1).
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/comprehension/glossary_miner.py`: regex **proposes** parenthetical/appositive/"stands for" candidates (the regex is a *proposal* signal, never the decision — mirror `column_role_resolver.py` LLM-confirms-classification). For each candidate call `select_model(task="glossary", ...)` → LLM confirm with the supporting span; on confirm, emit `GlossaryEntry(provenance=STATED, evidence_spans=[{chunk_id, page_num, bbox, text}])`. Inclusion threshold via `get_tunable("glossary_min_confidence", container_id)` + `log_gate_decision`. No hardcoded acronym/jargon list anywhere.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): glossary mining of LLM-confirmed explicit definitions`

### Task 5 — Glossary miner: distributional anomaly (`inferred`, never `stated`)

- [ ] **Write failing test** `test_comprehension_glossary.py::test_distributional_anomaly_is_inferred`: corpus repeats a coined term `"ZephyrFlow"` far above its background frequency; assert an entry is produced with `provenance == Provenance.INFERRED`, `definition` LLM-synthesized from usage context, evidence spans present, and **`provenance != Provenance.STATED`**. Add `::test_no_background_table_degrades_gracefully`: with the background table absent, the anomaly signal is skipped (logged via `log_gate_decision`), no crash, no fabricated `stated` entry. Add `::test_open_vocab_no_hardcoded_list`: a made-up term the system has never seen is mineable purely from signals — assert mining never consults a static term allow-list (the miner accepts an injected `background_freq` mapping and uses only corpus stats + that table).
- [ ] **Run (fails).**
- [ ] **Minimal impl:** add `_distributional_candidates(chunks, background_freq, container_id)` — compute corpus-internal term log-frequency, compare to injected background via `get_tunable("glossary_anomaly_zscore", container_id)` (+`log_gate_decision`); flag anomalies, LLM-synthesize a usage definition with `task="glossary"`, emit `provenance=INFERRED`. Load `background_freq.json` once; absence ⇒ return `[]`. The signal source is **injected data**, never an in-code dictionary.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): distributional-anomaly glossary mining (inferred provenance)`

### Task 6 — Glossary miner: co-reference variants + conflict surfacing

- [ ] **Write failing test** `test_comprehension_glossary.py::test_variants_merged` (alias `"CAC"`/`"Cust. Acq. Cost"` collapse into one entry's `variants[]`) and `::test_conflicting_definitions_surface_both`: two chunks define `"NRR"` incompatibly → resulting entry has `provenance == Provenance.CONFLICTING` and `evidence_spans` containing **both** definitions (never a silent pick — invariant 7).
- [ ] **Run (fails).**
- [ ] **Minimal impl:** `_coref_variants()` groups by `normalized_value`/expansion similarity (LLM adjudication via router, gated by `get_tunable`); a `_reconcile()` step: if a term has ≥2 incompatible confirmed expansions, set `provenance=CONFLICTING`, keep all spans, recency-tag by chunk's doc date. Single consistent definition stays `STATED`.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): co-reference variants + conflict-both-sides reconciliation`

### Task 7 — Ontology builder (versioned; bumps on rebuild)

- [ ] **Write failing test** `test_comprehension_ontology.py::test_build_then_rebuild_bumps_version`: build a `TenantOntology` from a fake `GraphReader`; assert version 1 persisted with entities/relationships/doc-taxonomy/key-metrics registries populated and `OntologyRelationship.state == "asserted"` for graph edges; rebuild → assert a NEW row with `version == 2` (old retained), and `source_graph_signature` recomputed. Add `::test_doc_taxonomy_open_vocab`: doc classes come from LLM clustering of `(:Document)` content, are arbitrary strings with confidence, and assert the builder never references an enumerated doc-type list.
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/comprehension/ontology_builder.py`: `async def build_tenant_ontology(reader: GraphReader, *, tenant_id, container_id, session) -> TenantOntology`. Read entities/edges/communities/documents/chunks; project into the registries (mirrors `semantic_layer_builder.infer_entity_spec` shape, `semantic_layer_builder.py:68`). Doc taxonomy: cluster docs via router LLM (`task="ontology"`) → open-vocab `doc_class` + confidence (gated). `version = max(existing)+1`. Persist all child rows under the new `ontology_id`. Three-state relationships preserved.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): versioned tenant-ontology builder from grounded graph`

### Task 8 — Temporal coverage + staleness annotation

- [ ] **Write failing test** `test_comprehension_ontology.py::test_temporal_coverage_and_staleness`: given chunks/docs with dates per entity, assert `compute_temporal_coverage` yields per-subject `min_date`/`max_date`/`density`/`last_mention_date`, and `staleness_annotation(last_mention_date, now)` returns a human note like `"most recent mention is 2025-09; may be outdated"` when older than `get_tunable("staleness_days", container_id)` (no literal), and `None` when fresh.
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/comprehension/temporal.py`: aggregate dates from `(:Document.doc_date)`/chunk dates per entity & per community topic; density = mentions / span. `staleness_annotation` compares to threshold via `get_tunable` + `log_gate_decision`.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): temporal coverage registry + staleness annotation`

### Task 9 — `glossary_lookup` Tool + query-expansion helper (Tool Protocol + register_tool)

- [ ] **Write failing test** `test_comprehension_tool.py`: `::test_lookup_returns_citation_or_refuses` — a known term returns expansion + definition + `provenance` label + a citation (`chunk_id`/`bbox`); an unknown term returns a single result with `provenance` `not found` and **no fabricated definition** (refuse, invariant 1/2). `::test_inferred_label_never_stated` — an `inferred` entry surfaces label `"inferred from usage"`, never `"stated in docs"`. `::test_conflict_surfaces_both`. `::test_tool_registers` — importing the module registers a `Tool` named `glossary_lookup` in `TOOL_REGISTRY`. `::test_expand_query_tenant_scoped_and_transparent` — `expand_query("what is CAC for tenant T")` returns `{original, added_terms, expansions:[{term, expansion, provenance}]}` using only tenant-T glossary rows, and adds nothing when no entry matches (transparent, no silent rewrite).
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/agent/tools_glossary.py`: a `GlossaryLookupTool` implementing the `Tool` Protocol (`name = "glossary_lookup"`, `async run(state, deps, *, term)`), reading via `reader.lookup_glossary(tenant_id, term)`; map provenance → label via `provenance.label_for`; refuse on miss. `register_tool(GlossaryLookupTool())` at import. `async def expand_query(query, *, tenant_id, container_id, reader) -> dict` returns the transparent expansion record (used by the Phase-3 planner). All gating via `get_tunable`.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): glossary_lookup tool + transparent query expansion`

### Task 10 — Onboarding surface (read-only projection endpoints)

- [ ] **Write failing test** `test_comprehension_onboarding.py` (FastAPI `TestClient`, fake reader/session injected): `GET /api/pdf/onboarding/topic-map` → community reports projected as the company's table-of-contents (cited); `GET .../entities` → paged entity browse; `GET .../glossary` → glossary browse with provenance labels (not raw confidence); `GET .../doc-taxonomy` → open-vocab classes; `GET .../ontology/version` → current `version` int. Assert every response is tenant-scoped and read-only (no write verb mounted), and the topic map carries citations (invariant 1).
- [ ] **Run (fails).**
- [ ] **Minimal impl** `pdf_chat/comprehension/reader.py` (the `GraphReader` Protocol + ontology/glossary read helpers) and `pdf_chat/api/onboarding.py` (self-prefixed `APIRouter`, late infra imports, principal via the same `_resolve_current_user` override pattern as `routes.py`). Projections are pure reads over the persisted artifact; topic map reads community reports. Note mounting in docstring (do NOT edit `main.py`).
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): read-only onboarding projection endpoints`

### Task 11 — Ingest-finalization wiring + exit acceptance test

- [ ] **Write failing test** `test_comprehension_ontology.py::test_finalization_builds_artifact_and_exit_flow`: simulate finalization calling `build_tenant_ontology` + `mine_glossary`; then drive the EXIT scenario end-to-end with fakes — browse topic map, call `glossary_lookup` for a company-specific term, assert a **cited company-specific** answer with a provenance label, and assert `ontology/version` is queryable. Assert glossary build uses `select_model(task="glossary")` (bulk `gpt-4o-mini`, never strong/Opus for bulk — C7).
- [ ] **Run (fails).**
- [ ] **Minimal impl:** an `async def finalize_comprehension(reader, *, tenant_id, container_id, session)` orchestrator (idempotent on `source_graph_signature`) that builds the ontology then mines the glossary stamped with the new `ontology_version`; document the finalization-task call site (Phase-1 state machine) without editing it here.
- [ ] **Run (passes).**
- [ ] **Commit:** `feat(pdf-phase5): finalization orchestrator + exit-criteria acceptance test`

### Task 12 — Full suite + static-sentinel self-check

- [ ] **Run:** `uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_comprehension_*.py -q`
- [ ] **Grep self-check (must return nothing):** numeric score-comparison literals in new `.py` files (`grep -rnE '[<>]=?\s*0?\.[0-9]' pdf_chat/comprehension pdf_chat/agent/tools_glossary.py pdf_chat/api/onboarding.py`); and no hardcoded jargon/doc-type dictionaries.
- [ ] **Commit:** `test(pdf-phase5): full comprehension suite green + no-literals self-check`

---

## Definition of Done (board gate)
- Inferred-definition entries are labelled `inferred`, never `stated`. ✅ T5/T9
- Glossary mining is open-vocab — no static jargon/doc-type list (signals only, injected data). ✅ T5/T7
- `glossary_lookup` returns a citation or refuses (no fabrication). ✅ T9
- Ontology `version` bumps on rebuild; old versions retained + queryable. ✅ T7/T10
- Conflicting definitions surface both sides (three-state, no silent pick). ✅ T6/T9
- Temporal staleness annotation computed from coverage. ✅ T8
- Onboarding surface is read-only, tenant-scoped, cited. ✅ T10
- No score literal in any new `.py`; all thresholds via `get_tunable` + `log_gate_decision`; bulk LLM = `gpt-4o-mini` via router. ✅ T12
- Routes through the reviewer board (architect, data-science, business-analyst, static-code-sentinel) + SME for the comprehension layer.

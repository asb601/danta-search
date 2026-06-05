# Phase 4 — Value-Evidenced Cross-Domain PDF↔CSV Bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Steps use `- [ ]` checkboxes. Agents do NOT git commit (manager commits after the gate).

## 🔒 ABSOLUTE CONSTRAINT — READ FIRST
The CSV/structured layer under `server/app/` is a **live, demo'd, working production system.** Phase 4 is **READ-ONLY toward it**: import and reuse `server/app/` modules, but **NEVER modify, refactor, or add to ANY file under `server/app/`.** ALL new code lives under `server/pdf_chat/`. **Any change to a `server/app/` file is a hard failure → reimplement.** The gate runs `git status server/app` and rejects the phase if anything there changed.

**Goal:** Let one question span a PDF (e.g. a contract) and the CSV vendor data, joined ONLY on value-evidenced master keys — never on names. Sub-threshold ⇒ refuse.

**Architecture:** A new `pdf_entity_bridge` table maps a PDF `Entity` to a `SemanticEntity` only when the PDF entity's literal *values* reconcile (value-overlap via `fingerprint_value`) against a real master key in the CSV `ColumnKeyRegistry`. A `structured_query` Tool (registered into the Phase-3 reserved seam) delegates to the read-only `run_agent_query`, inheriting the CSV feasibility + negative-claim gates, and runs strictly sequentially. Grain alignment + numeric reconciliation prove the join. Cross-domain answers carry a version-stamped cache key.

**Tech Stack:** Python 3.12, async SQLAlchemy 2.0 + asyncpg, PostgreSQL, the existing `relationship_index`/`run_agent_query` (read-only), `pdf_chat` Phase 1–3 modules. Tests: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q -p no:cacheprovider`. Baseline: 469 green.

---

## Entry Points (consumed — exact signatures)

**READ-ONLY (`server/app/` — never edit):**
- `app/agent/graph/graph.py::run_agent_query(query: str, db: AsyncSession, *, conversation_context="", user_id="", is_admin=True, allowed_domains: list[str]|None=None, container_id: str|None=None, prior_files: list[str]|None=None, actor_email="", actor_role="", org_id: str|None=None) -> dict` (line 1354). Returns `{answer, data, chart, route, row_count, files_used, tool_calls}`. Already inherits feasibility + negative-claim gates.
- `app/services/relationship_index.py::fingerprint_value(value) -> str|None` (line 71; sha256[:16] of normalized value). `find_fingerprint_matches(file_id, db) -> list[dict]` (223). `ColumnKeyRegistry` model (`app/models/column_key_registry.py`) — has `file_id`, `column_name`, `value_fingerprints` (array), `semantic_role`, `container_id`.
- `app/services/relationship_detector.py::join_confidence(overlap_pct, min_cardinality, policy) -> float` (line 51).
- `app/models/semantic_layer.py::SemanticEntity` (container_id, entity_name, aliases).
- `app/services/erp/feasibility_gate.py`, `negative_claim_gate.py` — already inside the `run_agent_query` path (confirm by reading; do not edit).

**EXTENDABLE (`server/pdf_chat/`):**
- `pdf_chat/agent/tools.py` — `Tool` Protocol (`name`; `async run(state, deps, **kw) -> list[dict]`), `register_tool(tool)`, `TOOL_REGISTRY`, `RESERVED_TOOL_NAMES` (reserves `structured_query`, line 74). `register_tool` accepts a reserved name.
- Phase-2 `Entity` node — `normalized_value` property (the fingerprint-style value), accessible via the searcher / a values reader over `(:Entity)`.
- `pdf_chat/tunables.py` — `get_tunable(container_id, key, default=_UNSET)`, `log_gate_decision(name, *, score, threshold, outcome, **ctx)`, `TUNABLE_DEFAULTS`.
- `pdf_chat/models/manifests.py` — ORM convention (shared `Base`); `pdf_chat/migrations/control_plane_upgrade.py` — idempotent migration style (`run_migration(engine)` + `upgrade` alias).
- The agent response-cache key builder (Phase 0/1) in `pdf_chat/retrieval/cache.py` / `pdf_chat/agent/graph.py`.

---

## File Structure

```
server/pdf_chat/
├── models/bridge.py                 CREATE  PdfEntityBridge + BridgeStatus
├── migrations/bridge_upgrade.py     CREATE  idempotent table+index migration
├── bridge/__init__.py               CREATE  exports
├── bridge/reconcile.py              CREATE  value-evidenced reconciliation + builder
├── bridge/grain.py                  CREATE  grain alignment + numeric reconciliation
├── agent/tools_structured.py        CREATE  structured_query Tool (delegates to run_agent_query)
├── agent/cross_domain_cache.py      CREATE  version-stamped cache key for cross-domain answers
└── testing/test_bridge.py, test_structured_query.py, test_grain.py, test_cross_domain_cache.py
```

All tunables registered in `pdf_chat/tunables.py::TUNABLE_DEFAULTS`:
`bridge.min_value_overlap_pct`=0.50, `bridge.min_overlap_count`=3, `bridge.min_confidence`=0.60,
`grain.numeric_tolerance_pct`=0.05.

---

## Task 1 — Bridge model + status enum

**Files:** Create `pdf_chat/models/bridge.py`; Test `pdf_chat/testing/test_bridge.py`.

- [ ] **Step 1 — failing test**
```python
# testing/test_bridge.py
from pdf_chat.models.bridge import PdfEntityBridge, BridgeStatus

def test_bridge_model_columns_and_status():
    cols = set(PdfEntityBridge.__table__.columns.keys())
    assert {"id","container_id","tenant_id","pdf_entity_id","semantic_entity_id",
            "resolved_master_file_id","resolved_master_column","resolved_semantic_role",
            "value_overlap_pct","confidence","overlap_count","pdf_value_count",
            "evidence","status","created_at"} <= cols
    assert BridgeStatus.LINKED.value == "linked" and BridgeStatus.REFUSED.value == "refused"
    # table key includes tenant
    assert "container_id" in cols
```
- [ ] **Step 2 — run, expect ImportError**
Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -q -p no:cacheprovider` → FAIL.
- [ ] **Step 3 — implement** `pdf_chat/models/bridge.py` mirroring `models/manifests.py` ORM style (shared `Base`, `__tablename__="pdf_entity_bridge"`): a `str`-Enum `BridgeStatus{LINKED="linked", REFUSED="refused"}`, and the model with the columns above (`value_overlap_pct`/`confidence` Float, `overlap_count`/`pdf_value_count` Integer, `evidence` JSONB, `status` Text, timestamps), indexed on `(container_id, pdf_entity_id)`. Register it in `pdf_chat/models/__init__.py` so it joins `Base.metadata`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit (manager).**

## Task 2 — Migration

**Files:** Create `pdf_chat/migrations/bridge_upgrade.py`; extend `test_bridge.py`.

- [ ] Test: `bridge_upgrade` exposes `run_migration` + `upgrade` alias; cypher/DDL idempotent (`CREATE TABLE IF NOT EXISTS` via `Base.metadata.create_all` for just this table + `CREATE INDEX IF NOT EXISTS`).
- [ ] Implement mirroring `control_plane_upgrade.py` exactly (idempotent, non-fatal). Do NOT wire into `app/main.py` (that's productionization, deferred — note it in the return).
- [ ] Run → PASS. Commit.

## Task 3 — Value-evidenced reconciliation (the core safety)

**Files:** Create `pdf_chat/bridge/reconcile.py`; Test `test_bridge.py`.

- [ ] **Step 1 — failing tests (the non-negotiables):**
```python
# name-equality alone must NOT create a bridge
async def test_name_equality_alone_refuses():
    # entity "Acme" with values that DON'T overlap any master key fingerprints
    verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e1", entity_name="Acme",
        samples=[EntityValueSample(value="ACME-XYZ")],
        master_columns=[MasterKeyColumn(file_id="f1", column="vendor_id",
                                        semantic_role="entity_key",
                                        value_fingerprints=[fingerprint_value("V-100")])])
    assert verdict.status == BridgeStatus.REFUSED
    assert "below" in verdict.reason.lower() or "overlap" in verdict.reason.lower()

# sub-threshold overlap refuses (never silently pick top match)
async def test_subthreshold_refuses(): ...

# real value overlap above threshold LINKS to the master key
async def test_value_overlap_links_master_key(): ...
```
- [ ] **Step 3 — implement** `reconcile_entity_to_master_keys(*, tenant_id, pdf_entity_id, entity_name, samples: list[EntityValueSample], master_columns: list[MasterKeyColumn]) -> ReconcileVerdict`:
  - Fingerprint the PDF entity's sample values via `fingerprint_value` (READ-ONLY import). For each `master_column`, compute `overlap_count = |pdf_fps ∩ column.value_fingerprints|`, `value_overlap_pct = overlap_count / max(1, len(pdf_fps))`.
  - `confidence = join_confidence(value_overlap_pct, overlap_count, policy)` (reuse the READ-ONLY detector; or a local mirror if the policy object is awkward to construct — but prefer reuse).
  - Gate via tunables (`bridge.min_value_overlap_pct`, `bridge.min_overlap_count`, `bridge.min_confidence`), log every decision via `log_gate_decision`. Pick the BEST qualifying master column; if none clears all gates ⇒ `ReconcileVerdict(status=REFUSED, reason=...)` (NEVER pick the top sub-threshold match). `entity_name` is NEVER used as a join signal — only as a label in evidence.
  - `build_bridge_for_entity(db, *, tenant_id, pdf_entity_id, entity_name, values_reader) -> ReconcileVerdict` persists a `PdfEntityBridge` row (LINKED or REFUSED) with evidence.
  - Dataclasses: `EntityValueSample(value)`, `MasterKeyColumn(file_id, column, semantic_role, value_fingerprints)`, `ReconcileVerdict(status, semantic_entity_id, master_file_id, master_column, value_overlap_pct, confidence, overlap_count, reason)`. `PdfEntityValuesReader = Callable[[str,str], Awaitable[list[EntityValueSample]]]`.
- [ ] Run → PASS. Commit.

## Task 4 — `structured_query` Tool (sequential, scope-passing)

**Files:** Create `pdf_chat/agent/tools_structured.py`; Test `test_structured_query.py`.

- [ ] **Step 1 — failing tests:**
```python
async def test_structured_query_passes_scope_and_is_sequential():
    calls = []
    async def fake_run_agent_query(query, db, **kw):
        calls.append(kw)
        return {"answer":"42","data":[],"row_count":0,"files_used":[]}
    deps = StructuredQueryDeps(run_agent_query=fake_run_agent_query, db=FakeSession(),
                               container_id="c1", allowed_domains=["finance"], user_id="u1")
    out = await structured_query(deps, "total spend for vendor V-100")
    assert calls[0]["container_id"]=="c1" and calls[0]["allowed_domains"]==["finance"] and calls[0]["user_id"]=="u1"
    assert out and out[0]["answer"]=="42"

async def test_tool_registers_into_reserved_seam():
    tool = build_structured_query_tool(deps)
    register_tool(tool)               # reserved name accepted
    assert TOOL_REGISTRY["structured_query"].name == "structured_query"
```
- [ ] **Step 3 — implement:** `StructuredQueryDeps(run_agent_query, db, container_id, allowed_domains, user_id, actor_email="", actor_role="", org_id=None)`. `async def structured_query(deps, query) -> list[dict]` calls `await deps.run_agent_query(query, deps.db, container_id=deps.container_id, allowed_domains=deps.allowed_domains, user_id=deps.user_id, actor_email=deps.actor_email, actor_role=deps.actor_role, org_id=deps.org_id)` and wraps the result dict in a one-element list shaped like other tool outputs (with `answer`, `data`, `files_used`, a `source="structured"` marker). `build_structured_query_tool(deps) -> Tool` returns an object implementing the Phase-3 `Tool` Protocol (`name="structured_query"`, `async run(self, state, deps2, **kw)` → calls `structured_query`). **Docstring + a comment must state: runs STRICTLY SEQUENTIALLY — the async DB session is not concurrency-safe; the Phase-3 loop must never dispatch it concurrently with another DB-touching tool.** (The loop is single-threaded sequential already; assert no asyncio.gather around DB tools.)
- [ ] Run → PASS. Commit.

## Task 5 — Grain alignment + numeric reconciliation

**Files:** Create `pdf_chat/bridge/grain.py`; Test `test_grain.py`.

- [ ] Test: `reconcile_grain(*, tenant_id, fact: GrainFact, aggregate: GrainAggregate) -> GrainResult` — align period + unit; the numeric reconciliation check passes when `abs(fact.rate*fact.volume - aggregate.total) / max(1, aggregate.total) <= get_tunable(tenant_id,"grain.numeric_tolerance_pct")`, else `GrainResult(reconciled=False, reason=...)`. Mismatched period or unit ⇒ not reconciled. Log via `log_gate_decision`.
- [ ] Implement. Run → PASS. Commit.

## Task 6 — Cross-domain cache invalidation

**Files:** Create `pdf_chat/agent/cross_domain_cache.py`; Test `test_cross_domain_cache.py`.

- [ ] Test: `build_cross_domain_cache_key(*, tenant_id, base_key, structured_query_used, version_stamps) -> str` — when `structured_query_used`, the key folds in `version_stamps["csv_semantic_layer"]` AND `version_stamps["graph_extraction"]`; when not used, equals `base_key`. Two different CSV-semantic versions ⇒ different keys. `version_stamps(tenant_id) -> dict` reads stamps WITHOUT importing/mutating `app/` (best-effort; missing stamp ⇒ "0").
- [ ] Implement. Run → PASS. Commit.

## Task 7 — Exit integration test (mocks-only)

**Files:** extend `test_bridge.py`.

- [ ] `test_exit_pdf_joins_vendor_csv_value_evidenced`: an entity whose values overlap a `vendor_id` master key ABOVE threshold builds a LINKED bridge; a `structured_query` through the bridge (mocked `run_agent_query`) returns the CSV answer; a low-overlap entity REFUSES and no cross-domain answer is produced. Assert ZERO files under `server/app/` were touched (`git status --porcelain server/app` is empty).
- [ ] Run full suite → still green (was 469). Commit.

---

## Self-review checklist (manager, before gate)
1. `git status --porcelain server/app` is EMPTY (no CSV-side edits).
2. No name-equality / embedding-cosine join path exists; only `fingerprint_value` value-overlap.
3. Sub-threshold ⇒ REFUSED, never silent top-match.
4. `structured_query` passes container_id/allowed_domains/user_id and is documented sequential.
5. No score-comparison literal in any `.py`; all thresholds in `TUNABLE_DEFAULTS`.
6. Full suite green; baseline 469 not regressed.

## Deferred (productionization, NOT this phase)
- Wire `bridge_upgrade.run_migration` + the bridge builder into the worker/lifespan (needs live infra).
- Register `structured_query` into the live agent deps (it's built+tested; activation is the turn-on step).

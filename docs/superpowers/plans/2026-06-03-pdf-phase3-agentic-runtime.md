# PDF Phase 3 — Agentic LangGraph Query Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `server/pdf_chat/`'s fixed 10-stage state machine with a LangGraph agent that plans typed intent, links entities, drives a capped tool loop (vector_search / graph_traverse / community_report_lookup / get_entity_neighbors), decomposes multi-part queries with a sufficiency check, synthesizes grounded answers with bbox citations, and refuses unproven absences via a ported negative-claim gate.

**Architecture:** Mirror the main system's patterns. The Planner/Router (`gpt-4o-mini`) emits a typed `PlannerResult` (intent + named confidence thresholds + typed `fallback_reason`); high-confidence simple/cached queries bypass the loop (mirrors `app/services/semantic_planner.py:378` `planner_fast_path_confidence`). An entity-linking node populates `state.entity` before `graph_traverse` is reachable. The tool loop enforces a hard total cap (mirrors `app/agent/state.py:22` `MAX_TOOL_CALLS` and `app/agent/graph/graph_builder.py:78`), a per-tool-type cap, decomposition depth ≤ N, and a monotonic-progress guard. Synthesis enforces a citation-density floor; a ported negative-claim gate (from `app/services/erp/negative_claim_gate.py`) blocks unproven "no data" claims. Every threshold resolves from `pdf_chat/tunables.py` (Phase 0) via `get_tunable` and is logged with its score via `log_gate_decision` — **no score-comparison literal in any `.py` file** (spec §3 invariant 4).

**Tech Stack:** Python 3.12, LangGraph (already a repo dep — guarded import, same as `pdf_chat/agent/graph.py:449`), `gpt-4o-mini` only, Neo4j (per-hop tenant isolation), `pytest` via `cd server && uv run --with pytest --with pytest-asyncio pytest`. Tests use in-memory fakes (mock LLM/searcher/tools) following `pdf_chat/testing/test_agent.py`.

---

## Assumptions & Cross-Phase Contracts

**Depends on (Phase 0 — assume exists):** `pdf_chat/tunables.py` exposes:
- `get_tunable(name: str, container_id: str | None = None) -> float | int` — resolves a per-container threshold (env/config defaulted, overridable).
- `log_gate_decision(gate: str, *, score: float, threshold: float, passed: bool, **fields) -> None` — structured log of every gate/cap/skip with its numeric score.

No `.py` file in this plan may contain a numeric score-comparison literal; every comparison reads its threshold from `get_tunable` and logs via `log_gate_decision`.

**Depends on (Phase 2 — assume exists, depend on signatures abstractly):**
- Neo4j graph has `(:Entity {name, tenant_id})`, `(:Entity)-[:RELATED_TO {desc, weight, confidence, evidence_count, src_chunk}]->(:Entity)`, `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:Entity)-[:IN_COMMUNITY]->(:Community {community_id, tenant_id})`, and `(:Community)` carries a cited `report` summary.
- `Neo4jSearcher.vector_search` / `graph_traversal` / `hybrid_search` keep their **current frozen signatures** (`pdf_chat/retrieval/neo4j_searcher.py:122,160,195`). Phase 3 ADDS three read methods on the searcher (community lookup, entity neighbors, entity name resolution); it does not change existing ones.
- Each candidate chunk dict carries `bbox` (a `[x0,y0,x1,y1]` list or `None`) and `page_num` alongside the existing `chunk_id/text/doc_id/element_type/acl/score` fields (Phase 1/2 extraction retained bbox).

**Exposes for Phase 4 (`structured_query`):** Task 11 lands a tool-registration **seam only** — `register_tool()` and a `TOOL_REGISTRY` keyed by tool name — so Phase 4 can register `structured_query` (CSV/DataFusion bridge, runs strictly sequentially, passes `container_id`/`allowed_domains`/`user_id` into `run_agent_query`) without touching the loop. Phase 3 does **not** implement `structured_query`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pdf_chat/agent/state.py` (modify) | Extend `PdfChatState` with agentic fields (intent, plan, sub-queries, tool-call counters, seen-chunk set, loop control). |
| `pdf_chat/agent/planner.py` (create) | `PlannerResult` type + `plan_query()` — LLM intent classification, named thresholds, typed fallback reasons, bypass decision. |
| `pdf_chat/agent/entity_linker.py` (create) | `link_entities()` — populate `state.entity` from the query before graph traversal. |
| `pdf_chat/agent/tools.py` (create) | `Tool` protocol + `register_tool`/`TOOL_REGISTRY` seam + the 4 Phase-3 tools (vector_search, graph_traverse, community_report_lookup, get_entity_neighbors). |
| `pdf_chat/agent/decompose.py` (create) | `decompose_query()` + `sufficiency_check()` — split multi-part asks, verify all output components covered. |
| `pdf_chat/agent/loop.py` (create) | `LoopBudget` + `tool_loop()` — hard caps, per-tool caps, monotonic-progress guard, drop logging. |
| `pdf_chat/agent/synthesis.py` (create) | `synthesize()` — grounded answer with bbox citations + citation-density floor. |
| `pdf_chat/agent/negative_claim_gate.py` (create) | Port of the ERP gate to PDF/graph (coverage = relevant pages/sections in-context via bbox/page coverage). |
| `pdf_chat/agent/graph.py` (modify) | Wire the new nodes into the LangGraph; keep the plain runner; preserve cache/ACL/audit short-circuits. |
| `pdf_chat/retrieval/neo4j_searcher.py` (modify) | Add `community_report_lookup`, `get_entity_neighbors`, `resolve_entities` (per-hop tenant isolation in Cypher). |
| `pdf_chat/testing/test_phase3_agent.py` (create) | All Phase-3 deterministic-seam tests (fakes). |

---

## NEW Public Types & Signatures (locked — later tasks must match exactly)

```python
# pdf_chat/agent/planner.py
from dataclasses import dataclass, field
from typing import Literal

QueryIntent = Literal["local", "global", "cross_domain"]

@dataclass
class PlannerResult:
    intent: QueryIntent = "local"
    confidence: float = 0.0
    bypass: bool = False                 # high-confidence simple/cached → skip the loop
    fallback_reason: str | None = None   # typed; mirrors semantic_planner.fallback_reason
    sub_queries: list[str] = field(default_factory=list)
    output_components: list[str] = field(default_factory=list)  # what a complete answer must contain

async def plan_query(state: "PdfChatState", deps: "Deps") -> PlannerResult: ...

# pdf_chat/agent/tools.py
from typing import Any, Protocol

class Tool(Protocol):
    name: str
    async def run(self, state: "PdfChatState", deps: "Deps", **kwargs: Any) -> list[dict]: ...

TOOL_REGISTRY: dict[str, Tool] = {}
def register_tool(tool: Tool) -> None: ...   # Phase-4 seam: structured_query registers here

# pdf_chat/agent/loop.py
@dataclass
class LoopBudget:
    total_calls: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    dropped: list[str] = field(default_factory=list)

async def tool_loop(state: "PdfChatState", deps: "Deps") -> "PdfChatState": ...

# pdf_chat/agent/negative_claim_gate.py
@dataclass
class PdfNegativeClaimVerdict:
    is_negative_claim: bool = False
    proven: bool = False
    coverage_complete: bool = False   # relevant pages/sections were in-context (bbox/page coverage)
    diagnosed: bool = False
    missing_diagnostics: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

def evaluate_pdf_negative_claim(answer: str, state: "PdfChatState") -> PdfNegativeClaimVerdict: ...
def honest_rewrite(verdict: PdfNegativeClaimVerdict) -> str: ...

# pdf_chat/agent/synthesis.py
async def synthesize(state: "PdfChatState", deps: "Deps") -> "PdfChatState": ...  # enforces citation-density floor
```

---

### Task 1: Extend `PdfChatState` for the agentic runtime

**Files:**
- Modify: `pdf_chat/agent/state.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# pdf_chat/testing/test_phase3_agent.py
from __future__ import annotations
from pdf_chat.agent.state import PdfChatState


def test_state_has_agentic_fields_with_defaults():
    s = PdfChatState(query="q", tenant_id="t1")
    assert s.intent == "local"
    assert s.sub_queries == []
    assert s.output_components == []
    assert s.tool_call_count == 0
    assert s.per_tool_calls == {}
    assert s.seen_chunk_ids == set()
    assert s.depth == 0
    assert s.dropped == []
    assert s.planner_confidence == 0.0
    assert s.fallback_reason is None
    assert s.bypass is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py::test_state_has_agentic_fields_with_defaults -v`
Expected: FAIL — `AttributeError: 'PdfChatState' object has no attribute 'intent'`

- [ ] **Step 3: Add the fields**

In `pdf_chat/agent/state.py`, add to the `from typing import` line `Literal`, and inside `@dataclass class PdfChatState` after the `acl_version` field add:

```python
    # --- Phase 3 agentic runtime (planner / loop / decomposition) ---
    intent: "Literal['local', 'global', 'cross_domain']" = "local"
    planner_confidence: float = 0.0
    bypass: bool = False
    fallback_reason: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    output_components: list[str] = field(default_factory=list)
    # loop control / cost ceilings
    tool_call_count: int = 0
    per_tool_calls: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    seen_chunk_ids: set[str] = field(default_factory=set)
    dropped: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py::test_state_has_agentic_fields_with_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/state.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): extend PdfChatState with agentic runtime fields

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Planner/Router — typed intent + bypass + typed fallback reasons

**Files:**
- Create: `pdf_chat/agent/planner.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Mirrors `app/services/semantic_planner.py`: named threshold (`planner_fast_path_confidence`) and typed `fallback_reason`. Bypass threshold reads `pdf_planner_bypass_confidence` from tunables and is logged.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
import asyncio
import pytest
from pdf_chat.agent.graph import Deps
from pdf_chat.agent.planner import PlannerResult, plan_query


class _FakeLlm:
    def __init__(self, payload: str):
        self._payload = payload
    async def generate(self, system: str, user: str) -> str:
        return self._payload


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_planner_high_confidence_simple_query_bypasses(monkeypatch):
    import pdf_chat.agent.planner as planner
    monkeypatch.setattr(planner, "get_tunable", lambda name, container_id=None: 0.7)
    payload = '{"intent":"local","confidence":0.95,"sub_queries":["q"],"output_components":["answer"]}'
    s = PdfChatState(query="q", tenant_id="t1")
    res = _run(plan_query(s, Deps(llm=_FakeLlm(payload))))
    assert res.intent == "local"
    assert res.bypass is True
    assert res.fallback_reason is None


def test_planner_low_confidence_does_not_bypass(monkeypatch):
    import pdf_chat.agent.planner as planner
    monkeypatch.setattr(planner, "get_tunable", lambda name, container_id=None: 0.7)
    payload = '{"intent":"global","confidence":0.30,"sub_queries":["a","b"],"output_components":["a","b"]}'
    s = PdfChatState(query="q", tenant_id="t1")
    res = _run(plan_query(s, Deps(llm=_FakeLlm(payload))))
    assert res.bypass is False
    assert res.intent == "global"


def test_planner_unparseable_llm_yields_typed_fallback():
    s = PdfChatState(query="q", tenant_id="t1")
    res = _run(plan_query(s, Deps(llm=_FakeLlm("not json"))))
    assert res.bypass is False
    assert res.fallback_reason == "planner_parse_error"


def test_planner_no_llm_yields_typed_fallback():
    s = PdfChatState(query="q", tenant_id="t1")
    res = _run(plan_query(s, Deps(llm=None)))
    assert res.fallback_reason == "no_planner_llm"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k planner -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.planner'`

- [ ] **Step 3: Implement the planner**

```python
# pdf_chat/agent/planner.py
"""Planner/Router (gpt-4o-mini) — typed intent + bypass + typed fallback reasons.

Mirrors app/services/semantic_planner.py: a named confidence threshold
(planner bypass, like planner_fast_path_confidence) and a typed fallback_reason
string. NO score-comparison literal lives here — the threshold is read from
tunables and the decision is logged with its score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pdf_chat.tunables import get_tunable, log_gate_decision

if TYPE_CHECKING:  # avoid infra import at module load
    from pdf_chat.agent.graph import Deps
    from pdf_chat.agent.state import PdfChatState

QueryIntent = Literal["local", "global", "cross_domain"]

_PLANNER_SYSTEM = (
    "You are a query router for an enterprise PDF knowledge graph. Classify the "
    "user query and return STRICT JSON with keys: intent (one of 'local','global',"
    "'cross_domain'), confidence (0..1 float), sub_queries (list of atomic sub-"
    "questions), output_components (list naming each thing a complete answer must "
    "contain). 'local' = fact in a few chunks; 'global' = needs community/theme "
    "summary; 'cross_domain' = needs structured/CSV data too. Return JSON only."
)
_VALID_INTENTS = ("local", "global", "cross_domain")


@dataclass
class PlannerResult:
    intent: QueryIntent = "local"
    confidence: float = 0.0
    bypass: bool = False
    fallback_reason: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    output_components: list[str] = field(default_factory=list)


async def plan_query(state: "PdfChatState", deps: "Deps") -> PlannerResult:
    """Classify intent, decide bypass. Never raises; failures return a typed reason."""
    if deps.llm is None:
        return PlannerResult(fallback_reason="no_planner_llm")
    try:
        raw = await deps.llm.generate(_PLANNER_SYSTEM, state.query)
        data = json.loads(raw)
    except Exception:
        return PlannerResult(fallback_reason="planner_parse_error")

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        return PlannerResult(fallback_reason="planner_unknown_intent")

    confidence = float(data.get("confidence", 0.0))
    sub_queries = [str(q) for q in data.get("sub_queries", []) if str(q).strip()]
    output_components = [str(c) for c in data.get("output_components", []) if str(c).strip()]

    bypass_threshold = float(get_tunable("pdf_planner_bypass_confidence", state.tenant_id))
    simple = len(sub_queries) <= 1
    bypass = confidence >= bypass_threshold and simple
    log_gate_decision(
        "planner_bypass",
        score=confidence,
        threshold=bypass_threshold,
        passed=bypass,
        intent=intent,
        simple=simple,
    )
    return PlannerResult(
        intent=intent,
        confidence=confidence,
        bypass=bypass,
        sub_queries=sub_queries or [state.query],
        output_components=output_components,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k planner -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/planner.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add Planner/Router with typed intent, bypass, typed fallback reasons

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Searcher extensions — community report, entity neighbors, entity resolution (per-hop tenant isolation)

**Files:**
- Modify: `pdf_chat/retrieval/neo4j_searcher.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Cypher must enforce tenant isolation on EVERY path hop — `ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)` (spec §2 L3, §3 invariant 3), not only endpoints. We assert the Cypher constants contain that clause (deterministic seam — no Neo4j needed).

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
def test_entity_neighbors_cypher_enforces_per_hop_tenant():
    from pdf_chat.retrieval import neo4j_searcher as ns
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in ns._NEIGHBORS_CYPHER


def test_community_report_cypher_is_tenant_scoped():
    from pdf_chat.retrieval import neo4j_searcher as ns
    assert "$tenant_id" in ns._COMMUNITY_CYPHER
    assert "Community" in ns._COMMUNITY_CYPHER


def test_resolve_entities_cypher_is_tenant_scoped():
    from pdf_chat.retrieval import neo4j_searcher as ns
    assert "e.tenant_id = $tenant_id" in ns._RESOLVE_ENTITY_CYPHER
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "cypher" -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_NEIGHBORS_CYPHER'`

- [ ] **Step 3: Add the Cypher constants + methods**

In `pdf_chat/retrieval/neo4j_searcher.py`, after `_GRAPH_CYPHER` add:

```python
# get_entity_neighbors — multi-hop entity walk. Tenant isolation is asserted on
# EVERY node of the path (not just endpoints): a single hop through a foreign-
# tenant entity would leak across tenants (spec §3 invariant 3; security, not
# hardening). The variable-length path is captured as `path` so ALL(...) can scan it.
_NEIGHBORS_CYPHER = """
MATCH path = (e:Entity {name: $entity})-[:RELATED_TO*1..2]-(n:Entity)
WHERE e.tenant_id = $tenant_id
  AND ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)
RETURN DISTINCT n.name AS name, n.entity_type AS entity_type
LIMIT $limit
"""

# community_report_lookup — global-intent routing. The report is CITED (it routes
# but is never the evidence of record). Tenant-scoped on the community node.
_COMMUNITY_CYPHER = """
MATCH (c:Community)
WHERE c.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)
RETURN c.community_id AS community_id, c.report AS report,
       c.src_chunks AS src_chunks
ORDER BY c.rank DESC
LIMIT $limit
"""

# resolve_entities — link query mentions to graph entities, tenant-scoped.
_RESOLVE_ENTITY_CYPHER = """
MATCH (e:Entity)
WHERE e.tenant_id = $tenant_id AND toLower(e.name) IN $names
RETURN e.name AS name, e.entity_type AS entity_type
LIMIT $limit
"""
```

Then add these methods to `class Neo4jSearcher` (after `graph_traversal`):

```python
    def get_entity_neighbors(
        self, entity: str, tenant_id: str, limit: int | None = None
    ) -> list[dict]:
        """Multi-hop neighbor entities, tenant-isolated on EVERY path hop."""
        if limit is None:
            limit = get_pdf_settings().graph_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _NEIGHBORS_CYPHER, entity=entity, tenant_id=tenant_id, limit=limit
            )
            return [dict(record) for record in result]

    def community_report_lookup(
        self, tenant_id: str, limit: int | None = None, doc_ids: list[str] | None = None
    ) -> list[dict]:
        """Cited community reports for global-intent routing (tenant-scoped)."""
        if limit is None:
            limit = get_pdf_settings().graph_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _COMMUNITY_CYPHER, tenant_id=tenant_id, limit=limit, doc_ids=doc_ids
            )
            return [dict(record) for record in result]

    def resolve_entities(
        self, names: list[str], tenant_id: str, limit: int = 8
    ) -> list[dict]:
        """Resolve lowercased candidate names to graph entities, tenant-scoped."""
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _RESOLVE_ENTITY_CYPHER,
                names=[n.lower() for n in names],
                tenant_id=tenant_id,
                limit=limit,
            )
            return [dict(record) for record in result]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "cypher" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/retrieval/neo4j_searcher.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add community/neighbor/resolve searcher methods with per-hop tenant isolation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Entity-linking node — populate `state.entity` before graph traverse

**Files:**
- Create: `pdf_chat/agent/entity_linker.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Resolves architect B2 (`state.entity` is never populated today — `pdf_chat/agent/state.py:35`). Candidate name extraction is the cheap step; the authoritative link is the tenant-scoped graph resolve.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent.entity_linker import link_entities


class _FakeSearcher:
    def __init__(self, resolved):
        self._resolved = resolved
        self.resolve_calls = []
    def resolve_entities(self, names, tenant_id, limit=8):
        self.resolve_calls.append((tuple(names), tenant_id))
        return self._resolved


def test_entity_linker_populates_state_entity():
    s = PdfChatState(query="What is the termination clause for Acme Corp?", tenant_id="t1")
    searcher = _FakeSearcher([{"name": "Acme Corp", "entity_type": "org"}])
    out = _run(link_entities(s, Deps(searcher=searcher)))
    assert out.entity == "Acme Corp"
    # tenant_id is passed into every resolve call (isolation)
    assert all(call[1] == "t1" for call in searcher.resolve_calls)


def test_entity_linker_no_match_leaves_entity_none():
    s = PdfChatState(query="general overview", tenant_id="t1")
    out = _run(link_entities(s, Deps(searcher=_FakeSearcher([]))))
    assert out.entity is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "entity_linker" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.entity_linker'`

- [ ] **Step 3: Implement the linker**

```python
# pdf_chat/agent/entity_linker.py
"""Entity-linking runtime step — populate state.entity from the query BEFORE
graph_traverse is reachable (resolves architect B2; state.entity was dead).

Candidate extraction is intentionally cheap (capitalized n-grams); the
AUTHORITATIVE link is the tenant-scoped graph resolve, so a hallucinated name
that does not exist in this tenant's graph simply yields no entity.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdf_chat.agent.graph import Deps
    from pdf_chat.agent.state import PdfChatState

# Capitalized word runs (e.g. "Acme Corp", "Section 4") — proper-noun candidates.
_CANDIDATE_RE = re.compile(r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*")


def _candidates(query: str) -> list[str]:
    seen: list[str] = []
    for m in _CANDIDATE_RE.findall(query):
        if m not in seen:
            seen.append(m)
    return seen


async def link_entities(state: "PdfChatState", deps: "Deps") -> "PdfChatState":
    """Resolve query mentions to a tenant-scoped graph entity → state.entity."""
    if deps.searcher is None or not getattr(deps.searcher, "resolve_entities", None):
        return state
    names = _candidates(state.query)
    if not names:
        return state
    import inspect

    resolved = deps.searcher.resolve_entities(names, state.tenant_id)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    if resolved:
        state.entity = resolved[0].get("name")
    return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "entity_linker" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/entity_linker.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add entity-linking node that populates state.entity tenant-scoped

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Tool registry + the four Phase-3 tools (with Phase-4 seam)

**Files:**
- Create: `pdf_chat/agent/tools.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

`structured_query` is Phase 4 — this task lands the `register_tool`/`TOOL_REGISTRY` seam ONLY, and implements the four read tools.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent.tools import TOOL_REGISTRY, register_tool, build_phase3_tools


def test_phase3_registers_exactly_four_tools():
    build_phase3_tools()  # idempotent registration
    assert set(TOOL_REGISTRY) >= {
        "vector_search", "graph_traverse", "community_report_lookup", "get_entity_neighbors"
    }
    # structured_query is Phase 4 — NOT registered here
    assert "structured_query" not in TOOL_REGISTRY


def test_register_tool_seam_accepts_new_tool():
    class _StubStructuredQuery:
        name = "structured_query"
        async def run(self, state, deps, **kwargs):
            return []
    register_tool(_StubStructuredQuery())
    assert "structured_query" in TOOL_REGISTRY
    del TOOL_REGISTRY["structured_query"]  # keep registry clean for other tests


def test_graph_traverse_tool_passes_tenant_and_entity():
    class _S:
        def __init__(self): self.calls = []
        def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
            self.calls.append((entity, tenant_id)); return [{"chunk_id": "c1"}]
    build_phase3_tools()
    s = PdfChatState(query="q", tenant_id="t9"); s.entity = "Acme"
    searcher = _S()
    out = _run(TOOL_REGISTRY["graph_traverse"].run(s, Deps(searcher=searcher)))
    assert out == [{"chunk_id": "c1"}]
    assert searcher.calls == [("Acme", "t9")]


def test_community_lookup_tool_marks_cited():
    class _S:
        def community_report_lookup(self, tenant_id, limit=None, doc_ids=None):
            return [{"community_id": "k1", "report": "summary", "src_chunks": ["c1", "c2"]}]
    build_phase3_tools()
    s = PdfChatState(query="themes", tenant_id="t1")
    out = _run(TOOL_REGISTRY["community_report_lookup"].run(s, Deps(searcher=_S())))
    assert out[0]["_cited_src_chunks"] == ["c1", "c2"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "tool or register or traverse or community_lookup" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.tools'`

- [ ] **Step 3: Implement tools + seam**

```python
# pdf_chat/agent/tools.py
"""Phase-3 agent tools + a registration seam for Phase 4 (structured_query).

Each tool is a small object satisfying the Tool protocol: name + async run().
Tools return a list of chunk-like dicts (chunk_id/text/doc_id/page_num/bbox/acl)
so the loop can fuse + dedup them uniformly. structured_query is NOT implemented
here — Phase 4 calls register_tool() with its own Tool to slot into the loop
without touching loop.py.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Protocol

from pdf_chat.config import get_pdf_settings

if TYPE_CHECKING:
    from pdf_chat.agent.graph import Deps
    from pdf_chat.agent.state import PdfChatState


class Tool(Protocol):
    name: str
    async def run(self, state: "PdfChatState", deps: "Deps", **kwargs: Any) -> list[dict]: ...


TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    """Phase-4 seam: register a tool by name. Idempotent (last write wins)."""
    TOOL_REGISTRY[tool.name] = tool


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class _VectorSearch:
    name = "vector_search"
    async def run(self, state, deps, **kwargs):
        if deps.searcher is None:
            return []
        top_k = kwargs.get("top_k") or get_pdf_settings().vector_top_k
        hits = await _maybe_await(
            deps.searcher.vector_search(
                state.query_vector or [], state.tenant_id, top_k=top_k, doc_ids=state.doc_ids
            )
        )
        return list(hits)


class _GraphTraverse:
    name = "graph_traverse"
    async def run(self, state, deps, **kwargs):
        if deps.searcher is None or not state.entity:
            return []
        hits = await _maybe_await(
            deps.searcher.graph_traversal(
                state.entity, state.tenant_id,
                limit=kwargs.get("limit"), doc_ids=state.doc_ids,
            )
        )
        return list(hits)


class _CommunityReportLookup:
    name = "community_report_lookup"
    async def run(self, state, deps, **kwargs):
        if deps.searcher is None:
            return []
        reports = await _maybe_await(
            deps.searcher.community_report_lookup(
                state.tenant_id, limit=kwargs.get("limit"), doc_ids=state.doc_ids
            )
        )
        # A report ROUTES but is never the evidence of record — carry its
        # drill-down citations forward so synthesis cites source chunks, not the report.
        out = []
        for r in reports:
            r = dict(r)
            r["_cited_src_chunks"] = list(r.get("src_chunks", []) or [])
            out.append(r)
        return out


class _GetEntityNeighbors:
    name = "get_entity_neighbors"
    async def run(self, state, deps, **kwargs):
        if deps.searcher is None or not state.entity:
            return []
        return list(await _maybe_await(
            deps.searcher.get_entity_neighbors(state.entity, state.tenant_id, limit=kwargs.get("limit"))
        ))


def build_phase3_tools() -> None:
    """Register the four Phase-3 read tools (idempotent)."""
    for tool in (_VectorSearch(), _GraphTraverse(), _CommunityReportLookup(), _GetEntityNeighbors()):
        register_tool(tool)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "tool or register or traverse or community_lookup" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/tools.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add four Phase-3 tools + register_tool seam for Phase-4 structured_query

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Capped tool loop with monotonic-progress guard

**Files:**
- Create: `pdf_chat/agent/loop.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Mirrors `MAX_TOOL_CALLS` (`app/agent/state.py:22`) and the budget-exhausted exit in `app/agent/graph/graph_builder.py:78`. All caps come from tunables; every drop is logged. The monotonic-progress guard aborts when a round adds no new accessible chunk ids (spec §2 L3, §3 invariant 4).

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent import loop as loopmod
from pdf_chat.agent.loop import tool_loop, LoopBudget


def _tunable_factory(values):
    def _get(name, container_id=None):
        return values[name]
    return _get


class _CountingSearcher:
    """Always returns the SAME chunk id → triggers the monotonic-progress abort."""
    def __init__(self): self.vector_calls = 0
    def vector_search(self, qv, tenant_id, top_k=None, doc_ids=None):
        self.vector_calls += 1
        return [{"chunk_id": "same", "text": "x", "tenant_id": tenant_id,
                 "acl": {"public": True}, "page_num": 1, "bbox": [0, 0, 1, 1]}]


def test_total_tool_call_cap_enforced(monkeypatch):
    monkeypatch.setattr(loopmod, "get_tunable", _tunable_factory({
        "pdf_max_tool_calls": 3, "pdf_max_calls_per_tool": 99, "pdf_max_depth": 3,
    }))
    s = PdfChatState(query="q", tenant_id="t1")
    s.sub_queries = ["q1", "q2", "q3", "q4", "q5"]
    out = _run(tool_loop(s, Deps(searcher=_CountingSearcher())))
    assert out.tool_call_count <= 3
    assert any("total_tool_call_cap" in d for d in out.dropped)


def test_monotonic_progress_aborts_when_no_new_chunks(monkeypatch):
    monkeypatch.setattr(loopmod, "get_tunable", _tunable_factory({
        "pdf_max_tool_calls": 10, "pdf_max_calls_per_tool": 10, "pdf_max_depth": 10,
    }))
    s = PdfChatState(query="q", tenant_id="t1")
    s.sub_queries = ["q1", "q2", "q3", "q4"]
    searcher = _CountingSearcher()
    out = _run(tool_loop(s, Deps(searcher=searcher)))
    # First round adds "same"; the next round adds nothing new → abort before cap.
    assert searcher.vector_calls < 10
    assert any("monotonic_progress" in d for d in out.dropped)


def test_per_tool_cap_enforced(monkeypatch):
    monkeypatch.setattr(loopmod, "get_tunable", _tunable_factory({
        "pdf_max_tool_calls": 99, "pdf_max_calls_per_tool": 1, "pdf_max_depth": 99,
    }))
    s = PdfChatState(query="q", tenant_id="t1")
    s.sub_queries = ["q1", "q2", "q3"]
    out = _run(tool_loop(s, Deps(searcher=_CountingSearcher())))
    assert out.per_tool_calls.get("vector_search", 0) <= 1
    assert any("per_tool_cap" in d for d in out.dropped)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "cap or monotonic" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.loop'`

- [ ] **Step 3: Implement the loop**

```python
# pdf_chat/agent/loop.py
"""Capped agent tool loop with a monotonic-progress guard.

Cost ceilings (all from tunables; every truncation logged with log_gate_decision):
  * pdf_max_tool_calls   — hard TOTAL tool-call cap (mirrors MAX_TOOL_CALLS).
  * pdf_max_calls_per_tool — per-tool-type cap.
  * pdf_max_depth        — decomposition depth (one round per sub-query, capped).
  * monotonic-progress   — abort a round that adds zero new accessible chunk ids.

The loop accumulates candidate chunks on state.candidates and tracks seen ids on
state.seen_chunk_ids. It never raises; on budget exhaustion it records what was
dropped and returns whatever it gathered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pdf_chat.agent.tools import TOOL_REGISTRY, build_phase3_tools
from pdf_chat.tunables import get_tunable, log_gate_decision

if TYPE_CHECKING:
    from pdf_chat.agent.graph import Deps
    from pdf_chat.agent.state import PdfChatState


@dataclass
class LoopBudget:
    total_calls: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)
    depth: int = 0
    dropped: list[str] = field(default_factory=list)


def _chunk_id(c: dict) -> str:
    return str(c.get("chunk_id", "")) if isinstance(c, dict) else ""


# Intent → which tools that round may call (community lookup only for global).
def _tools_for(intent: str) -> list[str]:
    base = ["vector_search", "graph_traverse", "get_entity_neighbors"]
    if intent == "global":
        return ["community_report_lookup", *base]
    return base


async def tool_loop(state: "PdfChatState", deps: "Deps") -> "PdfChatState":
    build_phase3_tools()
    max_total = int(get_tunable("pdf_max_tool_calls", state.tenant_id))
    max_per_tool = int(get_tunable("pdf_max_calls_per_tool", state.tenant_id))
    max_depth = int(get_tunable("pdf_max_depth", state.tenant_id))

    sub_queries = state.sub_queries or [state.query]
    for sub in sub_queries:
        if state.depth >= max_depth:
            msg = f"max_depth:{max_depth}:dropped_subquery:{sub}"
            state.dropped.append(msg)
            log_gate_decision("decomposition_depth", score=state.depth, threshold=max_depth, passed=False, sub=sub)
            continue
        state.depth += 1
        round_new = 0
        for tool_name in _tools_for(state.intent):
            if state.tool_call_count >= max_total:
                state.dropped.append(f"total_tool_call_cap:{max_total}:tool:{tool_name}")
                log_gate_decision("total_tool_call_cap", score=state.tool_call_count, threshold=max_total, passed=False, tool=tool_name)
                break
            used = state.per_tool_calls.get(tool_name, 0)
            if used >= max_per_tool:
                state.dropped.append(f"per_tool_cap:{max_per_tool}:tool:{tool_name}")
                log_gate_decision("per_tool_cap", score=used, threshold=max_per_tool, passed=False, tool=tool_name)
                continue
            tool = TOOL_REGISTRY.get(tool_name)
            if tool is None:
                continue
            results = await tool.run(state, deps)
            state.tool_call_count += 1
            state.per_tool_calls[tool_name] = used + 1
            for c in results:
                cid = _chunk_id(c)
                if cid and cid not in state.seen_chunk_ids:
                    state.seen_chunk_ids.add(cid)
                    state.candidates.append(c)
                    round_new += 1
        # Monotonic-progress guard: a round that added zero new accessible chunks
        # cannot make progress — abort rather than burn the remaining budget.
        progressed = round_new > 0
        log_gate_decision("monotonic_progress", score=round_new, threshold=1, passed=progressed, depth=state.depth)
        if not progressed:
            state.dropped.append(f"monotonic_progress:no_new_chunks_at_depth:{state.depth}")
            break
    return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "cap or monotonic" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/loop.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add capped tool loop with per-tool caps, depth limit, monotonic-progress guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Query decomposition + output-completeness sufficiency check

**Files:**
- Create: `pdf_chat/agent/decompose.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Sufficiency verifies ALL requested output components are present, not merely that *an* answer formed (spec §2 L3 — prevents dropping the 3rd part of a multi-part ask). Component coverage is detected by checking each component's salient tokens appear in the answer; the coverage threshold comes from tunables.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent import decompose as decmod
from pdf_chat.agent.decompose import sufficiency_check


def test_sufficiency_passes_when_all_components_present(monkeypatch):
    monkeypatch.setattr(decmod, "get_tunable", lambda name, container_id=None: 1.0)
    s = PdfChatState(query="q", tenant_id="t1")
    s.output_components = ["revenue", "headcount"]
    s.answer = "Revenue was 5M and headcount was 200."
    ok, missing = sufficiency_check(s)
    assert ok is True
    assert missing == []


def test_sufficiency_fails_and_lists_missing_component(monkeypatch):
    monkeypatch.setattr(decmod, "get_tunable", lambda name, container_id=None: 1.0)
    s = PdfChatState(query="q", tenant_id="t1")
    s.output_components = ["revenue", "headcount", "churn"]
    s.answer = "Revenue was 5M and headcount was 200."
    ok, missing = sufficiency_check(s)
    assert ok is False
    assert "churn" in missing
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "sufficiency" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.decompose'`

- [ ] **Step 3: Implement decomposition + sufficiency**

```python
# pdf_chat/agent/decompose.py
"""Query decomposition + output-completeness sufficiency check.

decompose_query() trusts the planner's sub_queries/output_components (the planner
already ran the LLM); it normalizes them onto state. sufficiency_check() verifies
EVERY requested output component is reflected in the answer — not merely that an
answer exists — so the 3rd part of a multi-part ask cannot be silently dropped.
The required coverage ratio per component comes from tunables (no literal here).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pdf_chat.tunables import get_tunable, log_gate_decision

if TYPE_CHECKING:
    from pdf_chat.agent.state import PdfChatState

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def decompose_query(state: "PdfChatState") -> "PdfChatState":
    if not state.sub_queries:
        state.sub_queries = [state.query]
    if not state.output_components:
        state.output_components = [state.query]
    return state


def sufficiency_check(state: "PdfChatState") -> tuple[bool, list[str]]:
    """Return (all_components_covered, missing_components)."""
    answer_tokens = _tokens(state.answer or "")
    required_ratio = float(get_tunable("pdf_component_coverage_ratio", state.tenant_id))
    missing: list[str] = []
    for component in state.output_components:
        ctoks = _tokens(component)
        if not ctoks:
            continue
        overlap = len(ctoks & answer_tokens) / len(ctoks)
        covered = overlap >= required_ratio
        log_gate_decision(
            "component_coverage", score=overlap, threshold=required_ratio,
            passed=covered, component=component,
        )
        if not covered:
            missing.append(component)
    return (len(missing) == 0, missing)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "sufficiency" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/decompose.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add decomposition + output-completeness sufficiency check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Adaptive rerank (skip threshold tunable + logged), reusing RRF + ACL

**Files:**
- Modify: `pdf_chat/agent/graph.py` (add an `adaptive_rerank` node function)
- Test: `pdf_chat/testing/test_phase3_agent.py`

Reuses `pdf_chat/retrieval/reranker.py` and ACL (unchanged). Skip happens when the top candidate's score already clears the skip threshold (from tunables) — logged via `log_gate_decision`.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
import pdf_chat.agent.graph as graphmod
from pdf_chat.agent.graph import adaptive_rerank, Deps


class _FakeReranker:
    def __init__(self): self.called = False
    async def rerank(self, query, candidates, top_n):
        self.called = True
        return list(reversed(candidates))[:top_n]


def test_adaptive_rerank_skips_above_threshold(monkeypatch):
    monkeypatch.setattr(graphmod, "get_tunable", lambda name, container_id=None: 0.5)
    s = PdfChatState(query="q", tenant_id="t1")
    s.candidates = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.1}]
    rr = _FakeReranker()
    out = _run(adaptive_rerank(s, Deps(reranker=rr)))
    assert rr.called is False         # top score 0.9 >= 0.5 → skip rerank
    assert out.reranked == s.candidates


def test_adaptive_rerank_runs_below_threshold(monkeypatch):
    monkeypatch.setattr(graphmod, "get_tunable", lambda name, container_id=None: 0.95)
    s = PdfChatState(query="q", tenant_id="t1")
    s.candidates = [{"chunk_id": "a", "score": 0.4}, {"chunk_id": "b", "score": 0.1}]
    rr = _FakeReranker()
    out = _run(adaptive_rerank(s, Deps(reranker=rr)))
    assert rr.called is True          # top score 0.4 < 0.95 → run rerank
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "adaptive_rerank" -v`
Expected: FAIL — `ImportError: cannot import name 'adaptive_rerank'`

- [ ] **Step 3: Add the node to `graph.py`**

In `pdf_chat/agent/graph.py`, add near the top with the other imports:

```python
from pdf_chat.tunables import get_tunable, log_gate_decision
```

Then add this node function (next to `rrf_rerank`):

```python
async def adaptive_rerank(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Adaptive rerank — skip the cross-encoder when the top candidate already
    clears the skip threshold (from tunables; the decision is logged). Reuses the
    existing reranker.rerank backend; RRF order is preserved on skip."""
    candidates = state.candidates
    if not candidates:
        state.reranked = []
        return state
    settings = get_pdf_settings()
    top_score = float(_attr(candidates[0], "score", 0.0) or 0.0)
    skip_threshold = float(get_tunable("pdf_rerank_skip_score", state.tenant_id))
    skip = top_score >= skip_threshold
    log_gate_decision("rerank_skip", score=top_score, threshold=skip_threshold, passed=skip)
    if skip or deps.reranker is None:
        state.reranked = candidates[: settings.rerank_top_n]
    else:
        state.reranked = await deps.reranker.rerank(state.query, candidates, settings.rerank_top_n)
    return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "adaptive_rerank" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/graph.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add adaptive rerank node with tunable, logged skip threshold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Grounded synthesis with bbox citations + citation-density floor

**Files:**
- Create: `pdf_chat/agent/synthesis.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Citations include `bbox`. The citation-density floor refuses to emit a claim (a sentence) with zero supporting citation — claims are sentences; a `[N]` marker is required (spec §2 L3, §3 invariant 1). Floor ratio comes from tunables.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
import pdf_chat.agent.synthesis as synthmod
from pdf_chat.agent.synthesis import synthesize


class _CitingLlm:
    def __init__(self, text): self._text = text
    async def generate(self, system, user): return self._text


def _chunk_bbox(cid, text, page=1, bbox=(0, 0, 1, 1)):
    return {"chunk_id": cid, "text": text, "doc_id": "d1", "page_num": page,
            "bbox": list(bbox), "acl": {"public": True}, "tenant_id": "t1"}


def test_synthesis_emits_bbox_citations(monkeypatch):
    monkeypatch.setattr(synthmod, "get_tunable", lambda name, container_id=None: 1.0)
    s = PdfChatState(query="q", tenant_id="t1")
    s.accessible_chunks = [_chunk_bbox("c1", "fact one")]
    out = _run(synthesize(s, Deps(llm=_CitingLlm("The fact is true [1]."))))
    assert out.citations[0]["bbox"] == [0, 0, 1, 1]
    assert out.citations[0]["page"] == 1


def test_citation_density_floor_rejects_uncited_claim(monkeypatch):
    monkeypatch.setattr(synthmod, "get_tunable", lambda name, container_id=None: 1.0)
    s = PdfChatState(query="q", tenant_id="t1")
    s.accessible_chunks = [_chunk_bbox("c1", "fact one")]
    # Two sentences, only one cited → density 0.5 < floor 1.0 → refuse.
    out = _run(synthesize(s, Deps(llm=_CitingLlm("Cited claim [1]. Uncited claim with no marker."))))
    assert out.answer == synthmod.UNGROUNDED_REFUSAL
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "synthesis or citation_density" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.synthesis'`

- [ ] **Step 3: Implement synthesis**

```python
# pdf_chat/agent/synthesis.py
"""Grounded synthesis (gpt-4o-mini) with bbox citations + citation-density floor.

Every accessible chunk becomes a numbered [N] context line carrying its bbox so
the answer's citations are click-to-highlight. After generation we measure
citation density (fraction of answer sentences carrying at least one [N] marker);
below the tunable floor we REFUSE rather than emit an ungrounded claim
(spec §3 invariant 1). No score-comparison literal here — the floor is a tunable.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pdf_chat.agent.prompts import SYSTEM_PROMPT, INSUFFICIENT_CONTEXT_MESSAGE, build_user_prompt
from pdf_chat.tunables import get_tunable, log_gate_decision

if TYPE_CHECKING:
    from pdf_chat.agent.graph import Deps
    from pdf_chat.agent.state import PdfChatState

UNGROUNDED_REFUSAL = (
    "I could not produce an answer in which every claim is supported by a cited "
    "source passage, so I am declining to answer rather than state something ungrounded."
)

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
_CITE_RE = re.compile(r"\[\d+\]")


def _attr(chunk: Any, name: str, default: Any = None) -> Any:
    return chunk.get(name, default) if isinstance(chunk, dict) else getattr(chunk, name, default)


def _citation_density(answer: str) -> float:
    sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITE_RE.search(s))
    return cited / len(sentences)


async def synthesize(state: "PdfChatState", deps: "Deps") -> "PdfChatState":
    if not state.accessible_chunks or deps.llm is None:
        state.answer = INSUFFICIENT_CONTEXT_MESSAGE
        state.citations = []
        return state

    lines: list[str] = []
    citations: list[dict] = []
    for i, chunk in enumerate(state.accessible_chunks, start=1):
        text = _attr(chunk, "text", "") or ""
        doc_id = _attr(chunk, "doc_id", "")
        page = _attr(chunk, "page_num", 0)
        bbox = _attr(chunk, "bbox", None)
        lines.append(f"[{i}] {text}    Source: {doc_id}, page {page}")
        citations.append({
            "n": i, "doc_id": str(doc_id), "page": int(page or 0),
            "bbox": bbox, "chunk_id": str(_attr(chunk, "chunk_id", "")),
        })
    state.context = "\n".join(lines)

    answer = await deps.llm.generate(SYSTEM_PROMPT, build_user_prompt(state.query, state.context))

    density = _citation_density(answer)
    floor = float(get_tunable("pdf_citation_density_floor", state.tenant_id))
    grounded = density >= floor
    log_gate_decision("citation_density", score=density, threshold=floor, passed=grounded)
    if not grounded:
        state.answer = UNGROUNDED_REFUSAL
        state.citations = []
        return state

    state.answer = answer
    state.citations = citations
    return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "synthesis or citation_density" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/synthesis.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): add grounded synthesis with bbox citations and citation-density floor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Negative-claim gate port (coverage = pages/sections in-context)

**Files:**
- Create: `pdf_chat/agent/negative_claim_gate.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Port of `app/services/erp/negative_claim_gate.py` (`NegativeClaimVerdict`, `evaluate_negative_claim`, `honest_rewrite`) to the PDF/graph world. `proven == coverage_complete AND diagnosed`. Coverage here = the relevant pages/sections were actually in-context (bbox/page coverage), so retrieval-empty ≠ absent (spec §3 invariant 2). Negative phrases are NOT a hardcoded business dictionary — they are linguistic negation markers, mirroring the ERP gate's `_NEGATIVE_PHRASES`.

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent.negative_claim_gate import (
    evaluate_pdf_negative_claim, honest_rewrite, PdfNegativeClaimVerdict,
)


def test_negative_claim_unproven_when_no_pages_in_context():
    s = PdfChatState(query="termination clause?", tenant_id="t1")
    s.accessible_chunks = []            # retrieval was empty → NOT proof of absence
    v = evaluate_pdf_negative_claim("There is no termination clause in the document.", s)
    assert v.is_negative_claim is True
    assert v.coverage_complete is False
    assert v.proven is False


def test_negative_claim_proven_when_pages_covered_and_diagnosed():
    s = PdfChatState(query="termination clause?", tenant_id="t1")
    s.accessible_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "page_num": 4, "bbox": [0, 0, 1, 1], "text": "section 4"},
        {"chunk_id": "c2", "doc_id": "d1", "page_num": 5, "bbox": [0, 0, 1, 1], "text": "section 5"},
    ]
    answer = "I reviewed pages 4-5 of d1 and found no termination clause; the contract omits it."
    v = evaluate_pdf_negative_claim(answer, s)
    assert v.coverage_complete is True
    assert v.diagnosed is True
    assert v.proven is True


def test_honest_rewrite_softens_unproven_absence():
    v = PdfNegativeClaimVerdict(is_negative_claim=True, proven=False, coverage_complete=False)
    out = honest_rewrite(v)
    assert "could not" in out.lower() or "not able to confirm" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "negative_claim" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.agent.negative_claim_gate'`

- [ ] **Step 3: Implement the gate**

```python
# pdf_chat/agent/negative_claim_gate.py
"""Negative-claim gate for PDF/graph answers (ported from
app/services/erp/negative_claim_gate.py).

A "no data / not found" claim is only honest if we PROVED coverage — the relevant
pages/sections were actually in-context (bbox/page coverage) — AND we diagnosed
what we scanned. Retrieval-empty is NOT proof of absence (spec §3 invariant 2).
proven == coverage_complete AND diagnosed.

_NEGATIVE_PHRASES are linguistic negation markers (not a business dictionary):
they mirror the ERP gate's marker list and detect the SHAPE of a no-data claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pdf_chat.agent.state import PdfChatState

_NEGATIVE_PHRASES = (
    "no data", "not found", "could not find", "there is no", "no such",
    "does not contain", "no information", "not available", "no record",
    "absent", "no mention", "no reference",
)


@dataclass
class PdfNegativeClaimVerdict:
    is_negative_claim: bool = False
    proven: bool = False
    coverage_complete: bool = False
    diagnosed: bool = False
    missing_diagnostics: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def _attr(chunk: Any, name: str, default: Any = None) -> Any:
    return chunk.get(name, default) if isinstance(chunk, dict) else getattr(chunk, name, default)


def _is_negative(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in _NEGATIVE_PHRASES)


def _coverage_complete(state: "PdfChatState") -> tuple[bool, dict]:
    """Coverage = relevant pages/sections were in-context (bbox/page coverage)."""
    pages = sorted({
        int(_attr(c, "page_num", 0) or 0) for c in state.accessible_chunks
        if _attr(c, "page_num", None) is not None
    } - {0})
    with_bbox = sum(1 for c in state.accessible_chunks if _attr(c, "bbox", None))
    signals = {"pages": pages, "chunks_in_context": len(state.accessible_chunks), "with_bbox": with_bbox}
    # Complete only if we actually had grounded pages in context (bbox present).
    complete = len(pages) > 0 and with_bbox > 0
    return complete, signals


def _diagnosed(answer: str, signals: dict) -> tuple[bool, list[str]]:
    """An honest absence names what was scanned (page numbers / 'reviewed')."""
    low = (answer or "").lower()
    missing: list[str] = []
    mentions_scan = "review" in low or "page" in low or "scan" in low or "section" in low
    if not mentions_scan:
        missing.append("no statement of which pages/sections were scanned")
    return (len(missing) == 0, missing)


def evaluate_pdf_negative_claim(answer: str, state: "PdfChatState") -> PdfNegativeClaimVerdict:
    """Return a verdict; never raises. proven == coverage_complete AND diagnosed."""
    try:
        if not _is_negative(answer):
            return PdfNegativeClaimVerdict(is_negative_claim=False)
        coverage, signals = _coverage_complete(state)
        diagnosed, missing = _diagnosed(answer, signals)
        proven = coverage and diagnosed
        return PdfNegativeClaimVerdict(
            is_negative_claim=True, proven=proven, coverage_complete=coverage,
            diagnosed=diagnosed, missing_diagnostics=missing, signals=signals,
        )
    except Exception:
        return PdfNegativeClaimVerdict(is_negative_claim=False)


def honest_rewrite(verdict: PdfNegativeClaimVerdict) -> str:
    """Replace an unproven absence with an honest 'could not confirm' statement."""
    if not verdict.coverage_complete:
        return (
            "I could not confirm whether this information is present: the relevant "
            "pages or sections were not retrieved into context, so I cannot say it is "
            "absent — only that I did not find it in what I could access."
        )
    return (
        "I was not able to confirm this from the passages I reviewed; the answer may "
        "exist in sections I did not have in context."
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "negative_claim" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/negative_claim_gate.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): port negative-claim gate to PDF/graph (coverage = pages in-context)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Wire the agentic graph (planner bypass + entity link + loop + gates)

**Files:**
- Modify: `pdf_chat/agent/graph.py`
- Test: `pdf_chat/testing/test_phase3_agent.py`

Adds `run_pdf_agent(state, deps)` orchestrating: embed → cache_check → (bypass? cache hit → audit) → plan → decompose → entity_link → tool_loop → adaptive_rerank → acl_filter → synthesize → negative-claim gate → sufficiency check → cache_write → audit. Reuses the existing nodes (`embed_query`, `cache_check`, `acl_filter`, `cache_write`, `audit`) unchanged. Wires the negative-claim gate exactly like `app/agent/graph/graph.py:1428` (`_gate_negative_claim`).

- [ ] **Step 1: Write the failing tests**

```python
# add to pdf_chat/testing/test_phase3_agent.py
from pdf_chat.agent.graph import run_pdf_agent


class _BypassLlm:
    """Planner says high-confidence simple; synthesis cites."""
    def __init__(self): self.synth_calls = 0
    async def generate(self, system, user):
        if "query router" in system.lower():
            return '{"intent":"local","confidence":0.99,"sub_queries":["q"],"output_components":["q"]}'
        self.synth_calls += 1
        return "Answer grounded [1]."


class _Emb:
    async def embed(self, text): return [0.1, 0.2]


class _OneChunkSearcher:
    def vector_search(self, qv, tenant_id, top_k=None, doc_ids=None):
        return [{"chunk_id": "c1", "text": "fact", "doc_id": "d1", "page_num": 1,
                 "bbox": [0, 0, 1, 1], "acl": {"public": True}, "tenant_id": tenant_id, "score": 0.9}]
    def resolve_entities(self, names, tenant_id, limit=8): return []


def test_bypass_skips_tool_loop(monkeypatch):
    # All thresholds permissive; bypass fires because confidence 0.99 + 1 sub-query.
    import pdf_chat.agent.planner as pl, pdf_chat.agent.graph as g, pdf_chat.agent.synthesis as sy
    monkeypatch.setattr(pl, "get_tunable", lambda n, container_id=None: 0.7)
    monkeypatch.setattr(g, "get_tunable", lambda n, container_id=None: 0.5)
    monkeypatch.setattr(sy, "get_tunable", lambda n, container_id=None: 1.0)
    s = PdfChatState(query="q", tenant_id="t1", user_id="u1", groups=["g1"])
    s.groups = ["g1"]
    out = _run(run_pdf_agent(s, Deps(embedder=_Emb(), searcher=_OneChunkSearcher(), llm=_BypassLlm())))
    # bypass path runs a single vector_search round, never escalates per-tool to graph etc.
    assert out.bypass is True
    assert "[1]" in out.answer


def test_full_path_runs_loop_when_not_bypass(monkeypatch):
    import pdf_chat.agent.planner as pl, pdf_chat.agent.graph as g, pdf_chat.agent.synthesis as sy
    import pdf_chat.agent.loop as lp
    monkeypatch.setattr(pl, "get_tunable", lambda n, container_id=None: 0.99)  # high bar → no bypass
    monkeypatch.setattr(g, "get_tunable", lambda n, container_id=None: 0.5)
    monkeypatch.setattr(sy, "get_tunable", lambda n, container_id=None: 1.0)
    monkeypatch.setattr(lp, "get_tunable", lambda n, container_id=None: 5)
    s = PdfChatState(query="q", tenant_id="t1", user_id="u1", groups=["g1"])
    out = _run(run_pdf_agent(s, Deps(embedder=_Emb(), searcher=_OneChunkSearcher(), llm=_BypassLlm())))
    assert out.bypass is False
    assert out.tool_call_count >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "bypass or full_path" -v`
Expected: FAIL — `ImportError: cannot import name 'run_pdf_agent'`

- [ ] **Step 3: Add the orchestrator to `graph.py`**

In `pdf_chat/agent/graph.py`, add imports near the top:

```python
from pdf_chat.agent.planner import plan_query
from pdf_chat.agent.entity_linker import link_entities
from pdf_chat.agent.decompose import decompose_query, sufficiency_check
from pdf_chat.agent.loop import tool_loop
from pdf_chat.agent.tools import build_phase3_tools
from pdf_chat.agent.synthesis import synthesize
from pdf_chat.agent.negative_claim_gate import evaluate_pdf_negative_claim, honest_rewrite
```

Add the bypass-loop helper and orchestrator (after `run_pdf_chat`):

```python
async def _bypass_retrieve(state: PdfChatState, deps: Deps) -> PdfChatState:
    """High-confidence simple path: a single vector_search round, no loop."""
    build_phase3_tools()
    from pdf_chat.agent.tools import TOOL_REGISTRY
    results = await TOOL_REGISTRY["vector_search"].run(state, deps)
    for c in results:
        cid = str(c.get("chunk_id", "")) if isinstance(c, dict) else ""
        if cid and cid not in state.seen_chunk_ids:
            state.seen_chunk_ids.add(cid)
            state.candidates.append(c)
    state.tool_call_count += 1
    return state


def _gate_negative_claim(state: PdfChatState) -> None:
    """Block confident-but-unproven 'no data' answers (mirror app graph wiring)."""
    verdict = evaluate_pdf_negative_claim(state.answer, state)
    if verdict.is_negative_claim and not verdict.proven:
        _logger.warning(
            "pdf_chat.negative_claim_unproven coverage=%s diagnosed=%s signals=%s",
            verdict.coverage_complete, verdict.diagnosed, verdict.signals,
        )
        state.answer = honest_rewrite(verdict)
        state.citations = []


async def run_pdf_agent(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Phase-3 agentic runtime. Plan → (bypass | loop) → rerank → ACL → synth → gates."""
    state = await embed_query(state, deps)
    if state.error:
        return state
    state = await cache_check(state, deps)
    if state.error:
        return state
    if state.cached:
        return await audit(state, deps)

    plan = await plan_query(state, deps)
    state.intent = plan.intent
    state.planner_confidence = plan.confidence
    state.bypass = plan.bypass
    state.fallback_reason = plan.fallback_reason
    state.sub_queries = plan.sub_queries
    state.output_components = plan.output_components
    state = decompose_query(state)

    if state.bypass:
        state = await _bypass_retrieve(state, deps)
    else:
        state = await link_entities(state, deps)
        state = await tool_loop(state, deps)

    state = await adaptive_rerank(state, deps)
    state.reranked = state.reranked or state.candidates
    state = await acl_filter(state, deps)
    state = await synthesize(state, deps)

    _gate_negative_claim(state)

    ok, missing = sufficiency_check(state)
    if not ok:
        log_gate_decision("sufficiency", score=len(state.output_components) - len(missing),
                          threshold=len(state.output_components), passed=False, missing=missing)
        state.answer = (
            state.answer
            + "\n\nNote: I could not find supporting evidence for: "
            + ", ".join(missing)
            + "."
        )

    state = await cache_write(state, deps)
    state = await audit(state, deps)
    return state
```

Note: `acl_filter` reads `state.reranked`; the orchestrator sets `state.reranked` before calling it.

- [ ] **Step 4: Run to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -k "bypass or full_path" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/graph.py server/pdf_chat/testing/test_phase3_agent.py
git commit -m "feat(pdf): wire agentic runtime — plan/bypass, entity-link, loop, gates, sufficiency

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Full-suite regression + invariant sweep

**Files:**
- Test: `pdf_chat/testing/test_phase3_agent.py` (no new module)

- [ ] **Step 1: Add a no-magic-literal guard test**

```python
# add to pdf_chat/testing/test_phase3_agent.py
import pathlib
import re

# Phase-3 modules that must contain NO bare score-comparison literal.
_PHASE3_FILES = [
    "planner.py", "loop.py", "decompose.py", "synthesis.py",
]
# A float/decimal literal directly on either side of a comparison operator is the smell.
_LITERAL_CMP = re.compile(r"(?:[<>]=?|==)\s*\d*\.\d+|\d*\.\d+\s*[<>]=?")


def test_phase3_modules_have_no_score_comparison_literal():
    base = pathlib.Path(__file__).resolve().parent.parent / "agent"
    offenders = []
    for fname in _PHASE3_FILES:
        text = (base / fname).read_text()
        for ln, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _LITERAL_CMP.search(line):
                offenders.append(f"{fname}:{ln}: {line.strip()}")
    assert offenders == [], "score-comparison literals found:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run the guard test**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py::test_phase3_modules_have_no_score_comparison_literal -v`
Expected: PASS (thresholds come from `get_tunable`, not literals)

- [ ] **Step 3: Run the entire Phase-3 suite**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_phase3_agent.py -v`
Expected: PASS (all tests across Tasks 1-12)

- [ ] **Step 4: Run the full pdf_chat suite (no regressions)**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing -v`
Expected: PASS — existing `test_agent.py` (the legacy linear `run_pdf_chat`) still green; new agentic path additive.

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/testing/test_phase3_agent.py
git commit -m "test(pdf): add no-magic-literal guard + Phase-3 regression sweep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (§5 Phase 3):**
- Planner/Router typed intent + named thresholds + typed fallback + bypass → Task 2.
- Entity-linking populates `state.entity` → Task 4.
- Tools (vector/graph/community/neighbors) + per-hop tenant isolation → Tasks 3, 5.
- `structured_query` Phase-4 seam only → Task 5 (`register_tool`/`TOOL_REGISTRY`).
- Loop caps (total/per-tool/depth) + monotonic-progress + drop logging → Task 6.
- Decomposition + output-completeness sufficiency → Task 7.
- Adaptive rerank skip (tunable + logged), reuse RRF/ACL → Task 8.
- Grounded synthesis + bbox + citation-density floor → Task 9.
- Negative-claim gate port (coverage = pages in-context) → Task 10.
- Wiring + bypass + gate placement → Task 11.
- Invariant 4 (no magic literals) enforced by a test → Task 12.

**Placeholder scan:** no TBD/TODO; every code step shows full code; every command has expected output.

**Type consistency:** `PlannerResult`, `Tool`/`TOOL_REGISTRY`/`register_tool`, `LoopBudget`, `PdfNegativeClaimVerdict`, `synthesize`, `adaptive_rerank` signatures are used identically across Tasks 2-12. `get_tunable`/`log_gate_decision` imported per-module so tests `monkeypatch` the module-local name.

**Note for the implementer:** Phase-0 `pdf_chat/tunables.py` and Phase-2's graph schema + searcher methods are assumed. If `tunables.py` is missing at execution time, Phase 0 must land first (it is a prerequisite, not part of this plan).

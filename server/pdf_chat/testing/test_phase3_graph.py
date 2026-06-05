"""Phase-3 Task 11 — ``run_pdf_query`` end-to-end (contract C4) over the wired
LangGraph agent. Zero infra: every backend (LLM, searcher, planner) is an
in-memory fake. Exercises the three governing paths:

  * **bypass path** — a high-confidence simple/cached query skips the tool loop
    entirely (the searcher's loop legs are never called) but still returns a
    grounded ``PdfQueryResult``;
  * **full loop path** — a non-bypass query links an entity, runs the capped
    ``multi_vector_search`` PRIMARY loop, ACL-filters, synthesizes, and returns
    citations + typed intent + provenance;
  * **proven-absence path** — a genuine "not found" over in-context evidence is
    kept (coverage proven), while a retrieval-empty "not found" is rewritten by
    the negative-claim gate (retrieval-empty ≠ absent).

These assert the C4 surface (``run_pdf_query(query, *, tenant_id, container_id,
...) -> PdfQueryResult``) and the BLOCKING ENTRY GATES the manager enforces:
multi_vector_search is PRIMARY, per-hop tenant isolation, the negative-claim
gate wraps the final answer, and bypass really skips the loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from pdf_chat.agent.graph import (
    AgentDeps,
    PdfQueryResult,
    build_agent_graph,
    build_default_deps,
    run_pdf_query,
)
from pdf_chat.agent.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from pdf_chat.agent.tools import TOOL_REGISTRY


# --------------------------------------------------------------------------- #
# In-memory fakes (mirror test_agent.py conventions)
# --------------------------------------------------------------------------- #
def _chunk(chunk_id, text, doc_id="doc1", page=1, acl=None, tenant="t1", etype="text"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "page_num": page,
        "tenant_id": tenant,
        "element_type": etype,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "acl": acl if acl is not None else {"public": True},
    }


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeSearcher:
    """Records which retrieval leg the loop calls and threads tenant per-hop.

    Returns a fixed candidate set so the monotonic-progress guard fires on the
    second round (no new chunk ids).
    """

    def __init__(self, results, *, neighbors=None):
        self._results = list(results)
        self._neighbors = neighbors or {}
        self.multi_calls = 0
        self.vector_calls = 0
        self.hybrid_calls = 0
        self.graph_calls = 0
        self.seen_tenants: list[str] = []

    def multi_vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.multi_calls += 1
        self.seen_tenants.append(tenant_id)
        return list(self._results)

    def vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.vector_calls += 1
        self.seen_tenants.append(tenant_id)
        return list(self._results)

    def hybrid_search(self, *a, **kw):  # must NEVER be called in Phase 3
        self.hybrid_calls += 1
        return list(self._results)

    def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
        self.graph_calls += 1
        self.seen_tenants.append(tenant_id)
        return []

    def entity_neighbors(self, entity, tenant_id, limit=None, doc_ids=None):
        self.seen_tenants.append(tenant_id)
        return self._neighbors.get(entity, [])

    def community_report_lookup(self, query_vec, tenant_id, limit=None):
        self.seen_tenants.append(tenant_id)
        return []


class FakePlannerLlm:
    """Returns a canned planner JSON; the synthesis call returns a grounded answer."""

    def __init__(self, *, intent="local", confidence=0.95, answer="The revenue grew 12% [1]."):
        self._intent = intent
        self._confidence = confidence
        self._answer = answer
        self.synth_calls = 0

    async def generate(self, system, user, *, container_id="", signals=None):
        if "query planner" in system.lower():
            return (
                '{"intent": "%s", "confidence": %s, "multi_part": false}'
                % (self._intent, self._confidence)
            )
        self.synth_calls += 1
        return self._answer


@dataclass
class FakeCache:
    store: dict = field(default_factory=dict)
    set_calls: int = 0

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl):
        self.set_calls += 1
        self.store[key] = value


@dataclass
class FakeAudit:
    rows: list = field(default_factory=list)

    async def write(self, **kwargs):
        self.rows.append(kwargs)


def _deps(searcher, *, llm=None, cache=None, audit=None):
    return AgentDeps(
        embedder=FakeEmbedder(),
        searcher=searcher,
        cache=cache,
        llm=llm or FakePlannerLlm(),
        audit_repo=audit,
    )


# --------------------------------------------------------------------------- #
# C4 surface
# --------------------------------------------------------------------------- #
def test_run_pdf_query_returns_contract_c4_result():
    searcher = FakeSearcher([_chunk("c1", "Revenue was $1.2M.", page=3)])
    deps = _deps(searcher, llm=FakePlannerLlm(intent="local", confidence=0.2))
    result = asyncio.run(
        run_pdf_query(
            "How did revenue do?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    assert isinstance(result, PdfQueryResult)
    assert hasattr(result, "answer")
    assert hasattr(result, "citations")
    assert hasattr(result, "intent")
    assert hasattr(result, "provenance")
    assert hasattr(result, "conflicts")
    assert result.intent == "local"


# --------------------------------------------------------------------------- #
# Full loop path — multi_vector_search is PRIMARY (blocking gate 2)
# --------------------------------------------------------------------------- #
def test_full_loop_uses_multi_vector_search_primary():
    searcher = FakeSearcher(
        [_chunk("c1", "Revenue was $1.2M.", page=3), _chunk("c2", "Costs were $900k.", page=4)]
    )
    deps = _deps(searcher, llm=FakePlannerLlm(intent="local", confidence=0.1))
    result = asyncio.run(
        run_pdf_query(
            "How did revenue and costs compare?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # PRIMARY retrieval is the fused multi_vector_search, NEVER plain vector / hybrid.
    assert searcher.multi_calls >= 1
    assert searcher.hybrid_calls == 0
    # Grounded answer with citations carrying bbox + page.
    assert result.answer == "The revenue grew 12% [1]."
    assert result.citations
    assert result.citations[0]["page"] == 3
    assert result.citations[0]["bbox"] == [0.0, 0.0, 1.0, 1.0]
    # Per-hop tenant isolation: every leg saw the tenant id.
    assert all(t == "t1" for t in searcher.seen_tenants)


# --------------------------------------------------------------------------- #
# Bypass path — high-confidence simple query skips the loop (blocking gate 6)
# --------------------------------------------------------------------------- #
def test_cached_bypass_skips_the_tool_loop():
    searcher = FakeSearcher([_chunk("c1", "x")])
    cache = FakeCache()
    deps = _deps(searcher, cache=cache, llm=FakePlannerLlm())
    # Pre-seed the version-keyed cache so the planner bypasses.
    from pdf_chat.agent.graph import _compute_cache_key
    from pdf_chat.agent.state import PdfChatState

    key = _compute_cache_key(PdfChatState(query="q", tenant_id="t1", groups=["g1"]))
    cache.store[key] = {
        "answer": "cached answer [1]",
        "citations": [{"n": 1, "doc_id": "d", "page": 2, "bbox": None}],
        "chunks_used": 1,
    }
    result = asyncio.run(
        run_pdf_query("q", tenant_id="t1", container_id="t1", groups=["g1"], deps=deps)
    )
    assert result.answer == "cached answer [1]"
    # The loop legs were never touched — the cache short-circuited retrieval.
    assert searcher.multi_calls == 0
    assert searcher.vector_calls == 0
    assert searcher.graph_calls == 0


def test_acl_empty_returns_insufficient_context_refusal():
    private = _chunk("c1", "secret", acl={"allowed_users": ["someone_else"]})
    searcher = FakeSearcher([private])
    deps = _deps(searcher, llm=FakePlannerLlm(intent="local", confidence=0.1))
    result = asyncio.run(
        run_pdf_query(
            "leak?", tenant_id="t1", container_id="t1", user_id="u1", groups=["g1"], deps=deps
        )
    )
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.citations == []


# --------------------------------------------------------------------------- #
# Proven-absence path — negative-claim gate wraps the final answer (gate 3)
# --------------------------------------------------------------------------- #
def test_retrieval_empty_negative_claim_is_rewritten():
    # The searcher returns nothing → a "not found" answer is a retrieval miss,
    # NEVER a proven absence → the gate rewrites it (retrieval-empty != absent).
    searcher = FakeSearcher([])
    deps = _deps(
        searcher,
        llm=FakePlannerLlm(intent="local", confidence=0.1, answer="There is no such data."),
    )
    result = asyncio.run(
        run_pdf_query(
            "Is there a 2099 budget?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # No accessible context → deterministic refusal (the gate never lets an
    # unproven "no data" stand).
    assert "could not confirm" in result.answer.lower() or result.answer == INSUFFICIENT_CONTEXT_MESSAGE


def test_proven_absence_over_in_context_evidence_is_kept():
    # Relevant evidence IS in context and the claimed item genuinely is not in
    # it → a "not found" is a PROVEN absence and is kept verbatim.
    chunks = [
        _chunk("c1", "The 2026 budget is $5M.", page=1),
        _chunk("c2", "The 2027 budget is $6M.", page=2),
    ]
    searcher = FakeSearcher(chunks)
    deps = _deps(
        searcher,
        llm=FakePlannerLlm(
            intent="local",
            confidence=0.1,
            answer="There is no 2099 budget mentioned in the documents [1].",
        ),
    )
    result = asyncio.run(
        run_pdf_query(
            "What is the 2099 budget?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # Coverage proven + diagnosed → the honest negative claim stands.
    assert "no 2099 budget" in result.answer.lower()


# --------------------------------------------------------------------------- #
# Per-hop tenant isolation is never dropped, even across the graph leg
# --------------------------------------------------------------------------- #
def test_entity_linked_graph_leg_threads_tenant():
    chunks = [_chunk("c1", "Acme Corporation reported strong growth.", page=1)]
    searcher = FakeSearcher(chunks, neighbors={"Acme Corporation": [{"name": "Acme", "etype": "ORG"}]})
    deps = _deps(searcher, llm=FakePlannerLlm(intent="local", confidence=0.1))
    asyncio.run(
        run_pdf_query(
            "Tell me about Acme Corporation",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # Whatever legs ran (multi + possibly graph), all were tenant-scoped to t1.
    assert searcher.seen_tenants  # at least one leg ran
    assert all(t == "t1" for t in searcher.seen_tenants)


# --------------------------------------------------------------------------- #
# deps=None wires defaults without raising; LangGraph builder is guarded
# --------------------------------------------------------------------------- #
def test_build_default_deps_returns_agent_deps():
    deps = build_default_deps()
    assert isinstance(deps, AgentDeps)


def test_build_agent_graph_is_guarded():
    # build_agent_graph either compiles a langgraph (langgraph installed) or
    # raises a RuntimeError pointing at run_pdf_query — never an ImportError.
    deps = _deps(FakeSearcher([]))
    try:
        compiled = build_agent_graph(deps)
        assert compiled is not None
    except RuntimeError as exc:
        assert "run_pdf_query" in str(exc)


def test_phase3_tools_registered_with_multi_vector_primary():
    assert "multi_vector_search" in TOOL_REGISTRY
    assert "vector_search" in TOOL_REGISTRY
    assert "graph_traverse" in TOOL_REGISTRY
    assert "community_report_lookup" in TOOL_REGISTRY
    assert "get_entity_neighbors" in TOOL_REGISTRY
    # Phase-4/5 seams stay UNregistered.
    assert "structured_query" not in TOOL_REGISTRY
    assert "glossary_lookup" not in TOOL_REGISTRY


# --------------------------------------------------------------------------- #
# Multi-part decomposition (BLOCKER fix) — components are populated and a
# partly-grounded multi-part ask is FLAGGED, never silently truncated.
# --------------------------------------------------------------------------- #
import json as _json


class MultiPartLlm:
    """Planner says multi_part=True; decompose returns components; synth answers.

    Distinguishes the three call kinds by their system prompt so one fake drives
    the whole pipeline (planner → decompose → synthesis).
    """

    def __init__(self, *, components, answer):
        self._components = components
        self._answer = answer
        self.synth_calls = 0
        self.decompose_calls = 0

    async def generate(self, system, user, *, container_id="", signals=None):
        s = system.lower()
        if "query planner" in s:
            return _json.dumps(
                {"intent": "global", "confidence": 0.9, "multi_part": True}
            )
        if "output components" in s:
            self.decompose_calls += 1
            return _json.dumps(self._components)
        self.synth_calls += 1
        return self._answer


def test_multi_part_partial_grounding_flags_ungrounded_components():
    # 3 requested components; only "revenue" is present in the evidence.
    chunks = [_chunk("c1", "Revenue grew to $1.2M this year.", page=1)]
    searcher = FakeSearcher(chunks)
    llm = MultiPartLlm(
        components=["revenue", "headcount", "market share"],
        answer="Revenue grew to $1.2M [1].",
    )
    deps = _deps(searcher, llm=llm)
    result = asyncio.run(
        run_pdf_query(
            "How did revenue, headcount and market share change?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # Decomposition actually ran (the planner's multi_part signal was wired).
    assert llm.decompose_calls == 1
    low = result.answer.lower()
    # The grounded part is asserted; the two ungrounded parts are EXPLICITLY
    # flagged (never silently dropped).
    assert "revenue" in low
    assert "headcount" in low
    assert "market share" in low
    assert "not fully address" in low or "not found in the retrieved evidence" in low


def test_fully_grounded_multi_part_answers_all_parts():
    chunks = [
        _chunk("c1", "Revenue grew to $1.2M.", page=1),
        _chunk("c2", "Headcount rose to 500 employees.", page=2),
    ]
    searcher = FakeSearcher(chunks)
    llm = MultiPartLlm(
        components=["revenue", "headcount"],
        answer="Revenue grew to $1.2M [1] and headcount rose to 500 [2].",
    )
    deps = _deps(searcher, llm=llm)
    result = asyncio.run(
        run_pdf_query(
            "How did revenue and headcount change?",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    assert llm.decompose_calls == 1
    # Every component grounded → NO partial-answer flag is appended.
    assert "not fully address" not in result.answer.lower()
    assert "revenue" in result.answer.lower()
    assert "headcount" in result.answer.lower()


# --------------------------------------------------------------------------- #
# Fix 2 — the bypass path also runs entity linking (graph leg reachable).
# --------------------------------------------------------------------------- #
def test_bypass_path_runs_entity_linking_and_graph_leg():
    chunks = [_chunk("c1", "Acme Corporation reported strong growth.", page=1)]
    # The entity "Acme Corporation" resolves in the graph (neighbors non-empty),
    # so the linker sets state.entity and the bypass retrieval includes the graph
    # leg rather than vector-only.
    searcher = FakeSearcher(
        chunks, neighbors={"Acme Corporation": [{"name": "Acme", "etype": "ORG"}]}
    )
    # High confidence + local intent → bypass=True (no cache hit).
    deps = _deps(searcher, llm=FakePlannerLlm(intent="local", confidence=0.99))
    asyncio.run(
        run_pdf_query(
            "Tell me about Acme Corporation",
            tenant_id="t1",
            container_id="t1",
            user_id="u1",
            groups=["g1"],
            deps=deps,
        )
    )
    # The graph leg ran on the bypass path (entity linking was NOT skipped).
    assert searcher.graph_calls >= 1
    assert all(t == "t1" for t in searcher.seen_tenants)

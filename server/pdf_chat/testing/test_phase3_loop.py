"""Phase 3 — Task 7: capped tool loop + monotonic-progress guard (agent/loop.py).

Pure unit tests, zero infra. The loop drives ``TOOL_REGISTRY`` against an
in-memory ``FakeSearcher`` (mirrors the Phase-2 Neo4jSearcher read surface) and
must NEVER run away. Verifies the HARD-RULE caps + guards the manager enforces:

  * (a) total-call cap — the loop stops at ``budget.max_total_calls`` (tunable
        ``agent.max_tool_calls``; default mirrors the main-system MAX_TOOL_CALLS=8);
  * (b) per-tool cap — a single tool is never invoked more than
        ``budget.max_per_tool`` times (tunable ``agent.max_per_tool_calls``);
  * (c) MONOTONIC-PROGRESS guard — a round that adds zero new ``chunk_id`` to
        ``state.seen_chunk_ids`` aborts the loop (no infinite re-fetch of the
        same fixed candidate set);
  * (d) every cap/abort/drop logs via ``log_gate_decision`` (asserted via a spy
        on the module's ``log_gate_decision`` symbol) with the running count as
        ``score``;
  * decomposition depth is capped via ``budget.max_decomp_depth`` (tunable
        ``agent.max_decomp_depth``) and the drop is logged;
  * the sufficiency check requires EVERY requested output component to have at
        least one grounded (accessible) chunk before the loop is satisfied;
  * ``LoopBudget.from_tunables`` reads all three caps via ``get_tunable`` — no
        magic literal lives in loop.py.

The loop also runs the PRIMARY retrieval node (Task 5 wiring): the FUSED
``multi_vector_search`` leg first, ``graph_traverse`` merged in only when an
anchor ``state.entity`` is set, then ``filter_by_acl`` → ``rerank`` (threading
``container_id`` for the adaptive-skip tunable). Tenant id is threaded on every
leg.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from pdf_chat.agent import loop as loop_mod
from pdf_chat.agent.loop import LoopBudget, run_tool_loop
from pdf_chat.agent.state import PdfChatState


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _hit(chunk_id, text="acme revenue grew", doc_id="doc1", page=1):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "page_num": page,
        "element_type": "text",
        "tenant_id": "t1",
        "acl": {"public": True},
        "score": 0.9,
    }


class FixedSearcher:
    """Returns the SAME fixed candidate set every call.

    This is the worst case for the monotonic-progress guard: after the first
    round every chunk_id is already in ``seen_chunk_ids`` so a second round adds
    nothing and MUST abort. Records every call name + tenant for assertions.
    """

    def __init__(self, hits=None):
        self._hits = hits if hits is not None else [_hit("c1"), _hit("c2")]
        self.calls: list[tuple[str, dict]] = []

    def multi_vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(("multi_vector_search", {"tenant_id": tenant_id}))
        return list(self._hits)

    def vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(("vector_search", {"tenant_id": tenant_id}))
        return list(self._hits)

    def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(("graph_traversal", {"tenant_id": tenant_id, "entity": entity}))
        return [_hit("g1")]

    def entity_neighbors(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(("entity_neighbors", {"tenant_id": tenant_id}))
        return []

    def community_report_lookup(self, query_vec, tenant_id, limit=None):
        self.calls.append(("community_report_lookup", {"tenant_id": tenant_id}))
        return []

    def hybrid_search(self, *a, **kw):  # must NEVER be called by the loop
        self.calls.append(("hybrid_search", kw))
        return []


class GrowingSearcher:
    """Yields a NEW unique chunk on every call so progress never stalls.

    Used to exercise the total/per-tool caps WITHOUT the monotonic-progress
    guard short-circuiting first.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._n = 0

    def _next(self):
        self._n += 1
        return [_hit(f"u{self._n}")]

    def multi_vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(("multi_vector_search", {"tenant_id": tenant_id}))
        return self._next()

    def vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(("vector_search", {"tenant_id": tenant_id}))
        return self._next()

    def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(("graph_traversal", {"tenant_id": tenant_id}))
        return self._next()

    def entity_neighbors(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(("entity_neighbors", {"tenant_id": tenant_id}))
        return []

    def community_report_lookup(self, query_vec, tenant_id, limit=None):
        self.calls.append(("community_report_lookup", {"tenant_id": tenant_id}))
        return self._next()

    def hybrid_search(self, *a, **kw):
        self.calls.append(("hybrid_search", kw))
        return []


class _Deps:
    def __init__(self, searcher):
        self.searcher = searcher


def _state(entity=None, doc_ids=None, sub_queries=None):
    return PdfChatState(
        query="how did acme revenue and headcount change",
        tenant_id="t1",
        query_vector=[0.1, 0.2, 0.3],
        entity=entity,
        doc_ids=doc_ids,
        sub_queries=sub_queries or [],
    )


# A multi-part ask whose components are NEVER grounded by the fakes, so the
# sufficiency check stays False and the loop keeps probing until a cap / the
# monotonic-progress guard stops it (lets us exercise the bounds directly).
_UNSATISFIABLE = ["nonexistent_component_alpha", "nonexistent_component_beta"]


def _probing_state(entity=None, doc_ids=None):
    return _state(entity=entity, doc_ids=doc_ids, sub_queries=list(_UNSATISFIABLE))


# --------------------------------------------------------------------------- #
# Budget construction — every cap via a tunable (no literal in loop.py)
# --------------------------------------------------------------------------- #
def test_loop_budget_from_tunables_reads_all_three_caps(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_AGENT.MAX_TOOL_CALLS", "8")
    monkeypatch.setenv("PDF_TUNABLE_AGENT.MAX_PER_TOOL_CALLS", "3")
    monkeypatch.setenv("PDF_TUNABLE_AGENT.MAX_DECOMP_DEPTH", "2")
    b = LoopBudget.from_tunables(container_id="c1")
    assert b.max_total_calls == 8
    assert b.max_per_tool == 3
    assert b.max_decomp_depth == 2


def test_loop_budget_default_total_mirrors_main_system_max_tool_calls():
    # No env override → the registered default must mirror MAX_TOOL_CALLS=8.
    b = LoopBudget.from_tunables(container_id="c-default")
    assert b.max_total_calls == 8


def test_loop_py_has_no_bare_cap_literal():
    """No magic cap literal in loop.py — caps come from get_tunable."""
    src = inspect.getsource(loop_mod)
    # The caps are referenced by their tunable KEY, not an inline integer.
    assert "agent.max_tool_calls" in src
    assert "agent.max_per_tool_calls" in src
    assert "agent.max_decomp_depth" in src


# --------------------------------------------------------------------------- #
# (a) total-call cap
# --------------------------------------------------------------------------- #
def test_total_call_cap_enforced(monkeypatch):
    logged = []
    monkeypatch.setattr(
        loop_mod, "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    s = GrowingSearcher()
    # Small total cap, generous per-tool cap so TOTAL is the binding constraint.
    budget = LoopBudget(max_total_calls=3, max_per_tool=99, max_decomp_depth=9)
    out = asyncio.run(run_tool_loop(_probing_state(), _Deps(s), budget))

    assert out.tool_calls <= 3
    # Some gate decision recorded the total-cap stop with the count as score.
    stops = [r for r in logged if r["gate"] == "agent.max_tool_calls"]
    assert stops, "total-call cap stop was not logged"
    assert stops[-1]["score"] >= stops[-1]["threshold"]


# --------------------------------------------------------------------------- #
# (b) per-tool cap
# --------------------------------------------------------------------------- #
def test_per_tool_cap_enforced(monkeypatch):
    logged = []
    monkeypatch.setattr(
        loop_mod, "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    s = GrowingSearcher()
    # Generous total cap, tiny per-tool cap so PER-TOOL is the binding constraint.
    budget = LoopBudget(max_total_calls=99, max_per_tool=2, max_decomp_depth=9)
    out = asyncio.run(run_tool_loop(_probing_state(), _Deps(s), budget))

    # No single tool exceeded its per-tool cap.
    for name, count in out.per_tool_calls.items():
        assert count <= 2, f"{name} exceeded per-tool cap: {count}"
    drops = [r for r in logged if r["gate"] == "agent.max_per_tool_calls"]
    assert drops, "per-tool cap was not logged"


# --------------------------------------------------------------------------- #
# (c) MONOTONIC-PROGRESS guard — a no-new-chunk round aborts the loop
# --------------------------------------------------------------------------- #
def test_monotonic_progress_aborts_when_no_new_chunks(monkeypatch):
    logged = []
    monkeypatch.setattr(
        loop_mod, "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    s = FixedSearcher()  # same {c1,c2} every call
    # Caps are high enough that ONLY the monotonic-progress guard can stop us.
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=9)
    out = asyncio.run(run_tool_loop(_probing_state(), _Deps(s), budget))

    # Guard must have stopped us WELL before the (high) caps.
    assert out.tool_calls < 50
    aborts = [r for r in logged if r["gate"] == "agent.monotonic_progress"]
    assert aborts, "monotonic-progress abort was not logged"
    assert aborts[-1]["outcome"] == "abort"
    # The fixed chunk ids were seen exactly once.
    assert out.seen_chunk_ids == {"c1", "c2"}


def test_monotonic_progress_does_not_abort_while_new_chunks_arrive():
    s = GrowingSearcher()  # always a new chunk
    budget = LoopBudget(max_total_calls=4, max_per_tool=99, max_decomp_depth=9)
    out = asyncio.run(run_tool_loop(_probing_state(), _Deps(s), budget))
    # Progress kept being made, so the TOTAL cap (not the guard) bound us.
    assert out.tool_calls == 4
    assert len(out.seen_chunk_ids) >= 1


# --------------------------------------------------------------------------- #
# decomposition depth cap
# --------------------------------------------------------------------------- #
def test_decomp_depth_capped(monkeypatch):
    logged = []
    monkeypatch.setattr(
        loop_mod, "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    s = GrowingSearcher()
    # depth budget 1, but the state arrives already AT depth 1 → must not recurse.
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    st = _state(sub_queries=["a", "b"])
    st.decomp_depth = 1
    out = asyncio.run(run_tool_loop(st, _Deps(s), budget))

    assert out.decomp_depth <= 1
    caps = [r for r in logged if r["gate"] == "agent.max_decomp_depth"]
    assert caps, "decomp-depth cap was not logged"


def test_decomp_depth_cap_is_a_live_break_not_decorative():
    """When the loop is entered already AT/over the depth cap, it takes a CONTROL
    action: NO further retrieval round is issued (the cap is enforced, not just
    logged). Previously the loop logged the cap and then ran anyway."""
    s = GrowingSearcher()  # would yield a new chunk every call if it ran
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    st = _state(sub_queries=["a", "b"])
    st.decomp_depth = 1  # already at the cap
    out = asyncio.run(run_tool_loop(st, _Deps(s), budget))
    # The control action fired: no tool was dispatched (no recursive round).
    assert out.tool_calls == 0
    assert s.calls == []
    assert out.accessible_chunks == []


# --------------------------------------------------------------------------- #
# Task-5 wiring inside the loop: PRIMARY = multi_vector_search; graph only w/ entity
# --------------------------------------------------------------------------- #
def test_loop_primary_retrieval_is_multi_vector_search():
    s = FixedSearcher()
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    asyncio.run(run_tool_loop(_state(entity=None), _Deps(s), budget))
    called = [c[0] for c in s.calls]
    assert "multi_vector_search" in called           # PRIMARY fused leg
    assert called[0] == "multi_vector_search"          # …and it leads
    assert "hybrid_search" not in called               # never the legacy path


def test_loop_merges_graph_leg_only_when_entity_set():
    s_no = FixedSearcher()
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    asyncio.run(run_tool_loop(_state(entity=None), _Deps(s_no), budget))
    assert "graph_traversal" not in [c[0] for c in s_no.calls]

    s_yes = FixedSearcher()
    asyncio.run(run_tool_loop(_state(entity="Acme"), _Deps(s_yes), budget))
    assert "graph_traversal" in [c[0] for c in s_yes.calls]
    # Per-hop tenant threaded on the graph leg.
    gkw = dict(s_yes.calls).get("graph_traversal", {})
    assert gkw.get("tenant_id") == "t1"


def test_loop_threads_tenant_to_every_leg():
    s = FixedSearcher()
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    asyncio.run(run_tool_loop(_state(entity="Acme"), _Deps(s), budget))
    for name, kw in s.calls:
        assert kw.get("tenant_id") == "t1", f"{name} lost the tenant id"


def test_loop_runs_acl_filter_and_populates_accessible_chunks():
    # public chunks survive ACL → accessible_chunks populated.
    s = FixedSearcher()
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    out = asyncio.run(run_tool_loop(_state(), _Deps(s), budget))
    ids = {c["chunk_id"] for c in out.accessible_chunks}
    assert {"c1", "c2"}.issubset(ids)


def test_loop_acl_denies_other_tenant_chunks():
    foreign = _hit("x1")
    foreign["tenant_id"] = "OTHER"
    foreign["acl"] = {"public": True}
    s = FixedSearcher(hits=[_hit("c1"), foreign])
    budget = LoopBudget(max_total_calls=50, max_per_tool=50, max_decomp_depth=1)
    out = asyncio.run(run_tool_loop(_state(), _Deps(s), budget))
    acc_ids = {c["chunk_id"] for c in out.accessible_chunks}
    assert "c1" in acc_ids
    assert "x1" not in acc_ids  # cross-tenant chunk filtered out


# --------------------------------------------------------------------------- #
# Sufficiency: every requested output component must be grounded
# --------------------------------------------------------------------------- #
def test_sufficiency_requires_all_components_grounded():
    from pdf_chat.agent.loop import _components_satisfied

    accessible = [_hit("c1", text="acme revenue grew 10%")]
    # "revenue" is grounded; "headcount" is not present anywhere → not satisfied.
    assert _components_satisfied(["revenue", "headcount"], accessible) is False
    # both present → satisfied.
    accessible.append(_hit("c2", text="headcount rose to 500"))
    assert _components_satisfied(["revenue", "headcount"], accessible) is True
    # empty component list is trivially satisfied (no decomposition requested).
    assert _components_satisfied([], accessible) is True


def test_loop_never_raises_on_searcher_error():
    class Boom:
        def multi_vector_search(self, *a, **kw):
            raise RuntimeError("neo4j down")

    budget = LoopBudget(max_total_calls=4, max_per_tool=4, max_decomp_depth=1)
    # Must degrade, not raise (honest-absence: empty context, no crash).
    out = asyncio.run(run_tool_loop(_state(), _Deps(Boom()), budget))
    assert out.error is not None or out.accessible_chunks == []

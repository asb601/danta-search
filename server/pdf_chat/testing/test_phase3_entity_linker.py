"""Phase-3 Task 3 — entity linking tests (mock searcher, zero infra).

``link_entities(state, deps)`` resolves the query to a graph entity and
populates ``state.entity`` *before* the graph tools become reachable. It is the
gate that makes ``graph_traverse`` / ``get_entity_neighbors`` usable: if no
entity links above the confidence floor, ``state.entity`` stays ``None`` and the
graph leg is skipped (vector-only retrieval), so a query that names nothing in
the tenant's graph can never trigger a (wasteful, possibly cross-tenant) walk.

These tests assert:

  * a query naming an entity present in the (mocked) graph resolver populates
    ``state.entity`` with the canonical graph name;
  * an unrecognized query leaves ``state.entity is None`` (graph tools unreachable);
  * the confidence floor is read via ``get_tunable(container_id,
    "agent.entity_link_min_confidence", ...)`` — no inline literal — and the
    decision is logged via ``log_gate_decision`` (assert the returned record's
    ``outcome``/``gate``);
  * ``tenant_id`` (and ``doc_ids``) are threaded to every searcher call
    (per-hop tenant isolation preserved — Spec §3 inv 3);
  * the linker NEVER raises (a searcher that blows up degrades to ``None``);
  * a pre-set ``state.entity`` is respected (idempotent — no re-resolution).
"""
from __future__ import annotations

import asyncio

import pytest

from pdf_chat.agent.entity_linker import link_entities
from pdf_chat.agent.graph import Deps
from pdf_chat.agent.state import PdfChatState


# --------------------------------------------------------------------------- #
# Fake resolver searcher — exposes the resolve_entity seam the linker prefers,
# AND records every call's kwargs so we can assert tenant/doc threading.
# --------------------------------------------------------------------------- #
class FakeResolverSearcher:
    """A graph that knows a fixed set of entities.

    ``resolve_entity(text, tenant_id, doc_ids=None, limit=None)`` returns the
    catalog rows whose ``name`` token-overlaps ``text``, each with a ``score``
    in [0, 1] (mocked similarity). Records calls for assertions.
    """

    def __init__(self, catalog):
        # catalog: list of {"name", "score"} the graph would return per probe,
        # keyed implicitly by lexical overlap against the candidate text.
        self._catalog = catalog
        self.calls = []

    def resolve_entity(self, text, tenant_id, doc_ids=None, limit=None):
        self.calls.append(
            {"text": text, "tenant_id": tenant_id, "doc_ids": doc_ids, "limit": limit}
        )
        toks = {t.lower() for t in text.split()}
        out = []
        for row in self._catalog:
            name_toks = {t.lower() for t in row["name"].split()}
            if toks & name_toks:
                out.append(dict(row))
        return sorted(out, key=lambda r: r["score"], reverse=True)


class BoomSearcher:
    def resolve_entity(self, *a, **k):
        raise RuntimeError("graph down")


def _state(query, tenant_id="t1", **kw):
    return PdfChatState(query=query, tenant_id=tenant_id, **kw)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_recognized_entity_populates_state_entity():
    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.9}])
    deps = Deps(searcher=searcher)
    state = _state("What was Acme Corporation revenue in Q3?")

    out = asyncio.run(link_entities(state, deps))

    assert out.entity == "Acme Corporation"


def test_unrecognized_query_leaves_entity_none():
    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.9}])
    deps = Deps(searcher=searcher)
    state = _state("summarize the document please")

    out = asyncio.run(link_entities(state, deps))

    assert out.entity is None  # graph tools stay unreachable


def test_low_confidence_match_left_none():
    # The only graph hit scores below the floor → not linked.
    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.05}])
    deps = Deps(searcher=searcher)
    state = _state("Acme Corporation outlook")

    out = asyncio.run(link_entities(state, deps))

    assert out.entity is None


def test_confidence_floor_read_from_tunable_and_logged(monkeypatch):
    import pdf_chat.agent.entity_linker as el

    seen_keys = []
    real_get = el.get_tunable

    def spy_get(container_id, key, *a, **k):
        seen_keys.append(key)
        return real_get(container_id, key, *a, **k)

    logged = []
    real_log = el.log_gate_decision

    def spy_log(name, **kw):
        rec = real_log(name, **kw)
        logged.append(rec)
        return rec

    monkeypatch.setattr(el, "get_tunable", spy_get)
    monkeypatch.setattr(el, "log_gate_decision", spy_log)

    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.9}])
    deps = Deps(searcher=searcher)
    out = asyncio.run(
        link_entities(_state("Acme Corporation revenue"), deps, container_id="c1")
    )

    assert "agent.entity_link_min_confidence" in seen_keys
    assert logged, "the link decision must be logged via log_gate_decision"
    rec = logged[-1]
    assert rec["gate"] == "agent.entity_link"
    assert rec["outcome"] == "linked"
    assert out.entity == "Acme Corporation"


def test_below_floor_logs_skip_outcome(monkeypatch):
    import pdf_chat.agent.entity_linker as el

    logged = []
    real_log = el.log_gate_decision
    monkeypatch.setattr(
        el, "log_gate_decision", lambda n, **kw: (logged.append(real_log(n, **kw)) or logged[-1])
    )

    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.01}])
    deps = Deps(searcher=searcher)
    out = asyncio.run(link_entities(_state("Acme Corporation"), deps, container_id="c1"))

    assert out.entity is None
    assert logged[-1]["outcome"] == "skip"


def test_tenant_and_docs_threaded_to_searcher():
    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.9}])
    deps = Deps(searcher=searcher)
    state = _state("Acme Corporation revenue", tenant_id="tenantX", doc_ids=["d1", "d2"])

    asyncio.run(link_entities(state, deps))

    assert searcher.calls, "searcher must be probed"
    for c in searcher.calls:
        assert c["tenant_id"] == "tenantX"
        assert c["doc_ids"] == ["d1", "d2"]


def test_never_raises_on_searcher_error():
    deps = Deps(searcher=BoomSearcher())
    out = asyncio.run(link_entities(_state("Acme Corporation revenue"), deps))
    assert out.entity is None  # degrades gracefully, no exception


def test_no_searcher_is_a_noop():
    deps = Deps(searcher=None)
    out = asyncio.run(link_entities(_state("Acme Corporation revenue"), deps))
    assert out.entity is None


def test_preset_entity_respected_no_reresolution():
    searcher = FakeResolverSearcher([{"name": "Acme Corporation", "score": 0.9}])
    deps = Deps(searcher=searcher)
    state = _state("Acme Corporation revenue", entity="Globex Inc")

    out = asyncio.run(link_entities(state, deps))

    assert out.entity == "Globex Inc"  # caller-supplied anchor wins
    assert searcher.calls == []  # no graph probe when entity already set


def test_best_scoring_candidate_wins():
    searcher = FakeResolverSearcher(
        [
            {"name": "Acme Corporation", "score": 0.6},
            {"name": "Acme Holdings", "score": 0.95},
        ]
    )
    deps = Deps(searcher=searcher)
    out = asyncio.run(link_entities(_state("Tell me about Acme"), deps))
    assert out.entity == "Acme Holdings"

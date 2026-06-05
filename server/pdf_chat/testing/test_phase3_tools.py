"""Phase 3 — Task 4: Tool Protocol + TOOL_REGISTRY + register_tool (contract C3).

Pure unit tests, zero infra. The 5 Phase-3 read tools wrap the Phase-2
``Neo4jSearcher`` surface via in-memory fakes. Verifies:

  * ``register_tool`` keys the registry by ``.name`` and returns the tool;
  * re-registering the same name raises (no silent shadowing);
  * the Phase-4 ``structured_query`` / Phase-5 ``glossary_lookup`` names are a
    reserved SEAM — present in the reservation set, ABSENT from the registry
    (no impl);
  * every Phase-3 tool satisfies the ``Tool`` Protocol (``name`` + awaitable
    ``run``);
  * ``multi_vector_search`` wraps the FUSED ``searcher.multi_vector_search`` leg
    (the PRIMARY retrieval — NOT plain ``vector_search`` / ``hybrid_search``);
  * per-hop tenant args (``tenant_id`` + ``doc_ids``) are threaded to every
    searcher leg the tools call.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from pdf_chat.agent import tools as tools_mod
from pdf_chat.agent.tools import (
    PHASE3_TOOL_NAMES,
    RESERVED_TOOL_NAMES,
    TOOL_REGISTRY,
    Tool,
    register_tool,
)
from pdf_chat.agent.state import PdfChatState


# --------------------------------------------------------------------------- #
# In-memory fake searcher — records which leg the tool called + the kwargs.
# --------------------------------------------------------------------------- #
def _hit(chunk_id, text="t", doc_id="doc1", page=1, etype="text"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "page_num": page,
        "element_type": etype,
        "acl": {"public": True},
    }


class RecordingSearcher:
    """Mirrors the Neo4jSearcher read surface; records call name + kwargs."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(
            ("vector_search", {"tenant_id": tenant_id, "top_k": top_k, "doc_ids": doc_ids})
        )
        return [_hit("v1")]

    def multi_vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.calls.append(
            (
                "multi_vector_search",
                {"tenant_id": tenant_id, "top_k": top_k, "doc_ids": doc_ids},
            )
        )
        return [_hit("m1"), _hit("m2")]

    def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(
            (
                "graph_traversal",
                {"entity": entity, "tenant_id": tenant_id, "limit": limit, "doc_ids": doc_ids},
            )
        )
        return [_hit("g1")]

    def entity_neighbors(self, entity, tenant_id, limit=None, doc_ids=None):
        self.calls.append(
            (
                "entity_neighbors",
                {"entity": entity, "tenant_id": tenant_id, "limit": limit, "doc_ids": doc_ids},
            )
        )
        return [{"name": "Acme", "etype": "ORG", "normalized_value": "acme"}]

    def community_report_lookup(self, query_vec, tenant_id, limit=None):
        self.calls.append(
            ("community_report_lookup", {"tenant_id": tenant_id, "limit": limit})
        )
        return [{"community_id": "c1", "report": "r", "citations": ["v1"], "score": 0.9}]

    def hybrid_search(self, *a, **kw):  # must NEVER be called by the primary tool
        self.calls.append(("hybrid_search", kw))
        return []


class _Deps:
    def __init__(self, searcher):
        self.searcher = searcher


def _state(entity=None, doc_ids=None):
    return PdfChatState(
        query="q",
        tenant_id="t1",
        doc_ids=doc_ids,
        entity=entity,
        query_vector=[0.1, 0.2, 0.3],
    )


# --------------------------------------------------------------------------- #
# Registry + Protocol (C3)
# --------------------------------------------------------------------------- #
def test_phase3_tools_registered_by_name():
    for name in PHASE3_TOOL_NAMES:
        assert name in TOOL_REGISTRY, f"{name} missing from TOOL_REGISTRY"
        assert TOOL_REGISTRY[name].name == name


def test_register_tool_returns_tool_and_keys_by_name():
    sentinel = list(TOOL_REGISTRY.keys())

    class _T:
        name = "tmp_register_probe"

        async def run(self, state, deps, **kw):
            return []

    t = _T()
    try:
        out = register_tool(t)
        assert out is t
        assert TOOL_REGISTRY["tmp_register_probe"] is t
    finally:
        TOOL_REGISTRY.pop("tmp_register_probe", None)
    # we did not perturb the pre-existing registrations
    assert set(sentinel).issubset(set(TOOL_REGISTRY.keys()))


def test_reregistering_same_name_raises():
    class _T:
        name = "multi_vector_search"

        async def run(self, state, deps, **kw):
            return []

    with pytest.raises((ValueError, KeyError)):
        register_tool(_T())


def test_phase4_phase5_names_are_seam_only_absent_from_registry():
    # Reserved by name (the loop/integration can discover the seam) ...
    assert "structured_query" in RESERVED_TOOL_NAMES
    assert "glossary_lookup" in RESERVED_TOOL_NAMES
    # ... but UNIMPLEMENTED here — never in the live registry.
    assert "structured_query" not in TOOL_REGISTRY
    assert "glossary_lookup" not in TOOL_REGISTRY


def test_every_phase3_tool_satisfies_tool_protocol():
    for name in PHASE3_TOOL_NAMES:
        tool = TOOL_REGISTRY[name]
        assert isinstance(tool, Tool)  # runtime_checkable Protocol
        assert isinstance(tool.name, str) and tool.name
        assert inspect.iscoroutinefunction(tool.run)


# --------------------------------------------------------------------------- #
# multi_vector_search is the PRIMARY retrieval — wraps the FUSED searcher call
# --------------------------------------------------------------------------- #
def test_multi_vector_search_tool_wraps_fused_searcher_call():
    s = RecordingSearcher()
    deps = _Deps(s)
    tool = TOOL_REGISTRY["multi_vector_search"]

    out = asyncio.run(tool.run(_state(), deps))

    called = [c[0] for c in s.calls]
    assert "multi_vector_search" in called
    assert "vector_search" not in called  # not the plain leg
    assert "hybrid_search" not in called  # not the legacy state-machine path
    assert [h["chunk_id"] for h in out] == ["m1", "m2"]


def test_multi_vector_search_threads_tenant_and_doc_ids():
    s = RecordingSearcher()
    deps = _Deps(s)
    tool = TOOL_REGISTRY["multi_vector_search"]

    asyncio.run(tool.run(_state(doc_ids=["doc1", "doc2"]), deps))

    kw = dict(s.calls)["multi_vector_search"]
    assert kw["tenant_id"] == "t1"
    assert kw["doc_ids"] == ["doc1", "doc2"]


def test_vector_search_tool_threads_tenant_and_doc_ids():
    s = RecordingSearcher()
    tool = TOOL_REGISTRY["vector_search"]
    asyncio.run(tool.run(_state(doc_ids=["d9"]), _Deps(s)))
    kw = dict(s.calls)["vector_search"]
    assert kw["tenant_id"] == "t1"
    assert kw["doc_ids"] == ["d9"]


# --------------------------------------------------------------------------- #
# graph_traverse / get_entity_neighbors — per-hop tenant + entity threading
# --------------------------------------------------------------------------- #
def test_graph_traverse_threads_per_hop_tenant_and_entity():
    s = RecordingSearcher()
    tool = TOOL_REGISTRY["graph_traverse"]
    out = asyncio.run(tool.run(_state(entity="Acme", doc_ids=["d1"]), _Deps(s)))
    kw = dict(s.calls)["graph_traversal"]
    assert kw["tenant_id"] == "t1"          # per-hop tenant arg threaded
    assert kw["entity"] == "Acme"
    assert kw["doc_ids"] == ["d1"]
    assert [h["chunk_id"] for h in out] == ["g1"]


def test_graph_traverse_without_entity_returns_empty_no_call():
    s = RecordingSearcher()
    tool = TOOL_REGISTRY["graph_traverse"]
    out = asyncio.run(tool.run(_state(entity=None), _Deps(s)))
    assert out == []
    assert "graph_traversal" not in [c[0] for c in s.calls]


def test_get_entity_neighbors_wraps_searcher_entity_neighbors():
    s = RecordingSearcher()
    tool = TOOL_REGISTRY["get_entity_neighbors"]
    out = asyncio.run(tool.run(_state(entity="Acme"), _Deps(s)))
    kw = dict(s.calls)["entity_neighbors"]
    assert kw["tenant_id"] == "t1"
    assert kw["entity"] == "Acme"
    assert out[0]["name"] == "Acme"


def test_community_report_lookup_tool_threads_tenant():
    s = RecordingSearcher()
    tool = TOOL_REGISTRY["community_report_lookup"]
    out = asyncio.run(tool.run(_state(), _Deps(s)))
    kw = dict(s.calls)["community_report_lookup"]
    assert kw["tenant_id"] == "t1"
    assert out[0]["community_id"] == "c1"


def test_tools_module_has_no_score_literal_seam_comment():
    # The Phase-4/5 seam must be a NAME reservation, not a stub impl.
    src = inspect.getsource(tools_mod)
    assert "structured_query" in src
    assert "glossary_lookup" in src

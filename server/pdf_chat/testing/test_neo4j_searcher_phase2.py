"""Phase-2 Task 1 (contract C2) — Neo4jSearcher rewrite tests.

These run with ZERO infra (no neo4j / azure / redis). They assert the C2
invariants the searcher MUST satisfy:

  * per-hop tenant isolation on EVERY traversal cypher — a variable-length walk
    must filter ``tenant_id`` on ALL path nodes
    (``ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)``), never just the
    anchor, so a graph walk can never straddle tenants via a shared entity name;
  * every cypher constant binds ``$tenant_id``;
  * ``multi_vector_search`` queries chunk + section-card + doc-card vector spaces
    and RRF-fuses the THREE rankings via ``retrieval.rrf.rrf`` (asserted: rrf is
    called with exactly 3 lists);
  * no bare score-comparison literal lives in the searcher source (Spec §3 inv 4);
  * the Phase-1 ``vector_search`` / ``hybrid_search`` surface still works.

The driver is mocked exactly like the writer/searcher seam: a fake driver →
session → ``run`` that records the (cypher, params) and replays canned records,
so the real ``neo4j`` package is never required.
"""
from __future__ import annotations

import inspect
import re

import pytest

from pdf_chat.retrieval import neo4j_searcher as S
from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher


# --------------------------------------------------------------------------- #
# Fake neo4j driver/session (records every cypher + params, replays records)
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, sink, rows_by_call):
        self._sink = sink
        self._rows_by_call = rows_by_call

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        self._sink.append((cypher, params))
        # Pop the next canned row-batch (default empty).
        rows = self._rows_by_call.pop(0) if self._rows_by_call else []
        return _FakeResult(rows)


class _FakeDriver:
    def __init__(self, sink, rows_by_call):
        self._sink = sink
        self._rows_by_call = rows_by_call

    def session(self, database=None):
        return _FakeSession(self._sink, self._rows_by_call)

    def close(self):
        pass


def _searcher(rows_by_call=None):
    s = Neo4jSearcher(uri="bolt://x", user="u", password="p", database="neo4j")
    sink: list = []
    s._driver = _FakeDriver(sink, list(rows_by_call or []))
    s._sink = sink  # type: ignore[attr-defined]
    return s, sink


# --------------------------------------------------------------------------- #
# Per-hop tenant isolation in the cypher constants (C2 invariant)
# --------------------------------------------------------------------------- #
def _all_cypher_constants() -> dict[str, str]:
    return {
        name: val
        for name, val in vars(S).items()
        if name.isupper() and name.endswith("_CYPHER") and isinstance(val, str)
    }


def test_there_are_traversal_cypher_constants():
    consts = _all_cypher_constants()
    assert consts, "expected *_CYPHER constants in the searcher module"


def test_graph_cypher_uses_per_hop_tenant_isolation():
    cy = S._GRAPH_CYPHER
    # Every node on the matched variable-length path must be tenant-filtered.
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cy
    assert ":MENTIONS" in cy and ":RELATED_TO" in cy


def test_entity_neighbors_cypher_uses_per_hop_tenant_isolation():
    cy = S._NEIGHBORS_CYPHER
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cy
    assert ":RELATED_TO" in cy


def test_entity_neighbors_doc_filter_targets_doc_id_not_src_chunk():
    # Regression: the doc subset previously compared a CHUNK id (rel.src_chunk)
    # against DOCUMENT ids, so it silently matched nothing. The fixed filter must
    # scope on the mentioning chunk's doc_id (chunks carry doc_id from Phase 1),
    # never on src_chunk.
    cy = S._NEIGHBORS_CYPHER
    assert "rel.src_chunk IN $doc_ids" not in cy  # the buggy comparison is gone
    assert "dc.doc_id IN $doc_ids" in cy           # filter on the chunk's document
    # the doc-scope hop stays tenant-isolated (never straddles tenants)
    assert "dc.tenant_id = $tenant_id" in cy
    assert "(dc:Chunk)-[:MENTIONS]->(other)" in cy


def test_every_variable_length_cypher_is_per_hop_isolated():
    # Any cypher that does a variable-length walk (``*1..``) MUST carry the
    # path-level ALL(...) tenant filter — never an anchor-only filter.
    for name, cy in _all_cypher_constants().items():
        if "*" in cy and "nodes(path)" in cy.lower() or "*1.." in cy:
            assert (
                "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cy
            ), f"{name} does a var-length walk without per-hop tenant isolation"


def test_every_cypher_binds_tenant_id():
    for name, cy in _all_cypher_constants().items():
        assert "$tenant_id" in cy, f"{name} does not bind $tenant_id"


def test_card_vector_cypher_filters_tenant_on_node():
    cy = S._CARD_VECTOR_CYPHER
    assert "node.tenant_id = $tenant_id" in cy


def test_community_report_cypher_filters_tenant_on_node():
    cy = S._COMMUNITY_REPORT_CYPHER
    assert "node.tenant_id = $tenant_id" in cy


# --------------------------------------------------------------------------- #
# No magic literals (Spec §3 inv 4)
# --------------------------------------------------------------------------- #
def test_no_score_literal_in_searcher_source():
    src = inspect.getsource(S)
    # No bare float comparison literal — thresholds come from get_tunable.
    assert not re.search(r"score\s*[<>]=?\s*0\.\d", src)


# --------------------------------------------------------------------------- #
# multi_vector_search fuses chunk + section-card + doc-card via RRF (3 spaces)
# --------------------------------------------------------------------------- #
def test_multi_vector_search_fuses_three_spaces_via_rrf(monkeypatch):
    # Stub the three legs with disjoint canned hits.
    s = Neo4jSearcher(uri="bolt://x", user="u", password="p")
    s.vector_search = lambda qv, tid, top_k=None, doc_ids=None: [  # type: ignore[assignment]
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
    ]
    s._card_vector_search = lambda qv, tid, index_name, top_k=None, doc_ids=None: (  # type: ignore[assignment]
        [{"chunk_id": "sec1"}] if "section" in index_name else [{"chunk_id": "doc1"}]
    )

    captured = {}

    def _spy_rrf(lists, k=60):
        captured["lists"] = lists
        captured["k"] = k
        # Flatten deterministically for the assertion below.
        flat: list[str] = []
        for lst in lists:
            for x in lst:
                if x not in flat:
                    flat.append(x)
        return flat

    monkeypatch.setattr(S, "rrf", _spy_rrf)

    out = s.multi_vector_search([0.1, 0.2], "t1")
    # rrf MUST be called with exactly 3 ranked id-lists (chunk/section/doc).
    assert len(captured["lists"]) == 3
    assert captured["lists"][0] == ["c1", "c2"]
    assert captured["lists"][1] == ["sec1"]
    assert captured["lists"][2] == ["doc1"]
    # The fused output returns chunk dicts in fused order.
    ids = [c["chunk_id"] for c in out]
    assert ids[:2] == ["c1", "c2"]
    assert "sec1" in ids and "doc1" in ids


def test_multi_vector_search_threads_doc_ids_to_every_leg():
    s = Neo4jSearcher(uri="bolt://x", user="u", password="p")
    seen = {"vector": None, "cards": []}

    def _vec(qv, tid, top_k=None, doc_ids=None):
        seen["vector"] = doc_ids
        return []

    def _card(qv, tid, index_name, top_k=None, doc_ids=None):
        seen["cards"].append((index_name, doc_ids))
        return []

    s.vector_search = _vec  # type: ignore[assignment]
    s._card_vector_search = _card  # type: ignore[assignment]
    s.multi_vector_search([0.1], "t1", doc_ids=["d9"])
    assert seen["vector"] == ["d9"]
    assert all(doc_ids == ["d9"] for _, doc_ids in seen["cards"])
    # Two card legs: section-card + doc-card.
    assert len(seen["cards"]) == 2


# --------------------------------------------------------------------------- #
# Methods exist and thread tenant_id into the cypher params
# --------------------------------------------------------------------------- #
def test_entity_neighbors_binds_tenant_and_doc_ids():
    s, sink = _searcher(rows_by_call=[[{"name": "Globex", "etype": "org"}]])
    out = s.entity_neighbors("Acme", "t1", doc_ids=["d1"])
    assert out == [{"name": "Globex", "etype": "org"}]
    cypher, params = sink[0]
    assert params["tenant_id"] == "t1"
    assert params["doc_ids"] == ["d1"]
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cypher


def test_community_report_lookup_binds_tenant():
    s, sink = _searcher(
        rows_by_call=[[{"report": "summary", "citations": ["c1"]}]]
    )
    out = s.community_report_lookup([0.1, 0.2], "t1")
    assert out == [{"report": "summary", "citations": ["c1"]}]
    _, params = sink[0]
    assert params["tenant_id"] == "t1"


def test_graph_traversal_still_works_per_hop_isolated():
    s, sink = _searcher(rows_by_call=[[{"chunk_id": "c1", "acl": None}]])
    out = s.graph_traversal("Acme", "t1", doc_ids=["d1"])
    assert out[0]["chunk_id"] == "c1"
    assert out[0]["acl"] == {}  # deserialized
    cypher, params = sink[0]
    assert params["tenant_id"] == "t1"
    assert "ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)" in cypher


def test_vector_search_still_works():
    s, sink = _searcher(rows_by_call=[[{"chunk_id": "c1", "acl": "{}"}]])
    out = s.vector_search([0.1], "t1")
    assert out[0]["chunk_id"] == "c1"
    _, params = sink[0]
    assert params["tenant_id"] == "t1"


def test_card_vector_search_binds_index_and_tenant():
    s, sink = _searcher(rows_by_call=[[{"chunk_id": "sec1", "acl": None}]])
    out = s._card_vector_search(
        [0.1], "t1", "section_card_vector_index", doc_ids=["d1"]
    )
    assert out[0]["chunk_id"] == "sec1"
    _, params = sink[0]
    assert params["tenant_id"] == "t1"
    assert params["index_name"] == "section_card_vector_index"
    assert params["doc_ids"] == ["d1"]


def test_hybrid_search_still_fuses():
    s = Neo4jSearcher(uri="bolt://x", user="u", password="p")
    s.vector_search = lambda qv, tid, top_k=None, doc_ids=None: [{"chunk_id": "a"}, {"chunk_id": "b"}]  # type: ignore[assignment]
    s.graph_traversal = lambda e, tid, limit=None, doc_ids=None: [{"chunk_id": "b"}, {"chunk_id": "c"}]  # type: ignore[assignment]
    out = s.hybrid_search([0.1], "t1", entity="VendorX")
    ids = [c["chunk_id"] for c in out]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_searcher_constructs_without_infra():
    # Constructing never opens a connection; pure-construct must not raise.
    Neo4jSearcher(uri="bolt://x", user="u", password="p")


def test_methods_raise_clear_error_without_driver():
    s = Neo4jSearcher(uri="bolt://x", user="u", password="p")
    # No driver injected and neo4j not installed → clear RuntimeError on call.
    if not S._HAS_NEO4J:
        with pytest.raises(RuntimeError):
            s.entity_neighbors("Acme", "t1")

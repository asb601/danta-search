"""Pure unit tests for Team C retrieval logic (Stages 3,4,5,7,8).

These run with ZERO infra: no neo4j, redis, cohere, sentence-transformers, or
openai required. They exercise the pure functions and the guarded-import
degradation paths.
"""
from __future__ import annotations

import json

from pdf_chat.ingestion.ton_schema import Chunk, ElementType
from pdf_chat.retrieval import (
    ROUTE_IMMEDIATE,
    ROUTE_ON_DEMAND_TABLE,
    ROUTE_ON_DEMAND_VISION,
    RedisCache,
    assemble_context,
    cache_key,
    filter_by_acl,
    insufficient_context,
    rerank,
    route_by_element_type,
    rrf,
)
from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher, deserialize_acl


# --------------------------------------------------------------------------- #
# RRF
# --------------------------------------------------------------------------- #
def test_rrf_single_list_preserves_order():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_merges_two_lists():
    # "b" appears highly in both lists → should rank first.
    vector = ["a", "b", "c"]
    graph = ["b", "d"]
    fused = rrf([vector, graph])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_formula_score_exact():
    # Single doc at rank 0 with k=60 → score = 1/(60+0+1) = 1/61.
    # Doc at rank 0 in two lists → 2/61, must outrank a rank-0 single doc.
    fused = rrf([["x", "y"], ["x"]], k=60)
    assert fused[0] == "x"


def test_rrf_tie_is_stable_by_first_seen():
    # Two disjoint lists, each one element at rank 0 → equal scores.
    # First-seen order ("p" before "q") breaks the tie deterministically.
    assert rrf([["p"], ["q"]]) == ["p", "q"]
    assert rrf([["q"], ["p"]]) == ["q", "p"]


def test_rrf_higher_rank_beats_lower():
    # Same doc set, different positions: rank-0 doc beats rank-2 doc.
    fused = rrf([["a", "z", "b"]])
    assert fused.index("a") < fused.index("b")


def test_rrf_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


# --------------------------------------------------------------------------- #
# ACL
# --------------------------------------------------------------------------- #
def _chunk(cid, tenant, acl):
    return Chunk(
        chunk_id=cid,
        doc_id="doc1",
        page_num=1,
        element_type=ElementType.TEXT,
        text="t",
        reading_order=0,
        tenant_id=tenant,
        acl=acl,
    )


def test_acl_allow_by_user():
    c = _chunk("c1", "t1", {"allowed_users": ["u1"]})
    acc, denied = filter_by_acl([c], "u1", [], "t1")
    assert acc == [c]
    assert denied == []


def test_acl_allow_by_group():
    c = _chunk("c1", "t1", {"allowed_groups": ["finance"]})
    acc, denied = filter_by_acl([c], "u9", ["finance", "hr"], "t1")
    assert acc == [c] and denied == []


def test_acl_allow_by_public():
    c = _chunk("c1", "t1", {"public": True})
    acc, denied = filter_by_acl([c], "anyone", [], "t1")
    assert acc == [c] and denied == []


def test_acl_deny_no_match():
    c = _chunk("c1", "t1", {"allowed_users": ["other"]})
    acc, denied = filter_by_acl([c], "u1", ["hr"], "t1")
    assert acc == []
    assert denied == ["c1"]


def test_acl_deny_cross_tenant_even_if_public():
    # Tenant mismatch denies unconditionally — public does NOT override.
    c = _chunk("c1", "t2", {"public": True})
    acc, denied = filter_by_acl([c], "u1", [], "t1")
    assert acc == []
    assert denied == ["c1"]


def test_acl_deny_tenant_mismatch_with_user_grant():
    c = _chunk("c1", "tOTHER", {"allowed_users": ["u1"]})
    acc, denied = filter_by_acl([c], "u1", [], "t1")
    assert acc == [] and denied == ["c1"]


def test_acl_handles_dict_chunks():
    c = {
        "chunk_id": "d1",
        "tenant_id": "t1",
        "acl": {"allowed_groups": ["eng"]},
    }
    acc, denied = filter_by_acl([c], "u1", ["eng"], "t1")
    assert acc == [c] and denied == []


def test_acl_mixed_accessible_and_denied_order_preserved():
    a = _chunk("a", "t1", {"public": True})
    b = _chunk("b", "t1", {"allowed_users": ["nope"]})
    cc = _chunk("c", "t1", {"allowed_groups": ["g1"]})
    acc, denied = filter_by_acl([a, b, cc], "u1", ["g1"], "t1")
    assert acc == [a, cc]
    assert denied == ["b"]


def test_insufficient_context():
    assert insufficient_context([], 1) is True
    assert insufficient_context(["x"], 1) is False
    assert insufficient_context(["x"], 2) is True
    assert insufficient_context(["x", "y"], 2) is False


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #
def test_cache_key_deterministic():
    k1 = cache_key("hello", "t1", ["a", "b"])
    k2 = cache_key("hello", "t1", ["a", "b"])
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_group_order_invariant():
    assert cache_key("q", "t1", ["a", "b", "c"]) == cache_key("q", "t1", ["c", "a", "b"])


def test_cache_key_differs_on_query_tenant_groups():
    base = cache_key("q", "t1", ["a"])
    assert cache_key("q2", "t1", ["a"]) != base
    assert cache_key("q", "t2", ["a"]) != base
    assert cache_key("q", "t1", ["b"]) != base


def test_cache_key_empty_groups():
    assert len(cache_key("q", "t1", [])) == 64


# --------------------------------------------------------------------------- #
# Cache key — acl_version + doc_ids (revocation surface, Security #5)
# --------------------------------------------------------------------------- #
def test_cache_key_acl_version_changes_key():
    base = cache_key("q", "t1", ["a"], acl_version="0")
    bumped = cache_key("q", "t1", ["a"], acl_version="1")
    assert base != bumped  # a revoke bumps acl_version → old answers unreachable


def test_cache_key_doc_ids_change_key_and_order_invariant():
    whole = cache_key("q", "t1", ["a"])  # doc_ids=None → whole tenant
    scoped = cache_key("q", "t1", ["a"], doc_ids=["d1", "d2"])
    assert whole != scoped
    # doc_id order does not matter
    assert cache_key("q", "t1", ["a"], doc_ids=["d1", "d2"]) == cache_key(
        "q", "t1", ["a"], doc_ids=["d2", "d1"]
    )
    # empty list (explicit no docs) is distinct from None (whole tenant)
    assert cache_key("q", "t1", ["a"], doc_ids=[]) != whole


def test_cache_key_defaults_backward_compatible_shape():
    # Default acl_version="0" + doc_ids=None still yields a 64-char hex digest.
    assert len(cache_key("q", "t1", ["a"])) == 64


# --------------------------------------------------------------------------- #
# RedisCache — dict value contract round-trip via a fake redis (Security A3)
# --------------------------------------------------------------------------- #
class _FakeRedisClient:
    """Minimal in-memory redis stand-in (string values, like decode_responses)."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.last_ex = None

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        # Mirror redis: values are strings.
        assert isinstance(value, str)
        self.store[key] = value
        self.last_ex = ex
        return True


def test_redis_cache_dict_roundtrip():
    cache = RedisCache(url="redis://x", ttl_seconds=123)
    fake = _FakeRedisClient()
    cache._client = fake  # inject the fake client (bypass lazy connect)

    payload = {"answer": "hi [1]", "citations": [{"n": 1}], "chunks_used": 1}
    assert cache.set("k", payload) is True
    # stored as JSON string
    assert json.loads(fake.store["k"]) == payload
    assert fake.last_ex == 123
    # get returns the deserialized dict
    assert cache.get("k") == payload


def test_redis_cache_get_miss_and_corrupt_value_return_none():
    cache = RedisCache(url="redis://x")
    fake = _FakeRedisClient()
    cache._client = fake
    assert cache.get("absent") is None  # miss
    fake.store["bad"] = "not-json{"
    assert cache.get("bad") is None  # corrupt → MISS, never raises


# --------------------------------------------------------------------------- #
# Neo4jSearcher — ACL deserialization helper (Agent↔retrieval contract A2)
# --------------------------------------------------------------------------- #
def test_deserialize_acl_from_json_string():
    raw = {"chunk_id": "c1", "acl": json.dumps({"public": True})}
    out = deserialize_acl(raw)
    assert out["acl"] == {"public": True}
    assert isinstance(out["acl"], dict)


def test_deserialize_acl_passthrough_dict():
    raw = {"chunk_id": "c1", "acl": {"allowed_groups": ["eng"]}}
    assert deserialize_acl(raw)["acl"] == {"allowed_groups": ["eng"]}


def test_deserialize_acl_none_and_malformed_fail_closed():
    assert deserialize_acl({"chunk_id": "c1", "acl": None})["acl"] == {}
    # malformed JSON → empty dict (chunk will be denied by filter_by_acl)
    assert deserialize_acl({"chunk_id": "c1", "acl": "{broken"})["acl"] == {}


def test_deserialize_acl_does_not_mutate_input():
    raw = {"chunk_id": "c1", "acl": json.dumps({"public": True})}
    deserialize_acl(raw)
    assert raw["acl"] == json.dumps({"public": True})  # original untouched


# --------------------------------------------------------------------------- #
# Neo4jSearcher.hybrid_search — vector + graph RRF fusion (Agent↔retrieval A1)
# --------------------------------------------------------------------------- #
class _StubSearcher(Neo4jSearcher):
    """Override the two legs with canned hits so fusion logic runs infra-free."""

    def __init__(self, vector_hits, graph_hits):
        # Skip the real __init__ (which reads settings); we only test fusion.
        self._vector_hits = vector_hits
        self._graph_hits = graph_hits
        self.vector_kwargs = None
        self.graph_called = False

    def vector_search(self, query_vec, tenant_id, top_k=None, doc_ids=None):
        self.vector_kwargs = {"tenant_id": tenant_id, "top_k": top_k, "doc_ids": doc_ids}
        return list(self._vector_hits)

    def graph_traversal(self, entity, tenant_id, limit=None, doc_ids=None):
        self.graph_called = True
        return list(self._graph_hits)


def test_hybrid_search_vector_only_when_no_entity():
    s = _StubSearcher(
        vector_hits=[{"chunk_id": "a"}, {"chunk_id": "b"}],
        graph_hits=[{"chunk_id": "z"}],
    )
    out = s.hybrid_search([0.1], "t1", doc_ids=["d1"])
    assert [c["chunk_id"] for c in out] == ["a", "b"]  # graph leg skipped
    assert s.graph_called is False
    assert s.vector_kwargs["doc_ids"] == ["d1"]  # doc_ids threaded to the leg


def test_hybrid_search_fuses_vector_and_graph_via_rrf():
    # "b" appears in BOTH legs near the top → RRF should rank it first.
    s = _StubSearcher(
        vector_hits=[{"chunk_id": "a"}, {"chunk_id": "b"}],
        graph_hits=[{"chunk_id": "b"}, {"chunk_id": "c"}],
    )
    out = s.hybrid_search([0.1], "t1", entity="VendorX")
    ids = [c["chunk_id"] for c in out]
    assert s.graph_called is True
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def test_route_text_immediate():
    assert route_by_element_type("text") == ROUTE_IMMEDIATE
    assert route_by_element_type(ElementType.TEXT) == ROUTE_IMMEDIATE


def test_route_table_on_demand():
    assert route_by_element_type("table") == ROUTE_ON_DEMAND_TABLE
    assert route_by_element_type(ElementType.TABLE) == ROUTE_ON_DEMAND_TABLE


def test_route_image_on_demand_vision():
    assert route_by_element_type("image") == ROUTE_ON_DEMAND_VISION
    assert route_by_element_type(ElementType.IMAGE) == ROUTE_ON_DEMAND_VISION


def test_route_unknown_defaults_immediate():
    assert route_by_element_type("formula") == ROUTE_IMMEDIATE
    assert route_by_element_type("bogus") == ROUTE_IMMEDIATE


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def test_assemble_numbers_text_chunks_and_sources():
    chunks = [
        {"text": "alpha", "doc_id": "INV-1", "page_num": 3},
        {"text": "beta", "doc_id": "INV-2", "page_num": 7},
    ]
    out = assemble_context(chunks)
    assert "[1] alpha" in out
    assert "[2] beta" in out
    assert "Source: INV-1, page 3" in out
    assert "Source: INV-2, page 7" in out


def test_assemble_table_image_graph_tags():
    out = assemble_context(
        text_chunks=[{"text": "t", "doc_id": "d", "page_num": 1}],
        table_results=["| a | b |"],
        image_descriptions=["a chart of revenue"],
        graph_nodes=["VendorX -> InvoiceY"],
    )
    assert "[1] t" in out
    assert "[TABLE-1] | a | b |" in out
    assert "[IMAGE-1] a chart of revenue" in out
    assert "[GRAPH-1] VendorX -> InvoiceY" in out


def test_assemble_empty_sections_omitted():
    out = assemble_context(text_chunks=[{"text": "only", "doc_id": "d", "page_num": 1}])
    assert "[TABLE" not in out
    assert "[IMAGE" not in out
    assert "[GRAPH" not in out


def test_assemble_deterministic():
    chunks = [{"text": "x", "doc_id": "d", "page_num": 2}]
    assert assemble_context(chunks) == assemble_context(chunks)


def test_assemble_dataclass_chunks():
    c = Chunk(
        chunk_id="c1",
        doc_id="DOC9",
        page_num=5,
        element_type=ElementType.TEXT,
        text="from dataclass",
        reading_order=0,
        tenant_id="t1",
    )
    out = assemble_context([c])
    assert "[1] from dataclass" in out
    assert "Source: DOC9, page 5" in out


# --------------------------------------------------------------------------- #
# Reranker (pure fallback path — no cohere / sentence-transformers installed)
# --------------------------------------------------------------------------- #
def test_rerank_fallback_preserves_order_and_truncates():
    cands = [{"text": f"c{i}"} for i in range(20)]
    out = rerank("q", cands, top_n=5)
    assert out == cands[:5]


def test_rerank_empty():
    assert rerank("q", [], top_n=5) == []

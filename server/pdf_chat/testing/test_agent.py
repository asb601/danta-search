"""Pure unit tests for Team D — schemas + the PDF chat agent.

Zero infra: the agent runs against in-memory fake adapters. No Neo4j, Redis,
OpenAI, Celery, or other teams' modules are required. Verifies:

  * the node sequence produces a grounded answer with citations,
  * a cache hit short-circuits retrieval/synthesis,
  * ACL-empty → deterministic "insufficient accessible context" (no hallucination),
  * Pydantic schemas round-trip.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from pdf_chat.agent.graph import Deps, run_pdf_chat
from pdf_chat.agent.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from pdf_chat.agent.state import PdfChatState
from pdf_chat.schemas.pdf_schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DeleteResponse,
    DocumentSummary,
    StatusResponse,
    UploadResponse,
)
from pdf_chat.models.enums import DocStatus


# --------------------------------------------------------------------------- #
# In-memory fakes
# --------------------------------------------------------------------------- #
def _chunk(chunk_id, text, doc_id="doc1", page=1, acl=None, tenant="t1", etype="text"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "doc_id": doc_id,
        "page_num": page,
        "tenant_id": tenant,
        "element_type": etype,
        "acl": acl if acl is not None else {"public": True},
    }


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]


class FakeSearcher:
    """Matches the real Neo4jSearcher.hybrid_search signature (vector+graph fused).

    Sync method (the real Neo4j driver is sync) — the agent awaits only if the
    result is awaitable, so a plain list return works.
    """

    def __init__(self, results):
        self._results = results
        self.calls = 0
        self.last_kwargs = None

    def hybrid_search(
        self,
        query_vector,
        tenant_id,
        doc_ids=None,
        vector_top_k=None,
        graph_top_k=None,
        entity=None,
    ):
        self.calls += 1
        self.last_kwargs = {
            "query_vector": query_vector,
            "tenant_id": tenant_id,
            "doc_ids": doc_ids,
            "vector_top_k": vector_top_k,
            "graph_top_k": graph_top_k,
            "entity": entity,
        }
        return list(self._results)


class FakeReranker:
    def __init__(self):
        self.calls = 0

    async def rerank(self, query, candidates, top_n):
        self.calls += 1
        return list(candidates)[:top_n]


@dataclass
class FakeCache:
    store: dict = field(default_factory=dict)
    set_calls: int = 0

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl):
        self.set_calls += 1
        self.store[key] = value


class FakeLlm:
    def __init__(self):
        self.calls = 0
        self.last_system = None

    async def generate(self, system, user):
        self.calls += 1
        self.last_system = system
        return "The revenue grew 12% [1]."


@dataclass
class FakeAudit:
    rows: list = field(default_factory=list)

    async def write(self, **kwargs):
        self.rows.append(kwargs)


def _full_deps(searcher_results, cache=None):
    return Deps(
        embedder=FakeEmbedder(),
        searcher=FakeSearcher(searcher_results),
        reranker=FakeReranker(),
        cache=cache or FakeCache(),
        extractor=None,
        llm=FakeLlm(),
        audit_repo=FakeAudit(),
    )


# --------------------------------------------------------------------------- #
# Agent tests
# --------------------------------------------------------------------------- #
def test_happy_path_produces_answer_with_citations():
    chunks = [
        _chunk("c1", "Revenue was $1.2M.", page=3),
        _chunk("c2", "Costs were $900k.", page=4),
    ]
    deps = _full_deps(chunks)
    state = PdfChatState(query="How did revenue do?", tenant_id="t1", user_id="u1", groups=["g1"])

    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.error is None
    assert result.cached is False
    assert result.answer == "The revenue grew 12% [1]."
    assert result.chunks_used() == 2
    # citations are numbered [N] with doc + page
    assert result.citations[0] == {"n": 1, "doc_id": "doc1", "page": 3}
    assert result.citations[1]["page"] == 4
    # full node sequence ran
    assert deps.embedder.calls == 1
    assert deps.searcher.calls == 1
    assert deps.reranker.calls == 1
    assert deps.llm.calls == 1
    assert deps.cache.set_calls == 1          # answer cached
    assert len(deps.audit_repo.rows) == 1     # audit written
    assert deps.audit_repo.rows[0]["returned_chunks"] == ["c1", "c2"]


def test_cache_hit_short_circuits():
    cache = FakeCache()
    deps = _full_deps([_chunk("c1", "x")], cache=cache)

    # Pre-seed the cache under the key the agent will compute.
    state0 = PdfChatState(query="q", tenant_id="t1", groups=["g1"])
    from pdf_chat.agent.graph import _compute_cache_key

    key = _compute_cache_key(state0)
    cache.store[key] = {"answer": "cached answer [1]", "citations": [{"n": 1, "doc_id": "d", "page": 2}], "chunks_used": 1}

    state = PdfChatState(query="q", tenant_id="t1", groups=["g1"])
    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.cached is True
    assert result.answer == "cached answer [1]"
    assert result.citations == [{"n": 1, "doc_id": "d", "page": 2}]
    # retrieval + synthesis were skipped
    assert deps.searcher.calls == 0
    assert deps.reranker.calls == 0
    assert deps.llm.calls == 0
    assert cache.set_calls == 0  # never re-write on a hit
    # SECURITY (#6): a cache hit must still be audited, marked cache_hit=True.
    assert len(deps.audit_repo.rows) == 1
    assert deps.audit_repo.rows[0]["cache_hit"] is True
    assert deps.audit_repo.rows[0]["query_hash"] == key


def test_acl_empty_returns_insufficient_context_no_hallucination():
    # Chunk is private to another user/group → denied for this principal.
    private = _chunk("c1", "secret", acl={"allowed_users": ["someone_else"]})
    deps = _full_deps([private])
    state = PdfChatState(query="leak?", tenant_id="t1", user_id="u1", groups=["g1"])

    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.citations == []
    assert result.chunks_used() == 0
    assert result.denied_ids == ["c1"]
    # LLM must NOT be called (no hallucination)
    assert deps.llm.calls == 0
    # refusal is not cached
    assert deps.cache.set_calls == 0
    # but the denial IS audited
    assert deps.audit_repo.rows[0]["denied_chunks"] == ["c1"]


def test_tenant_mismatch_is_denied():
    foreign = _chunk("c9", "other tenant data", tenant="t2", acl={"public": True})
    deps = _full_deps([foreign])
    state = PdfChatState(query="q", tenant_id="t1", user_id="u1", groups=["g1"])

    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.denied_ids == ["c9"]
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE


def test_runs_with_no_searcher_degrades_gracefully():
    # Minimal deps (no infra at all) — must not raise.
    state = PdfChatState(query="q", tenant_id="t1")
    result = asyncio.run(run_pdf_chat(state, Deps()))
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.error is None


def test_hybrid_retrieve_calls_hybrid_search_with_contract_signature():
    # A4: the node must call searcher.hybrid_search with the frozen kwargs.
    chunks = [_chunk("c1", "x"), _chunk("c2", "y")]
    deps = _full_deps(chunks)
    state = PdfChatState(
        query="q", tenant_id="t1", user_id="u1", groups=["g1"],
        doc_ids=["d1"], entity="VendorX", top_k=7,
    )
    asyncio.run(run_pdf_chat(state, deps))
    kw = deps.searcher.last_kwargs
    assert kw["tenant_id"] == "t1"
    assert kw["doc_ids"] == ["d1"]
    assert kw["entity"] == "VendorX"
    assert kw["vector_top_k"] == 7  # state.top_k override flows to vector_top_k


def test_fail_closed_missing_tenant_denies_all(monkeypatch):
    # B8: a missing tenant_id must deny every chunk (default None != any tenant).
    public = _chunk("c1", "data", acl={"public": True}, tenant="")
    deps = _full_deps([public])
    state = PdfChatState(query="q", tenant_id="", user_id="u1", groups=["g1"])

    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.denied_ids == ["c1"]
    assert deps.llm.calls == 0


def test_insufficient_context_below_floor_refuses(monkeypatch):
    # B9: with min_accessible_chunks=2, a single accessible chunk must refuse.
    import pdf_chat.agent.graph as graph_mod
    from pdf_chat.config import PdfSettings

    # Bump the floor to 2 for this test only.
    monkeypatch.setattr(
        graph_mod, "get_pdf_settings",
        lambda: PdfSettings(min_accessible_chunks=2),
    )
    one = _chunk("c1", "only one accessible", acl={"public": True})
    deps = _full_deps([one])
    state = PdfChatState(query="q", tenant_id="t1", user_id="u1", groups=["g1"])

    result = asyncio.run(run_pdf_chat(state, deps))

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert deps.llm.calls == 0          # no hallucination on thin context
    assert deps.cache.set_calls == 0    # below-floor refusal is not cached


# --------------------------------------------------------------------------- #
# Schema round-trip tests
# --------------------------------------------------------------------------- #
def test_upload_response_roundtrip():
    r = UploadResponse(upload_id="abc", status=DocStatus.UPLOADED, deduplicated=True)
    dumped = r.model_dump()
    assert dumped["status"] == "uploaded"
    assert UploadResponse.model_validate(dumped) == r


def test_status_response_defaults_and_roundtrip():
    s = StatusResponse(upload_id="abc", status=DocStatus.PARTIALLY_INDEXED, page_count=10,
                       pages_succeeded=8, pages_failed=1, pages_pending=1)
    assert StatusResponse.model_validate(s.model_dump()) == s


def test_chat_request_response_roundtrip():
    req = ChatRequest(query="hi", tenant_id="t1", doc_ids=["d1"], top_k=5)
    assert ChatRequest.model_validate(req.model_dump()) == req

    resp = ChatResponse(
        answer="a [1]",
        citations=[Citation(n=1, doc_id="d1", page=2)],
        chunks_used=1,
        cached=False,
    )
    back = ChatResponse.model_validate(resp.model_dump())
    assert back == resp
    assert back.citations[0].page == 2


def test_chat_request_rejects_empty_query():
    with pytest.raises(Exception):
        ChatRequest(query="", tenant_id="t1")


def test_document_summary_and_delete_roundtrip():
    d = DocumentSummary(upload_id="u", status=DocStatus.INDEXED, page_count=3, mime_type="application/pdf")
    assert DocumentSummary.model_validate(d.model_dump()) == d

    dr = DeleteResponse(upload_id="u", deleted=True, chunks_removed=42)
    assert DeleteResponse.model_validate(dr.model_dump()) == dr

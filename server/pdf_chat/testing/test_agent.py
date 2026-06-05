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
        self.last_container_id = None
        self.last_signals = None

    async def generate(self, system, user, *, container_id="", signals=None):
        self.calls += 1
        self.last_system = system
        self.last_container_id = container_id
        self.last_signals = signals
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


# ---------------------------------------------------------------------------
# Task 6 — PdfLlm synthesis adapter (gpt-4o-mini only, prompt caching)
# ---------------------------------------------------------------------------
def test_pdf_llm_routes_model_via_router_and_prompt_caches(monkeypatch):
    # B1: the adapter must resolve the model through model_router.select_model
    # (NOT a direct deployment lookup) with the QUERY_SYNTHESIS task, and use the
    # returned ModelChoice.model_id. I1: a stable prompt-cache routing hint is
    # passed and the system prompt is the cacheable first message.
    from pdf_chat.retrieval import llm as llm_mod
    from pdf_chat.model_router import ModelChoice, TaskClass

    captured = {}

    class _FakeMsgs:
        def create(self, **kwargs):
            captured.update(kwargs)
            class _R:
                choices = [type("C", (), {"message": type("M", (), {"content": "grounded answer"})()})()]
            return _R()

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeMsgs()})()

    select_calls = {}

    def _fake_select(*, task, container_id, signals, **_):
        select_calls["task"] = task
        select_calls["container_id"] = container_id
        select_calls["signals"] = signals
        return ModelChoice(provider="azure", model_id="gpt-4o-mini", is_strong=False)

    monkeypatch.setattr(llm_mod, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(llm_mod, "select_model", _fake_select)

    adapter = llm_mod.PdfLlm()
    out = asyncio.run(adapter.generate("SYS", "USER", container_id="c-1", signals={"x": 1}))
    assert out == "grounded answer"
    # Routed through select_model with the synthesis task + tenant scope + signals.
    assert select_calls["task"] == TaskClass.QUERY_SYNTHESIS
    assert select_calls["container_id"] == "c-1"
    assert select_calls["signals"] == {"x": 1}
    # The router's chosen model id is the one actually called (never gpt-4o).
    assert captured["model"] == "gpt-4o-mini"
    # System prompt sent as a cacheable first message (prompt caching).
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "SYS"
    # A stable prompt-cache routing hint is passed (keyed on the system prompt).
    assert captured["user"] == llm_mod._prompt_cache_key("SYS")


# ---------------------------------------------------------------------------
# Fix 10 — PdfLlm.generate wires cost tracking into the synthesis call
# ---------------------------------------------------------------------------
def _fake_llm_client_with_usage(model_id: str, *, prompt_tokens: int,
                                completion_tokens: int):
    usage = type("Usage", (), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })()

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ans"})()})()]

    _Resp.usage = usage
    _Resp.model = model_id

    class _FakeMsgs:
        def create(self, **kwargs):
            return _Resp()

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeMsgs()})()

    return lambda: _FakeClient()


def test_pdf_llm_records_synthesis_cost(monkeypatch):
    """generate() must record one 'synthesis' cost-tracker call with the response's
    usage tokens + the router-selected model id (Fix 10)."""
    from pdf_chat.retrieval import llm as llm_mod
    from pdf_chat.model_router import ModelChoice
    from pdf_chat.observability.cost_tracker import get_cost_tracker

    get_cost_tracker().reset("c-cost")

    monkeypatch.setattr(
        llm_mod, "_build_client",
        _fake_llm_client_with_usage("gpt-4o-mini", prompt_tokens=120,
                                    completion_tokens=30),
    )
    monkeypatch.setattr(
        llm_mod, "select_model",
        lambda *, task, container_id, signals, **_: ModelChoice(
            provider="azure", model_id="gpt-4o-mini", is_strong=False
        ),
    )

    out = asyncio.run(llm_mod.PdfLlm().generate("SYS", "USER", container_id="c-cost"))
    assert out == "ans"

    snap = get_cost_tracker().snapshot("c-cost")
    assert snap["llm_calls"] == 1
    assert snap["prompt_tokens"] == 120
    assert snap["completion_tokens"] == 30
    assert snap["by_phase"]["synthesis"]["llm_calls"] == 1
    assert snap["policy_violations"] == 0  # gpt-4o-mini is allowed


def test_pdf_llm_flags_gpt4o_policy_violation(monkeypatch):
    """If the router somehow selected a gpt-4o (non-mini) model, generate() still
    records the cost AND flags the policy violation (Fix 10)."""
    from pdf_chat.retrieval import llm as llm_mod
    from pdf_chat.model_router import ModelChoice
    from pdf_chat.observability.cost_tracker import get_cost_tracker

    get_cost_tracker().reset("c-viol")

    monkeypatch.setattr(
        llm_mod, "_build_client",
        _fake_llm_client_with_usage("gpt-4o", prompt_tokens=10, completion_tokens=5),
    )
    monkeypatch.setattr(
        llm_mod, "select_model",
        lambda *, task, container_id, signals, **_: ModelChoice(
            provider="azure", model_id="gpt-4o", is_strong=True
        ),
    )

    asyncio.run(llm_mod.PdfLlm().generate("SYS", "USER", container_id="c-viol"))
    snap = get_cost_tracker().snapshot("c-viol")
    assert snap["llm_calls"] == 1
    assert snap["policy_violations"] == 1


def test_pdf_llm_generate_does_not_raise_when_response_lacks_usage(monkeypatch):
    """A fake response without .usage must never raise (getattr-guarded) and must
    still return the answer unchanged (Fix 10)."""
    from pdf_chat.retrieval import llm as llm_mod
    from pdf_chat.model_router import ModelChoice
    from pdf_chat.observability.cost_tracker import get_cost_tracker

    get_cost_tracker().reset("c-nousage")

    class _FakeMsgs:
        def create(self, **kwargs):
            class _R:
                choices = [type("C", (), {"message": type("M", (), {"content": "ans"})()})()]
            return _R()  # no .usage attribute at all

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeMsgs()})()

    monkeypatch.setattr(llm_mod, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(
        llm_mod, "select_model",
        lambda *, task, container_id, signals, **_: ModelChoice(
            provider="azure", model_id="gpt-4o-mini", is_strong=False
        ),
    )

    out = asyncio.run(llm_mod.PdfLlm().generate("SYS", "USER", container_id="c-nousage"))
    assert out == "ans"


# ---------------------------------------------------------------------------
# Task 7 — OnDemandExtractor + QueryAuditRepo adapters
# ---------------------------------------------------------------------------
def test_on_demand_extractor_passthrough_for_text_chunk():
    from pdf_chat.retrieval.extractor import OnDemandExtractor

    chunk = {"chunk_id": "c1", "element_type": "text", "text": "already here"}
    out = asyncio.run(OnDemandExtractor().extract(chunk))
    assert out["text"] == "already here"   # text chunks are returned unchanged


def test_query_audit_repo_writes_via_injected_sink():
    from pdf_chat.agent.audit import QueryAuditRepo

    rows = []
    repo = QueryAuditRepo(sink=lambda row: rows.append(row))
    asyncio.run(repo.write(
        user_id="u1", tenant_id="t1", query_hash="h", query_text="q",
        returned_chunks=["c1"], denied_chunks=["c2"], cache_hit=True,
    ))
    assert rows[0]["tenant_id"] == "t1"
    assert rows[0]["cache_hit"] is True
    assert rows[0]["returned_chunks"] == ["c1"]


# ---------------------------------------------------------------------------
# Task 5 — context token budget in assemble_context
# ---------------------------------------------------------------------------
from pdf_chat.agent.graph import assemble_context


def _budget_chunk(cid, text, page=1):
    return {"chunk_id": cid, "text": text, "doc_id": "d1", "page_num": page,
            "tenant_id": "t1"}


def test_assemble_context_truncates_to_token_budget(monkeypatch):
    # 4 chunks of ~10 "tokens" each; budget low enough to admit only the first 2.
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "25")
    chunks = [_budget_chunk(f"c{i}", " ".join(["word"] * 10)) for i in range(4)]
    state = PdfChatState(query="q", tenant_id="t1")
    state.accessible_chunks = chunks
    out = asyncio.run(assemble_context(state, Deps()))
    # Only the chunks that fit under the budget are cited; nothing crashes.
    assert len(out.citations) < 4
    assert len(out.citations) >= 1
    assert all(f"[{c['n']}]" in out.context for c in out.citations)


def test_assemble_context_keeps_all_when_under_budget(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "100000")
    chunks = [_budget_chunk(f"c{i}", "short") for i in range(3)]
    state = PdfChatState(query="q", tenant_id="t1")
    state.accessible_chunks = chunks
    out = asyncio.run(assemble_context(state, Deps()))
    assert len(out.citations) == 3

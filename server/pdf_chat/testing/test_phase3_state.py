"""Phase 3 Task 1 — PdfChatState agentic-field extension.

Asserts the locked agentic fields exist with safe defaults and that the
pre-Phase-3 fields + `chunks_used()` remain intact (back-compat). The state is
a pure dataclass with no infra imports, so this runs with zero infra installed.
"""
from __future__ import annotations

from pdf_chat.agent.state import PdfChatState


def _fresh() -> PdfChatState:
    return PdfChatState(query="q", tenant_id="t")


# --- New agentic-field defaults (locked in the plan) ---

def test_intent_defaults_local():
    assert _fresh().intent == "local"


def test_planner_confidence_defaults_zero():
    assert _fresh().planner_confidence == 0.0


def test_fallback_reason_defaults_none():
    assert _fresh().fallback_reason is None


def test_bypass_defaults_false():
    assert _fresh().bypass is False


def test_sub_queries_defaults_empty_list():
    assert _fresh().sub_queries == []


def test_decomp_depth_defaults_zero():
    assert _fresh().decomp_depth == 0


def test_tool_calls_defaults_zero():
    assert _fresh().tool_calls == 0


def test_per_tool_calls_defaults_empty_dict():
    assert _fresh().per_tool_calls == {}


def test_seen_chunk_ids_defaults_empty_set():
    s = _fresh()
    assert s.seen_chunk_ids == set()
    assert isinstance(s.seen_chunk_ids, set)


def test_router_signals_defaults_empty_dict():
    assert _fresh().router_signals == {}


def test_provenance_defaults_empty_dict():
    assert _fresh().provenance == {}


def test_conflicts_defaults_empty_list():
    assert _fresh().conflicts == []


# --- Mutable defaults are per-instance (no shared aliasing) ---

def test_mutable_defaults_are_not_shared():
    a = _fresh()
    b = _fresh()
    a.sub_queries.append("x")
    a.per_tool_calls["multi_vector_search"] = 1
    a.seen_chunk_ids.add("c1")
    a.router_signals["cross_domain"] = True
    a.provenance[1] = "stated"
    a.conflicts.append({"k": "v"})
    assert b.sub_queries == []
    assert b.per_tool_calls == {}
    assert b.seen_chunk_ids == set()
    assert b.router_signals == {}
    assert b.provenance == {}
    assert b.conflicts == []


# --- Back-compat: existing fields + method unchanged ---

def test_back_compat_existing_fields():
    s = PdfChatState(query="q", tenant_id="t")
    assert s.query == "q"
    assert s.tenant_id == "t"
    assert s.user_id == ""
    assert s.groups == []
    assert s.doc_ids is None
    assert s.top_k is None
    assert s.entity is None
    assert s.acl_version == "0"
    assert s.query_vector is None
    assert s.candidates == []
    assert s.reranked == []
    assert s.accessible_chunks == []
    assert s.denied_ids == []
    assert s.context == ""
    assert s.answer == ""
    assert s.citations == []
    assert s.cached is False
    assert s.cache_key is None
    assert s.error is None


def test_back_compat_chunks_used():
    s = PdfChatState(query="q", tenant_id="t")
    assert s.chunks_used() == 0
    s.accessible_chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert s.chunks_used() == 2


def test_query_and_tenant_are_required_positional():
    # The two required inputs must still be settable positionally.
    s = PdfChatState("hello", "tenant-1")
    assert s.query == "hello"
    assert s.tenant_id == "tenant-1"

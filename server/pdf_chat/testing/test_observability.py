"""Phase 6 (hardening) observability tests — trace + cost tracker + metrics.

Every test here runs with ZERO infra installed: the observability surface is a
pure, import-safe set of in-process accumulators (mirrors the main app's
core/{orchestration_trace,cost_tracker,metrics}.py but tenant-scoped and living
inside pdf_chat so the two pipelines stay independent — pdf_chat/CLAUDE.md #7).
"""
from __future__ import annotations

import uuid

from pdf_chat.observability.trace import PdfTrace, new_trace_id


# --------------------------------------------------------------------------- #
# Task 1 — trace IDs + JSON-safe trace
# --------------------------------------------------------------------------- #
def test_new_trace_id_is_unique_uuid_hex():
    a, b = new_trace_id(), new_trace_id()
    assert a != b
    assert uuid.UUID(hex=a)  # parses as a valid uuid


def test_trace_records_stages_and_is_json_safe():
    t = PdfTrace(trace_id="t1", tenant_id="tenant-A")
    t.set_stage("retrieval", {"chunks": 5, "secret": object()})  # non-serializable value
    d = t.as_dict()
    assert d["trace_id"] == "t1"
    assert d["tenant_id"] == "tenant-A"
    assert d["stages"]["retrieval"]["chunks"] == 5
    # non-serializable coerced to str, never raises
    assert isinstance(d["stages"]["retrieval"]["secret"], str)


def test_trace_setters_never_raise():
    t = PdfTrace(trace_id="t1", tenant_id="tenant-A")
    t.set_stage("x", None)  # must not raise
    t.emit()  # must not raise even with no logger configured


# --------------------------------------------------------------------------- #
# Task 2 — per-tenant cost tracker (extraction + synthesis)
# --------------------------------------------------------------------------- #
from pdf_chat.observability.cost_tracker import PdfCostTracker, get_cost_tracker


def test_cost_tracker_accumulates_per_tenant():
    ct = PdfCostTracker()
    ct.track_llm("tenant-A", "extraction", "gpt-4o-mini",
                 prompt_tokens=1000, completion_tokens=200,
                 cost_usd=0.0003, document_id="doc1", trace_id="t1")
    ct.track_llm("tenant-A", "synthesis", "gpt-4o-mini",
                 prompt_tokens=500, completion_tokens=100,
                 cost_usd=0.0001, document_id=None, trace_id="t2")
    ct.track_llm("tenant-B", "synthesis", "gpt-4o-mini",
                 prompt_tokens=10, completion_tokens=5,
                 cost_usd=0.00001, document_id=None, trace_id="t3")

    a = ct.snapshot("tenant-A")
    assert a["llm_calls"] == 2
    assert a["prompt_tokens"] == 1500
    assert a["completion_tokens"] == 300
    assert round(a["cost_usd"], 6) == 0.0004
    # per-phase breakdown
    assert a["by_phase"]["extraction"]["llm_calls"] == 1
    assert a["by_phase"]["synthesis"]["llm_calls"] == 1

    b = ct.snapshot("tenant-B")
    assert b["llm_calls"] == 1
    # tenant isolation: A's totals never leak into B
    assert b["prompt_tokens"] == 10


def test_cost_tracker_rejects_non_mini_model():
    ct = PdfCostTracker()
    # spec §6: no gpt-4o anywhere. A gpt-4o call is recorded but flagged.
    ct.track_llm("tenant-A", "synthesis", "gpt-4o",
                 prompt_tokens=1, completion_tokens=1, cost_usd=0.01,
                 document_id=None, trace_id="t1")
    a = ct.snapshot("tenant-A")
    assert a["policy_violations"] == 1
    # the call is still recorded — cost is never silently dropped
    assert a["llm_calls"] == 1
    assert round(a["cost_usd"], 6) == 0.01


def test_cost_tracker_allows_gpt_4o_mini():
    """gpt-4o-mini IS the allowed bulk model — never a policy violation."""
    ct = PdfCostTracker()
    ct.track_llm("tenant-A", "synthesis", "gpt-4o-mini",
                 prompt_tokens=1, completion_tokens=1, cost_usd=0.001,
                 document_id=None, trace_id="t1")
    a = ct.snapshot("tenant-A")
    assert a["policy_violations"] == 0
    assert a["llm_calls"] == 1


def test_cost_tracker_allows_claude_escalation_tier():
    """The model_router legitimately escalates to claude-sonnet-4-6 / claude-opus-4-8;
    the real policy is 'no gpt-4o', so a claude model is NOT a violation."""
    ct = PdfCostTracker()
    ct.track_llm("tenant-A", "synthesis", "claude-sonnet-4-6",
                 prompt_tokens=1, completion_tokens=1, cost_usd=0.02,
                 document_id=None, trace_id="t1")
    ct.track_llm("tenant-A", "synthesis", "claude-opus-4-8",
                 prompt_tokens=1, completion_tokens=1, cost_usd=0.03,
                 document_id=None, trace_id="t2")
    a = ct.snapshot("tenant-A")
    assert a["policy_violations"] == 0
    assert a["llm_calls"] == 2


def test_cost_tracker_reset_clears_tenant():
    ct = PdfCostTracker()
    ct.track_llm("tenant-A", "synthesis", "gpt-4o-mini",
                 prompt_tokens=5, completion_tokens=2, cost_usd=0.0001,
                 document_id=None, trace_id="t1")
    ct.reset("tenant-A")
    a = ct.snapshot("tenant-A")
    assert a["llm_calls"] == 0
    assert a["prompt_tokens"] == 0


def test_cost_tracker_snapshot_unknown_tenant_is_blank():
    ct = PdfCostTracker()
    a = ct.snapshot("never-seen")
    assert a["llm_calls"] == 0
    assert a["policy_violations"] == 0
    assert a["by_phase"] == {}


def test_get_cost_tracker_is_singleton():
    assert get_cost_tracker() is get_cost_tracker()


# --------------------------------------------------------------------------- #
# Task 3 — per-tenant metrics surface
# --------------------------------------------------------------------------- #
from pdf_chat.observability import metrics as pdf_metrics


def test_metrics_inc_and_latency_snapshot():
    pdf_metrics.reset()
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    pdf_metrics.inc("tenant-A", "pdf_fallback_count")
    pdf_metrics.observe_latency("tenant-A", 120.0)
    pdf_metrics.observe_latency("tenant-A", 240.0)

    snap = pdf_metrics.get_snapshot("tenant-A")
    assert snap["pdf_query_total"] == 2
    assert snap["pdf_fallback_count"] == 1
    assert snap["latency_p50_ms"] is not None
    assert snap["latency_p95_ms"] is not None
    assert snap["latency_p99_ms"] is not None
    assert snap["latency_sample_count"] == 2


def test_metrics_are_tenant_scoped():
    pdf_metrics.reset()
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    snap_b = pdf_metrics.get_snapshot("tenant-B")
    assert snap_b["pdf_query_total"] == 0


def test_metrics_empty_snapshot_has_null_percentiles():
    pdf_metrics.reset()
    snap = pdf_metrics.get_snapshot("tenant-X")
    assert snap["latency_p50_ms"] is None
    assert snap["latency_sample_count"] == 0


def test_metrics_inc_amount_and_lazy_counter():
    pdf_metrics.reset()
    pdf_metrics.inc("tenant-A", "pdf_orphan_entity_deleted_count", 3)
    pdf_metrics.inc("tenant-A", "lazily_created_counter")
    snap = pdf_metrics.get_snapshot("tenant-A")
    assert snap["pdf_orphan_entity_deleted_count"] == 3
    assert snap["lazily_created_counter"] == 1


def test_metrics_route_is_registered_and_tenant_scoped():
    import pdf_chat.api.routes as r

    paths = [route.path for route in r.pdf_router.routes]
    assert "/api/pdf/metrics" in paths


# --------------------------------------------------------------------------- #
# Task 9 — trace + metrics threaded through the LIVE chat/delete routes.
#
# Mocks-only / zero infra: we override get_principal (no JWT backend), monkeypatch
# the agent runtime + delete service, and drive the routes via FastAPI's TestClient.
# These assert the OBSERVABILITY wiring (counters/latency/background cascade) WITHOUT
# changing query semantics — the runtime call itself is faked.
# --------------------------------------------------------------------------- #
def _client_with_principal(tenant_id: str, *, raise_server_exceptions: bool = True):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import pdf_chat.api.routes as routes

    app = FastAPI()
    app.include_router(routes.pdf_router)
    app.dependency_overrides[routes.get_principal] = lambda: routes.Principal(
        user_id="u1", tenant_id=tenant_id, groups=[]
    )
    return (
        TestClient(app, raise_server_exceptions=raise_server_exceptions),
        routes,
    )


def test_chat_route_counts_query_and_observes_latency(monkeypatch):
    from pdf_chat.observability import metrics as pdf_metrics

    pdf_metrics.reset()
    client, routes = _client_with_principal("tenant-A")

    class _Result:
        answer = "grounded answer"
        citations = []  # chunks_used is derived from len(citations) now

    import pdf_chat.agent.graph as graph

    async def _fake_run(query, **kwargs):
        return _Result()

    monkeypatch.setattr(graph, "run_pdf_query", _fake_run)
    monkeypatch.setattr(graph, "build_default_deps", lambda: object())

    resp = client.post(
        "/api/pdf/chat", json={"query": "hi", "tenant_id": "tenant-A"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "grounded answer"

    snap = pdf_metrics.get_snapshot("tenant-A")
    assert snap["pdf_query_total"] == 1
    assert snap["pdf_query_errors"] == 0
    # latency was observed in the finally block
    assert snap["latency_sample_count"] == 1


def test_chat_route_increments_errors_on_runtime_failure(monkeypatch):
    from pdf_chat.observability import metrics as pdf_metrics

    pdf_metrics.reset()
    client, routes = _client_with_principal("tenant-B", raise_server_exceptions=False)

    import pdf_chat.agent.graph as graph

    async def _boom(query, **kwargs):
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(graph, "run_pdf_query", _boom)
    monkeypatch.setattr(graph, "build_default_deps", lambda: object())

    resp = client.post(
        "/api/pdf/chat", json={"query": "hi", "tenant_id": "tenant-B"}
    )
    assert resp.status_code == 500

    snap = pdf_metrics.get_snapshot("tenant-B")
    assert snap["pdf_query_total"] == 1
    assert snap["pdf_query_errors"] == 1
    # latency still observed despite the failure (finally block)
    assert snap["latency_sample_count"] == 1


def test_delete_route_soft_deletes_and_schedules_cascade(monkeypatch):
    client, routes = _client_with_principal("tenant-A")

    import pdf_chat.control_plane.delete_service as delete_service

    async def _fake_delete(*, upload_id, tenant_id):
        return {"upload_id": upload_id, "status": "deleted", "cleanup": "scheduled"}

    monkeypatch.setattr(delete_service, "delete_document", _fake_delete)

    scheduled = {}

    async def _fake_cleanup(upload_id, tenant_id):
        scheduled["upload_id"] = upload_id
        scheduled["tenant_id"] = tenant_id

    # Replace the background cascade with a recorder so no Neo4j infra is touched.
    monkeypatch.setattr(routes, "_run_graph_cleanup", _fake_cleanup)

    resp = client.delete("/api/pdf/documents/doc1")
    assert resp.status_code == 200
    body = resp.json()
    # DeleteResponse shape preserved (Task 9 must not change it)
    assert body == {"upload_id": "doc1", "deleted": True, "chunks_removed": 0}
    # the out-of-band cascade was scheduled with the tenant from the principal
    assert scheduled == {"upload_id": "doc1", "tenant_id": "tenant-A"}

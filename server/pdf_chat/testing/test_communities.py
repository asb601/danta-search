"""Phase-2 Task 10/11 — tests for COMMUNITIES + PageRank + cited reports.

Asserts the spec §1b/§9 contract:
  * ``detect_communities`` runs Leiden over the GROUNDED-edge graph, reads its
    resolution + min-size dials via ``get_tunable`` (NOT inline literals), and
    drops communities below the per-container min-size floor.
  * ``pagerank_confidence`` weights edges by their ``confidence`` so a high-
    confidence hub outranks a low-confidence one.
  * ``CommunityReporter.report`` (the ``TaskClass.SYNTHESIS`` call site) routes
    through the bulk ``gpt-4o-mini`` (escalation OFF), SUPPRESSES any report not
    traceable to ``>= kg.report.min_grounded_edges`` grounded edges, and a
    produced report carries citations drilling down to chunk ids.
  * GUARDED networkx: when networkx is absent both graph functions degrade to an
    empty result with no crash.

Pure-testable with zero infra: the LLM is injected; networkx is the only real
dep and it is monkeypatched absent for the degrade path. ``GroundedEdge`` is the
real artifact from ``grounding_gate`` (the gate's output is this module's input).
"""
from __future__ import annotations

import pytest

from pdf_chat.ingestion import communities as C
from pdf_chat.ingestion.communities import (
    Community,
    CommunityReporter,
    detect_communities,
    pagerank_confidence,
)
from pdf_chat.ingestion.grounding_gate import GroundedEdge


def _edge(subject, obj, confidence=0.9, src="c1"):
    return GroundedEdge(
        subject=subject,
        predicate="related_to",
        obj=obj,
        confidence=confidence,
        span=f"{subject} related_to {obj}",
        src_chunk_id=src,
        evidence_count=1,
    )


# ── community detection ──────────────────────────────────────────────────────
def test_detect_communities_returns_communities_for_a_dense_cluster():
    # A 4-node clique → one community of size 4 (>= default min_size of 3).
    edges = [
        _edge("A", "B"),
        _edge("B", "C"),
        _edge("C", "A"),
        _edge("A", "D"),
        _edge("B", "D"),
    ]
    comms = detect_communities(edges, container_id="t1")
    assert comms, "a dense cluster should yield at least one community"
    assert all(isinstance(c, Community) for c in comms)
    biggest = max(comms, key=lambda c: len(c.members))
    assert {"A", "B", "C", "D"} <= set(biggest.members)
    assert biggest.community_id  # deterministic id present
    assert biggest.src_chunk_ids  # traces back to grounding chunks


def test_detect_communities_drops_below_min_size(monkeypatch):
    # Raise min_size so the only small cluster is dropped.
    def fake_tunable(container_id, key, default=None):
        if key == "kg.community.min_size":
            return 5
        if key == "kg.community.resolution":
            return 1.0
        return default

    monkeypatch.setattr(C, "get_tunable", fake_tunable)
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("C", "A")]  # size 3 < 5
    assert detect_communities(edges, container_id="t1") == []


def test_detect_communities_reads_dials_via_get_tunable(monkeypatch):
    seen_keys = []

    def spy_tunable(container_id, key, default=None):
        seen_keys.append(key)
        return default

    monkeypatch.setattr(C, "get_tunable", spy_tunable)
    detect_communities([_edge("A", "B"), _edge("B", "C"), _edge("C", "A")], container_id="t1")
    assert "kg.community.resolution" in seen_keys
    assert "kg.community.min_size" in seen_keys


def test_detect_communities_degrades_when_networkx_absent(monkeypatch):
    monkeypatch.setattr(C, "_HAS_NETWORKX", False)
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("C", "A")]
    assert detect_communities(edges, container_id="t1") == []  # no crash


def test_detect_communities_empty_edges():
    assert detect_communities([], container_id="t1") == []


# ── confidence-weighted PageRank ─────────────────────────────────────────────
def test_pagerank_weights_by_confidence():
    # One connected graph. The shared spokes link to H_high with HIGH-confidence
    # edges and to H_low with LOW-confidence edges, so rank flows preferentially
    # to H_high — confidence-as-weight makes the high-confidence hub outrank the
    # low-confidence one. (Standard PageRank cannot separate two ISOLATED
    # symmetric stars, so the hubs must share spokes to actually compete.)
    edges = [
        _edge("s1", "H_high", confidence=0.95),
        _edge("s2", "H_high", confidence=0.95),
        _edge("s3", "H_high", confidence=0.95),
        _edge("s1", "H_low", confidence=0.05),
        _edge("s2", "H_low", confidence=0.05),
        _edge("s3", "H_low", confidence=0.05),
    ]
    ranks = pagerank_confidence(edges)
    assert ranks["H_high"] > ranks["H_low"]


def test_pagerank_degrades_when_networkx_absent(monkeypatch):
    monkeypatch.setattr(C, "_HAS_NETWORKX", False)
    assert pagerank_confidence([_edge("A", "B")]) == {}


def test_pagerank_empty_edges():
    assert pagerank_confidence([]) == {}


# ── cited community reports (the SYNTHESIS call site) ─────────────────────────
class _FakeLLM:
    """Records calls + returns a fixed synthesis payload."""

    def __init__(self, summary="Cluster summary."):
        self.summary = summary
        self.calls = []

    def synthesize(self, prompt, *, model_id, container_id, **kw):
        self.calls.append({"model_id": model_id, "container_id": container_id})
        return {"summary": self.summary}


def _community(members, src_chunk_ids):
    return Community(
        community_id="t1::comm0",
        members=tuple(members),
        src_chunk_ids=tuple(src_chunk_ids),
    )


def test_report_suppressed_below_min_grounded_edges():
    # Only ONE grounded edge traces this community; default floor is 2 → suppress.
    comm = _community(["A", "B"], ["c1"])
    edges = [_edge("A", "B", src="c1")]
    llm = _FakeLLM()
    out = CommunityReporter(llm).report(comm, edges, container_id="t1")
    assert out is None
    assert llm.calls == []  # never spend an LLM call on a suppressed community


def test_report_produced_carries_citations():
    comm = _community(["A", "B", "C"], ["c1", "c2"])
    edges = [
        _edge("A", "B", src="c1"),
        _edge("B", "C", src="c2"),
    ]
    llm = _FakeLLM(summary="A, B and C are tightly related.")
    out = CommunityReporter(llm).report(comm, edges, container_id="t1")
    assert out is not None
    assert out.summary == "A, B and C are tightly related."
    # cited drill-down to the grounding chunks
    assert set(out.citations) == {"c1", "c2"}
    assert out.community_id == comm.community_id


def test_report_uses_bulk_synthesis_model_no_escalation(monkeypatch):
    seen = {}

    def spy_select(*, task, container_id, signals, **kw):
        from pdf_chat.model_router import ModelChoice

        seen["task"] = task
        seen["signals"] = signals
        return ModelChoice("azure", "gpt-4o-mini", is_strong=False)

    monkeypatch.setattr(C, "select_model", spy_select)
    comm = _community(["A", "B", "C"], ["c1", "c2"])
    edges = [_edge("A", "B", src="c1"), _edge("B", "C", src="c2")]
    llm = _FakeLLM()
    CommunityReporter(llm).report(comm, edges, container_id="t1")
    from pdf_chat.model_router import TaskClass

    assert seen["task"] == TaskClass.SYNTHESIS
    assert seen["signals"] == {}  # escalation OFF for bulk synthesis
    assert llm.calls and llm.calls[0]["model_id"] == "gpt-4o-mini"


def test_report_reads_min_edges_via_get_tunable(monkeypatch):
    seen_keys = []

    def spy_tunable(container_id, key, default=None):
        seen_keys.append(key)
        return default

    monkeypatch.setattr(C, "get_tunable", spy_tunable)
    comm = _community(["A", "B", "C"], ["c1", "c2"])
    edges = [_edge("A", "B", src="c1"), _edge("B", "C", src="c2")]
    CommunityReporter(_FakeLLM()).report(comm, edges, container_id="t1")
    assert "kg.report.min_grounded_edges" in seen_keys


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

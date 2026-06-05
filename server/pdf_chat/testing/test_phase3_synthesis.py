"""Phase-3 Task 8/9 — tests for grounded SYNTHESIS (the HARD ENTRY GATE).

Spec §1b (misleading-tag safeguard), §3 invariant 1 (grounding), §4
(faithfulness for a non-expert: provenance labels + staleness). Contract under
test (``pdf_chat/agent/synthesis.py``):

  * **citation-density floor** — a synthesized claim with ZERO citations is
    refused; the floor resolves via ``get_tunable("agent.min_citations_per_claim")``
    and the decision is logged.
  * **tag_as_answer HARD GATE** — every tag-/card-derived claim is routed
    through ``grounding_gate.tag_as_answer(tag, supporting_chunks)``; a tag whose
    label appears in NO supporting chunk is DROPPED (the gate returns ``None``).
    The test asserts the gate IS actually called and that an unsupported tag is
    dropped.
  * **card demotion** — a section/doc CARD hit contributes only its
    ``src_chunk_ids`` to context; it is NEVER emitted as a quotable citation.
  * **bbox + page** — emitted citations carry ``bbox`` and ``page``.
  * **provenance labels** — ``stated`` / ``inferred`` / ``conflicting`` /
    ``not_found`` are emitted (NOT a raw confidence number).
  * **staleness hook** — ``staleness_annotation(latest_date)`` returns a
    human-readable "may be outdated" string.

Deterministic seams only — a ``FakeLlm`` returns a canned answer; the searcher /
Neo4j / Redis are not touched. Mirrors ``test_agent.py`` fake conventions.
"""
from __future__ import annotations

import asyncio

import pytest

from pdf_chat.agent.state import PdfChatState
from pdf_chat.agent.synthesis import (
    SynthesisResult,
    synthesize,
    staleness_annotation,
)
from pdf_chat.ingestion import grounding_gate
from pdf_chat.ingestion.grounding_gate import GroundedTag


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLlm:
    """Records the (system, user) it was called with and returns a canned answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.last_signals: dict | None = None
        self.last_container_id: str | None = None

    async def generate(self, system, user, *, container_id="", signals=None):
        self.calls.append((system, user))
        self.last_signals = signals
        self.last_container_id = container_id
        return self.answer


class Deps:
    def __init__(self, llm):
        self.llm = llm


def _chunk(chunk_id, text, *, page=1, doc_id="d1", bbox=None, element_type="text"):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "page_num": page,
        "doc_id": doc_id,
        "bbox": bbox or [0.0, 0.0, 1.0, 1.0],
        "element_type": element_type,
    }


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 8a — citation-density floor: a zero-citation claim is refused
# --------------------------------------------------------------------------- #
def test_zero_citation_answer_is_refused_via_floor():
    # The LLM produces an answer that cites NOTHING ([N] absent). The citation
    # floor must refuse it rather than emit an ungrounded claim.
    llm = FakeLlm("The revenue grew significantly last year.")  # no [N]
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert isinstance(res, SynthesisResult)
    # No grounded citation → refusal, not the ungrounded LLM text.
    assert res.citations == []
    low = res.answer.lower()
    assert "could not" in low or "not" in low or "insufficient" in low


def test_cited_answer_passes_the_floor():
    llm = FakeLlm("Revenue grew 12% in FY2025 [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert res.citations, "a [1]-cited answer must keep its citation"
    assert res.answer.strip()


# --------------------------------------------------------------------------- #
# 8b — tag_as_answer IS CALLED and an unsupported tag is DROPPED
# --------------------------------------------------------------------------- #
def test_tag_as_answer_called_and_unsupported_tag_dropped(monkeypatch):
    calls: list[tuple[str, int]] = []
    real = grounding_gate.tag_as_answer

    def spy(tag, supporting_chunks, *, container_id=""):
        calls.append((getattr(tag, "label", None), len(list(supporting_chunks or []))))
        return real(tag, supporting_chunks, container_id=container_id)

    # synthesis.py imports the symbol; patch it where it is looked up.
    import pdf_chat.agent.synthesis as synth
    monkeypatch.setattr(synth, "tag_as_answer", spy)

    llm = FakeLlm("Topic is revenue [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    # One supported tag ("revenue" appears in c1) + one UNSUPPORTED tag
    # ("acquisition" appears in NO chunk → must be dropped by the gate).
    supported = GroundedTag(
        label="revenue", scope="section", confidence=0.9, span="", src_chunk_id="c1"
    )
    unsupported = GroundedTag(
        label="acquisition", scope="doc", confidence=0.9, span="", src_chunk_id="c1"
    )
    state.candidate_tags = [supported, unsupported]

    res = _run(synthesize(state, Deps(llm), container_id="t"))

    labels_called = {label for label, _ in calls}
    assert "revenue" in labels_called, "tag_as_answer must be called per tag"
    assert "acquisition" in labels_called, "the unsupported tag must also be gated"
    # Surviving (admitted) tags only — the unsupported one is dropped.
    assert "revenue" in res.admitted_tags
    assert "acquisition" not in res.admitted_tags


def test_tag_with_no_supporting_chunk_returns_none_from_gate():
    # Direct contract: the gate suppresses a label absent from supporting chunks.
    tag = GroundedTag(
        label="phantom", scope="doc", confidence=0.9, span="", src_chunk_id="c1"
    )
    out = grounding_gate.tag_as_answer(
        tag, [{"text": "nothing relevant here"}], container_id="t"
    )
    assert out is None


# --------------------------------------------------------------------------- #
# 8c — card demotion: a card hit only pulls src_chunk_ids; never a citation
# --------------------------------------------------------------------------- #
def test_card_hit_is_demoted_to_src_chunk_ids_not_quoted():
    llm = FakeLlm("Revenue grew [1].")
    state = PdfChatState(query="q", tenant_id="t")
    chunk = _chunk("c1", "Revenue grew 12% in FY2025.")
    card = {
        "chunk_id": "sec_card_1",
        "text": "SUMMARY: this section covers revenue.",
        "element_type": "section_card",
        "src_chunk_ids": ["c1"],
        "doc_id": "d1",
        "page_num": 2,
    }
    state.accessible_chunks = [card, chunk]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    cited_ids = {c.get("chunk_id") for c in res.citations}
    # The card itself is NEVER a quotable citation.
    assert "sec_card_1" not in cited_ids
    # Its src chunk IS eligible.
    assert "c1" in cited_ids


# --------------------------------------------------------------------------- #
# 8d — citations carry bbox + page
# --------------------------------------------------------------------------- #
def test_citations_carry_bbox_and_page():
    llm = FakeLlm("Revenue grew [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [
        _chunk("c1", "Revenue grew 12% in FY2025.", page=7, bbox=[1, 2, 3, 4])
    ]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert res.citations
    cit = res.citations[0]
    assert "bbox" in cit and "page" in cit
    assert cit["page"] == 7
    assert cit["bbox"] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# 9 — provenance labels (NOT raw confidence)
# --------------------------------------------------------------------------- #
def test_provenance_label_stated_for_grounded_citation():
    llm = FakeLlm("Revenue grew 12% in FY2025 [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert res.provenance, "provenance labels must be emitted"
    assert set(res.provenance.values()) <= {
        "stated", "inferred", "conflicting", "not_found"
    }
    # A directly-grounded citation is "stated".
    assert "stated" in res.provenance.values()


def test_provenance_conflicting_when_sources_contradict():
    llm = FakeLlm("Acme is the parent of Globex [1][2].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [
        _chunk("c1", "Acme is the parent company of Globex.", page=1, doc_id="dA"),
        _chunk("c2", "Acme is not the parent company of Globex.", page=5, doc_id="dB"),
    ]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert "conflicting" in res.provenance.values()


def test_provenance_not_found_for_uncited_index():
    # The model cites [3] but only 1 grounded chunk exists → that index has no
    # backing evidence → labelled not_found (never silently treated as stated).
    llm = FakeLlm("It says X [3].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12%.")]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert res.provenance.get(3) == "not_found"


# --------------------------------------------------------------------------- #
# 9 — staleness annotation hook
# --------------------------------------------------------------------------- #
def test_staleness_annotation_returns_outdated_hint():
    out = staleness_annotation("2025-09")
    assert out and "2025-09" in out
    assert "outdated" in out.lower() or "recent" in out.lower()


def test_staleness_annotation_empty_for_no_date():
    assert staleness_annotation(None) == ""
    assert staleness_annotation("") == ""


# --------------------------------------------------------------------------- #
# never-raise / refusal on no accessible context
# --------------------------------------------------------------------------- #
def test_insufficient_context_refuses_without_calling_llm():
    llm = FakeLlm("should not be called")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = []  # ACL-empty / below floor
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert res.citations == []
    assert llm.calls == [], "LLM must not be called with no accessible context"


# --------------------------------------------------------------------------- #
# Fix 4 — staleness annotation is WIRED into synthesis (latest cited-chunk date).
# --------------------------------------------------------------------------- #
def test_synthesis_attaches_staleness_from_cited_chunk_date():
    llm = FakeLlm("Revenue grew 12% [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [
        dict(_chunk("c1", "Revenue grew 12% in FY2019."), date="2019-03")
    ]
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert "may be outdated" in res.answer.lower()
    assert "2019-03" in res.answer


def test_synthesis_degrades_silently_without_chunk_date():
    llm = FakeLlm("Revenue grew 12% [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12%.")]  # no date field
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    assert "may be outdated" not in res.answer.lower()


# --------------------------------------------------------------------------- #
# Fix 3 — the negative-claim/conflict verdict is computed ONCE per query (the
# memoized verdict is reused by negative_claim_node, not recomputed).
# --------------------------------------------------------------------------- #
def test_conflict_detection_runs_once_per_query(monkeypatch):
    from pdf_chat.agent import negative_claim as nc_mod
    from pdf_chat.agent.graph import AgentDeps, negative_claim_node, synthesize_node

    calls = {"n": 0}
    orig = nc_mod._detect_conflicts

    def _spy(accessible_chunks, *, container_id):
        calls["n"] += 1
        return orig(accessible_chunks, container_id=container_id)

    monkeypatch.setattr(nc_mod, "_detect_conflicts", _spy)

    llm = FakeLlm("Revenue grew 12% [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    deps = AgentDeps(llm=llm)

    # synthesize memoizes the verdict; negative_claim_node must REUSE it.
    asyncio.run(synthesize_node(state, deps))
    asyncio.run(negative_claim_node(state, deps))

    assert calls["n"] == 1, f"_detect_conflicts ran {calls['n']}x; expected 1 (memoized)"


# --------------------------------------------------------------------------- #
# Multi-part honesty — partly-grounded components are flagged at synthesis time.
# --------------------------------------------------------------------------- #
def test_partial_component_grounding_flagged_in_synthesis():
    llm = FakeLlm("Revenue grew 12% [1].")
    state = PdfChatState(query="q", tenant_id="t")
    state.accessible_chunks = [_chunk("c1", "Revenue grew 12% in FY2025.")]
    state.output_components = ["revenue", "headcount"]  # headcount not grounded
    res = _run(synthesize(state, Deps(llm), container_id="t"))
    low = res.answer.lower()
    assert "not fully address" in low
    assert "headcount" in low
    # per-component grounding recorded on state.
    assert state.component_grounding == {"revenue": True, "headcount": False}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

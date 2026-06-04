"""Phase-2 Task 6/11 — tests for the blocking GROUNDING GATE.

Asserts the faithfulness contract: an ungrounded edge AND an ungrounded tag are
REJECTED; grounded ones are admitted; and a tag never surfaces as an answer
without a grounded supporting chunk (misleading-tag safeguard, spec §1b).

The extracted-input dataclasses are defined locally with the EXACT positional
constructor signatures the plan specifies for ``ExtractedRelation`` /
``ExtractedTag`` (kg_extraction.py, Task 4). The gate is duck-typed on
attributes, so these stand-ins exercise the real contract without coupling this
test to another agent's module.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pdf_chat.ingestion.grounding_gate import (
    GroundedEdge,
    GroundedTag,
    GroundingGate,
    tag_as_answer,
)


# ── plan-exact stand-in inputs ──────────────────────────────────────────────
@dataclass(frozen=True)
class _Relation:
    subject: str
    predicate: str
    obj: str
    confidence: float
    span: str
    src_chunk_id: str


@dataclass(frozen=True)
class _Tag:
    label: str
    scope: str
    confidence: float
    span: str
    src_chunk_id: str


# ── edges ───────────────────────────────────────────────────────────────────
def test_rejects_edge_absent_from_span():
    # "Globex" (the object) is NOT in the cited span → ungrounded → reject.
    g = GroundingGate()
    rel = _Relation(
        "Acme", "acquired", "Globex", 0.9, "Acme bought Beta Corp in 2025", "c1"
    )
    assert g.admit_edge(rel, cited_text="Acme bought Beta Corp in 2025", container_id="t1") is None


def test_admits_grounded_edge():
    rel = _Relation("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1")
    out = GroundingGate().admit_edge(
        rel, cited_text="Acme acquired Globex", container_id="t1"
    )
    assert isinstance(out, GroundedEdge)
    assert out.subject == "Acme" and out.obj == "Globex"
    assert out.src_chunk_id == "c1"
    assert out.evidence_count == 1


def test_rejects_edge_with_empty_endpoint():
    # An empty object slot is a fabricated claim → reject (never "trivially" in).
    rel = _Relation("Acme", "acquired", "", 0.9, "Acme acquired Globex", "c1")
    assert GroundingGate().admit_edge(
        rel, cited_text="Acme acquired Globex", container_id="t1"
    ) is None


def test_edge_grounding_is_case_and_whitespace_robust():
    rel = _Relation("Acme", "acquired", "Globex", 0.9, "ACME   acquired\nGLOBEX", "c1")
    assert GroundingGate().admit_edge(
        rel, cited_text="ACME   acquired\nGLOBEX", container_id="t1"
    ) is not None


def test_rejects_edge_whose_predicate_span_is_absent_even_if_endpoints_cooccur():
    # Both endpoints (Acme, Globex) co-occur in the cited text, but the asserted
    # relation's verbatim span ("Acme acquired Globex") is NOT present — the text
    # only puts them in a list. Endpoint co-presence must NOT ground a fabricated
    # predicate: the gate verifies the PER-RELATION span, not just the endpoints.
    g = GroundingGate()
    cited = "Acme and Globex both attended the 2025 industry conference."
    rel = _Relation("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1")
    assert g.admit_edge(rel, cited_text=cited, container_id="t1") is None


def test_admits_edge_when_predicate_span_is_present():
    # Same endpoints, but now the relation's span IS in the cited text → admit.
    g = GroundingGate()
    cited = "In 2025, Acme acquired Globex for an undisclosed sum."
    rel = _Relation("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1")
    assert g.admit_edge(rel, cited_text=cited, container_id="t1") is not None


def test_short_token_uses_word_boundary_not_incidental_substring():
    # The 2-char object "HP" must NOT ground against "HP" buried inside "sHiPment"
    # (no such substring here) / "championship". The span is present but the short
    # endpoint only appears as an incidental substring → reject.
    g = GroundingGate()
    cited = "The championship was won decisively; Acme led the standings."
    # span present, subject present, but obj "HP" only as a substring of championship
    rel = _Relation("Acme", "noted", "HP", 0.9, "Acme led the standings", "c1")
    assert g.admit_edge(rel, cited_text=cited, container_id="t1") is None
    # When "HP" appears as a standalone token, the same edge grounds.
    cited2 = "Acme led the standings ahead of HP this quarter."
    rel2 = _Relation("Acme", "noted", "HP", 0.9, "Acme led the standings", "c1")
    assert g.admit_edge(rel2, cited_text=cited2, container_id="t1") is not None


# ── tags ────────────────────────────────────────────────────────────────────
def test_rejects_tag_absent_from_span():
    # tag claims "describes Product Z" but the span only mentions "Product Y".
    tag = _Tag("describes Product Z", "doc", 0.8, "discusses Product Y", "c1")
    assert GroundingGate().admit_tag(
        tag, cited_text="discusses Product Y", container_id="t1"
    ) is None


def test_admits_grounded_tag():
    tag = _Tag("discusses Product Y", "doc", 0.8, "discusses Product Y", "c1")
    out = GroundingGate().admit_tag(
        tag, cited_text="discusses Product Y", container_id="t1"
    )
    assert isinstance(out, GroundedTag)
    assert out.label == "discusses Product Y" and out.scope == "doc"


def test_rejects_grounded_tag_below_confidence_floor():
    # Grounded in span but confidence under the tag floor (0.50 default) → reject.
    tag = _Tag("discusses Product Y", "section", 0.10, "discusses Product Y", "c1")
    assert GroundingGate().admit_tag(
        tag, cited_text="discusses Product Y", container_id="t1"
    ) is None


# ── misleading-tag safeguard (spec §1b) ─────────────────────────────────────
def test_tag_never_an_answer_without_supporting_chunk():
    tag = GroundedTag("revenue grew 20%", "doc", 0.9, "revenue grew 20%", "c1")
    assert tag_as_answer(tag, supporting_chunks=[]) is None


def test_tag_surfaces_as_answer_when_a_chunk_text_contains_the_label():
    tag = GroundedTag("revenue grew 20%", "doc", 0.9, "revenue grew 20%", "c1")
    # A supporting chunk whose TEXT actually contains the label → surface.
    chunks = [{"chunk_id": "c1", "text": "In FY25 revenue grew 20% year over year."}]
    assert tag_as_answer(tag, supporting_chunks=chunks) == "revenue grew 20%"
    # A plain string chunk text containing the label also works.
    assert (
        tag_as_answer(tag, supporting_chunks=["revenue grew 20% in 2025"])
        == "revenue grew 20%"
    )


def test_tag_not_an_answer_when_no_chunk_text_contains_the_label():
    # A NON-EMPTY supporting list is not enough: no chunk's text mentions the tag
    # label, so surfacing it would fabricate an answer the evidence never states.
    tag = GroundedTag("revenue grew 20%", "doc", 0.9, "revenue grew 20%", "c1")
    chunks = [
        {"chunk_id": "c1", "text": "Costs were flat this year."},
        "Headcount increased.",
    ]
    assert tag_as_answer(tag, supporting_chunks=chunks) is None
    # Bare ids (no text) likewise cannot back the tag (the old weak contract).
    assert tag_as_answer(tag, supporting_chunks=["c1", "c2"]) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

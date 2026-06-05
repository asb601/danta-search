"""Phase-3 Task 10 — tests for the ported NEGATIVE-CLAIM + CONFLICT gate.

Spec §3 invariant 2 (honest absence) + invariant 7 (three-state relationships) +
§4 (faithfulness for a non-expert). Mirrors the never-raise discipline of the
ERP ``negative_claim_gate`` it ports from.

Contract under test (``pdf_chat/agent/negative_claim.py``):

  * A "no data / not found" claim is PROVEN only when coverage is complete
    (the relevant query pages/sections were actually in-context) AND the
    absence is diagnosed. ``retrieval-empty != absent``: an answer that claims
    absence with ZERO accessible chunks is NOT proven and is rewritten.
  * Conflicting relationships are surfaced in ``verdict.conflicts`` WITH
    provenance (chunk_id/page) — never silently resolved/picked.
  * ``pdf_honest_rewrite`` returns a scoped, honest replacement for an
    unproven negative claim.
  * The gate never raises.

Deterministic seams only — pure inputs (plain dicts), zero infra.
"""
from __future__ import annotations

import pytest

from pdf_chat.agent.negative_claim import (
    PdfNegativeVerdict,
    evaluate_pdf_negative_claim,
    pdf_honest_rewrite,
)


def _chunk(chunk_id, text, *, page=1, doc_id="d1"):
    return {"chunk_id": chunk_id, "text": text, "page_num": page, "doc_id": doc_id}


# ── non-negative answers pass straight through ───────────────────────────────
def test_non_negative_answer_is_not_a_negative_claim():
    v = evaluate_pdf_negative_claim(
        answer="The contract was signed on 2025-03-01.",
        accessible_chunks=[_chunk("c1", "signed on 2025-03-01")],
        container_id="t1",
    )
    assert v.is_negative_claim is False
    assert v.proven is True  # nothing to prove → not blocked


# ── retrieval-empty != absent ────────────────────────────────────────────────
def test_unproven_absence_with_no_accessible_chunks_is_blocked_and_rewritten():
    # The model claims "not found" but NOTHING was accessible. That is a
    # retrieval miss, not a proven absence → unproven → must be rewritten.
    v = evaluate_pdf_negative_claim(
        answer="There is no information about the 2024 budget in the documents.",
        accessible_chunks=[],
        query_pages=[3, 4],
        container_id="t1",
    )
    assert v.is_negative_claim is True
    assert v.coverage_complete is False
    assert v.proven is False
    rewrite = pdf_honest_rewrite(v)
    assert rewrite  # a non-empty honest replacement
    assert "could not" in rewrite.lower() or "not verify" in rewrite.lower() \
        or "did not retrieve" in rewrite.lower()


def test_unproven_absence_when_query_pages_not_in_context_is_blocked():
    # Chunks WERE accessible, but none of them cover the pages the question is
    # about → coverage not complete → unproven.
    v = evaluate_pdf_negative_claim(
        answer="No such clause is found in the agreement.",
        accessible_chunks=[_chunk("c1", "unrelated boilerplate", page=9)],
        query_pages=[2, 3],
        container_id="t1",
    )
    assert v.is_negative_claim is True
    assert v.coverage_complete is False
    assert v.proven is False


# ── proven absence: coverage + diagnosis ─────────────────────────────────────
def test_proven_absence_when_relevant_pages_in_context_and_diagnosed():
    # The relevant query pages ARE in-context and the claimed item genuinely is
    # not present in any of them → coverage complete + diagnosed → proven.
    v = evaluate_pdf_negative_claim(
        answer="The documents do not state a termination fee.",
        accessible_chunks=[
            _chunk("c1", "Section 2 covers payment terms and schedules.", page=2),
            _chunk("c2", "Section 3 covers delivery and acceptance.", page=3),
        ],
        query_pages=[2, 3],
        container_id="t1",
    )
    assert v.is_negative_claim is True
    assert v.coverage_complete is True
    assert v.diagnosed is True
    assert v.proven is True


def test_proven_absence_with_chunks_and_no_query_pages_specified():
    # No explicit query_pages: coverage is proven by the presence of accessible,
    # in-context chunks that were actually scanned for the claim.
    v = evaluate_pdf_negative_claim(
        answer="No mention of a warranty is found.",
        accessible_chunks=[
            _chunk("c1", "The agreement covers scope, fees, and timeline."),
            _chunk("c2", "Termination requires 30 days written notice."),
        ],
        container_id="t1",
    )
    assert v.is_negative_claim is True
    assert v.coverage_complete is True
    assert v.proven is True


# ── three-state conflict surfacing (invariant 7) ─────────────────────────────
def test_conflicting_relationship_is_surfaced_with_provenance_not_resolved():
    # Two accessible chunks make directly contradictory statements about the
    # same relationship. The gate must SURFACE the conflict with provenance,
    # never silently pick one.
    v = evaluate_pdf_negative_claim(
        answer="Acme is the parent company of Globex.",
        accessible_chunks=[
            _chunk("c1", "Acme is the parent company of Globex.", page=1, doc_id="dA"),
            _chunk("c2", "Acme is not the parent company of Globex.", page=5, doc_id="dB"),
        ],
        container_id="t1",
    )
    assert v.conflicts, "a contradiction must be surfaced, never silently resolved"
    prov = {(c.get("chunk_id"), c.get("page"), c.get("doc_id")) for c in v.conflicts}
    # Both sides of the contradiction carry provenance (chunk_id + page + doc).
    assert ("c1", 1, "dA") in prov
    assert ("c2", 5, "dB") in prov


def test_no_conflict_surfaced_when_sources_agree():
    v = evaluate_pdf_negative_claim(
        answer="Acme acquired Globex in 2025.",
        accessible_chunks=[
            _chunk("c1", "Acme acquired Globex in 2025.", page=1),
            _chunk("c2", "The 2025 acquisition of Globex by Acme closed in Q3.", page=2),
        ],
        container_id="t1",
    )
    assert v.conflicts == []


# ── never raises (mirrors evaluate_negative_claim) ───────────────────────────
def test_gate_never_raises_on_malformed_input():
    v = evaluate_pdf_negative_claim(
        answer=None,  # type: ignore[arg-type]
        accessible_chunks=[None, 123, {"no_text": True}],  # type: ignore[list-item]
        query_pages="oops",  # type: ignore[arg-type]
        container_id="t1",
    )
    assert isinstance(v, PdfNegativeVerdict)


def test_honest_rewrite_is_a_nonempty_string_for_unproven_verdict():
    v = PdfNegativeVerdict(is_negative_claim=True, proven=False, coverage_complete=False)
    out = pdf_honest_rewrite(v)
    assert isinstance(out, str) and out.strip()


# --------------------------------------------------------------------------- #
# Fix 5 — negative-phrase list is a TUNABLE a tenant can extend.
# --------------------------------------------------------------------------- #
def test_tenant_extended_phrase_triggers_the_gate(monkeypatch):
    # A phrase that is NOT in the canonical default ("rien trouvé" — French
    # "nothing found"). With no override it is NOT a negative claim; with a tenant
    # override extending the phrase list it IS detected and gated as unproven.
    answer = "Rien trouvé concerning the 2024 budget."

    base = evaluate_pdf_negative_claim(
        answer=answer, accessible_chunks=[], container_id="t-base"
    )
    assert base.is_negative_claim is False  # canonical list doesn't know it

    # Tenant override (env tier yields a comma/newline-separated string).
    monkeypatch.setenv(
        "PDF_TUNABLE_AGENT.NEG_CLAIM.PHRASES", "no data,not found,rien trouvé"
    )
    extended = evaluate_pdf_negative_claim(
        answer=answer, accessible_chunks=[], container_id="t-ext"
    )
    assert extended.is_negative_claim is True
    assert extended.proven is False  # retrieval-empty → unproven


def test_phrase_list_default_is_unchanged_without_override():
    # Without an override the canonical English phrasing still fires (behavior
    # unchanged from before the tunable was introduced).
    v = evaluate_pdf_negative_claim(
        answer="There is no such record in the documents.",
        accessible_chunks=[],
        container_id="t1",
    )
    assert v.is_negative_claim is True


# --------------------------------------------------------------------------- #
# Fix 5 — a NON-negative answer that cited ZERO chunks logs an auditable miss.
# --------------------------------------------------------------------------- #
def test_zero_citation_non_negative_answer_logs_a_miss(monkeypatch):
    from pdf_chat.agent import negative_claim as nc_mod

    logged = []
    monkeypatch.setattr(
        nc_mod,
        "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    # A confident, non-negative answer with NO [N] citation — a potential silent
    # miss the phrase list failed to catch.
    evaluate_pdf_negative_claim(
        answer="Revenue grew significantly last year.",
        accessible_chunks=[_chunk("c1", "Revenue grew 12%.")],
        container_id="t1",
    )
    misses = [r for r in logged if r["gate"] == "agent.neg_claim.zero_citation_miss"]
    assert misses, "a zero-citation non-negative answer must log an auditable miss"
    assert misses[-1]["outcome"] == "non_negative_but_uncited"


def test_cited_non_negative_answer_logs_no_miss(monkeypatch):
    from pdf_chat.agent import negative_claim as nc_mod

    logged = []
    monkeypatch.setattr(
        nc_mod,
        "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    evaluate_pdf_negative_claim(
        answer="Revenue grew 12% [1].",
        accessible_chunks=[_chunk("c1", "Revenue grew 12%.")],
        container_id="t1",
    )
    misses = [r for r in logged if r["gate"] == "agent.neg_claim.zero_citation_miss"]
    assert not misses, "a cited answer must NOT log a zero-citation miss"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

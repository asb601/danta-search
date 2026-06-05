"""Phase 5 — corpus-learned glossary miner (Tasks 4/5/6), mocks-only.

The miner turns grounded corpus chunks into ``GlossaryEntry`` candidates from
THREE grounded signals, each carrying a ``provenance`` + evidence spans:

  * Task 4 — explicit definitions: a regex PROPOSES parenthetical/appositive/
    "stands for" candidates (proposal only, never the decision — mirrors
    ``column_role_resolver`` LLM-confirms-classification); the injected LLM
    CONFIRMS with the supporting span → ``provenance == STATED``. An unconfirmed
    candidate is DROPPED (grounding gate, invariant 1).
  * Task 5 — distributional anomaly: a coined term used far above its INJECTED
    background frequency → ``provenance == INFERRED`` (never ``STATED``); the
    definition is LLM-synthesized from usage context. No background table ⇒
    the signal degrades gracefully (logged, no crash, no fabricated ``STATED``).
    Open-vocab: a never-before-seen term is mineable purely from signals +
    injected data — NO static jargon allow-list anywhere.
  * Task 6 — co-reference variants + conflict: alias variants collapse into one
    entry's ``variants[]``; a term with >=2 incompatible confirmed expansions →
    ``provenance == CONFLICTING`` keeping ALL spans (never a silent pick,
    invariant 7).

Pure, infra-free: the LLM is a fake, ``select_model`` is spied so we assert the
bulk-only routing contract (escalation OFF; the strong tier is NEVER reached for
bulk mining — contract C7). Thresholds resolve via ``get_tunable`` — no literal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf_chat.comprehension import glossary_miner as G
from pdf_chat.comprehension.glossary_miner import mine_glossary
from pdf_chat.comprehension.provenance import Provenance


# ── chunk fixture (dict OR object; the miner accepts both) ─────────────────────
def _chunk(chunk_id, text, *, page_num=1, bbox=None, doc_id="doc1", doc_date=None):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "page_num": page_num,
        "bbox": bbox or [0, 0, 1, 1],
        "doc_id": doc_id,
        "doc_date": doc_date,
    }


# ── fake LLM seam (records calls; returns a programmed verdict) ────────────────
class _FakeGlossaryLLM:
    """Injected LLM seam. Each method is async and records its call.

    * ``confirm_definition`` — Task 4: confirms (or declines) a regex-proposed
      explicit definition for the supporting span.
    * ``synthesize_definition`` — Task 5: writes a usage definition for a coined
      term from its in-corpus contexts.
    * ``adjudicate_variants`` — Task 6: groups candidate variants/expansions and
      flags incompatible expansions.
    """

    def __init__(self, *, confirm=None, synthesize=None, adjudicate=None):
        self.confirm_calls: list[dict] = []
        self.synthesize_calls: list[dict] = []
        self.adjudicate_calls: list[dict] = []
        self._confirm = confirm
        self._synthesize = synthesize
        self._adjudicate = adjudicate

    async def confirm_definition(self, *, term, expansion, span, model_id, container_id):
        self.confirm_calls.append(
            {"term": term, "expansion": expansion, "span": span,
             "model_id": model_id, "container_id": container_id}
        )
        if callable(self._confirm):
            return self._confirm(term=term, expansion=expansion, span=span)
        # Default: confirm with the proposed expansion + a high confidence.
        return {
            "confirmed": True,
            "expansion": expansion,
            "definition": f"{expansion} — definition.",
            "confidence": 0.9,
        }

    async def synthesize_definition(self, *, term, contexts, model_id, container_id):
        self.synthesize_calls.append(
            {"term": term, "contexts": list(contexts),
             "model_id": model_id, "container_id": container_id}
        )
        if callable(self._synthesize):
            return self._synthesize(term=term, contexts=contexts)
        return {"definition": f"usage-derived meaning of {term}", "confidence": 0.8}

    async def adjudicate_variants(self, *, term, candidates, model_id, container_id):
        self.adjudicate_calls.append(
            {"term": term, "candidates": list(candidates),
             "model_id": model_id, "container_id": container_id}
        )
        if callable(self._adjudicate):
            return self._adjudicate(term=term, candidates=candidates)
        return {"same": True}


def _spy_select_model(monkeypatch):
    """Spy ``select_model`` so tests assert the bulk-only routing contract."""
    from pdf_chat import model_router

    recorded: list[dict] = []
    real = model_router.select_model

    def _spy(*, task, container_id, signals, store=None):
        choice = real(task=task, container_id=container_id, signals=signals, store=store)
        recorded.append({"task": task, "is_strong": choice.is_strong})
        return choice

    monkeypatch.setattr(G, "select_model", _spy)
    return recorded


def _by_term(entries):
    return {e.term: e for e in entries}


# ── Task 4 — explicit definitions (LLM-confirmed → STATED) ─────────────────────
@pytest.mark.asyncio
async def test_explicit_definition_is_stated_with_span(monkeypatch):
    recorded = _spy_select_model(monkeypatch)
    chunks = [
        _chunk("ch1",
               "The Customer Acquisition Cost (CAC) measures spend per new customer.",
               page_num=3, bbox=[10, 20, 30, 40]),
    ]
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", ontology_version=1,
        background_freq={},
    )
    by_term = _by_term(entries)
    assert "CAC" in by_term
    cac = by_term["CAC"]
    assert cac.provenance == Provenance.STATED.value
    assert cac.expansion == "Customer Acquisition Cost"
    # Evidence span carries the verbatim sentence + chunk_id + page/bbox.
    assert cac.evidence_spans, "STATED entry must carry an evidence span"
    span = cac.evidence_spans[0]
    assert span["chunk_id"] == "ch1"
    assert span["page_num"] == 3
    assert "Customer Acquisition Cost (CAC)" in span["text"]
    # Confidence is a resolved float (came from get_tunable gate, not a literal).
    assert isinstance(cac.confidence, float)
    # Bulk-only routing: every recorded choice is the bulk tier (never strong).
    assert recorded and all(r["is_strong"] is False for r in recorded)


@pytest.mark.asyncio
async def test_unconfirmed_candidate_is_dropped(monkeypatch):
    _spy_select_model(monkeypatch)
    chunks = [
        _chunk("ch1", "The Customer Acquisition Cost (CAC) measures spend."),
    ]
    # LLM declines to confirm → the candidate is a grounding-gate DROP.
    llm = _FakeGlossaryLLM(confirm=lambda **kw: {"confirmed": False})
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    assert all(e.term != "CAC" for e in entries)


@pytest.mark.asyncio
async def test_below_confidence_floor_is_dropped(monkeypatch):
    """A confirmed candidate under the tunable inclusion floor is excluded."""
    _spy_select_model(monkeypatch)
    chunks = [_chunk("ch1", "The Net Revenue Retention (NRR) tracks retained revenue.")]
    # Confirmed, but confidence below glossary.min_confidence (0.60 default).
    llm = _FakeGlossaryLLM(
        confirm=lambda **kw: {"confirmed": True, "expansion": kw["expansion"],
                              "definition": "x", "confidence": 0.10}
    )
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    assert all(e.term != "NRR" for e in entries)


@pytest.mark.asyncio
async def test_stands_for_pattern_is_proposed(monkeypatch):
    """The regex proposer also recognises the "X stands for Y" pattern."""
    _spy_select_model(monkeypatch)
    chunks = [_chunk("ch1", "ARR stands for Annual Recurring Revenue.")]
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    by_term = _by_term(entries)
    assert "ARR" in by_term
    assert by_term["ARR"].expansion == "Annual Recurring Revenue"
    assert by_term["ARR"].provenance == Provenance.STATED.value


# ── Task 5 — distributional anomaly (INFERRED, never STATED) ───────────────────
@pytest.mark.asyncio
async def test_distributional_anomaly_is_inferred(monkeypatch):
    recorded = _spy_select_model(monkeypatch)
    # A coined term repeated far above its (tiny) background frequency.
    coined = "ZephyrFlow"
    chunks = [
        _chunk(f"ch{i}", f"Our {coined} pipeline processes {coined} events nightly.")
        for i in range(8)
    ]
    # Background table: common words are frequent; the coined term is absent/rare.
    background_freq = {"the": -1.0, "our": -2.0, "pipeline": -5.0,
                       "events": -5.0, "processes": -6.0, "nightly": -7.0}
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1",
        background_freq=background_freq,
    )
    by_term = _by_term(entries)
    assert coined in by_term, "coined anomaly term should be mined"
    zf = by_term[coined]
    assert zf.provenance == Provenance.INFERRED.value
    assert zf.provenance != Provenance.STATED.value
    assert zf.definition, "an inferred entry carries an LLM-synthesized definition"
    assert zf.evidence_spans, "an inferred entry carries usage evidence spans"
    # The synthesize seam was actually used for the coined term.
    assert any(c["term"] == coined for c in llm.synthesize_calls)
    # Still bulk-only.
    assert recorded and all(r["is_strong"] is False for r in recorded)


@pytest.mark.asyncio
async def test_no_background_table_degrades_gracefully(monkeypatch):
    """With the background table absent, the anomaly signal is skipped (no crash,
    no fabricated STATED entry)."""
    _spy_select_model(monkeypatch)
    coined = "ZephyrFlow"
    chunks = [
        _chunk(f"ch{i}", f"Our {coined} pipeline processes {coined} events.")
        for i in range(8)
    ]
    llm = _FakeGlossaryLLM()
    # background_freq=None ⇒ anomaly disabled. No explicit definition present, so
    # NO entry should be fabricated for the coined term.
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq=None,
    )
    assert all(e.term != coined for e in entries)
    # And specifically: nothing STATED was fabricated from the anomaly path.
    assert all(e.provenance != Provenance.STATED.value
               or e.evidence_spans for e in entries)


@pytest.mark.asyncio
async def test_distributional_candidates_helper_uses_injected_table(monkeypatch):
    """``_distributional_candidates`` flags anomalies from corpus stats + the
    INJECTED background only — the open-vocab signal source."""
    _spy_select_model(monkeypatch)
    coined = "Qwizmo"
    chunks = [_chunk(f"ch{i}", f"{coined} {coined} {coined} drives growth.")
              for i in range(6)]
    background = {"drives": -5.0, "growth": -5.0}
    cands = G._distributional_candidates(chunks, background, "c1")
    flagged = {c["term"] for c in cands}
    assert coined in flagged


@pytest.mark.asyncio
async def test_lift_uses_background_values_not_membership(monkeypatch):
    """On a MIXED-frequency corpus the anomaly statistic USES the background
    log-frequency VALUES (lift), not mere key membership.

    A coined term ("Snorktel") absent from the background recurs far above what
    general usage predicts (its OOV floor is very rare) ⇒ high positive lift ⇒
    PASSES. A common in-background word ("revenue") at a SIMILAR in-corpus count
    has a high background log-frequency, so its lift is low ⇒ FAILS — function
    words / common business words can no longer dominate the gate. This proves the
    .json VALUES are consulted (the old z-score against a cross-token baseline
    would have flagged BOTH equally)."""
    _spy_select_model(monkeypatch)
    coined = "Snorktel"
    # Both the coined term and the common word "revenue" appear the SAME number of
    # times across the corpus (5 chunks → 5 mentions each), so a count-only or
    # cross-token z-score statistic could not tell them apart. Only a VALUE-using
    # lift (coined is OOV/rare in general usage; "revenue" is a common business
    # word with a high background log-freq) separates them.
    chunks = [_chunk(f"ch{i}", f"{coined} revenue grew this period.")
              for i in range(5)]
    # "revenue" is a common business word (high background log-freq, near the
    # corpus's own); "grew"/"this"/"period" are likewise in-background. "Snorktel"
    # is ABSENT (OOV ⇒ assumed very rare via the oov floor).
    background = {"revenue": -0.6, "grew": -1.0, "this": -0.5, "period": -1.2}
    cands = G._distributional_candidates(chunks, background, "c1")
    flagged = {c["term"] for c in cands}
    assert coined in flagged, "coined OOV term recurs far above general usage ⇒ lift passes"
    assert "revenue" not in flagged, (
        "a common in-background word at the SAME count must FAIL the lift gate "
        "(its high background log-freq cancels its corpus frequency)"
    )


@pytest.mark.asyncio
async def test_open_vocab_no_hardcoded_list(monkeypatch):
    """A made-up term never seen before is mineable purely from signals —
    mining consults NO static term allow-list (only corpus stats + injected
    background_freq)."""
    _spy_select_model(monkeypatch)
    nonce = "Blorptastic"
    chunks = [_chunk(f"ch{i}", f"The {nonce} index summarises {nonce} health.")
              for i in range(7)]
    background = {"the": -1.0, "index": -5.0, "health": -5.0, "summarises": -6.0}
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1",
        background_freq=background,
    )
    assert any(e.term == nonce for e in entries), "novel term must be mineable"


def test_glossary_miner_source_has_no_jargon_dictionary():
    """Static guard: the miner source contains no hardcoded jargon/acronym list.

    Open-vocab (invariant 6): the only domain vocabulary the miner ever sees is
    INJECTED (``background_freq``) or proposed by regex from the corpus + the LLM.
    A literal business-term dictionary in the source would violate that — assert
    the well-known seed acronyms used in OTHER tests do not appear hardcoded here.
    """
    src = Path(G.__file__).read_text(encoding="utf-8")
    for jargon in ("ZephyrFlow", "Qwizmo", "Blorptastic", "Annual Recurring Revenue",
                   "Customer Acquisition Cost", "Net Revenue Retention"):
        assert jargon not in src, f"hardcoded jargon found in miner: {jargon!r}"


# ── Task 6 — co-reference variants + conflict surfacing ────────────────────────
@pytest.mark.asyncio
async def test_variants_merged(monkeypatch):
    """An alias variant of a confirmed term collapses into ONE entry's variants[]."""
    _spy_select_model(monkeypatch)
    chunks = [
        _chunk("ch1", "The Customer Acquisition Cost (CAC) is spend per new customer."),
        _chunk("ch2", "Cust. Acq. Cost (CAC) rose last quarter."),
    ]

    def _confirm(*, term, expansion, span):
        # Both chunks confirm CAC → same expansion (consistent).
        return {"confirmed": True, "expansion": "Customer Acquisition Cost",
                "definition": "spend per new customer", "confidence": 0.9}

    llm = _FakeGlossaryLLM(
        confirm=_confirm,
        adjudicate=lambda **kw: {"same": True},
    )
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    by_term = _by_term(entries)
    assert "CAC" in by_term
    cac = by_term["CAC"]
    # Exactly one entry per term — the alias is a variant, not a duplicate row.
    assert sum(1 for e in entries if e.term == "CAC") == 1
    variants = cac.variants or []
    assert any("Cust. Acq. Cost" in v for v in variants)


@pytest.mark.asyncio
async def test_conflicting_definitions_surface_both(monkeypatch):
    """Two incompatible confirmed expansions for the same term → CONFLICTING,
    keeping BOTH spans (never a silent pick — invariant 7)."""
    _spy_select_model(monkeypatch)
    chunks = [
        _chunk("ch1", "NRR (Net Revenue Retention) tracks retained revenue.",
               doc_date="2024-01-01"),
        _chunk("ch2", "NRR (Net Run Rate) is the annualised run rate.",
               doc_date="2025-06-01"),
    ]

    def _confirm(*, term, expansion, span):
        return {"confirmed": True, "expansion": expansion,
                "definition": f"{expansion} def", "confidence": 0.9}

    # The adjudicator says the two expansions are NOT the same → conflict.
    llm = _FakeGlossaryLLM(
        confirm=_confirm,
        adjudicate=lambda **kw: {"same": False},
    )
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    by_term = _by_term(entries)
    assert "NRR" in by_term
    nrr = by_term["NRR"]
    assert nrr.provenance == Provenance.CONFLICTING.value
    # BOTH definitions/spans are retained (both sides surfaced).
    assert len(nrr.evidence_spans) >= 2
    expansions_in_spans = {s.get("expansion") for s in nrr.evidence_spans}
    assert "Net Revenue Retention" in expansions_in_spans
    assert "Net Run Rate" in expansions_in_spans


@pytest.mark.asyncio
async def test_single_consistent_definition_stays_stated(monkeypatch):
    """A term defined consistently across chunks stays STATED (not CONFLICTING)."""
    _spy_select_model(monkeypatch)
    chunks = [
        _chunk("ch1", "ARR (Annual Recurring Revenue) is recurring revenue."),
        _chunk("ch2", "ARR (Annual Recurring Revenue) grew 30%."),
    ]
    llm = _FakeGlossaryLLM(
        confirm=lambda **kw: {"confirmed": True,
                              "expansion": "Annual Recurring Revenue",
                              "definition": "recurring revenue", "confidence": 0.9},
        adjudicate=lambda **kw: {"same": True},
    )
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    by_term = _by_term(entries)
    assert by_term["ARR"].provenance == Provenance.STATED.value


# ── stamping + shape contracts ─────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_entries_are_stamped_with_tenant_and_version(monkeypatch):
    _spy_select_model(monkeypatch)
    chunks = [_chunk("ch1", "The Gross Margin (GM) is revenue minus COGS.")]
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        chunks, llm=llm, tenant_id="t9", container_id="c9", ontology_version=4,
        background_freq={},
    )
    assert entries
    for e in entries:
        assert e.tenant_id == "t9"
        assert e.container_id == "c9"
        assert e.ontology_version == 4


@pytest.mark.asyncio
async def test_empty_corpus_returns_empty(monkeypatch):
    _spy_select_model(monkeypatch)
    llm = _FakeGlossaryLLM()
    entries = await mine_glossary(
        [], llm=llm, tenant_id="t1", container_id="c1", background_freq={},
    )
    assert entries == []


# ── shipped background_freq.json data file ─────────────────────────────────────
def test_background_freq_json_ships_as_data():
    """The background frequency table ships as DATA (a JSON file), not a .py dict."""
    path = Path(G.__file__).with_name("background_freq.json")
    assert path.exists(), "background_freq.json must ship alongside the miner"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, "must be a non-empty unigram table"
    # Every UNIGRAM value is a log-frequency (numeric); underscore-prefixed keys
    # (e.g. an "_comment" provenance note) are metadata, not unigrams.
    unigrams = {k: v for k, v in data.items() if not k.startswith("_")}
    assert unigrams, "must contain unigram entries"
    assert all(isinstance(v, (int, float)) for v in unigrams.values())


def test_load_background_freq_reads_shipped_table():
    table = G.load_background_freq()
    assert isinstance(table, dict) and table

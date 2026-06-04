"""Tests for the no-LLM NER + value-overlap backbone (Phase-2 Task 3).

Covers BOTH the spaCy-present and spaCy-absent code paths (guarded import must
degrade to an empty list with no crash), plus the value-overlap link proposer
that reuses the ``fingerprint_value`` concept (no live DB).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pdf_chat.ingestion import ner_backbone as N
from pdf_chat.ingestion.ner_backbone import EntityCandidate, propose_entities, propose_links


# ── A tiny duck-typed Section stand-in (the real Section is owned by another
# agent's sectionizer.py; propose_links only reads .section_id/.text/.tenant_id).
@dataclass
class _Section:
    section_id: str
    tenant_id: str
    text: str


# ── A fake spaCy nlp + doc/ent so the present-path needs no real spaCy install.
class _FakeEnt:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    def __init__(self, ents):
        self._ents = ents

    def __call__(self, text):  # spaCy nlp(text) -> Doc
        return _FakeDoc(self._ents)


# ───────────────────────────── spaCy-ABSENT path ─────────────────────────────

def test_propose_entities_returns_empty_when_spacy_absent(monkeypatch):
    """Guarded import degrades to [] (no crash) when spaCy is unavailable."""
    monkeypatch.setattr(N, "_HAS_SPACY", False, raising=False)
    monkeypatch.setattr(N, "_DEFAULT_NLP", None, raising=False)
    out = propose_entities("Acme signed with Globex", container_id="t1")
    assert out == []


def test_propose_entities_no_crash_on_empty_text_spacy_absent(monkeypatch):
    monkeypatch.setattr(N, "_HAS_SPACY", False, raising=False)
    monkeypatch.setattr(N, "_DEFAULT_NLP", None, raising=False)
    assert propose_entities("", container_id="t1") == []


# ───────────────────────────── spaCy-PRESENT path ────────────────────────────

def test_propose_entities_uses_injected_nlp(monkeypatch):
    """An injected nlp produces candidates carrying text + label, source='ner'."""
    monkeypatch.setattr(N, "_HAS_SPACY", True, raising=False)
    nlp = _FakeNlp([_FakeEnt("Acme", "ORG"), _FakeEnt("Globex", "ORG")])
    out = propose_entities("Acme signed with Globex", container_id="t1", nlp=nlp)
    assert all(isinstance(c, EntityCandidate) for c in out)
    texts = {c.text for c in out}
    assert {"Acme", "Globex"} <= texts
    for c in out:
        assert c.label  # spaCy ent label preserved
        assert c.source == "ner"


def test_propose_entities_injected_nlp_works_even_if_has_spacy_false(monkeypatch):
    """Explicitly injecting nlp bypasses the global guard (testability)."""
    monkeypatch.setattr(N, "_HAS_SPACY", False, raising=False)
    nlp = _FakeNlp([_FakeEnt("Beta Corp", "ORG")])
    out = propose_entities("Beta Corp grows", container_id="t1", nlp=nlp)
    assert [c.text for c in out] == ["Beta Corp"]
    assert out[0].source == "ner"


def test_propose_entities_dedupes_repeated_mentions(monkeypatch):
    monkeypatch.setattr(N, "_HAS_SPACY", True, raising=False)
    nlp = _FakeNlp([_FakeEnt("Acme", "ORG"), _FakeEnt("acme", "ORG")])
    out = propose_entities("Acme and acme", container_id="t1", nlp=nlp)
    # case-insensitive dedupe → one candidate
    assert len(out) == 1


# ───────────────────────────── value-overlap links ───────────────────────────

def test_propose_links_finds_value_overlap_across_sections():
    """Two sections sharing a normalized value → one undirected link with evidence."""
    secs = [
        _Section("d::s0", "t1", "Invoice INV-0001 was issued to Acme."),
        _Section("d::s1", "t1", "Payment cleared for invoice inv-0001 last week."),
    ]
    links = propose_links(secs, container_id="t1")
    assert len(links) >= 1
    a, b, evidence = links[0]
    assert {a, b} == {"d::s0", "d::s1"}
    assert isinstance(evidence, str) and evidence  # the shared value as evidence


def test_propose_links_empty_when_no_overlap():
    secs = [
        _Section("d::s0", "t1", "Apples are red."),
        _Section("d::s1", "t1", "Bananas are yellow."),
    ]
    assert propose_links(secs, container_id="t1") == []


def test_propose_links_single_section_yields_no_links():
    assert propose_links([_Section("d::s0", "t1", "only one")], container_id="t1") == []


def test_propose_links_ignores_null_like_tokens():
    """Null-like / empty fingerprints must not create spurious links."""
    secs = [
        _Section("d::s0", "t1", "n/a n/a null"),
        _Section("d::s1", "t1", "null n/a none"),
    ]
    assert propose_links(secs, container_id="t1") == []


# ───────────────────────────── no magic literals ─────────────────────────────

def test_no_bare_score_literal_in_source():
    import inspect

    src = inspect.getsource(N)
    assert not re.search(r"\b(score|sim|threshold)\s*[<>]=?\s*0\.\d", src)

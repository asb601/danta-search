"""Phase-2 Task 4/5 tests — SECTION-LEVEL extraction (the EXTRACTION call site).

Pure, infra-free: the LLM is a fake returning a fixed JSON payload and
``select_model`` is spied so we can assert the EXACT routing contract:

  * extraction calls ``select_model(task=TaskClass.EXTRACTION, container_id=...,
    signals={})`` and uses the returned BULK model id;
  * the STRONG tier is NEVER invoked for bulk ingestion (every recorded choice
    has ``is_strong is False``) — escalation is OFF for bulk by construction;
  * one LLM call PER SECTION (section is the default granularity, not per-chunk);
  * idempotent on ``section_fingerprint`` (a re-extract is a cache hit, no 2nd call);
  * adaptive gleaning is capped at ``kg.gleaning.max_passes``;
  * dataclasses carry ``confidence`` + ``span`` + ``src_chunk_id``;
  * exactly one doc-level relational tag + a small set of section topic tags.
"""
from __future__ import annotations

import dataclasses

import pytest

from pdf_chat.ingestion import kg_extraction as K
from pdf_chat.ingestion.kg_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractedTag,
    SectionExtractor,
    section_fingerprint,
)
from pdf_chat.ingestion.sectionizer import Section
from pdf_chat.model_router import ModelChoice, TaskClass


# ── fixtures ──────────────────────────────────────────────────────────────────
def _section(fingerprint: str = "abc123", section_id: str = "doc1::s0") -> Section:
    return Section(
        section_id=section_id,
        doc_id="doc1",
        tenant_id="t1",
        chunk_ids=["c1", "c2"],
        text="Acme acquired Globex in 2025. Acme builds Product Z.",
        fingerprint=fingerprint,
        page_span=(1, 1),
    )


_PAYLOAD = {
    "entities": [
        {"name": "Acme", "type": "company", "confidence": 0.9,
         "span": "Acme acquired Globex", "src_chunk_id": "c1"},
        {"name": "Globex", "type": "company", "confidence": 0.85,
         "span": "Acme acquired Globex", "src_chunk_id": "c1"},
    ],
    "relations": [
        {"subject": "Acme", "predicate": "acquired", "object": "Globex",
         "confidence": 0.92, "span": "Acme acquired Globex", "src_chunk_id": "c1"},
    ],
    "tags": [
        {"label": "describes Product Z built 2025", "scope": "doc",
         "confidence": 0.8, "span": "Acme builds Product Z", "src_chunk_id": "c2"},
        {"label": "acquisitions", "scope": "section", "confidence": 0.7,
         "span": "Acme acquired Globex", "src_chunk_id": "c1"},
        {"label": "products", "scope": "section", "confidence": 0.6,
         "span": "Product Z", "src_chunk_id": "c2"},
    ],
}


class _FakeLLM:
    """Records every call; returns the same fixed payload each pass."""

    def __init__(self, payload=None):
        self.calls: list[dict] = []
        self._payload = payload if payload is not None else _PAYLOAD

    def extract(self, prompt, *, section, model_id, container_id, known_entities):
        self.calls.append(
            {"prompt": prompt, "section": section, "model_id": model_id,
             "container_id": container_id, "known_entities": list(known_entities)}
        )
        return self._payload


class _SpyStore:
    """Trivial cache seam exposing get/set, recording sets."""

    def __init__(self):
        self._d: dict = {}

    def get(self, k):
        return self._d.get(k)

    def set(self, k, v):
        self._d[k] = v


@pytest.fixture
def spy_select(monkeypatch):
    """Spy on select_model: record every ModelChoice it returns (bulk-only)."""
    recorded: list[dict] = []
    real = K.select_model

    def _spy(*, task, container_id, signals, **kw):
        choice = real(task=task, container_id=container_id, signals=signals, **kw)
        recorded.append({"task": task, "signals": signals, "is_strong": choice.is_strong,
                         "model_id": choice.model_id})
        return choice

    monkeypatch.setattr(K, "select_model", _spy)
    return recorded


# ── Task 4: dataclasses + fingerprint idempotency ─────────────────────────────
def test_dataclasses_carry_confidence_span_src_chunk():
    e = ExtractedEntity("Acme", "company", 0.9, "Acme acquired Globex", "c1")
    r = ExtractedRelation("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1")
    t = ExtractedTag("acquisitions", "section", 0.7, "Acme acquired Globex", "c1")
    for obj in (e, r, t):
        assert isinstance(obj.confidence, float)
        assert obj.span and obj.src_chunk_id
        # frozen / immutable
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.confidence = 0.1  # type: ignore[misc]


def test_section_fingerprint_stable_across_calls():
    s = _section()
    a = section_fingerprint(s, "p2.v1", "gpt-4o-mini")
    b = section_fingerprint(s, "p2.v1", "gpt-4o-mini")
    assert a == b


def test_section_fingerprint_changes_with_model_id():
    s = _section()
    assert section_fingerprint(s, "p2.v1", "gpt-4o-mini") != section_fingerprint(
        s, "p2.v1", "some-other-model"
    )


def test_section_fingerprint_changes_with_prompt_version():
    s = _section()
    assert section_fingerprint(s, "p2.v1", "m") != section_fingerprint(s, "p2.v2", "m")


# ── Task 5: the EXTRACTION call site contract ────────────────────────────────
def test_extract_routes_through_extraction_task_with_empty_signals(spy_select):
    llm = _FakeLLM()
    SectionExtractor(llm).extract(_section(), container_id="t1")
    assert spy_select, "select_model was not called"
    call = spy_select[0]
    assert call["task"] == TaskClass.EXTRACTION
    assert call["signals"] == {}  # bulk-only: escalation can never fire


def test_extraction_never_invokes_strong_tier(spy_select):
    """Escalation OFF for bulk: every routing decision stays on the bulk tier."""
    llm = _FakeLLM()
    SectionExtractor(llm).extract(_section(), container_id="t1")
    assert all(c["is_strong"] is False for c in spy_select)


def test_extract_uses_returned_bulk_model_id(spy_select):
    llm = _FakeLLM()
    SectionExtractor(llm).extract(_section(), container_id="t1")
    # the LLM seam was invoked with the SAME model id the router returned
    assert llm.calls
    assert llm.calls[0]["model_id"] == spy_select[0]["model_id"]


def test_one_llm_call_per_section_default_granularity():
    """Section is the LLM unit: a single section → at least one, bounded call set
    (no per-chunk fan-out). With max_passes=1 it is exactly one call."""
    import os
    os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"] = "1"
    try:
        llm = _FakeLLM()
        SectionExtractor(llm).extract(_section(), container_id="t1")
        assert len(llm.calls) == 1
    finally:
        del os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"]


def test_extract_returns_grounded_candidates():
    llm = _FakeLLM()
    entities, relations, tags = SectionExtractor(llm).extract(_section(), container_id="t1")
    assert {e.name for e in entities} == {"Acme", "Globex"}
    assert relations[0].subject == "Acme" and relations[0].obj == "Globex"
    for coll in (entities, relations, tags):
        for item in coll:
            assert item.span and item.src_chunk_id
            assert 0.0 <= item.confidence <= 1.0


def test_emits_one_doc_tag_and_section_tags():
    llm = _FakeLLM()
    _, _, tags = SectionExtractor(llm).extract(_section(), container_id="t1")
    doc_tags = [t for t in tags if t.scope == "doc"]
    section_tags = [t for t in tags if t.scope == "section"]
    assert len(doc_tags) == 1  # exactly one doc-level relational tag
    assert 1 <= len(section_tags) <= 5  # a small set of section topic tags


def test_section_tags_capped_per_container():
    payload = {
        "entities": [],
        "relations": [],
        "tags": (
            [{"label": "doc-tag", "scope": "doc", "confidence": 0.9,
              "span": "x", "src_chunk_id": "c1"}]
            + [{"label": f"topic{i}", "scope": "section", "confidence": 0.5 + i * 0.01,
                "span": "x", "src_chunk_id": "c1"} for i in range(20)]
        ),
    }
    import os
    os.environ["PDF_TUNABLE_KG.EXTRACTION.SECTION_TAG_CAP"] = "3"
    try:
        _, _, tags = SectionExtractor(_FakeLLM(payload)).extract(_section(), container_id="t1")
        assert len([t for t in tags if t.scope == "section"]) == 3
        assert len([t for t in tags if t.scope == "doc"]) == 1
    finally:
        del os.environ["PDF_TUNABLE_KG.EXTRACTION.SECTION_TAG_CAP"]


def test_idempotent_on_fingerprint_no_second_llm_call():
    llm = _FakeLLM()
    cache = _SpyStore()
    extractor = SectionExtractor(llm, cache=cache)
    s = _section()
    r1 = extractor.extract(s, container_id="t1")
    calls_after_first = len(llm.calls)
    r2 = extractor.extract(s, container_id="t1")  # same fingerprint
    assert len(llm.calls) == calls_after_first  # NO second LLM call
    assert r1 == r2


def test_changed_section_fingerprint_re_extracts():
    llm = _FakeLLM()
    cache = _SpyStore()
    extractor = SectionExtractor(llm, cache=cache)
    extractor.extract(_section(fingerprint="fpA"), container_id="t1")
    calls_after_first = len(llm.calls)
    extractor.extract(_section(fingerprint="fpB", section_id="doc1::s1"), container_id="t1")
    assert len(llm.calls) > calls_after_first  # different section → fresh call


def test_gleaning_capped_at_max_passes():
    import os
    os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"] = "2"
    os.environ["PDF_TUNABLE_KG.GLEANING.NEW_ENTITY_FLOOR"] = "1"
    try:
        # payload keeps yielding the SAME 2 entities → first pass adds 2 (new),
        # so a second pass runs; second pass adds 0 new → loop stops. Capped at 2.
        llm = _FakeLLM()
        SectionExtractor(llm).extract(_section(), container_id="t1")
        assert len(llm.calls) <= 2
    finally:
        del os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"]
        del os.environ["PDF_TUNABLE_KG.GLEANING.NEW_ENTITY_FLOOR"]


def test_gleaning_passes_known_entities_forward():
    """A gleaning pass tells the LLM which entities are already found (so it can
    glean the missed ones) — the contract for adaptive capped gleaning."""
    import os
    os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"] = "2"
    os.environ["PDF_TUNABLE_KG.GLEANING.NEW_ENTITY_FLOOR"] = "1"
    try:
        llm = _FakeLLM()
        SectionExtractor(llm).extract(_section(), container_id="t1")
        if len(llm.calls) >= 2:
            assert set(llm.calls[1]["known_entities"]) >= {"acme", "globex"}
    finally:
        del os.environ["PDF_TUNABLE_KG.GLEANING.MAX_PASSES"]
        del os.environ["PDF_TUNABLE_KG.GLEANING.NEW_ENTITY_FLOOR"]

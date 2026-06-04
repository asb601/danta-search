"""Tests — multi-representation card builders (Phase 2, Task 9).

MOCKS-ONLY: no live infra. The embedding seam (``retrieval.embeddings``) and the
model seam (``model_router.embedding_model``) are monkeypatched so these tests run
with zero Azure / Neo4j / Redis. Asserts (spec §1b multi-representation index):
  * a section card's text includes the section summary AND its section tag labels;
  * a doc card's text includes the doc-level tag label;
  * cards embed via ``embed_texts_batched(..., container_id=...)`` (the batched,
    per-container seam) using the per-container ``embedding_model``;
  * doc-scope tags are excluded from section cards and vice-versa (provenance
    stays correct);
  * every card carries ``tenant_id`` + provenance (misleading-tag safeguard:
    a card is a retrieval signal, traceable to its source chunks).
"""
from __future__ import annotations

import pdf_chat.ingestion.card_builder as cb
from pdf_chat.ingestion.card_builder import (
    DocCard,
    SectionCard,
    build_doc_card,
    build_section_card,
)
from pdf_chat.ingestion.kg_extraction import ExtractedTag
from pdf_chat.ingestion.sectionizer import Section


# ── fixtures ──────────────────────────────────────────────────────────────────
def _section(text: str = "Acme acquired Globex in 2025. Revenue rose sharply.") -> Section:
    return Section(
        section_id="doc-1::s0",
        doc_id="doc-1",
        tenant_id="t1",
        chunk_ids=["c0", "c1"],
        text=text,
        fingerprint="deadbeef",
        page_span=(1, 1),
    )


def _tag(label: str, scope: str, src: str = "c0") -> ExtractedTag:
    return ExtractedTag(label=label, scope=scope, confidence=0.9, span=label, src_chunk_id=src)


class _EmbedSpy:
    """Records the calls to embed_texts_batched so the seam is asserted."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, container_id, batch_size=None, model=None):
        self.calls.append({"texts": list(texts), "container_id": container_id, "model": model})
        return [[float(len(t))] for t in texts]


def _patch_embed(monkeypatch):
    spy = _EmbedSpy()
    monkeypatch.setattr(cb, "_embed_card_texts", cb._embed_card_texts)  # keep real wrapper
    # Patch the deepest seams the wrapper imports lazily.
    import pdf_chat.retrieval.embeddings as emb
    import pdf_chat.model_router as mr

    monkeypatch.setattr(emb, "embed_texts_batched", spy)
    monkeypatch.setattr(mr, "embedding_model", lambda container_id: "test-embed-model")
    return spy


# ── section card ──────────────────────────────────────────────────────────────
def test_section_card_text_includes_summary_and_tag_labels(monkeypatch):
    spy = _patch_embed(monkeypatch)
    tags = [_tag("growth metrics", "section"), _tag("acquisitions", "section")]
    card = build_section_card(_section(), tags, container_id="t1")

    assert isinstance(card, SectionCard)
    assert "Acme acquired Globex" in card.text  # the section summary
    assert "growth metrics" in card.text and "acquisitions" in card.text
    # embedded via the batched seam with the container_id + per-container model
    assert spy.calls and spy.calls[0]["container_id"] == "t1"
    assert spy.calls[0]["model"] == "test-embed-model"
    assert spy.calls[0]["texts"][0] == card.text
    assert card.embedding is not None


def test_section_card_carries_tenant_and_provenance(monkeypatch):
    _patch_embed(monkeypatch)
    card = build_section_card(_section(), [_tag("topic a", "section", src="c1")], container_id="t1")
    assert card.tenant_id == "t1"
    assert card.section_id == "doc-1::s0"
    assert card.doc_id == "doc-1"
    assert card.card_id == "doc-1::s0::card"
    assert "c1" in card.src_chunk_ids
    assert card.tag_labels == ("topic a",)


def test_section_card_ignores_doc_scope_tags(monkeypatch):
    _patch_embed(monkeypatch)
    tags = [_tag("section topic", "section"), _tag("describes Product A", "doc")]
    card = build_section_card(_section(), tags, container_id="t1")
    assert "section topic" in card.text
    assert "describes Product A" not in card.text


def test_section_card_caps_section_tags_via_tunable(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setenv("PDF_TUNABLE_KG.CARD.SECTION_TAG_CAP", "2")
    tags = [_tag(f"tag{i}", "section") for i in range(5)]
    card = build_section_card(_section(), tags, container_id="t1")
    assert len(card.tag_labels) == 2  # capped by the tunable, not a literal


def test_section_card_no_tags_still_embeds_summary(monkeypatch):
    spy = _patch_embed(monkeypatch)
    card = build_section_card(_section(), [], container_id="t1")
    assert "Acme acquired Globex" in card.text
    assert "Tags:" not in card.text
    assert card.embedding is not None
    assert spy.calls[0]["container_id"] == "t1"


def test_section_card_summary_truncated_via_tunable(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setenv("PDF_TUNABLE_KG.CARD.SUMMARY_MAX_CHARS", "10")
    long_section = _section(text="A" * 200)
    card = build_section_card(long_section, [], container_id="t1")
    assert len(card.text) <= 10


# ── doc card ──────────────────────────────────────────────────────────────────
def test_doc_card_text_includes_doc_level_tag(monkeypatch):
    spy = _patch_embed(monkeypatch)
    tags = [_tag("describes Product A, built 2025", "doc", src="c9")]
    card = build_doc_card(tags, container_id="t1", doc_id="doc-1", tenant_id="t1")

    assert isinstance(card, DocCard)
    assert "describes Product A, built 2025" in card.text
    assert card.tenant_id == "t1"
    assert card.doc_id == "doc-1"
    assert card.card_id == "doc-1::doccard"
    assert "c9" in card.src_chunk_ids
    assert spy.calls[0]["container_id"] == "t1"
    assert spy.calls[0]["model"] == "test-embed-model"
    assert card.embedding is not None


def test_doc_card_ignores_section_scope_tags(monkeypatch):
    _patch_embed(monkeypatch)
    tags = [_tag("section topic", "section"), _tag("doc relational tag", "doc")]
    card = build_doc_card(tags, container_id="t1", doc_id="doc-1", tenant_id="t1")
    assert "doc relational tag" in card.text
    assert "section topic" not in card.text


def test_doc_card_empty_tags_produces_empty_card_no_embed(monkeypatch):
    spy = _patch_embed(monkeypatch)
    card = build_doc_card([], container_id="t1", doc_id="doc-1", tenant_id="t1")
    assert card.text == ""
    assert card.embedding is None
    assert spy.calls == []  # no oversized empty embedding request


def test_doc_card_is_tenant_scoped(monkeypatch):
    _patch_embed(monkeypatch)
    card = build_doc_card(
        [_tag("x", "doc")], container_id="t1", doc_id="doc-9", tenant_id="t9"
    )
    assert card.tenant_id == "t9"
    assert card.doc_id == "doc-9"

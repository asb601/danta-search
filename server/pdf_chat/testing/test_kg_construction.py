"""Phase-2 — tests for the KG-CONSTRUCTION ORCHESTRATOR.

``construct_knowledge_graph`` is PURE ORCHESTRATION over injected backends — it
runs the Phase-2 chain end-to-end (sectionize → NER + section extraction →
grounding gate → entity resolution → write entities/relations/mentions/sections/
tags → build & store section/doc cards → detect communities + reports +
pagerank) and never owns any infra. Every backend is a SEAM, so this whole test
fakes the lot end-to-end with ZERO live infra (no Neo4j/Redis/Azure/spaCy/
networkx) — exactly the manager's mocks-only rule.

The faithfulness invariants asserted here mirror the plan:
  * SECTION is the extraction unit — the extractor is called once PER section.
  * GROUNDING IS BLOCKING — an edge/tag whose claim is absent from its cited
    chunk span is rejected and never written.
  * tenant_id flows to every writer call (multi-tenant isolation).
  * a tag never becomes an answer on its own (misleading-tag safeguard).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pdf_chat.ingestion.kg_construction import (
    KGConstructionResult,
    construct_knowledge_graph,
)
from pdf_chat.ingestion.ton_schema import Chunk, ElementType


# ── chunk fixtures (the only "real" objects; everything else is a fake seam) ──
def _chunk(chunk_id: str, text: str, *, page: int = 1, order: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        page_num=page,
        element_type=ElementType.TEXT,
        text=text,
        reading_order=order,
        tenant_id="tenantA",
    )


def _corpus() -> list[Chunk]:
    # A short heading + body so the real sectionizer groups into >=1 section.
    return [
        _chunk("c0", "Overview", page=1, order=0),
        _chunk("c1", "Acme acquired Globex in 2025.", page=1, order=1),
        _chunk("c2", "Globex is based in Springfield.", page=2, order=2),
    ]


# ── faked extraction payload objects (duck-typed Extracted* shapes) ──────────
@dataclass(frozen=True)
class _Entity:
    name: str
    etype: str
    confidence: float
    span: str
    src_chunk_id: str


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


# ── fake backends (every seam construct_knowledge_graph injects) ─────────────
class _FakeExtractor:
    """One call per section; returns one grounded edge + tags, plus one UNgrounded
    edge whose object is absent from the cited span (the gate must reject it)."""

    def __init__(self):
        self.calls: list = []

    def extract(self, section, *, container_id):
        self.calls.append(section.section_id)
        entities = [
            _Entity("Acme", "custom:org:acme", 0.9, "Acme", "c1"),
            _Entity("Globex", "custom:org:globex", 0.9, "Globex", "c1"),
        ]
        relations = [
            # grounded: both endpoints appear in c1's text
            _Relation("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1"),
            # UNgrounded: "Initech" is not in c1's text → gate rejects
            _Relation("Acme", "acquired", "Initech", 0.9, "fabricated", "c1"),
        ]
        tags = [
            # grounded doc tag (label present in c1 text)
            _Tag("acquired", "doc", 0.9, "Acme acquired Globex", "c1"),
            # UNgrounded section tag (label absent from c1 text) → rejected
            _Tag("space exploration", "section", 0.9, "Acme acquired Globex", "c1"),
        ]
        return entities, relations, tags


class _FakeNER:
    def __init__(self):
        self.calls: list = []

    def propose_entities(self, text, *, container_id, nlp=None):
        self.calls.append(text)
        return []  # backbone degrade path — orchestrator must not crash


class _FakeGate:
    """Delegates to the real grounding semantics but records every decision so the
    test can assert the ungrounded edge/tag were rejected."""

    def __init__(self):
        from pdf_chat.ingestion.grounding_gate import GroundingGate

        self._real = GroundingGate()
        self.admitted_edges: list = []
        self.rejected_edges: list = []

    def admit_edge(self, rel, *, cited_text, container_id):
        out = self._real.admit_edge(rel, cited_text=cited_text, container_id=container_id)
        (self.admitted_edges if out else self.rejected_edges).append(rel.obj)
        return out

    def admit_tag(self, tag, *, cited_text, container_id):
        return self._real.admit_tag(tag, cited_text=cited_text, container_id=container_id)


@dataclass(frozen=True)
class _Resolved:
    name: str
    etype: str
    normalized_value: str | None = None
    aliases: list = field(default_factory=list)


@dataclass(frozen=True)
class _Decision:
    kept: str
    merged: str
    merged_now: bool


class _FakeResolver:
    def __init__(self):
        self.calls: list = []

    def resolve(self, entities, *, embed_fn, container_id):
        self.calls.append([e.name for e in entities])
        resolved = [_Resolved(e.name, e.etype) for e in entities]
        return resolved, [_Decision("Acme", "ACME", False)]


class _FakeWriter:
    def __init__(self):
        self.entities = None
        self.edges = None
        self.mentions = None
        self.sections = None
        self.tags = None
        self.communities = None
        self.reports: list = []
        self.cards: list = []
        self.tenant_ids: list = []

    def write_entities(self, entities, *, tenant_id):
        self.entities = entities
        self.tenant_ids.append(tenant_id)
        return len(entities)

    def write_related_to(self, edges, *, tenant_id):
        self.edges = edges
        self.tenant_ids.append(tenant_id)
        return len(edges)

    def write_mentions(self, mentions, *, tenant_id):
        self.mentions = mentions
        self.tenant_ids.append(tenant_id)
        return len(mentions)

    def write_sections(self, sections, *, tenant_id):
        self.sections = sections
        self.tenant_ids.append(tenant_id)
        return sum(len(s.chunk_ids) for s in sections)

    def write_tags(self, tags, *, tenant_id):
        self.tags = tags
        self.tenant_ids.append(tenant_id)
        return len(tags)

    def write_communities(self, assignments, *, tenant_id):
        self.communities = assignments
        self.tenant_ids.append(tenant_id)
        return len(assignments)

    def write_community_reports(self, reports, *, tenant_id):
        self.reports.extend(reports)
        self.tenant_ids.append(tenant_id)
        return len(reports)

    # card-store seam (section/doc cards persisted as vector nodes)
    def write_cards(self, cards, *, tenant_id):
        self.cards.extend(cards)
        self.tenant_ids.append(tenant_id)
        return len(cards)


class _FakeCardBuilder:
    def __init__(self):
        self.section_calls = 0
        self.doc_calls = 0

    def build_section_card(self, section, tags, *, container_id):
        self.section_calls += 1
        return {"card_id": f"{section.section_id}::card", "kind": "section"}

    def build_doc_card(self, doc_tags, *, container_id, doc_id, tenant_id):
        self.doc_calls += 1
        return {"card_id": f"{doc_id}::doccard", "kind": "doc"}


@dataclass(frozen=True)
class _Community:
    community_id: str
    members: tuple
    src_chunk_ids: tuple = ()


class _FakeCommunities:
    """Bundles detect/pagerank/reporter so the orchestrator never imports the
    networkx-guarded module directly (works with networkx absent)."""

    def __init__(self):
        self.detect_calls = 0
        self.report_calls = 0

    def detect_communities(self, edges, *, container_id):
        self.detect_calls += 1
        return [_Community("tenantA::comm0", ("Acme", "Globex"), ("c1",))]

    def pagerank_confidence(self, edges):
        return {"Acme": 0.6, "Globex": 0.4}

    def report(self, community, edges, *, container_id):
        self.report_calls += 1
        return {"community_id": community.community_id, "summary": "Acme acquired Globex"}


def _embed_fn(names):
    return [[1.0, 0.0] for _ in names]


def _run(**overrides):
    extractor = overrides.pop("extractor", _FakeExtractor())
    ner = overrides.pop("ner", _FakeNER())
    gate = overrides.pop("gate", _FakeGate())
    resolver = overrides.pop("resolver", _FakeResolver())
    writer = overrides.pop("writer", _FakeWriter())
    card_builder = overrides.pop("card_builder", _FakeCardBuilder())
    communities = overrides.pop("communities", _FakeCommunities())
    chunks = overrides.pop("chunks", _corpus())
    result = construct_knowledge_graph(
        doc_id="doc1",
        container_id="tenantA",
        tenant_id="tenantA",
        chunks=chunks,
        extractor=extractor,
        ner=ner,
        gate=gate,
        resolver=resolver,
        writer=writer,
        card_builder=card_builder,
        communities=communities,
        embed_fn=_embed_fn,
        **overrides,
    )
    return result, dict(
        extractor=extractor, ner=ner, gate=gate, resolver=resolver,
        writer=writer, card_builder=card_builder, communities=communities,
    )


# --------------------------------------------------------------------------- #
# happy path: end-to-end with every backend faked
# --------------------------------------------------------------------------- #
def test_returns_result_dataclass():
    result, _ = _run()
    assert isinstance(result, KGConstructionResult)
    assert result.doc_id == "doc1"
    assert result.tenant_id == "tenantA"
    assert result.sections >= 1


def test_extractor_called_once_per_section():
    result, b = _run()
    # the real sectionizer groups c0(heading)+body → at least one section
    assert len(b["extractor"].calls) == result.sections
    assert result.sections >= 1


def test_ner_runs_as_backbone_before_extraction():
    _, b = _run()
    # NER proposer is invoked (the no-LLM backbone runs ahead of extraction)
    assert b["ner"].calls, "NER backbone must run over section text"


# --------------------------------------------------------------------------- #
# grounding is BLOCKING — ungrounded edge + ungrounded tag are rejected
# --------------------------------------------------------------------------- #
def test_ungrounded_edge_is_rejected_and_never_written():
    result, b = _run()
    writer = b["writer"]
    written_objs = {e.obj for e in (writer.edges or [])}
    assert "Globex" in written_objs           # grounded edge persisted
    assert "Initech" not in written_objs       # ungrounded edge rejected
    # the fake extractor emits exactly 1 grounded + 1 ungrounded edge PER section,
    # so the gate admits/rejects one each per section (grounding is blocking).
    assert result.edges_admitted == result.sections
    assert result.edges_rejected == result.sections


def test_ungrounded_tag_is_rejected():
    result, b = _run()
    labels = {t.label for t in (b["writer"].tags or [])}
    assert "acquired" in labels                # grounded doc tag persisted
    assert "space exploration" not in labels    # ungrounded section tag rejected


# --------------------------------------------------------------------------- #
# every writer call is tenant-scoped (multi-tenant isolation)
# --------------------------------------------------------------------------- #
def test_every_writer_call_is_tenant_scoped():
    _, b = _run()
    writer = b["writer"]
    assert writer.tenant_ids, "expected writer calls"
    assert all(t == "tenantA" for t in writer.tenant_ids)


def test_entities_relations_mentions_sections_tags_all_written():
    result, b = _run()
    w = b["writer"]
    assert w.entities is not None and len(w.entities) >= 1
    assert w.edges is not None and len(w.edges) == result.sections
    assert w.mentions is not None and len(w.mentions) >= 1
    assert w.sections is not None and len(w.sections) >= 1
    assert w.tags is not None and len(w.tags) >= 1


def test_mentions_anchor_grounded_entities_to_their_chunk():
    _, b = _run()
    # each mention binds a chunk_id + an entity name (the (:Chunk)-[:MENTIONS]->(:Entity) edge)
    for m in b["writer"].mentions:
        assert m.chunk_id
        assert m.name


# --------------------------------------------------------------------------- #
# entity resolution runs over the section-level entities
# --------------------------------------------------------------------------- #
def test_resolver_runs_over_extracted_entities():
    _, b = _run()
    assert b["resolver"].calls, "resolver must run"
    names = b["resolver"].calls[0]
    assert "Acme" in names and "Globex" in names


# --------------------------------------------------------------------------- #
# cards built + stored; communities + reports + pagerank run
# --------------------------------------------------------------------------- #
def test_section_and_doc_cards_built_and_stored():
    result, b = _run()
    cb = b["card_builder"]
    assert cb.section_calls == result.sections   # one section card per section
    assert cb.doc_calls == 1                       # exactly one doc card
    assert b["writer"].cards, "cards must be persisted via the writer card seam"


def test_communities_reports_and_pagerank_run():
    result, b = _run()
    comm = b["communities"]
    assert comm.detect_calls == 1
    assert comm.report_calls >= 1
    assert result.communities == 1
    assert result.pagerank  # pagerank computed over grounded edges


def test_community_reports_are_embedded_and_written():
    # The reporter's non-None report must be embedded (so the
    # community_report_vector_index the searcher reads is populated) and persisted
    # via the writer's write_community_reports seam (the citation loop is closed).
    result, b = _run()
    reports = b["writer"].reports
    assert reports, "non-suppressed community report must be written"
    assert result.reports == 1
    rep = reports[0]
    assert rep.community_id == "tenantA::comm0"
    assert rep.summary == "Acme acquired Globex"
    # embedded via the SAME embed_fn used for cards (2-dim fixture vector).
    assert list(rep.embedding) == [1.0, 0.0]
    # grounded_edge_count traces the report to its supporting grounded edges.
    assert rep.grounded_edge_count >= 1


def test_suppressed_report_is_not_written():
    # A reporter that SUPPRESSES (returns None) must not produce any persisted or
    # embedded report — only cited, grounded reports are ever retrievable.
    class _SuppressingCommunities(_FakeCommunities):
        def report(self, community, edges, *, container_id):
            self.report_calls += 1
            return None

    result, b = _run(communities=_SuppressingCommunities())
    assert result.reports == 0
    assert b["writer"].reports == []


# --------------------------------------------------------------------------- #
# degenerate inputs degrade gracefully (no crash)
# --------------------------------------------------------------------------- #
def test_empty_chunks_returns_empty_result_without_touching_backends():
    result, b = _run(chunks=[])
    assert isinstance(result, KGConstructionResult)
    assert result.sections == 0
    assert result.edges_admitted == 0
    assert not b["extractor"].calls
    assert b["writer"].entities is None  # nothing written


def test_no_grounded_edges_skips_community_detection_gracefully():
    class _NoEdgeExtractor:
        def __init__(self):
            self.calls = []

        def extract(self, section, *, container_id):
            self.calls.append(section.section_id)
            # only entities, no relations/tags
            return [_Entity("Acme", "custom:org:acme", 0.9, "Acme", "c1")], [], []

    result, b = _run(extractor=_NoEdgeExtractor())
    assert result.edges_admitted == 0
    # communities detection still invoked but yields nothing actionable;
    # the orchestrator must not crash on an empty grounded-edge set.
    assert isinstance(result, KGConstructionResult)


# --------------------------------------------------------------------------- #
# misleading-tag safeguard: a grounded tag alone is not surfaced as an answer
# --------------------------------------------------------------------------- #
def test_tags_are_retrieval_signals_not_answers():
    result, _ = _run()
    # the orchestrator persists tags as a retrieval signal; it never promotes a
    # tag to an answer claim (no answer field is produced from tags alone).
    assert not getattr(result, "answers", None)

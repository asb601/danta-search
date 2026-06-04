"""Phase-2 — the KG-CONSTRUCTION ORCHESTRATOR (ties the phase together).

``construct_knowledge_graph`` is the single entry point that runs the whole
Phase-2 ingestion chain end-to-end, AFTER all of a document's pages have settled
(the Phase-1 writer has already persisted the structural
``(:Document)-[:CONTAINS]->(:Page)-[:CONTAINS]->(:Chunk)`` spine; this builds the
SEMANTIC layer on top of those chunks):

    chunks
      → [sectionize]                 group chunks into SECTIONS (the LLM unit)
      → per section:
          [NER backbone]             cheap, no-LLM entity candidates (degrade ok)
          [section extraction]       ONE bulk gpt-4o-mini call → entities/relations/tags
      → [grounding gate]             BLOCKING: reject any edge/tag whose claim is
                                     absent from its cited chunk span
      → [entity resolution]          collapse open-vocab names → canonical entities
      → [write]                      entities / relations / mentions / sections / tags
      → [cards]                      build + store section-cards & the doc-card
      → [communities]                detect communities + cited reports + pagerank

This module is PURE ORCHESTRATION over INJECTED backends — it owns no infra and
no model/threshold literal. Every backend (extractor, ner, gate, resolver,
writer, card_builder, communities) is a SEAM passed in by the caller, so the
whole phase is end-to-end mockable with zero live infra (Neo4j/Redis/Azure/
spaCy/networkx). The worker bootstrap wires the real implementations; tests wire
fakes.

GOVERNING CRITERIA (many tenants, millions of files, cross-context):
  * cost-at-scale — extraction is SECTION-level (one bulk call per section, not
    per chunk); the orchestrator never adds a second model brain.
  * multi-tenant isolation — ``tenant_id`` is threaded onto EVERY writer call and
    onto every artifact; the orchestrator never widens a tenant boundary.
  * grounded faithfulness — the grounding gate is BLOCKING: an ungrounded edge or
    tag is rejected here and is never written, embedded, or reported. Tags are a
    retrieval signal, never an answer (no answer is ever synthesized from a tag).
  * per-client tunability — every threshold lives in the injected backends (each
    resolves via ``get_tunable``); the orchestrator passes ``container_id``
    through so per-container dials apply uniformly.

Idempotency: each backend is itself idempotent (the extractor caches on
``section_fingerprint``; every writer statement is a MERGE keyed on
``(<businessKey>, tenant_id)``), so a re-run of ``construct_knowledge_graph`` for
the same document overwrites rather than duplicating — safe under retry/DLQ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..tunables import log_gate_decision

# The orchestrator owns NO threshold; it only LOGS what it routed/skipped so a
# decision is never silent (spec §3 inv 4). These are structural sentinels for
# the log harness (score/threshold), not per-container dials.
_PRESENT = 1.0
_ABSENT = 0.0


# ── lightweight carriers the orchestrator derives between seams ───────────────
@dataclass(frozen=True)
class _Mention:
    """A ``(:Chunk)-[:MENTIONS]->(:Entity)`` candidate the writer persists.

    Derived from a GROUNDED edge's endpoints (and the section's grounded
    entities), so a mention only ever anchors an entity that survived the gate.
    """

    chunk_id: str
    name: str
    confidence: float
    span: str


@dataclass(frozen=True)
class _CommunityReport:
    """A cited community report carrier the writer persists as ``(:CommunityReport)``.

    Derived from the reporter's ``CommunityReport`` plus an ``embedding`` of its
    summary (so the ``community_report_vector_index`` the searcher reads is
    populated) and the ``grounded_edge_count`` it was traced to. Built only for
    NON-suppressed reports — a suppressed (``None``) report is never carried here,
    so it is never embedded, written, or made retrievable (faithfulness).
    """

    community_id: str
    summary: str
    citations: tuple[str, ...]
    grounded_edge_count: int
    embedding: Sequence[float] | None


@dataclass(frozen=True)
class _Assignment:
    """An ``(:Entity)-[:IN_COMMUNITY]->(:Community)`` assignment for the writer.

    Carries the per-community size + the entity's confidence-weighted pagerank so
    the persisted community node reflects the grounded-edge importance signal.
    """

    community_id: str
    name: str
    size: int
    pagerank: float


@dataclass(frozen=True)
class KGConstructionResult:
    """A pure, infra-free summary of one document's KG construction.

    Returned to the caller (finalize/run_page_pipeline) so the control plane can
    record what was built without re-reading Neo4j. Counts are the audit trail the
    Phase-2 exit gate / faithfulness eval inspect.
    """

    doc_id: str
    container_id: str
    tenant_id: str
    sections: int = 0
    entities_resolved: int = 0
    edges_admitted: int = 0
    edges_rejected: int = 0
    tags_admitted: int = 0
    tags_rejected: int = 0
    mentions_written: int = 0
    cards_written: int = 0
    communities: int = 0
    reports: int = 0
    pagerank: dict[str, float] = field(default_factory=dict)


def _report_field(report: Any, field_name: str, default: Any = None) -> Any:
    """Read ``field_name`` off a report that may be an attribute-object OR a dict.

    The reporter seam returns a duck-typed ``CommunityReport`` (attribute access)
    in production, but a test fake may return a plain dict — support both so the
    orchestrator never couples to one shape.
    """
    if isinstance(report, dict):
        return report.get(field_name, default)
    return getattr(report, field_name, default)


def _chunk_text_index(chunks: Sequence[Any]) -> dict[str, str]:
    """Map ``chunk_id -> text`` so the grounding gate can fetch a cited span.

    The extractor cites a ``src_chunk_id`` for every claim; the gate needs that
    chunk's verbatim text to verify the claim is present. We index once, O(n).
    """
    return {c.chunk_id: (c.text or "") for c in chunks if getattr(c, "chunk_id", None)}


def construct_knowledge_graph(
    doc_id: str,
    container_id: str,
    tenant_id: str,
    chunks: Sequence[Any],
    *,
    extractor: Any,
    ner: Any,
    gate: Any,
    resolver: Any,
    writer: Any,
    card_builder: Any,
    communities: Any,
    embed_fn: Callable[[list[str]], list[Sequence[float]]],
    sectionize: Callable[..., list[Any]] | None = None,
) -> KGConstructionResult:
    """Run the full Phase-2 KG construction for ONE settled document.

    Args:
        doc_id: the document whose chunks are being graph-ified.
        container_id: the per-container dial scope (tunables/routing).
        tenant_id: the tenant boundary — threaded onto EVERY writer call so no
            node/edge can cross tenants (spec §3 inv 3).
        chunks: the document's Phase-1 chunks (each carrying ``chunk_id``/``text``
            /``tenant_id``). These are the grounding substrate.
        extractor: a ``SectionExtractor``-like seam exposing
            ``extract(section, *, container_id) -> (entities, relations, tags)``.
            One call PER section (SECTION is the extraction unit).
        ner: the no-LLM backbone exposing
            ``propose_entities(text, *, container_id, nlp=None)``. Degrades to
            ``[]`` when spaCy is absent — the orchestrator must not crash.
        gate: a ``GroundingGate``-like seam exposing ``admit_edge`` / ``admit_tag``
            (BLOCKING — returns ``None`` for an ungrounded claim).
        resolver: an ``EntityResolver``-like seam exposing
            ``resolve(entities, *, embed_fn, container_id)``.
        writer: a ``Neo4jKGWriter``-like seam exposing ``write_entities`` /
            ``write_related_to`` / ``write_mentions`` / ``write_sections`` /
            ``write_tags`` / ``write_communities`` /
            ``write_community_reports`` (+ ``write_cards`` for the
            multi-representation card nodes). Every method takes ``tenant_id``.
        card_builder: a seam exposing ``build_section_card(section, tags, *,
            container_id)`` and ``build_doc_card(doc_tags, *, container_id,
            doc_id, tenant_id)``.
        communities: a seam exposing ``detect_communities(edges, *, container_id)``,
            ``pagerank_confidence(edges)`` and ``report(community, edges, *,
            container_id)``. Bundling these lets the orchestrator stay free of the
            networkx-guarded import (works with networkx absent).
        embed_fn: the per-name embedding callable passed through to the resolver
            (the SAME model used at ingest/query time).
        sectionize: the sectionizer callable; defaults to the real
            ``ingestion.sectionizer.sectionize`` (injected only for tests).

    Returns:
        A :class:`KGConstructionResult` summarizing what was built.
    """
    result_skeleton = KGConstructionResult(
        doc_id=doc_id, container_id=container_id, tenant_id=tenant_id
    )

    if not chunks:
        log_gate_decision(
            "kg.construct.empty",
            score=_ABSENT,
            threshold=_PRESENT,
            outcome="skip",
            container_id=container_id,
            doc_id=doc_id,
        )
        return result_skeleton

    # ── 1. sectionize — group chunks into the LLM extraction units ────────────
    if sectionize is None:
        from .sectionizer import sectionize as _sectionize
    else:
        _sectionize = sectionize
    sections = _sectionize(list(chunks), container_id=container_id)
    if not sections:
        log_gate_decision(
            "kg.construct.no_sections",
            score=_ABSENT,
            threshold=_PRESENT,
            outcome="skip",
            container_id=container_id,
            doc_id=doc_id,
        )
        return result_skeleton

    chunk_text = _chunk_text_index(chunks)

    # Accumulators across all sections (grounded only — the gate is the choke).
    all_entities: list[Any] = []          # extracted entities (pre-resolution)
    grounded_edges: list[Any] = []        # GroundedEdge after the gate
    grounded_tags: list[Any] = []         # GroundedTag after the gate (all scopes)
    doc_tags: list[Any] = []              # doc-scope grounded tags (for the doc card)
    mentions: list[_Mention] = []         # (:Chunk)-[:MENTIONS]->(:Entity)
    section_cards: list[Any] = []         # built once per section
    edges_rejected = 0
    tags_rejected = 0

    # ── 2. per-section: NER backbone → section extraction → grounding gate ─────
    for section in sections:
        # 2a. no-LLM backbone (cheap candidates; degrade to [] when spaCy absent).
        # The orchestrator does not consume the candidates as facts — they prime
        # the extractor's recall; here we run them so the backbone always fires.
        try:
            ner.propose_entities(section.text, container_id=container_id)
        except Exception:  # a broken NER must never abort KG construction
            log_gate_decision(
                "kg.construct.ner_error",
                score=_ABSENT,
                threshold=_PRESENT,
                outcome="degrade",
                container_id=container_id,
                section_id=section.section_id,
            )

        # 2b. section-level extraction — ONE bulk call per section.
        entities, relations, tags = extractor.extract(
            section, container_id=container_id
        )
        all_entities.extend(entities)

        # 2c. grounding gate (BLOCKING) for every edge.
        section_grounded_entity_names: set[str] = set()
        for rel in relations:
            cited = chunk_text.get(rel.src_chunk_id, section.text)
            edge = gate.admit_edge(rel, cited_text=cited, container_id=container_id)
            if edge is None:
                edges_rejected += 1
                continue
            grounded_edges.append(edge)
            # a grounded edge's endpoints are, by construction, present in the
            # cited chunk → they are safe MENTIONS anchors.
            for name in (edge.subject, edge.obj):
                section_grounded_entity_names.add(name)
                mentions.append(
                    _Mention(
                        chunk_id=edge.src_chunk_id,
                        name=name,
                        confidence=edge.confidence,
                        span=edge.span,
                    )
                )

        # 2d. grounding gate (BLOCKING) for every tag.
        section_grounded_tags: list[Any] = []
        for tag in tags:
            cited = chunk_text.get(tag.src_chunk_id, section.text)
            gtag = gate.admit_tag(tag, cited_text=cited, container_id=container_id)
            if gtag is None:
                tags_rejected += 1
                continue
            grounded_tags.append(gtag)
            section_grounded_tags.append(gtag)
            if getattr(gtag, "scope", "section") == "doc":
                doc_tags.append(gtag)

        # 2e. build the section card from the section + its grounded tags. A card
        # is a RETRIEVAL signal (its tags already cleared the gate); it is never
        # an answer. Built per section so the multi-representation index is dense.
        section_cards.append(
            card_builder.build_section_card(
                section, section_grounded_tags, container_id=container_id
            )
        )

    log_gate_decision(
        "kg.construct.grounding",
        score=float(len(grounded_edges)),
        threshold=float(len(grounded_edges) + edges_rejected),
        outcome="gated",
        container_id=container_id,
        doc_id=doc_id,
        edges_admitted=len(grounded_edges),
        edges_rejected=edges_rejected,
        tags_admitted=len(grounded_tags),
        tags_rejected=tags_rejected,
    )

    # ── 3. entity resolution — collapse open-vocab names → canonical entities ──
    resolved: list[Any] = []
    if all_entities:
        resolved, _decisions = resolver.resolve(
            all_entities, embed_fn=embed_fn, container_id=container_id
        )

    # ── 4. write the grounded graph (tenant_id on EVERY call) ─────────────────
    if resolved:
        writer.write_entities(resolved, tenant_id=tenant_id)
    if grounded_edges:
        writer.write_related_to(grounded_edges, tenant_id=tenant_id)
    if mentions:
        writer.write_mentions(mentions, tenant_id=tenant_id)
    writer.write_sections(sections, tenant_id=tenant_id)
    if grounded_tags:
        writer.write_tags(grounded_tags, tenant_id=tenant_id)

    # ── 5. multi-representation cards: section cards + the single doc card ─────
    doc_card = card_builder.build_doc_card(
        doc_tags, container_id=container_id, doc_id=doc_id, tenant_id=tenant_id
    )
    cards = list(section_cards) + [doc_card]
    cards_written = 0
    if cards:
        cards_written = writer.write_cards(cards, tenant_id=tenant_id) or 0

    # ── 6. communities + cited reports + confidence-weighted pagerank ─────────
    detected = communities.detect_communities(
        grounded_edges, container_id=container_id
    )
    pagerank = communities.pagerank_confidence(grounded_edges) or {}

    reports = 0
    community_reports: list[_CommunityReport] = []
    assignments: list[_Assignment] = []
    for community in detected:
        report = communities.report(
            community, grounded_edges, container_id=container_id
        )
        if report is not None:
            reports += 1
            # Embed the summary via the SAME embedding seam used for cards so the
            # community_report_vector_index the searcher reads is populated (the
            # citation loop is otherwise broken — nothing else writes that index).
            # A suppressed (None) report is NEVER carried here → never embedded or
            # written (faithfulness: only cited, grounded reports are retrievable).
            summary = _report_field(report, "summary", "") or ""
            embedding: Sequence[float] | None = None
            if summary:
                embedded = embed_fn([summary]) or []
                embedding = embedded[0] if embedded else None
            members = set(getattr(community, "members", ()) or ())
            grounded_edge_count = sum(
                1
                for e in grounded_edges
                if e.subject in members and e.obj in members
            )
            community_reports.append(
                _CommunityReport(
                    community_id=_report_field(
                        report, "community_id", community.community_id
                    ),
                    summary=summary,
                    citations=tuple(_report_field(report, "citations", ()) or ()),
                    grounded_edge_count=grounded_edge_count,
                    embedding=embedding,
                )
            )
        size = len(getattr(community, "members", ()) or ())
        for name in getattr(community, "members", ()) or ():
            assignments.append(
                _Assignment(
                    community_id=community.community_id,
                    name=name,
                    size=size,
                    pagerank=float(pagerank.get(name, 0.0)),
                )
            )
    if assignments:
        writer.write_communities(assignments, tenant_id=tenant_id)
    if community_reports:
        writer.write_community_reports(community_reports, tenant_id=tenant_id)

    log_gate_decision(
        "kg.construct.done",
        score=float(len(detected)),
        threshold=_ABSENT,
        outcome="constructed",
        container_id=container_id,
        doc_id=doc_id,
        sections=len(sections),
        entities=len(resolved),
        edges=len(grounded_edges),
        communities=len(detected),
        reports=reports,
    )

    return KGConstructionResult(
        doc_id=doc_id,
        container_id=container_id,
        tenant_id=tenant_id,
        sections=len(sections),
        entities_resolved=len(resolved),
        edges_admitted=len(grounded_edges),
        edges_rejected=edges_rejected,
        tags_admitted=len(grounded_tags),
        tags_rejected=tags_rejected,
        mentions_written=len(mentions),
        cards_written=cards_written,
        communities=len(detected),
        reports=reports,
        pagerank=dict(pagerank),
    )

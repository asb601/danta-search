"""Phase-2 Task 8 — the Neo4j KNOWLEDGE-GRAPH writer.

Persists the grounded, section-level knowledge graph produced by the Phase-2
ingestion chain (sectionizer → NER backbone → section extractor → grounding gate
→ entity resolver → communities). The Phase-1 writer (``neo4j_writer.py``) owns
the structural ``(:Document)-[:CONTAINS]->(:Page)-[:CONTAINS]->(:Chunk)`` spine;
this writer adds the *semantic* layer the searcher actually traverses (spec §1b,
schema item 6):

    (:Entity {name, tenant_id, etype, normalized_value})
    (:Entity)-[:RELATED_TO {desc, weight, confidence, evidence_count, src_chunk}]->(:Entity)
    (:Chunk)-[:MENTIONS]->(:Entity)
    (:Section)-[:HAS_CHUNK]->(:Chunk)
    (:Tag {label, scope, tenant_id})  +  (:Entity|:Section|:Document)-[:HAS_TAG]->(:Tag)
    (:Entity)-[:IN_COMMUNITY]->(:Community {community_id, tenant_id})

INVARIANTS (spec §3 inv 3, manager hard rules):
    * Every node carries ``tenant_id``.
    * EVERY write is a MERGE keyed on ``(<businessKey>, tenant_id)`` — never a bare
      CREATE — so retry/DLQ replay OVERWRITES rather than duplicating, and a leaked
      key for one tenant can never collide with another tenant's node.
    * Grounding props (``src_chunk``/``confidence``/``span``/``evidence_count``)
      ride on every edge and tag so nothing ungrounded is ever persisted (the
      GroundingGate is the upstream choke point; this writer only binds what it
      already vetted).

The ``neo4j`` driver is imported behind a guard (mirrors ``neo4j_writer.py``):
``Neo4jKGWriter`` is always constructible (so callers / tests can wire it without
infra), but its write methods raise a clear ``RuntimeError`` when the driver is
absent. The Cypher lives in small **static** helper methods so it is reviewable,
reusable, and assertable in tests WITHOUT touching infra.

GOVERNING CRITERIA (millions of files, many tenants): writes are per-statement
MERGE (idempotent, O(1) per artifact), tenant_id is bound on every statement, and
no model/threshold literal lives here (this module persists; it does not gate).

Inputs are duck-typed (attribute-carrying objects) exactly like ``grounding_gate``
so this module never imports another agent's not-yet-present file
(``entity_resolution``/``kg_extraction``/``communities``): a ``GroundedEdge`` /
``GroundedTag`` / ``ResolvedEntity`` is consumed by attribute access only.
"""
from __future__ import annotations

try:
    from neo4j import GraphDatabase  # type: ignore

    _HAS_NEO4J = True
except ImportError:  # pragma: no cover - exercised only without infra
    GraphDatabase = None  # type: ignore
    _HAS_NEO4J = False


class Neo4jKGWriter:
    """Writes the grounded knowledge-graph layer into Neo4j with tenant isolation.

    Every public ``write_*`` method binds ``$tenant_id`` on every MERGE and keys
    each node on its business key + ``tenant_id``. The writer is stateless beyond
    its connection; construct once per ingestion and reuse.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None  # lazily opened; None until connected

    # ---- lifecycle --------------------------------------------------------
    def _require_driver(self):
        # An already-connected/injected driver wins (so a caller — or a test
        # mocking the session — can supply its own). Only when none is present do
        # we fall back to opening one, which requires the real neo4j package.
        if self._driver is not None:
            return self._driver
        if not _HAS_NEO4J:
            raise RuntimeError(
                "The neo4j driver is required to write the knowledge graph but is "
                "not installed. Install it with `pip install neo4j`."
            )
        if self._driver is None:
            self._driver = GraphDatabase.driver(  # type: ignore[union-attr]
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jKGWriter":
        self._require_driver()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Cypher (STATIC helpers — pure strings, reusable + reviewable) -----
    # Every constant below MERGEs on (<businessKey>, tenant_id) and binds
    # $tenant_id, so the static-cypher audit (Task 12 exit gate) and this
    # module's own tests can assert tenant isolation WITHOUT any infra.

    @staticmethod
    def _entity_cypher() -> str:
        # Entity business key = (name, tenant_id). normalized_value is the
        # Phase-4 bridge; etype is the open-vocab custom:<kind>:<slug> type.
        # Mutable props converge on BOTH branches so a resolver re-run (alias
        # discovery, type refinement) overwrites rather than duplicating.
        return (
            "MERGE (e:Entity {name: $name, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    e.etype = $etype, e.normalized_value = $normalized_value, "
            "    e.aliases = $aliases "
            "  ON MATCH SET "
            "    e.etype = $etype, e.normalized_value = $normalized_value, "
            "    e.aliases = $aliases"
        )

    @staticmethod
    def _related_to_cypher() -> str:
        # MERGE both endpoint Entities on (name, tenant_id) FIRST so the edge can
        # never bridge two tenants, then MERGE the RELATED_TO edge itself keyed on
        # the (subject, predicate, object, tenant) triple. Grounding props ride on
        # the edge: desc/weight/confidence/evidence_count/src_chunk (spec schema 6).
        return (
            "MERGE (a:Entity {name: $subject, tenant_id: $tenant_id}) "
            "MERGE (b:Entity {name: $obj, tenant_id: $tenant_id}) "
            "MERGE (a)-[r:RELATED_TO {predicate: $predicate, tenant_id: $tenant_id}]->(b) "
            "  ON CREATE SET "
            "    r.desc = $desc, r.weight = $weight, r.confidence = $confidence, "
            "    r.evidence_count = $evidence_count, r.src_chunk = $src_chunk, "
            "    r.span = $span "
            "  ON MATCH SET "
            "    r.desc = $desc, r.weight = $weight, r.confidence = $confidence, "
            "    r.evidence_count = $evidence_count, r.src_chunk = $src_chunk, "
            "    r.span = $span"
        )

    @staticmethod
    def _mentions_cypher() -> str:
        # (:Chunk)-[:MENTIONS]->(:Entity). Both nodes MERGE on their business key +
        # tenant_id; the Chunk is matched/merged on (chunk_id, tenant_id) (the same
        # key the Phase-1 writer used) so this never forks the chunk identity.
        return (
            "MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id}) "
            "MERGE (e:Entity {name: $name, tenant_id: $tenant_id}) "
            "MERGE (c)-[m:MENTIONS {tenant_id: $tenant_id}]->(e) "
            "  ON CREATE SET m.confidence = $confidence, m.span = $span "
            "  ON MATCH SET m.confidence = $confidence, m.span = $span"
        )

    @staticmethod
    def _section_cypher() -> str:
        # (:Section)-[:HAS_CHUNK]->(:Chunk). Section business key = (section_id,
        # tenant_id); each member chunk MERGEs on (chunk_id, tenant_id). Called
        # once per member chunk so the HAS_CHUNK fan-out is idempotent.
        return (
            "MERGE (s:Section {section_id: $section_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    s.doc_id = $doc_id, s.fingerprint = $fingerprint, "
            "    s.page_start = $page_start, s.page_end = $page_end "
            "  ON MATCH SET "
            "    s.doc_id = $doc_id, s.fingerprint = $fingerprint, "
            "    s.page_start = $page_start, s.page_end = $page_end "
            "MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id}) "
            "MERGE (s)-[:HAS_CHUNK]->(c)"
        )

    @staticmethod
    def _tag_cypher() -> str:
        # Tag node business key = (label, scope, tenant_id); the owner (Entity for
        # section/entity scope or Document for doc scope) attaches via HAS_TAG. The
        # owner is matched generically on (owner_key, tenant_id). Grounding props
        # (confidence/span/src_chunk) ride on the HAS_TAG edge so a tag is never
        # persisted without its provenance (spec §1b: tags are a retrieval signal).
        return (
            "MERGE (t:Tag {label: $label, scope: $scope, tenant_id: $tenant_id}) "
            "MERGE (o {owner_key: $owner_key, tenant_id: $tenant_id}) "
            "MERGE (o)-[h:HAS_TAG {tenant_id: $tenant_id}]->(t) "
            "  ON CREATE SET "
            "    h.confidence = $confidence, h.span = $span, h.src_chunk = $src_chunk "
            "  ON MATCH SET "
            "    h.confidence = $confidence, h.span = $span, h.src_chunk = $src_chunk"
        )

    @staticmethod
    def _community_cypher() -> str:
        # (:Entity)-[:IN_COMMUNITY]->(:Community). Community business key =
        # (community_id, tenant_id); the member Entity MERGEs on (name, tenant_id).
        # Called once per member so reassignment (re-run of Leiden) overwrites.
        return (
            "MERGE (k:Community {community_id: $community_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET k.size = $size, k.pagerank = $pagerank "
            "  ON MATCH SET k.size = $size, k.pagerank = $pagerank "
            "MERGE (e:Entity {name: $name, tenant_id: $tenant_id}) "
            "MERGE (e)-[:IN_COMMUNITY]->(k)"
        )

    @staticmethod
    def _community_report_cypher() -> str:
        # (:CommunityReport) vector node, business key = (community_id, tenant_id).
        # Holds the CITED Leiden report and an embedding of its summary, indexed by
        # ``community_report_vector_index`` (the searcher's report leg). The searcher
        # reads ``node.report`` + ``node.citations`` — we set ``report`` to the
        # summary text (and keep ``summary`` as an alias) so the citation loop is
        # closed. ``grounded_edge_count`` rides on the node as the report's
        # grounding-strength provenance.
        return (
            "MERGE (n:CommunityReport {community_id: $community_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    n.report = $report, n.summary = $report, n.citations = $citations, "
            "    n.grounded_edge_count = $grounded_edge_count, n.embedding = $embedding "
            "  ON MATCH SET "
            "    n.report = $report, n.summary = $report, n.citations = $citations, "
            "    n.grounded_edge_count = $grounded_edge_count, n.embedding = $embedding"
        )

    @staticmethod
    def _section_card_cypher() -> str:
        # (:SectionCard) vector node, business key = (card_id, tenant_id). Carries
        # the embedded summary+tags text and an embedding indexed by
        # ``section_card_vector_index`` (the searcher's card leg). Provenance
        # (section_id/doc_id/tag_labels/src_chunk_ids) rides on the node so a card
        # hit stays traceable to its grounding chunks.
        return (
            "MERGE (n:SectionCard {card_id: $card_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    n.section_id = $section_id, n.doc_id = $doc_id, n.text = $text, "
            "    n.embedding = $embedding, n.tag_labels = $tag_labels, "
            "    n.src_chunk_ids = $src_chunk_ids "
            "  ON MATCH SET "
            "    n.section_id = $section_id, n.doc_id = $doc_id, n.text = $text, "
            "    n.embedding = $embedding, n.tag_labels = $tag_labels, "
            "    n.src_chunk_ids = $src_chunk_ids"
        )

    @staticmethod
    def _doc_card_cypher() -> str:
        # (:DocCard) vector node, business key = (card_id, tenant_id). Indexed by
        # ``doc_card_vector_index`` (the searcher's doc-card leg).
        return (
            "MERGE (n:DocCard {card_id: $card_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    n.doc_id = $doc_id, n.text = $text, n.embedding = $embedding, "
            "    n.tag_labels = $tag_labels, n.src_chunk_ids = $src_chunk_ids "
            "  ON MATCH SET "
            "    n.doc_id = $doc_id, n.text = $text, n.embedding = $embedding, "
            "    n.tag_labels = $tag_labels, n.src_chunk_ids = $src_chunk_ids"
        )

    # ---- public surface ----------------------------------------------------
    # All write_* methods are idempotent (MERGE-only) and short-circuit on empty
    # input before touching the driver.

    def write_cards(self, cards: list, *, tenant_id: str) -> int:
        """MERGE each multi-representation card node (SectionCard / DocCard).

        ``cards`` is any iterable of ``SectionCard``/``DocCard``-shaped objects.
        A card carrying a ``section_id`` is written as a ``:SectionCard`` (indexed
        by ``section_card_vector_index``); otherwise as a ``:DocCard`` (indexed by
        ``doc_card_vector_index``). Both MERGE on (card_id, tenant_id) and bind
        ``$tenant_id``. Returns the write count.
        """
        if not cards:
            return 0
        driver = self._require_driver()
        section_cypher = self._section_card_cypher()
        doc_cypher = self._doc_card_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for card in cards:
                section_id = getattr(card, "section_id", None)
                tag_labels = list(getattr(card, "tag_labels", ()) or ())
                src_chunk_ids = list(getattr(card, "src_chunk_ids", ()) or ())
                if section_id is not None:
                    session.run(
                        section_cypher,
                        card_id=card.card_id,
                        section_id=section_id,
                        doc_id=card.doc_id,
                        tenant_id=tenant_id,
                        text=card.text,
                        embedding=getattr(card, "embedding", None),
                        tag_labels=tag_labels,
                        src_chunk_ids=src_chunk_ids,
                    )
                else:
                    session.run(
                        doc_cypher,
                        card_id=card.card_id,
                        doc_id=card.doc_id,
                        tenant_id=tenant_id,
                        text=card.text,
                        embedding=getattr(card, "embedding", None),
                        tag_labels=tag_labels,
                        src_chunk_ids=src_chunk_ids,
                    )
                written += 1
        return written

    def write_entities(self, entities: list, *, tenant_id: str) -> int:
        """MERGE each resolved entity on (name, tenant_id).

        ``entities`` is any iterable of objects exposing ``name``/``etype`` and
        (optionally) ``normalized_value``/``aliases`` (duck-typed ``ResolvedEntity``).
        Returns the number of entity writes issued.
        """
        if not entities:
            return 0
        driver = self._require_driver()
        cypher = self._entity_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for ent in entities:
                session.run(
                    cypher,
                    name=ent.name,
                    tenant_id=tenant_id,
                    etype=getattr(ent, "etype", "") or "",
                    normalized_value=getattr(ent, "normalized_value", None),
                    aliases=list(getattr(ent, "aliases", []) or []),
                )
                written += 1
        return written

    def write_related_to(self, edges: list, *, tenant_id: str) -> int:
        """MERGE each grounded edge as ``(:Entity)-[:RELATED_TO {...}]->(:Entity)``.

        ``edges`` is any iterable of ``GroundedEdge``-shaped objects (subject /
        predicate / obj / confidence / span / src_chunk_id / evidence_count). The
        edge carries the full grounding provenance. Returns the write count.
        """
        if not edges:
            return 0
        driver = self._require_driver()
        cypher = self._related_to_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for edge in edges:
                session.run(
                    cypher,
                    subject=edge.subject,
                    obj=edge.obj,
                    predicate=edge.predicate,
                    tenant_id=tenant_id,
                    desc=getattr(edge, "desc", None) or edge.predicate,
                    weight=getattr(edge, "weight", None)
                    if getattr(edge, "weight", None) is not None
                    else edge.confidence,
                    confidence=edge.confidence,
                    evidence_count=edge.evidence_count,
                    src_chunk=edge.src_chunk_id,
                    span=edge.span,
                )
                written += 1
        return written

    def write_mentions(self, mentions: list, *, tenant_id: str) -> int:
        """MERGE each ``(:Chunk)-[:MENTIONS]->(:Entity)`` edge.

        ``mentions`` is any iterable of objects exposing ``chunk_id`` + ``name``
        (and optional ``confidence``/``span``). Returns the write count.
        """
        if not mentions:
            return 0
        driver = self._require_driver()
        cypher = self._mentions_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for m in mentions:
                session.run(
                    cypher,
                    chunk_id=m.chunk_id,
                    name=m.name,
                    tenant_id=tenant_id,
                    confidence=getattr(m, "confidence", None),
                    span=getattr(m, "span", None),
                )
                written += 1
        return written

    def write_sections(self, sections: list, *, tenant_id: str) -> int:
        """MERGE each Section and its ``HAS_CHUNK`` edges to member chunks.

        ``sections`` is any iterable of ``Section``-shaped objects (section_id /
        doc_id / chunk_ids / fingerprint / page_span). One statement per member
        chunk. Returns the number of HAS_CHUNK writes issued.
        """
        if not sections:
            return 0
        driver = self._require_driver()
        cypher = self._section_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for sec in sections:
                page_span = getattr(sec, "page_span", (0, 0)) or (0, 0)
                for chunk_id in sec.chunk_ids:
                    session.run(
                        cypher,
                        section_id=sec.section_id,
                        doc_id=sec.doc_id,
                        tenant_id=tenant_id,
                        fingerprint=getattr(sec, "fingerprint", None),
                        page_start=page_span[0],
                        page_end=page_span[1],
                        chunk_id=chunk_id,
                    )
                    written += 1
        return written

    def write_tags(self, tags: list, *, tenant_id: str) -> int:
        """MERGE each grounded tag node + its ``HAS_TAG`` edge to its owner.

        ``tags`` is any iterable of ``GroundedTag``-shaped objects (label / scope /
        confidence / span / src_chunk_id) plus an ``owner_key`` attribute that
        identifies the Entity/Section/Document the tag describes. When ``owner_key``
        is absent we fall back to the tag's ``src_chunk_id`` so the tag is always
        anchored to a tenant-scoped node. Returns the write count.
        """
        if not tags:
            return 0
        driver = self._require_driver()
        cypher = self._tag_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for tag in tags:
                owner_key = getattr(tag, "owner_key", None) or tag.src_chunk_id
                session.run(
                    cypher,
                    label=tag.label,
                    scope=tag.scope,
                    tenant_id=tenant_id,
                    owner_key=owner_key,
                    confidence=tag.confidence,
                    span=tag.span,
                    src_chunk=tag.src_chunk_id,
                )
                written += 1
        return written

    def write_communities(self, assignments: list, *, tenant_id: str) -> int:
        """MERGE each ``(:Entity)-[:IN_COMMUNITY]->(:Community)`` assignment.

        ``assignments`` is any iterable of objects exposing ``community_id`` +
        ``name`` (and optional ``size``/``pagerank``). Returns the write count.
        """
        if not assignments:
            return 0
        driver = self._require_driver()
        cypher = self._community_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for a in assignments:
                session.run(
                    cypher,
                    community_id=a.community_id,
                    name=a.name,
                    tenant_id=tenant_id,
                    size=getattr(a, "size", None),
                    pagerank=getattr(a, "pagerank", None),
                )
                written += 1
        return written

    def write_community_reports(self, reports: list, *, tenant_id: str) -> int:
        """MERGE each cited :class:`CommunityReport` as a ``(:CommunityReport)`` node.

        ``reports`` is any iterable of ``CommunityReport``-shaped objects exposing
        ``community_id`` + ``summary`` + ``citations`` (and an optional
        ``embedding`` of the summary + ``grounded_edge_count``). Each node MERGEs on
        ``(community_id, tenant_id)`` and binds ``$tenant_id`` so a re-run
        OVERWRITES rather than duplicating and a report can never cross tenants.

        The ``report`` property is set to the summary text — the SAME property the
        searcher's ``community_report_lookup`` reads — so the citation loop the
        Phase-2 reporter starts is actually closed (the searcher's
        ``community_report_vector_index`` is populated by this node's
        ``embedding``). Returns the write count.

        ``None`` entries (suppressed/ungrounded reports) are skipped: an ungrounded
        report is never persisted, embedded, or made retrievable.
        """
        if not reports:
            return 0
        driver = self._require_driver()
        cypher = self._community_report_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for report in reports:
                if report is None:
                    continue
                citations = list(getattr(report, "citations", ()) or ())
                session.run(
                    cypher,
                    community_id=report.community_id,
                    tenant_id=tenant_id,
                    report=getattr(report, "summary", "") or "",
                    citations=citations,
                    grounded_edge_count=getattr(report, "grounded_edge_count", None),
                    embedding=getattr(report, "embedding", None),
                )
                written += 1
        return written

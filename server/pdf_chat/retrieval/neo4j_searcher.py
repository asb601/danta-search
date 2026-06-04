"""Stage 3 — Neo4j Hybrid Retrieval (vector ANN + grounded graph traversal).

Phase-2 (contract C2) rewrite. The searcher now traverses the *semantic* graph
the Phase-2 writer persists (``(:Entity)-[:RELATED_TO]->(:Entity)``,
``(:Chunk)-[:MENTIONS]->(:Entity)``) and exposes the multi-representation
retrieval surface the agentic runtime consumes:

* :meth:`Neo4jSearcher.vector_search` — ANN over the chunk HNSW vector index.
* :meth:`Neo4jSearcher.graph_traversal` — entity-relationship walk to the chunks
  that mention the entities reachable from an anchor (relational queries).
* :meth:`Neo4jSearcher.entity_neighbors` — the related-entity neighbourhood of an
  anchor entity (wraps to the Phase-3 ``get_entity_neighbors`` tool, contract C2).
* :meth:`Neo4jSearcher.community_report_lookup` — ANN over the (cited) community
  report vector space.
* :meth:`Neo4jSearcher.multi_vector_search` — ANN over chunk + section-card +
  doc-card vector spaces, RRF-fused via :func:`retrieval.rrf.rrf`.
* :meth:`Neo4jSearcher.hybrid_search` — vector + graph legs fused via RRF.

PER-HOP TENANT ISOLATION (spec §3 inv 3 — non-negotiable). Every traversal that
walks a variable-length path filters ``tenant_id`` on ALL path nodes via
``ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)`` — never just the anchor —
so a walk can NEVER straddle tenants through a shared entity name. Every vector
leg filters ``node.tenant_id = $tenant_id``. ``tenant_id`` is bound as a
parameter on every statement.

The ``neo4j`` driver import is GUARDED (Hard rule #6). The class CONSTRUCTS fine
with no infra; the search methods raise a clear :class:`RuntimeError` only when
actually CALLED without the driver installed. No score-comparison literal lives
in this module — fan-outs resolve through config (Spec §3 inv 4).
"""
from __future__ import annotations

import json
from typing import Any

from pdf_chat.config import get_pdf_settings
from pdf_chat.retrieval.rrf import rrf

try:
    from neo4j import GraphDatabase  # type: ignore

    _HAS_NEO4J = True
except ImportError:  # pragma: no cover - exercised only without infra
    GraphDatabase = None  # type: ignore
    _HAS_NEO4J = False


_VECTOR_INDEX = "chunk_vector_index"
# Multi-representation index names (Phase-2 schema item 7). The card spaces are
# written by the CardBuilder/KG writer; the community-report space holds cited
# Leiden reports. Passed as ``$index_name`` so a single cypher serves every leg.
_SECTION_CARD_INDEX = "section_card_vector_index"
_DOC_CARD_INDEX = "doc_card_vector_index"
_COMMUNITY_REPORT_INDEX = "community_report_vector_index"

# tenant_id is bound as a parameter and asserted in WHERE on every query.
# The optional `$doc_ids IS NULL OR node.doc_id IN $doc_ids` clause scopes
# retrieval to a caller-supplied document subset without forking the query.
_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
WHERE node.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR node.doc_id IN $doc_ids)
RETURN node.chunk_id AS chunk_id, node.text AS text,
       node.doc_id AS doc_id, node.page_num AS page_num,
       node.element_type AS element_type, node.acl AS acl, score
ORDER BY score DESC
"""

# ANN over a CARD vector space (section-card / doc-card). The card node carries
# its own tenant_id (every node does) so the leg is per-card tenant-isolated. The
# card's source chunk/section id is returned as ``chunk_id`` so the result merges
# into the same id-space the chunk leg uses for RRF fusion.
_CARD_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
WHERE node.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR node.doc_id IN $doc_ids)
RETURN coalesce(node.chunk_id, node.section_id, node.doc_id) AS chunk_id,
       node.text AS text, node.doc_id AS doc_id, node.page_num AS page_num,
       node.element_type AS element_type, node.acl AS acl, score
ORDER BY score DESC
"""

# ANN over the CITED community-report vector space. Reports that could not be
# traced to ≥N grounded edges are never written, so this only ever returns cited
# reports (the suppression happens upstream in the CommunityReporter). The node
# carries tenant_id like every other node.
_COMMUNITY_REPORT_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
WHERE node.tenant_id = $tenant_id
RETURN node.community_id AS community_id, node.report AS report,
       node.citations AS citations, score
ORDER BY score DESC
LIMIT $limit
"""

# PER-HOP TENANT ISOLATION: the Entity anchor AND every node on the
# variable-length RELATED_TO path are filtered on tenant_id via
# ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id), so the walk can never
# straddle tenants through a shared entity name (spec §3 inv 3). The terminal
# (:Chunk)-[:MENTIONS]->(other) hop is tenant-filtered on the chunk too.
_GRAPH_CYPHER = """
MATCH path = (e:Entity {name: $entity})-[:RELATED_TO*1..2]-(other:Entity)
WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)
MATCH (c:Chunk)-[:MENTIONS]->(other)
WHERE c.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)
RETURN c.chunk_id AS chunk_id, c.text AS text, c.doc_id AS doc_id,
       c.page_num AS page_num, c.element_type AS element_type, c.acl AS acl
LIMIT $limit
"""

# PER-HOP TENANT ISOLATION: the related-entity neighbourhood of an anchor. Every
# node on the 1..2-hop RELATED_TO path is tenant-filtered via the path-level
# ALL(...) predicate (never anchor-only). Returns the neighbour entities with the
# connecting edge's grounding props so the caller can show provenance.
#
# DOC FILTER (fixed): the previous form filtered ``rel.src_chunk IN $doc_ids``,
# comparing a CHUNK id against DOCUMENT ids — so the doc subset silently matched
# nothing. A neighbour is in-scope only when it is MENTIONED by a tenant-scoped
# chunk whose ``doc_id`` is in the requested subset (chunks carry ``doc_id`` from
# the Phase-1 writer). The mention hop is tenant-filtered too, so the doc scope
# never straddles tenants.
_NEIGHBORS_CYPHER = """
MATCH path = (e:Entity {name: $entity})-[r:RELATED_TO*1..2]-(other:Entity)
WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)
  AND ($doc_ids IS NULL OR EXISTS {
        MATCH (dc:Chunk)-[:MENTIONS]->(other)
        WHERE dc.tenant_id = $tenant_id AND dc.doc_id IN $doc_ids
      })
RETURN DISTINCT other.name AS name, other.etype AS etype,
       other.normalized_value AS normalized_value
LIMIT $limit
"""


def deserialize_acl(chunk: dict) -> dict:
    """Return ``chunk`` with its ``acl`` field guaranteed to be a dict.

    The ingestion writer persists ``acl`` as a JSON *string*
    (``Chunk.to_neo4j_props`` → ``json.dumps``). Neo4j therefore returns it as a
    string. ``filter_by_acl`` expects a dict, so we deserialize here at the
    retrieval boundary. Pure + side-effect free on the input mapping (returns a
    shallow copy) so it is trivially unit-testable.

    Robust to: already-a-dict, missing key, ``None``, and malformed JSON (which
    fails CLOSED to an empty ACL → the chunk will be denied by ACL filtering).
    """
    out = dict(chunk)
    acl = out.get("acl")
    if isinstance(acl, str):
        try:
            parsed = json.loads(acl)
        except (ValueError, TypeError):
            parsed = {}
        out["acl"] = parsed if isinstance(parsed, dict) else {}
    elif acl is None:
        out["acl"] = {}
    return out


class Neo4jSearcher:
    """Hybrid searcher over the Neo4j graph + vector store.

    Constructing this object never opens a connection — the driver is created
    lazily on first use so the class loads with zero infra. The methods raise a
    clear error if the ``neo4j`` driver is missing.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        settings = get_pdf_settings()
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._database = database or settings.neo4j_database
        self._driver: Any = None

    def _require_driver(self) -> Any:
        # An already-connected/injected driver wins (so a caller — or a test
        # mocking the session — can supply its own) before we require the real
        # neo4j package. This mirrors the writer's seam (kg_writer.py).
        if self._driver is not None:
            return self._driver
        if not _HAS_NEO4J:
            raise RuntimeError(
                "neo4j driver is not installed. Install `neo4j` to run "
                "Neo4jSearcher search methods. Pure retrieval logic (rrf, acl, "
                "cache_key, context assembly) runs without it."
            )
        self._driver = GraphDatabase.driver(  # type: ignore[union-attr]
            self._uri, auth=(self._user, self._password)
        )
        return self._driver

    def vector_search(
        self,
        query_vec: list[float],
        tenant_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """ANN search over the chunk vector index, tenant-scoped.

        Args:
            query_vec: query embedding (same model/dim as ingestion).
            tenant_id: enforced in the Cypher WHERE clause.
            top_k: number of candidates (defaults to ``vector_top_k`` config).
            doc_ids: optional document subset; when given, only chunks whose
                ``doc_id`` is in this list are returned.

        Returns:
            A list of chunk dicts (``chunk_id``, ``text``, ``doc_id``,
            ``page_num``, ``element_type``, ``acl``, ``score``) ranked by
            similarity desc. ``acl`` is deserialized to a dict.

        Raises:
            RuntimeError: if the ``neo4j`` driver is not installed.
        """
        if top_k is None:
            top_k = get_pdf_settings().vector_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _VECTOR_CYPHER,
                index_name=_VECTOR_INDEX,
                top_k=top_k,
                query_vector=query_vec,
                tenant_id=tenant_id,
                doc_ids=doc_ids,
            )
            return [deserialize_acl(dict(record)) for record in result]

    def graph_traversal(
        self,
        entity: str,
        tenant_id: str,
        limit: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """1–2 hop entity-relationship walk to related chunks, tenant-scoped.

        Args:
            entity: entity name to anchor the traversal.
            tenant_id: enforced in the Cypher WHERE clause (anchor + chunks).
            limit: max related chunks (defaults to ``graph_top_k`` config).
            doc_ids: optional document subset filter.

        Returns:
            A list of chunk dicts (``chunk_id``, ``text``, ``doc_id``,
            ``page_num``, ``element_type``, ``acl``). ``acl`` is deserialized.

        Raises:
            RuntimeError: if the ``neo4j`` driver is not installed.
        """
        if limit is None:
            limit = get_pdf_settings().graph_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _GRAPH_CYPHER,
                entity=entity,
                tenant_id=tenant_id,
                limit=limit,
                doc_ids=doc_ids,
            )
            return [deserialize_acl(dict(record)) for record in result]

    def entity_neighbors(
        self,
        entity: str,
        tenant_id: str,
        limit: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Related-entity neighbourhood of ``entity`` (1–2 hops), tenant-scoped.

        Contract C2: the Phase-3 ``get_entity_neighbors`` tool wraps this. The
        walk is PER-HOP tenant-isolated — every node on the variable-length
        ``RELATED_TO`` path is filtered on ``tenant_id`` (not just the anchor),
        so the neighbourhood can never include another tenant's entity.

        Args:
            entity: anchor entity name.
            tenant_id: enforced on every node of the matched path.
            limit: max neighbours (defaults to ``graph_top_k`` config).
            doc_ids: optional document subset — keep only neighbours MENTIONED by a
                tenant-scoped chunk whose ``doc_id`` is in this subset (filters on
                the chunk's document, not on a chunk id).

        Returns:
            A list of neighbour dicts (``name``, ``etype``, ``normalized_value``).

        Raises:
            RuntimeError: if the ``neo4j`` driver is not installed.
        """
        if limit is None:
            limit = get_pdf_settings().graph_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _NEIGHBORS_CYPHER,
                entity=entity,
                tenant_id=tenant_id,
                limit=limit,
                doc_ids=doc_ids,
            )
            return [dict(record) for record in result]

    def community_report_lookup(
        self,
        query_vec: list[float],
        tenant_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """ANN over the CITED community-report vector space, tenant-scoped.

        Only cited reports are ever written (the CommunityReporter suppresses any
        report not traceable to enough grounded edges), so this returns grounded
        community summaries with their ``citations`` (chunk ids). The report node
        carries ``tenant_id`` so the leg is tenant-isolated.

        Args:
            query_vec: query embedding (same model as ingestion).
            tenant_id: enforced in the Cypher WHERE clause.
            limit: number of reports (defaults to ``graph_top_k`` config).

        Returns:
            A list of report dicts (``community_id``, ``report``, ``citations``,
            ``score``) ranked by similarity desc.

        Raises:
            RuntimeError: if the ``neo4j`` driver is not installed.
        """
        if limit is None:
            limit = get_pdf_settings().graph_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _COMMUNITY_REPORT_CYPHER,
                index_name=_COMMUNITY_REPORT_INDEX,
                top_k=limit,
                query_vector=query_vec,
                tenant_id=tenant_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    def _card_vector_search(
        self,
        query_vec: list[float],
        tenant_id: str,
        index_name: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """ANN over a single CARD vector space (section-card / doc-card).

        Tenant-scoped (``node.tenant_id = $tenant_id``) and doc-scoped. The
        card's source chunk/section/doc id is projected as ``chunk_id`` so card
        hits share the chunk leg's id-space for RRF fusion.
        """
        if top_k is None:
            top_k = get_pdf_settings().vector_top_k
        driver = self._require_driver()
        with driver.session(database=self._database) as session:
            result = session.run(
                _CARD_VECTOR_CYPHER,
                index_name=index_name,
                top_k=top_k,
                query_vector=query_vec,
                tenant_id=tenant_id,
                doc_ids=doc_ids,
            )
            return [deserialize_acl(dict(record)) for record in result]

    def multi_vector_search(
        self,
        query_vec: list[float],
        tenant_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Multi-representation ANN: chunk + section-card + doc-card, RRF-fused.

        Runs THREE tenant-isolated, doc-scoped vector legs over the chunk index,
        the section-card index, and the doc-card index, then fuses their ranked
        id-lists with :func:`retrieval.rrf.rrf` (Phase-2 schema item 7). Querying
        all three representations lets a query match a precise chunk, a
        summarised section, or a whole-document card — and RRF reconciles them
        without one space dominating.

        Args:
            query_vec: query embedding (same model/dim as ingestion).
            tenant_id: enforced on every leg's Cypher WHERE clause.
            top_k: per-leg fan-out (defaults to ``vector_top_k`` config).
            doc_ids: optional document subset — threaded to every leg.

        Returns:
            Fused list of hit dicts in RRF order. Each dict has a dict ``acl``.

        Raises:
            RuntimeError: if the ``neo4j`` driver is not installed.
        """
        chunk_hits = self.vector_search(query_vec, tenant_id, top_k, doc_ids)
        section_hits = self._card_vector_search(
            query_vec, tenant_id, _SECTION_CARD_INDEX, top_k, doc_ids
        )
        doc_hits = self._card_vector_search(
            query_vec, tenant_id, _DOC_CARD_INDEX, top_k, doc_ids
        )

        # Index every hit by id (chunk leg wins on a duplicate — it carries the
        # finest-grained provenance). Then fuse the three ranked id-lists.
        by_id: dict[str, dict] = {}
        for hit in doc_hits:
            cid = str(hit.get("chunk_id", ""))
            if cid:
                by_id[cid] = hit
        for hit in section_hits:
            cid = str(hit.get("chunk_id", ""))
            if cid:
                by_id[cid] = hit
        for hit in chunk_hits:
            cid = str(hit.get("chunk_id", ""))
            if cid:
                by_id[cid] = hit

        def _ids(hits: list[dict]) -> list[str]:
            return [str(h.get("chunk_id", "")) for h in hits if h.get("chunk_id")]

        fused_ids = rrf(
            [_ids(chunk_hits), _ids(section_hits), _ids(doc_hits)],
            k=get_pdf_settings().rrf_k,
        )
        return [by_id[cid] for cid in fused_ids if cid in by_id]

    def hybrid_search(
        self,
        query_vector: list[float],
        tenant_id: str,
        doc_ids: list[str] | None = None,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        entity: str | None = None,
    ) -> list[dict]:
        """Stage 3 — run vector ANN + (optional) graph traversal, fuse via RRF.

        Both legs are tenant-scoped and doc_id-scoped (when ``doc_ids`` is
        given). Their ranked id-lists are fused with :func:`rrf` and the chunk
        dicts are returned in fused order. Each chunk dict has a dict ``acl``
        (already deserialized by the leg methods).

        Args:
            query_vector: query embedding for the ANN leg.
            tenant_id: tenant isolation key (enforced in every Cypher WHERE).
            doc_ids: optional document subset filter for both legs.
            vector_top_k: ANN fan-out (defaults to ``vector_top_k`` config).
            graph_top_k: graph fan-out (defaults to ``graph_top_k`` config).
            entity: anchor entity for the graph leg. When ``None`` the graph leg
                is skipped and the result is the vector ranking alone.

        Returns:
            Fused list of chunk dicts in RRF order.
        """
        vector_hits = self.vector_search(
            query_vector, tenant_id, top_k=vector_top_k, doc_ids=doc_ids
        )
        graph_hits: list[dict] = []
        if entity:
            graph_hits = self.graph_traversal(
                entity, tenant_id, limit=graph_top_k, doc_ids=doc_ids
            )

        # Index every chunk by id (vector leg wins on duplicate id — it carries
        # the similarity score). Then fuse the two ranked id-lists via RRF.
        by_id: dict[str, dict] = {}
        for chunk in graph_hits:
            cid = str(chunk.get("chunk_id", ""))
            if cid:
                by_id[cid] = chunk
        for chunk in vector_hits:
            cid = str(chunk.get("chunk_id", ""))
            if cid:
                by_id[cid] = chunk

        vector_ids = [str(c.get("chunk_id", "")) for c in vector_hits if c.get("chunk_id")]
        graph_ids = [str(c.get("chunk_id", "")) for c in graph_hits if c.get("chunk_id")]
        fused_ids = rrf([vector_ids, graph_ids], k=get_pdf_settings().rrf_k)
        return [by_id[cid] for cid in fused_ids if cid in by_id]

    def close(self) -> None:
        """Close the underlying driver if one was opened."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

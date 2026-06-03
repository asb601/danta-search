"""Stage 3 — Neo4j Hybrid Retrieval (vector ANN + graph traversal).

Two searches that the pipeline runs and then fuses with :func:`rrf`:

* :meth:`Neo4jSearcher.vector_search` — ANN over the chunk HNSW vector index.
* :meth:`Neo4jSearcher.graph_traversal` — entity-relationship walk for
  relational queries.

Every Cypher statement carries ``tenant_id`` in its WHERE clause (Hard rule #3 —
multi-tenant isolation is non-negotiable).

The ``neo4j`` driver import is GUARDED (Hard rule #6). The class CONSTRUCTS fine
with no infra; the search methods raise a clear :class:`RuntimeError` only when
actually CALLED without the driver installed.
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

# The Entity anchor is tenant-scoped too (e.tenant_id) so a graph walk can never
# straddle tenants via a shared entity name (Hard rule #3 — tenant isolation).
_GRAPH_CYPHER = """
MATCH (e:Entity {name: $entity})-[:RELATED_TO*1..2]-(c:Chunk)
WHERE e.tenant_id = $tenant_id AND c.tenant_id = $tenant_id
  AND ($doc_ids IS NULL OR c.doc_id IN $doc_ids)
RETURN c.chunk_id AS chunk_id, c.text AS text,
       c.doc_id AS doc_id, c.page_num AS page_num,
       c.element_type AS element_type, c.acl AS acl
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
        if not _HAS_NEO4J:
            raise RuntimeError(
                "neo4j driver is not installed. Install `neo4j` to run "
                "Neo4jSearcher.vector_search / graph_traversal. Pure retrieval "
                "logic (rrf, acl, cache_key, context assembly) runs without it."
            )
        if self._driver is None:
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

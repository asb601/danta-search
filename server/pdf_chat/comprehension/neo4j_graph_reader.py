"""Phase 5 — the GraphReader→Neo4jSearcher adapter (seals the C2 read seam).

``reader.py`` defines the ``GraphReader`` Protocol (six tenant-scoped async
iterators) that the comprehension layer consumes — the ontology builder, the
temporal-coverage computation, and the onboarding topic-map projection. In
PRODUCTION that interface is satisfied by ``Neo4jSearcher``; but ``Neo4jSearcher``
implements the multi-representation RETRIEVAL surface (vector/graph/hybrid), NOT
the six comprehension iterators — so a raw ``Neo4jSearcher`` handed to the
comprehension layer would ``AttributeError`` on ``iter_entities`` etc.

``Neo4jGraphReader`` is the thin adapter that closes that gap. It wraps an
injected ``Neo4jSearcher`` and implements every ``GraphReader`` method as a
tenant-scoped async iterator over Cypher run against the searcher's driver:

  * PER-HOP TENANT ISOLATION (spec §3 inv 3 / contract C2): EVERY Cypher filters
    ``WHERE n.tenant_id = $tenant_id`` (or the node's ``tenant_id``) — never an
    unscoped scan — so a read can never straddle tenants through a shared name.
    ``tenant_id`` is bound as a parameter on every statement.
  * ASYNC-SAFE: the searcher runs SYNC Cypher
    (``with driver.session(database=...) as s: s.run(...)``). Each async iterator
    wraps that blocking call in ``await asyncio.to_thread(...)`` so it never
    blocks the event loop, materialises the rows, then ``yield``s the mapped dicts.
  * MAPPING: each row is projected onto the field names the comprehension
    consumers read (``_field``-compatible dicts). ``iter_communities`` rows carry
    a ``citations`` field (so ``reader.topic_map``'s grounded TOC is non-empty).

No score-comparison literal lives in this module — it is a pure read adapter, not
a gate (so no ``get_tunable`` / ``log_gate_decision`` call).

Wiring (deferred, documented only): ``api/onboarding.py::get_onboarding_graph_reader``
returns ``Neo4jGraphReader(Neo4jSearcher())``; the Phase-1 finalize call site
passes a ``Neo4jGraphReader`` as the ``reader`` argument. We do NOT edit that
state machine here.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher

# --------------------------------------------------------------------------- #
# Tenant-scoped Cypher (every statement filters $tenant_id — contract C2).
# Each query returns rows already projected onto the comprehension field names so
# the async iterator can yield them unchanged.
# --------------------------------------------------------------------------- #
_ENTITIES_CYPHER = """
MATCH (e:Entity)
WHERE e.tenant_id = $tenant_id
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
WHERE c.tenant_id = $tenant_id
WITH e, collect(DISTINCT c.chunk_id) AS evidence_chunk_ids
RETURN e.name AS name,
       e.normalized_value AS normalized_value,
       e.etype AS type,
       e.pagerank AS pagerank,
       e.mention_count AS mention_count,
       e.definition AS definition,
       evidence_chunk_ids AS evidence_chunk_ids
"""

_RELATIONSHIPS_CYPHER = """
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id
  AND r.tenant_id = $tenant_id
RETURN a.name AS src_name,
       b.name AS dst_name,
       coalesce(r.predicate, r.desc) AS relation,
       r.state AS state,
       r.confidence AS confidence,
       r.evidence AS evidence
"""

_COMMUNITIES_CYPHER = """
MATCH (n:CommunityReport)
WHERE n.tenant_id = $tenant_id
RETURN n.community_id AS community_id,
       coalesce(n.report, n.summary) AS report,
       n.level AS level,
       n.citations AS citations
"""

_DOCUMENTS_CYPHER = """
MATCH (d:Document)
WHERE d.tenant_id = $tenant_id
RETURN d.doc_id AS doc_id,
       d.title AS title,
       d.doc_date AS doc_date,
       d.created_at AS created_at
"""

_CHUNKS_CYPHER = """
MATCH (c:Chunk)
WHERE c.tenant_id = $tenant_id
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WHERE e.tenant_id = $tenant_id
WITH c, collect(DISTINCT e.name) AS entities
RETURN c.chunk_id AS chunk_id,
       c.doc_id AS doc_id,
       c.doc_date AS doc_date,
       c.text AS text,
       c.page_num AS page_num,
       c.bbox AS bbox,
       c.element_type AS element_type,
       entities AS entities
"""

_ENTITY_CHUNKS_CYPHER = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $entity_name})
WHERE c.tenant_id = $tenant_id AND e.tenant_id = $tenant_id
RETURN c.chunk_id AS chunk_id,
       c.doc_id AS doc_id,
       c.doc_date AS doc_date,
       c.text AS text,
       c.page_num AS page_num,
       c.bbox AS bbox,
       c.element_type AS element_type
"""


def _maybe_json(value: Any) -> Any:
    """Best-effort decode a JSON-string property (Neo4j stores some props as str).

    ``evidence``/``citations`` may round-trip as JSON strings; decode them so the
    consumer sees a dict/list. A non-string (or non-JSON string) passes through
    unchanged. Never raises.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


class Neo4jGraphReader:
    """``GraphReader`` adapter over an injected ``Neo4jSearcher`` (contract C2).

    Construct with ``Neo4jGraphReader(Neo4jSearcher())``. Every iterator is a
    tenant-scoped async generator running its Cypher off the event loop via
    ``asyncio.to_thread`` (the searcher's driver/session call is synchronous).
    """

    def __init__(self, searcher: Neo4jSearcher | None = None):
        self._searcher = searcher or Neo4jSearcher()

    def _run_sync(self, cypher: str, params: dict) -> list[dict]:
        """Run ``cypher`` SYNCHRONOUSLY on the searcher's driver, returning rows.

        Reuses the searcher's lazy ``_require_driver`` + ``_database`` (so a test
        can inject a fake driver) and the same ``with driver.session(...)`` seam
        the searcher's own methods use. Called only from inside ``to_thread``.
        """
        driver = self._searcher._require_driver()
        with driver.session(database=self._searcher._database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    async def _iter(self, cypher: str, params: dict) -> AsyncIterator[dict]:
        """Run tenant-scoped ``cypher`` off the loop, then yield each mapped row."""
        rows = await asyncio.to_thread(self._run_sync, cypher, params)
        for row in rows:
            yield row

    # ---- GraphReader Protocol — six tenant-scoped async iterators -----------
    async def iter_entities(self, tenant_id: str) -> AsyncIterator[dict]:
        async for row in self._iter(_ENTITIES_CYPHER, {"tenant_id": tenant_id}):
            yield row

    async def iter_relationships(self, tenant_id: str) -> AsyncIterator[dict]:
        async for row in self._iter(_RELATIONSHIPS_CYPHER, {"tenant_id": tenant_id}):
            row["evidence"] = _maybe_json(row.get("evidence"))
            yield row

    async def iter_communities(self, tenant_id: str) -> AsyncIterator[dict]:
        async for row in self._iter(_COMMUNITIES_CYPHER, {"tenant_id": tenant_id}):
            # Citations may round-trip as a JSON string; normalise to a list so
            # the grounded TOC (reader.topic_map) is non-empty.
            citations = _maybe_json(row.get("citations"))
            row["citations"] = list(citations) if citations else []
            yield row

    async def iter_documents(self, tenant_id: str) -> AsyncIterator[dict]:
        async for row in self._iter(_DOCUMENTS_CYPHER, {"tenant_id": tenant_id}):
            yield row

    async def iter_chunks(self, tenant_id: str) -> AsyncIterator[dict]:
        async for row in self._iter(_CHUNKS_CYPHER, {"tenant_id": tenant_id}):
            yield row

    async def entity_chunks(
        self, tenant_id: str, entity_name: str
    ) -> AsyncIterator[dict]:
        async for row in self._iter(
            _ENTITY_CHUNKS_CYPHER,
            {"tenant_id": tenant_id, "entity_name": entity_name},
        ):
            yield row


__all__ = ["Neo4jGraphReader"]

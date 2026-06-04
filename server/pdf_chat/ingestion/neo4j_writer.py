"""Stage 13 — Store in Neo4j.

Writes chunks as a graph + vector store:

    (:Document)-[:CONTAINS]->(:Page)-[:CONTAINS]->(:Chunk)

Every node carries ``tenant_id`` and ``acl`` (Hard rule #3: tenant isolation —
every query filters on ``tenant_id``; the chunk's acl is inherited from the
upload manifest). The chunk ``embedding`` property is indexed via an HNSW vector
index for ANN search at query time.

The ``neo4j`` driver is imported behind a guard. ``Neo4jWriter`` is always
constructible (so callers can wire it up without infra), but its write/index
methods raise a clear ``RuntimeError`` when the driver is missing. The Cypher
lives in small helper methods so it is reviewable and reusable.
"""
from __future__ import annotations

from .ton_schema import Chunk

try:
    from neo4j import GraphDatabase  # type: ignore

    _HAS_NEO4J = True
except ImportError:  # pragma: no cover - exercised only without infra
    GraphDatabase = None  # type: ignore
    _HAS_NEO4J = False


class Neo4jWriter:
    """Writes Document→Page→Chunk graphs into Neo4j with tenant isolation."""

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
        if not _HAS_NEO4J:
            raise RuntimeError(
                "The neo4j driver is required to write chunks but is not "
                "installed. Install it with `pip install neo4j`."
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

    def __enter__(self) -> "Neo4jWriter":
        self._require_driver()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Cypher (helpers — pure strings, reusable + reviewable) -----------
    @staticmethod
    def _vector_index_cypher(dim: int) -> str:
        return (
            "CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine' } }"
        )

    @staticmethod
    def _write_chunk_cypher() -> str:
        # MERGE the document + page (tenant scoped), then MERGE the chunk keyed on
        # (chunk_id, tenant_id) so re-extraction (retry/DLQ replay) OVERWRITES the
        # chunk rather than duplicating it (idempotent write). The mutable props
        # (text/embedding/confidence/low_confidence/page_num/element_type/
        # reading_order/acl) are set identically on both the ON CREATE and
        # ON MATCH branches so a replay always converges to the latest extraction.
        return (
            "MERGE (d:Document {doc_id: $doc_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET d.acl = $acl "
            "MERGE (p:Page {doc_id: $doc_id, page_num: $page_num, tenant_id: $tenant_id}) "
            "  ON CREATE SET p.acl = $acl "
            "MERGE (d)-[:CONTAINS]->(p) "
            "MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id}) "
            "  ON CREATE SET "
            "    c.doc_id = $doc_id, c.page_num = $page_num, "
            "    c.element_type = $element_type, c.text = $text, "
            "    c.reading_order = $reading_order, c.acl = $acl, "
            "    c.embedding = $embedding, c.confidence = $confidence, "
            "    c.low_confidence = $low_confidence "
            "  ON MATCH SET "
            "    c.doc_id = $doc_id, c.page_num = $page_num, "
            "    c.element_type = $element_type, c.text = $text, "
            "    c.reading_order = $reading_order, c.acl = $acl, "
            "    c.embedding = $embedding, c.confidence = $confidence, "
            "    c.low_confidence = $low_confidence "
            "MERGE (p)-[:CONTAINS]->(c)"
        )

    # ---- public surface ---------------------------------------------------
    def ensure_vector_index(self, dim: int) -> None:
        """Create the HNSW chunk vector index (idempotent) for ``dim`` dims."""
        driver = self._require_driver()
        with driver.session(database=self.database) as session:
            session.run(self._vector_index_cypher(dim), dim=dim)

    def write_chunks(self, chunks: list[Chunk]) -> int:
        """Write chunks and their Document→Page→Chunk relationships.

        Returns the number of chunks written. Each chunk's flattened props
        (via :meth:`Chunk.to_neo4j_props`) carry ``tenant_id`` + ``acl``.
        """
        if not chunks:
            return 0
        driver = self._require_driver()
        cypher = self._write_chunk_cypher()
        written = 0
        with driver.session(database=self.database) as session:
            for chunk in chunks:
                session.run(cypher, **chunk.to_neo4j_props())
                written += 1
        return written

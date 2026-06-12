"""Neo4j vector-index schema — the single source of truth for index names.

The retrieval searcher queries four HNSW vector indexes by name
(``db.index.vector.queryNodes($index_name, …)``) but nothing created them at
runtime — they were only ever created in a test. This module is the ONE place
that declares each index (its name, the node label it covers, and the embedding
property) and creates them idempotently. Both the writer (creation) and the
searcher (querying) import the names from here, so the index identity is defined
once and can never drift between the write side and the read side.

Creation is idempotent (``CREATE VECTOR INDEX … IF NOT EXISTS``) and dimension-
agnostic: the embedding dimension is bound as the ``$dim`` parameter (sourced from
``PdfSettings.embedding_dim``), never hardcoded. The similarity function is cosine
to match the embedding model contract (text-embedding-3-small, 1536-dim).

The ``neo4j`` driver is imported behind a guard so this module is import-safe with
zero infra; ``ensure_vector_indexes`` is only reached with a live driver.
"""
from __future__ import annotations

from typing import Any

# ── Index names (THE single source of truth — imported by writer + searcher) ───
CHUNK_VECTOR_INDEX = "chunk_vector_index"
SECTION_CARD_VECTOR_INDEX = "section_card_vector_index"
DOC_CARD_VECTOR_INDEX = "doc_card_vector_index"
COMMUNITY_REPORT_VECTOR_INDEX = "community_report_vector_index"

# (index_name, node_label, embedding_property). One row per vector space the
# searcher reads. Adding a new vector-indexed node type is a one-line registry
# change here — both creation and (by importing the name) querying follow.
VECTOR_INDEXES: tuple[tuple[str, str, str], ...] = (
    (CHUNK_VECTOR_INDEX, "Chunk", "embedding"),               # Phase-1 chunk ANN
    (SECTION_CARD_VECTOR_INDEX, "SectionCard", "embedding"),  # Phase-2 section card
    (DOC_CARD_VECTOR_INDEX, "DocCard", "embedding"),          # Phase-2 doc card
    (COMMUNITY_REPORT_VECTOR_INDEX, "CommunityReport", "embedding"),  # Phase-2 reports
)


def vector_index_cypher(index_name: str, label: str, prop: str) -> str:
    """Idempotent cosine-HNSW vector index DDL; dimension bound as ``$dim``."""
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{prop}) "
        "OPTIONS { indexConfig: { "
        "`vector.dimensions`: $dim, "
        "`vector.similarity_function`: 'cosine' } }"
    )


def ensure_vector_indexes(driver: Any, *, dim: int, database: str = "neo4j") -> list[str]:
    """Create every vector index in :data:`VECTOR_INDEXES` (idempotent).

    Returns the list of index names ensured. Safe to call repeatedly and on every
    worker start: ``IF NOT EXISTS`` makes re-creation a no-op, and an empty index
    (no nodes of that label yet) is valid — the searcher's leg just returns nothing
    until Phase-2 populates it. Requires a live ``neo4j`` driver (the caller owns
    its lifecycle).
    """
    ensured: list[str] = []
    with driver.session(database=database) as session:
        for index_name, label, prop in VECTOR_INDEXES:
            session.run(vector_index_cypher(index_name, label, prop), dim=dim)
            ensured.append(index_name)
    return ensured

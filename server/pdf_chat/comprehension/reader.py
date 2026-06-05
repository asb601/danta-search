"""Phase 5 — GraphReader Protocol + read-only comprehension query helpers.

This is the FOUNDATION every other Phase-5 worker imports:

* ``GraphReader`` — the abstract read interface over the Phase-2 Neo4j graph,
  implemented in production by ``Neo4jSearcher`` and by an in-memory fake in
  tests. We INJECT it (Protocol) so nothing here touches live Neo4j; per-hop
  tenant isolation is the searcher's responsibility (contract C2).
* Read-only persisted-artifact helpers the ``glossary_lookup`` tool and the
  onboarding API call: pure reads over the comprehension ORM (``AsyncSession``)
  and the injected ``GraphReader``.

Import-safe: zero infra at import (SQLAlchemy ``select``/``func`` are pure; the
ORM models register on the shared Base; the session/reader are arguments). No
score-comparison literal lives here — these are reads, not gates.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from sqlalchemy import func, select

from pdf_chat.models.comprehension import (
    DocTaxonomyClass,
    GlossaryEntry,
    OntologyEntity,
    TemporalCoverage,
    TenantOntology,
)


# --------------------------------------------------------------------------- #
# GraphReader Protocol (contract C2) — implemented by Neo4jSearcher / a fake.
# --------------------------------------------------------------------------- #
@runtime_checkable
class GraphReader(Protocol):
    """Read interface over the grounded Phase-2 graph (per-hop tenant-isolated).

    Every method is an async iterator scoped to ``tenant_id`` (the searcher's
    Cypher carries the tenant predicate). The ontology builder + temporal +
    topic-map projections consume this; tests inject an in-memory fake.
    """

    def iter_entities(self, tenant_id: str) -> AsyncIterator[Any]: ...
    def iter_relationships(self, tenant_id: str) -> AsyncIterator[Any]: ...
    def iter_communities(self, tenant_id: str) -> AsyncIterator[Any]: ...
    def iter_documents(self, tenant_id: str) -> AsyncIterator[Any]: ...
    def iter_chunks(self, tenant_id: str) -> AsyncIterator[Any]: ...
    def entity_chunks(self, tenant_id: str, entity_name: str) -> AsyncIterator[Any]: ...


# --------------------------------------------------------------------------- #
# Field-access helper (rows may be ORM objects, dicts, or namespaces).
# --------------------------------------------------------------------------- #
def _field(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict OR an attribute-bearing row, with a default."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


# --------------------------------------------------------------------------- #
# Persisted-artifact read helpers (pure reads over the comprehension ORM).
# --------------------------------------------------------------------------- #
async def current_ontology_version(session, tenant_id: str) -> int | None:
    """Return the highest persisted ontology ``version`` for ``tenant_id``.

    ``None`` when no artifact has been built yet (the onboarding API surfaces
    this as "not built"). Tenant-scoped (the WHERE clause carries ``tenant_id``).
    """
    stmt = (
        select(func.max(TenantOntology.version))
        .where(TenantOntology.tenant_id == tenant_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def lookup_glossary(session, tenant_id: str, term: str) -> GlossaryEntry | None:
    """Return the latest-version glossary row for ``term`` in ``tenant_id``, else None.

    Tenant-scoped, ordered by ``ontology_version`` descending so the most recent
    re-version wins (old versions are retained but the lookup reads the newest).
    The tool maps a hit → expansion/definition/provenance label + citation, and
    REFUSES on a miss (no fabrication, invariant 1/2).
    """
    stmt = (
        select(GlossaryEntry)
        .where(
            GlossaryEntry.tenant_id == tenant_id,
            GlossaryEntry.term == term,
        )
        .order_by(GlossaryEntry.ontology_version.desc())
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def list_glossary(
    session, tenant_id: str, *, ontology_version: int | None = None,
    limit: int | None = None, offset: int | None = None,
) -> list[GlossaryEntry]:
    """Glossary browse for ``tenant_id`` (optionally pinned to one version)."""
    stmt = select(GlossaryEntry).where(GlossaryEntry.tenant_id == tenant_id)
    if ontology_version is not None:
        stmt = stmt.where(GlossaryEntry.ontology_version == ontology_version)
    stmt = stmt.order_by(GlossaryEntry.term)
    if offset is not None:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_entities(
    session, ontology_id: str, *, limit: int | None = None, offset: int | None = None,
) -> list[OntologyEntity]:
    """Paged entity browse scoped to ONE ontology version (by ``ontology_id``)."""
    stmt = (
        select(OntologyEntity)
        .where(OntologyEntity.ontology_id == ontology_id)
        .order_by(OntologyEntity.pagerank.desc().nullslast(), OntologyEntity.name)
    )
    if offset is not None:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_doc_taxonomy(session, ontology_id: str) -> list[DocTaxonomyClass]:
    """Open-vocab doc-taxonomy classes for ONE ontology version."""
    stmt = (
        select(DocTaxonomyClass)
        .where(DocTaxonomyClass.ontology_id == ontology_id)
        .order_by(DocTaxonomyClass.confidence.desc().nullslast())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_temporal_coverage(session, ontology_id: str) -> list[TemporalCoverage]:
    """Per-subject temporal-coverage rows for ONE ontology version (pure read).

    The coverage (min/max/density/last_mention per subject) is computed + persisted
    by the ontology builder; this read surfaces it so the onboarding API can attach
    a human staleness note. Scoped by ``ontology_id`` (which is itself the latest
    tenant version the caller resolved). Ordered most-recently-mentioned first so a
    browse leads with the freshest subjects.
    """
    stmt = (
        select(TemporalCoverage)
        .where(TemporalCoverage.ontology_id == ontology_id)
        .order_by(TemporalCoverage.last_mention_date.desc().nullslast())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# --------------------------------------------------------------------------- #
# Topic map — the company's grounded table-of-contents from community reports.
# --------------------------------------------------------------------------- #
async def topic_map(reader: GraphReader, tenant_id: str) -> list[dict]:
    """Project the persisted (cited) community reports as a table-of-contents.

    Only cited community reports are ever written to the graph, so each topic
    carries its ``citations`` (chunk ids) — the onboarding topic map is grounded
    (invariant 1). Accepts dict OR object community rows. Tenant-scoped via the
    reader's ``iter_communities`` (per-hop isolation in the searcher Cypher).
    """
    topics: list[dict] = []
    async for community in reader.iter_communities(tenant_id):
        topics.append(
            {
                "community_id": _field(community, "community_id"),
                "report": _field(community, "report"),
                "level": _field(community, "level"),
                "citations": list(_field(community, "citations", []) or []),
            }
        )
    return topics


__all__ = [
    "GraphReader",
    "current_ontology_version",
    "lookup_glossary",
    "list_glossary",
    "list_entities",
    "list_doc_taxonomy",
    "list_temporal_coverage",
    "topic_map",
]

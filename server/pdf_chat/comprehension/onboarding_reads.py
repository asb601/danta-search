"""Phase 5 Task 10 — onboarding read facade + the one helper wave-1 didn't ship.

Wave 1's ``reader.py`` provides the per-entity read helpers (``lookup_glossary``,
``list_glossary``, ``list_entities``, ``list_doc_taxonomy``,
``current_ontology_version``, ``topic_map``). The onboarding API needs ONE more
read that wave 1 didn't provide — resolving the LATEST ontology's ``ontology_id``
for a tenant (``list_entities`` / ``list_doc_taxonomy`` are scoped by
``ontology_id``, not ``tenant_id``). It is added HERE (a NEW module) rather than
by editing ``reader.py`` so there is no write conflict with the wave-1 worker.

``OnboardingReads`` is a thin FACADE bundling the wave-1 module functions plus
``latest_ontology_id`` into one injectable object. The onboarding router depends
on this facade (``get_onboarding_reader``); tests inject an in-memory fake with
the same method surface. Pure module — zero infra at import (SQLAlchemy
``select``/``desc`` are pure; the session is an argument).

No score-comparison literal lives here — these are reads, not gates.
"""
from __future__ import annotations

from sqlalchemy import select

from pdf_chat.comprehension import reader as _reader
from pdf_chat.models.comprehension import TenantOntology


async def latest_ontology_id(session, tenant_id: str) -> str | None:
    """Return the ``ontology_id`` of the latest (highest-version) ontology.

    ``None`` when no artifact has been built for ``tenant_id`` yet (the onboarding
    API surfaces this as an empty browse, never a 500). Tenant-scoped — the WHERE
    clause carries ``tenant_id`` and we order by ``version`` descending so a
    re-version's newest row wins (old versions are retained but not browsed here).
    """
    stmt = (
        select(TenantOntology.ontology_id)
        .where(TenantOntology.tenant_id == tenant_id)
        .order_by(TenantOntology.version.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


class OnboardingReads:
    """Injectable read facade over the wave-1 helpers + ``latest_ontology_id``.

    Every method is tenant-scoped (or ontology-scoped after resolving the latest
    ontology). The facade exists so the router depends on a single object that a
    test can replace wholesale with an in-memory fake.
    """

    async def topic_map(self, reader, tenant_id: str) -> list[dict]:
        return await _reader.topic_map(reader, tenant_id)

    async def current_ontology_version(self, session, tenant_id: str) -> int | None:
        return await _reader.current_ontology_version(session, tenant_id)

    async def latest_ontology_id(self, session, tenant_id: str) -> str | None:
        return await latest_ontology_id(session, tenant_id)

    async def list_entities(self, session, ontology_id: str, *,
                            limit: int | None = None, offset: int | None = None):
        return await _reader.list_entities(session, ontology_id, limit=limit, offset=offset)

    async def list_glossary(self, session, tenant_id: str, *,
                            ontology_version: int | None = None,
                            limit: int | None = None, offset: int | None = None):
        return await _reader.list_glossary(
            session, tenant_id, ontology_version=ontology_version,
            limit=limit, offset=offset,
        )

    async def list_doc_taxonomy(self, session, ontology_id: str):
        return await _reader.list_doc_taxonomy(session, ontology_id)

    async def list_temporal_coverage(self, session, ontology_id: str):
        return await _reader.list_temporal_coverage(session, ontology_id)


__all__ = ["OnboardingReads", "latest_ontology_id"]

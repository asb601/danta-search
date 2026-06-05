"""Runtime migration — create the Phase-5 comprehension tables + indexes.

Idempotent, non-fatal, additive (same convention as ``bridge_upgrade.py`` and
``app/migrations/``). Importing the models registers them on the shared ``Base``;
this migration creates ONLY the comprehension tables (table-scoped
``__table__.create(checkfirst=True)`` — IF-NOT-EXISTS), then adds the explicit
secondary indexes the plan calls for.

DDL is scoped to the comprehension tables alone — we deliberately do NOT
``Base.metadata.create_all`` on the shared Base, which would also try to
materialize every CSV-side (server/app) table.

NOT wired into ``app/main.py`` — productionization (lifespan wiring) is deferred
(needs live infra). To mount, call in the app lifespan AFTER the other pdf_chat
migrations::

    from pdf_chat.migrations.comprehension_upgrade import apply_comprehension_migration
    await apply_comprehension_migration(engine)
"""
from __future__ import annotations

import structlog
from sqlalchemy import text

# Importing registers the tables on the shared Base metadata.
from pdf_chat.models.comprehension import (
    DocTaxonomyClass,
    GlossaryEntry,
    KeyMetric,
    OntologyEntity,
    OntologyRelationship,
    TemporalCoverage,
    TenantOntology,
)

_log = structlog.get_logger("pdf_chat.migrations.comprehension")

# Create order respects the FK to pdf_tenant_ontology (parent first).
_TABLES = (
    TenantOntology,
    OntologyEntity,
    OntologyRelationship,
    DocTaxonomyClass,
    TemporalCoverage,
    KeyMetric,
    GlossaryEntry,
)

# Explicit secondary indexes (the plan calls out (tenant_id, term) and
# (tenant_id, version)). All IF-NOT-EXISTS so re-running is a no-op.
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pdf_glossary_tenant_term "
    "ON pdf_glossary_entry(tenant_id, term)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_ontology_tenant_version "
    "ON pdf_tenant_ontology(tenant_id, version)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_glossary_tenant_version "
    "ON pdf_glossary_entry(tenant_id, ontology_version)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_onto_entity_ontology "
    "ON pdf_ontology_entity(ontology_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_doc_taxonomy_ontology "
    "ON pdf_doc_taxonomy_class(ontology_id)",
]


def _create_tables(sync_conn) -> None:
    """Table-scoped create (checkfirst=True ⇒ IF-NOT-EXISTS) for each table."""
    for model in _TABLES:
        model.__table__.create(sync_conn, checkfirst=True)


async def apply_comprehension_migration(engine) -> None:
    """Create the comprehension tables + indexes only. Safe to run repeatedly.

    Non-fatal: any failure logs a warning and is swallowed (matching the
    runtime-migration convention) so a degraded DB never blocks app startup.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(_create_tables)
            for stmt in _INDEXES:
                await conn.execute(text(stmt))
    except Exception as exc:  # pragma: no cover - DB best-effort, never fatal
        _log.warning("pdf_comprehension_migration_failed", error=str(exc))


# Alias so callers can use either name (mirrors bridge_upgrade.py style).
upgrade = apply_comprehension_migration

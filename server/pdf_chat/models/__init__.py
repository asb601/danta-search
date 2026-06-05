"""pdf_chat ORM models — import side-effect registers them on the shared Base.

Importing this package ensures every pdf_chat table is registered on
``app.core.database.Base.metadata`` so the app lifespan ``create_all`` and the
runtime migrations create them. Tenant isolation is via ``container_id`` /
``tenant_id`` on each table.
"""
from __future__ import annotations

from pdf_chat.models.bridge import BridgeStatus, PdfEntityBridge
from pdf_chat.models.comprehension import (
    DocTaxonomyClass,
    GlossaryEntry,
    KeyMetric,
    OntologyEntity,
    OntologyRelationship,
    TemporalCoverage,
    TenantOntology,
)
from pdf_chat.models.manifests import (
    PageManifest,
    QueryAuditLog,
    UploadManifest,
)
from pdf_chat.models.tunable import PdfGraphRagTunable

__all__ = [
    "UploadManifest",
    "PageManifest",
    "QueryAuditLog",
    "PdfGraphRagTunable",
    "PdfEntityBridge",
    "BridgeStatus",
    "TenantOntology",
    "OntologyEntity",
    "OntologyRelationship",
    "DocTaxonomyClass",
    "TemporalCoverage",
    "KeyMetric",
    "GlossaryEntry",
]

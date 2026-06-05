"""Phase 5 — comprehension ORM models (versioned ontology + glossary).

These tables turn the grounded Phase-2 Neo4j graph into a per-tenant, browsable,
VERSIONED comprehension artifact (mirroring the structured side's
``app/services/semantic_layer_builder.py`` — a queryable object built from a graph
substrate). Registered on the app's shared ``Base`` so ``create_all`` / the
runtime migration create them in the same PostgreSQL database. Tenant-isolated via
``tenant_id`` (and ``container_id`` on the artifact + glossary rows).

Open-vocabulary rule (invariant 6): every learned-meaning column —
``provenance``, relationship ``state`` (asserted|not_stated|conflicting),
``doc_class``, ``relation``, ``subject_kind`` — is a free-text ``Text`` column,
NEVER a SQLAlchemy ``Enum``, so the vocabulary stays open (mirrors
``semantic_roles`` minting ``custom:<kind>:<slug>`` dynamically and ``manifests.py``
storing ``status`` as Text). Provenance values come from
``pdf_chat.comprehension.provenance.Provenance`` but persist as their bare value.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TenantOntology(Base):
    """One row per built ontology VERSION — the versioned artifact header.

    A rebuild (re-ingestion) inserts a NEW row with ``version = max+1`` and a
    recomputed ``source_graph_signature``; old versions are retained + queryable
    (Definition of Done: version bumps on rebuild, old versions kept).
    """
    __tablename__ = "pdf_tenant_ontology"

    ontology_id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    container_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Signature of the graph substrate this artifact was built from — used for
    # idempotent finalization (skip rebuild when the graph is unchanged).
    source_graph_signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="built")

    __table_args__ = (
        UniqueConstraint("tenant_id", "version", name="uq_pdf_ontology_tenant_version"),
        Index("idx_pdf_ontology_tenant_version", "tenant_id", "version"),
    )


class OntologyEntity(Base):
    """A projected entity registry row (mirrors semantic_layer_builder entity spec)."""
    __tablename__ = "pdf_ontology_entity"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    ontology_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pdf_tenant_ontology.ontology_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    entity_type: Mapped[str | None] = mapped_column(Text, nullable=True)  # open-vocab
    pagerank: Mapped[float | None] = mapped_column(Float, nullable=True)
    mention_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_chunk_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_pdf_onto_entity_onto_name", "ontology_id", "name"),
    )


class OntologyRelationship(Base):
    """A three-state relationship row (asserted|not_stated|conflicting — Text)."""
    __tablename__ = "pdf_ontology_relationship"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    ontology_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pdf_tenant_ontology.ontology_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    src_name: Mapped[str] = mapped_column(Text, nullable=False)
    dst_name: Mapped[str] = mapped_column(Text, nullable=False)
    relation: Mapped[str | None] = mapped_column(Text, nullable=True)  # open-vocab
    # asserted | not_stated | conflicting — open Text, never silently resolved.
    state: Mapped[str] = mapped_column(Text, nullable=False, default="asserted")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class DocTaxonomyClass(Base):
    """An OPEN-VOCAB learned document class (arbitrary string + confidence).

    Doc classes are LLM-clustered from ``(:Document)`` content — never an
    enumerated list. ``doc_class`` is free-text Text exactly as semantic roles are
    minted dynamically.
    """
    __tablename__ = "pdf_doc_taxonomy_class"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    ontology_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pdf_tenant_ontology.ontology_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    doc_class: Mapped[str] = mapped_column(Text, nullable=False)  # open-vocab
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    member_doc_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)


class TemporalCoverage(Base):
    """Per-subject temporal coverage (date span + density + last mention)."""
    __tablename__ = "pdf_temporal_coverage"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    ontology_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pdf_tenant_ontology.ontology_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    subject_kind: Mapped[str | None] = mapped_column(Text, nullable=True)  # open-vocab
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    min_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    max_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    density: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_mention_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class KeyMetric(Base):
    """A learned key metric (name + grounded definition + evidence)."""
    __tablename__ = "pdf_key_metric"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    ontology_id: Mapped[str] = mapped_column(
        Text, ForeignKey("pdf_tenant_ontology.ontology_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    metric: Mapped[str] = mapped_column(Text, nullable=False)
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class GlossaryEntry(Base):
    """A corpus-learned glossary term — grounded (cited span) or refused.

    Stamped with the ``ontology_version`` it was mined under so a re-version is a
    NEW row (``UniqueConstraint(tenant_id, term, ontology_version)``), keeping old
    versions queryable. ``provenance`` is open Text (the ``Provenance`` value:
    stated|inferred|conflicting|not_found). An ``inferred`` entry is NEVER labelled
    ``stated``; conflicting definitions keep ALL spans (both sides surfaced).
    """
    __tablename__ = "pdf_glossary_entry"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    container_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    ontology_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    term: Mapped[str] = mapped_column(Text, nullable=False)
    expansion: Mapped[str | None] = mapped_column(Text, nullable=True)
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance: Mapped[str] = mapped_column(Text, nullable=False)  # open-vocab
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    variants: Mapped[list | None] = mapped_column(ARRAY(Text), nullable=True)
    evidence_spans: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "term", "ontology_version",
            name="uq_pdf_glossary_tenant_term_version",
        ),
        Index("idx_pdf_glossary_tenant_term", "tenant_id", "term"),
    )

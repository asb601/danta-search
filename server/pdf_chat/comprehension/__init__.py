"""Phase 5 — the comprehension ("superhuman memory") layer.

Turns the grounded Phase-2 Neo4j graph into a per-tenant, browsable, versioned
comprehension artifact: a Postgres-backed tenant ontology, a corpus-learned
glossary (each entry grounded or refused), faithfulness labels, and a read-only
onboarding surface.

Pure-import package: importing any module here touches NO infra (no Neo4j /
Postgres / Azure). Backends are injected (GraphReader Protocol, AsyncSession).
"""
from __future__ import annotations

from pdf_chat.comprehension.finalize import finalize_comprehension
from pdf_chat.comprehension.glossary_miner import load_background_freq, mine_glossary
from pdf_chat.comprehension.ontology_builder import build_tenant_ontology
from pdf_chat.comprehension.provenance import Provenance, label_for
from pdf_chat.comprehension.reader import (
    GraphReader,
    current_ontology_version,
    list_doc_taxonomy,
    list_entities,
    list_glossary,
    lookup_glossary,
    topic_map,
)
from pdf_chat.comprehension.temporal import (
    compute_temporal_coverage,
    staleness_annotation,
)

__all__ = [
    "Provenance",
    "label_for",
    "GraphReader",
    "current_ontology_version",
    "lookup_glossary",
    "list_glossary",
    "list_entities",
    "list_doc_taxonomy",
    "topic_map",
    "mine_glossary",
    "load_background_freq",
    "build_tenant_ontology",
    "compute_temporal_coverage",
    "staleness_annotation",
    "finalize_comprehension",
]

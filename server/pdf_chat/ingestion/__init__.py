"""Team B — Ingestion public surface.

Pure logic + guarded infra adapters for the PDF ingestion pipeline. Importing
this package pulls in only pure code; infra libraries (pikepdf, pdfplumber,
python-magic, fitz, neo4j, celery, redis, openai, llama-index) are behind
guarded imports and are only required when their feature is actually invoked.
"""
from __future__ import annotations

from .chunker import chunk_elements
from .embeddings import embed_texts
from .fingerprint import compute_sha256
from .neo4j_writer import Neo4jWriter
from .page_reader import stream_pages
# ── Phase 2 — Knowledge Graph public surface ────────────────────────────────
from .card_builder import (
    DocCard,
    SectionCard,
    build_doc_card,
    build_section_card,
)
from .communities import (
    Community,
    CommunityReport,
    CommunityReporter,
    detect_communities,
    pagerank_confidence,
)
from .entity_resolution import EntityResolver, MergeDecision, ResolvedEntity
from .grounding_gate import GroundedEdge, GroundedTag, GroundingGate, tag_as_answer
from .kg_construction import KGConstructionResult, construct_knowledge_graph
from .kg_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractedTag,
    SectionExtractor,
    section_fingerprint,
)
from .kg_writer import Neo4jKGWriter
from .ner_backbone import (
    EntityCandidate,
    fingerprint_value,
    propose_entities,
    propose_links,
)
from .sectionizer import Section, sectionize
from .parser_router import route_parser
from .preflight import (
    PreflightResult,
    classify_page,
    evaluate_preflight,
    run_preflight,
)
from .tasks import (
    PermanentError,
    TransientError,
    _run_page_extraction,
    dlq_key,
    process_page_task,
    push_to_dlq,
    retry_countdown,
)
from .ton_schema import BBox, Chunk, ElementType, UnifiedElement

__all__ = [
    "compute_sha256",
    "PreflightResult",
    "evaluate_preflight",
    "classify_page",
    "run_preflight",
    "route_parser",
    "stream_pages",
    "chunk_elements",
    "embed_texts",
    "Neo4jWriter",
    "process_page_task",
    "_run_page_extraction",
    "retry_countdown",
    "dlq_key",
    "push_to_dlq",
    "TransientError",
    "PermanentError",
    "UnifiedElement",
    "Chunk",
    "ElementType",
    "BBox",
    # Phase 2 — Knowledge Graph
    "Section",
    "sectionize",
    "EntityCandidate",
    "propose_entities",
    "propose_links",
    "fingerprint_value",
    "GroundedEdge",
    "GroundedTag",
    "GroundingGate",
    "tag_as_answer",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractedTag",
    "SectionExtractor",
    "section_fingerprint",
    "ResolvedEntity",
    "MergeDecision",
    "EntityResolver",
    "Neo4jKGWriter",
    "SectionCard",
    "DocCard",
    "build_section_card",
    "build_doc_card",
    "Community",
    "CommunityReport",
    "CommunityReporter",
    "detect_communities",
    "pagerank_confidence",
    "construct_knowledge_graph",
    "KGConstructionResult",
]

"""Phase-2 + Phase-5 document finalization — the post-page-settle graph build.

When every page of a document has settled, ``control_plane.finalizer`` flips the
document to a terminal indexed status. THIS module is what turns that settled,
chunk-only document into the actual knowledge graph + tenant comprehension
artifact that GraphRAG retrieval reads:

  Phase 2 (per document) — ``ingestion.construct_knowledge_graph``: sectionize the
    document's chunks, extract grounded entities/relations/tags, resolve entities,
    write the graph + multi-representation cards, detect communities and write
    cited community reports.

  Phase 5 (per tenant) — ``comprehension.finalize_comprehension``: build/version
    the tenant ontology from the grounded graph and mine the corpus glossary.

Both phases are IDEMPOTENT (Phase-2 MERGEs on business-key+tenant; Phase-5 is a
no-op when the graph signature is unchanged), so the at-least-once Celery delivery
and the reconciler can safely re-run this for a document that already settled.

This is the SINGLE place the deferred Phase-2/Phase-5 backends are assembled with
real infra (Neo4j + the bulk Azure ``gpt-4o-mini`` deployment). The model id is
always chosen by the consumers via ``model_router.select_model`` — never here.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..config import get_pdf_settings


def _container_scope(container_id: str | None, tenant_id: str) -> str:
    """The per-container dial scope for tunables/embeddings.

    Mirrors the chat route's ``container_id or tenant_id`` fallback so a document
    ingested before ``container_id`` was recorded still resolves real tunables.
    """
    return container_id or tenant_id


def build_document_kg(upload_id: str, tenant_id: str, container_id: str | None) -> Any:
    """Phase-2: read the settled document's chunks and construct its KG.

    Returns the :class:`KGConstructionResult`, or ``None`` when the document has
    no chunks (nothing to build). Synchronous — ``construct_knowledge_graph`` is a
    pure sync orchestrator over sync backends.
    """
    from ..ingestion import ner_backbone, card_builder
    from ..ingestion.communities import (
        CommunityReporter,
        detect_communities,
        pagerank_confidence,
    )
    from ..ingestion.entity_resolution import EntityResolver
    from ..ingestion.grounding_gate import GroundingGate
    from ..ingestion.ingest_llm import KgIngestionLlm
    from ..ingestion.kg_construction import construct_knowledge_graph
    from ..ingestion.kg_writer import Neo4jKGWriter
    from ..ingestion.neo4j_writer import Neo4jWriter
    from ..ingestion.kg_extraction import SectionExtractor
    from ..retrieval.embeddings import embed_texts_batched

    scope = _container_scope(container_id, tenant_id)
    s = get_pdf_settings()

    # Re-read the document's chunks (written page-by-page at ingest time).
    reader = Neo4jWriter(s.neo4j_uri, s.neo4j_user, s.neo4j_password, database=s.neo4j_database)
    try:
        chunks = reader.read_chunks_for_doc(upload_id, tenant_id)
    finally:
        reader.close()
    if not chunks:
        return None

    llm = KgIngestionLlm()
    kg_writer = Neo4jKGWriter(
        s.neo4j_uri, s.neo4j_user, s.neo4j_password, database=s.neo4j_database
    )
    # The ``communities`` seam is a single object bundling the two pure functions
    # (networkx-guarded) with the LLM-backed reporter — kept out of the
    # orchestrator so it never imports networkx directly.
    communities = SimpleNamespace(
        detect_communities=detect_communities,
        pagerank_confidence=pagerank_confidence,
        report=CommunityReporter(llm).report,
    )
    try:
        return construct_knowledge_graph(
            doc_id=upload_id,
            container_id=scope,
            tenant_id=tenant_id,
            chunks=chunks,
            extractor=SectionExtractor(llm),
            ner=ner_backbone,
            gate=GroundingGate(),
            resolver=EntityResolver(),
            writer=kg_writer,
            card_builder=card_builder,
            communities=communities,
            # construct's embed_fn is a single-arg callable; bind the scope.
            embed_fn=lambda texts: embed_texts_batched(texts, container_id=scope),
        )
    finally:
        kg_writer.close()


async def run_comprehension(tenant_id: str, container_id: str | None) -> Any:
    """Phase-5: build/version the tenant ontology + mine the glossary.

    Idempotent on the graph signature, so this is safe to run after EVERY
    document settles for the tenant (an unchanged graph is a no-op). Returns the
    ``TenantOntology`` header, or ``None`` on any best-effort failure.
    """
    from app.core.database import async_session  # type: ignore

    from ..comprehension.comprehension_llm import ComprehensionLlm
    from ..comprehension.finalize import finalize_comprehension
    from ..comprehension.neo4j_graph_reader import Neo4jGraphReader

    scope = _container_scope(container_id, tenant_id)
    async with async_session() as session:
        onto = await finalize_comprehension(
            Neo4jGraphReader(),
            tenant_id=tenant_id,
            container_id=scope,
            session=session,
            llm=ComprehensionLlm(),
        )
        await session.commit()
        return onto


async def build_document_graph(
    upload_id: str, tenant_id: str, container_id: str | None
) -> dict:
    """Run Phase-2 then Phase-5 for one settled document (the task entry point).

    Each phase is independently best-effort: a Phase-2 failure must not skip
    Phase-5's idempotency check, and vice versa. Returns a small summary dict for
    the task result / audit.
    """
    summary: dict[str, Any] = {"upload_id": upload_id, "kg": None, "comprehension": False}
    try:
        result = build_document_kg(upload_id, tenant_id, container_id)
        if result is not None:
            summary["kg"] = {
                "entities": result.entities_resolved,
                "edges": result.edges_admitted,
                "communities": result.communities,
                "reports": result.reports,
            }
    except Exception as exc:  # pragma: no cover - best-effort; retried by the task
        summary["kg_error"] = str(exc)

    try:
        onto = await run_comprehension(tenant_id, container_id)
        summary["comprehension"] = onto is not None
    except Exception as exc:  # pragma: no cover - best-effort; retried by the task
        summary["comprehension_error"] = str(exc)

    return summary

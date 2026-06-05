"""Phase 5 — the comprehension FINALIZATION orchestrator (Task 11).

``finalize_comprehension`` is the single ingest-finalization entry point that
turns the freshly-written, grounded Phase-2 graph into the per-tenant
comprehension artifact, in the correct order and exactly once per graph state:

  1. Build (and version) the tenant ontology from the grounded graph
     (``ontology_builder.build_tenant_ontology``) — entities, three-state
     relationships, open-vocab doc taxonomy, key metrics, temporal coverage.
  2. Mine the corpus glossary (``glossary_miner.mine_glossary``) from the same
     graph's chunks, STAMPED with the new ontology's ``version`` so a glossary
     row always points at the ontology it was learned against. Each mined row is
     persisted under that version.

IDEMPOTENT on ``source_graph_signature`` (intelligence-at-ingest, spec §0
principle 1): the orchestrator computes the signature of the current graph
substrate (the same stable identity hash the builder stamps onto the header) and,
when the LATEST persisted ontology already carries that signature, SKIPS the
rebuild and returns the existing header unchanged. A changed graph ⇒ a new
version is built (old versions retained + queryable — Definition of Done). This
makes re-running finalization on an unchanged corpus a no-op rather than a
version churn.

ROUTING (contract C7): both phases are INGESTION BULK. The ontology builder and
the glossary miner each route their LLM through
``model_router.select_model(task=SYNTHESIS, signals={})`` — bulk ``gpt-4o-mini``;
the strong tier (Opus) is structurally unreachable for ``SYNTHESIS``. This
orchestrator makes NO direct LLM call (it delegates), and never requests a strong
tier.

No score-comparison literal lives here (spec §3 invariant 4): the only decision
is signature EQUALITY (a string compare for idempotency), not a tunable score
gate.

Call site (DEFERRED WIRING — not edited here): the Phase-1 ingest-finalization
state machine calls ``finalize_comprehension`` once, after the Phase-2 graph for
the upload has been written. PRODUCTION passes a ``Neo4jGraphReader`` (the
``comprehension.neo4j_graph_reader`` adapter wrapping the Phase-2
``Neo4jSearcher``) as the ``reader`` — it implements the six ``GraphReader``
iterators this orchestrator drains, which the raw ``Neo4jSearcher`` does not.
Plus ``tenant_id``/``container_id``, an injected bulk ``llm`` seam, and the
request ``AsyncSession``. We document the call site here but do NOT edit that
state machine (or ``agent/graph.py`` / ``app/main.py``).
"""
from __future__ import annotations

from .glossary_miner import load_background_freq, mine_glossary
from .ontology_builder import _compute_signature, build_tenant_ontology
from .reader import GraphReader


async def _latest_signature(session, tenant_id: str) -> str | None:
    """Return the ``source_graph_signature`` of the LATEST ontology for a tenant.

    ``None`` when no artifact has been built yet (so the first finalization always
    builds). Tenant-scoped (the WHERE clause carries ``tenant_id``); ordered by
    ``version`` descending so a re-version's newest header wins. The query mirrors
    ``reader.current_ontology_version`` / ``onboarding_reads.latest_ontology_id``
    (kept here, not in ``reader.py``, to avoid a write conflict with that module).
    """
    from sqlalchemy import select

    from pdf_chat.models.comprehension import TenantOntology

    stmt = (
        select(TenantOntology.source_graph_signature)
        .where(TenantOntology.tenant_id == tenant_id)
        .order_by(TenantOntology.version.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _existing_latest(session, tenant_id: str):
    """Return the latest persisted ``TenantOntology`` header for ``tenant_id`` or None.

    Used to return the existing artifact unchanged when finalization is a no-op
    (the graph signature is unchanged), so callers always get a header back.
    """
    from sqlalchemy import select

    from pdf_chat.models.comprehension import TenantOntology

    stmt = (
        select(TenantOntology)
        .where(TenantOntology.tenant_id == tenant_id)
        .order_by(TenantOntology.version.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def finalize_comprehension(
    reader: GraphReader,
    *,
    tenant_id: str,
    container_id: str,
    session,
    llm,
    background_freq: dict[str, float] | None = None,
):
    """Build the ontology then mine the glossary; idempotent on graph signature.

    Returns the ``TenantOntology`` header (newly built, or the existing latest one
    when finalization is a no-op because the graph is unchanged).

    ``llm`` is the injected bulk seam shared by both phases (doc-taxonomy
    clustering AND glossary mining) — production wires the prompt-cached
    ``gpt-4o-mini`` client behind it; tests inject a fake. ``background_freq`` is
    the injected distributional-anomaly signal source (the shipped
    ``background_freq.json`` is loaded when not supplied) — never an in-code
    jargon list.

    Persistence: ``build_tenant_ontology`` persists the ontology + child rows; this
    orchestrator then persists the mined glossary rows (stamped with the new
    ``version``) via ``session.add`` and a final ``flush``. Committing the
    transaction is the caller's (finalization state machine's) responsibility.
    """
    # ── Idempotency: skip a rebuild when the graph substrate is unchanged ──────
    # Compute the signature of the CURRENT graph (same identity hash the builder
    # stamps) and compare it (string equality, not a score gate) to the latest
    # persisted signature. Unchanged ⇒ no-op, return the existing header. The
    # CHUNKS are drained here (not only for mining) because the signature folds in
    # a chunk-content digest (FIX G) — so the idempotency probe matches what the
    # builder stamps and a chunk-text edit correctly forces a re-version.
    entities = [e async for e in reader.iter_entities(tenant_id)]
    relationships = [r async for r in reader.iter_relationships(tenant_id)]
    documents = [d async for d in reader.iter_documents(tenant_id)]
    chunks = [c async for c in reader.iter_chunks(tenant_id)]
    new_signature = _compute_signature(entities, relationships, documents, chunks)

    prior_signature = await _latest_signature(session, tenant_id)
    if prior_signature is not None and prior_signature == new_signature:
        return await _existing_latest(session, tenant_id)

    # ── 1) Build + version the tenant ontology from the grounded graph ─────────
    onto = await build_tenant_ontology(
        reader, tenant_id=tenant_id, container_id=container_id,
        session=session, llm=llm,
    )

    # ── 2) Mine the glossary from the same graph's chunks, stamped with the new
    #       ontology version, and persist each grounded row under that version ──
    bg = background_freq if background_freq is not None else load_background_freq()
    glossary_entries = await mine_glossary(
        chunks,
        llm=llm,
        tenant_id=tenant_id,
        container_id=container_id,
        ontology_version=onto.version,
        background_freq=bg,
    )
    for entry in glossary_entries:
        session.add(entry)
    await session.flush()

    return onto


__all__ = ["finalize_comprehension"]

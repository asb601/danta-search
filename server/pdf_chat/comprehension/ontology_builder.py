"""Phase 5 — versioned tenant-ontology builder (Task 7).

Turns the grounded Phase-2 Neo4j graph into a per-tenant, VERSIONED, queryable
comprehension artifact — mirroring the structured side's
``app/services/semantic_layer_builder.py`` (build registries from a graph
substrate as a queryable object, versioned). One ``build_tenant_ontology`` call
reads the graph through an injected ``GraphReader`` and projects it into:

* ``OntologyEntity``      — the entity registry (name/normalized/type/pagerank/
  mention_count/evidence chunk ids), mirroring the entity-spec shape.
* ``OntologyRelationship``— graph edges land three-state ``asserted`` (open Text;
  ``not_stated``/``conflicting`` are surfaced elsewhere, never silently resolved).
* ``DocTaxonomyClass``    — OPEN-VOCAB document classes LLM-clustered from
  ``(:Document)`` content (arbitrary strings + confidence), NEVER an enumerated
  doc-type list. A class below ``ontology.doc_taxonomy_min_confidence`` is gated
  out (``get_tunable`` + ``log_gate_decision``).
* ``KeyMetric``           — metric-typed entities promoted to grounded metric rows.
* ``TemporalCoverage``    — per-subject date span (delegated to ``temporal.py``).

VERSIONING: a rebuild reads ``max(existing version)`` and inserts a NEW header at
``version = max + 1`` (old versions retained + queryable — Definition of Done).
The ``source_graph_signature`` is recomputed each build (idempotent-finalization
key). All child rows are persisted under the new ``ontology_id``.

ROUTING (contract C7): doc-taxonomy clustering is INGESTION BULK ⇒
``select_model(task="synthesis")`` → bulk ``gpt-4o-mini`` with ``signals={}`` so
escalation can never fire; Opus is structurally unreachable. The LLM is INJECTED
(``llm``) so this module is import-safe with zero infra; production wires the
prompt-cached Azure ``gpt-4o-mini`` client behind it (same convention as
``ingestion/communities.py::CommunityReporter``).

NO score-comparison literal lives here (spec §3 invariant 4): the only threshold
(doc-taxonomy confidence) resolves via ``get_tunable`` and is decided by
``log_gate_decision``.

Call site (deferred wiring): the Phase-1 ingest-finalization state machine calls
``build_tenant_ontology`` after the graph is written; we document it here but do
NOT edit that state machine.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pdf_chat.model_router import TaskClass, select_model
from pdf_chat.models.comprehension import (
    DocTaxonomyClass,
    KeyMetric,
    OntologyEntity,
    OntologyRelationship,
    TemporalCoverage,
    TenantOntology,
)
from pdf_chat.tunables import get_tunable, log_gate_decision

from .reader import GraphReader, _field, current_ontology_version
from .temporal import compute_temporal_coverage

# Tunable key (registered in tunables.py::TUNABLE_DEFAULTS — single source).
TUN_DOC_TAXONOMY_MIN_CONFIDENCE = "ontology.doc_taxonomy_min_confidence"

# A metric-typed graph entity becomes a KeyMetric. The type string is OPEN-VOCAB
# (learned upstream), so we match on a normalized contains, never an allow-list.
_METRIC_TYPE_HINT = "metric"

_DOC_CLUSTER_SYSTEM_PROMPT = (
    "You are a grounded document-taxonomy clusterer. Group the supplied documents "
    "into a small number of OPEN-VOCABULARY classes that describe what kind of "
    "document each is, using ONLY their titles/content. Invent the class names "
    "from the documents themselves — do NOT map onto any fixed list of document "
    "types. Return strict JSON: {\"classes\": [{\"doc_class\": str, "
    "\"confidence\": number, \"member_doc_ids\": [str]}]}."
)


def _compute_signature(
    entities: list, relationships: list, documents: list, chunks: list | None = None,
) -> str:
    """A stable signature of the graph substrate this artifact was built from.

    Used for idempotent finalization (skip rebuild when the graph is unchanged).
    Built from sorted identity tuples so it is order-independent.

    FIX G: the signature ALSO folds in a CHUNK-CONTENT digest — a stable hash over
    sorted ``chunk_id + "|" + text`` for every chunk. The glossary is mined from
    chunk TEXT, so a chunk whose text changed (same id, edited content) MUST force
    a re-version; keying only on entity names + edge tuples + doc ids would miss
    that and leave finalization (idempotent on the signature) stale. ``chunks`` is
    optional/defaulted so any legacy 3-arg call still works (it then contributes
    an empty chunk digest).
    """
    payload = {
        "entities": sorted(str(_field(e, "name")) for e in entities),
        "relationships": sorted(
            f"{_field(r, 'src_name')}|{_field(r, 'dst_name')}|{_field(r, 'relation')}"
            for r in relationships
        ),
        "documents": sorted(str(_field(d, "doc_id")) for d in documents),
        "chunks": sorted(
            f"{_field(c, 'chunk_id')}|{_field(c, 'text', '') or ''}"
            for c in (chunks or [])
        ),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_cluster_prompt(documents: list) -> str:
    """Render the documents (id + title) into the open-vocab clustering prompt."""
    lines = [_DOC_CLUSTER_SYSTEM_PROMPT, "", "Documents:"]
    for d in documents:
        lines.append(
            f"- id={_field(d, 'doc_id')} title=\"{_field(d, 'title', '') or ''}\""
        )
    return "\n".join(lines)


async def build_tenant_ontology(
    reader: GraphReader,
    *,
    tenant_id: str,
    container_id: str,
    session,
    llm,
) -> TenantOntology:
    """Build + persist a NEW versioned ontology from the grounded graph.

    Reads entities/relationships/communities/documents/chunks through the injected
    ``GraphReader`` (per-hop tenant isolation in the searcher), projects them into
    the registries, clusters an open-vocab doc taxonomy via the bulk SYNTHESIS
    model, persists every child row under a freshly versioned ``ontology_id``, and
    returns the header. Old versions are retained (a rebuild bumps the version).
    """
    # Drain the graph (the reader's async iterators are tenant-scoped).
    entities = [e async for e in reader.iter_entities(tenant_id)]
    relationships = [r async for r in reader.iter_relationships(tenant_id)]
    documents = [d async for d in reader.iter_documents(tenant_id)]
    # Chunks feed the signature's chunk-content digest (FIX G): the glossary is
    # mined from chunk text, so chunk-text changes must change the stamped
    # signature and force a re-version.
    chunks = [c async for c in reader.iter_chunks(tenant_id)]

    # ── version: max(existing)+1 (old retained, queryable) ──────────────────
    prior = await current_ontology_version(session, tenant_id)
    version = (prior or 0) + 1
    signature = _compute_signature(entities, relationships, documents, chunks)

    onto = TenantOntology(
        tenant_id=tenant_id,
        container_id=container_id,
        version=version,
        source_graph_signature=signature,
        status="built",
    )
    session.add(onto)
    await session.flush()  # materialize ontology_id for child FKs

    # ── entity registry (mirror semantic_layer_builder entity-spec shape) ───
    for e in entities:
        session.add(
            OntologyEntity(
                ontology_id=onto.ontology_id,
                tenant_id=tenant_id,
                name=_field(e, "name"),
                normalized_value=_field(e, "normalized_value"),
                entity_type=_field(e, "type") or _field(e, "entity_type"),
                pagerank=_field(e, "pagerank"),
                mention_count=_field(e, "mention_count"),
                evidence_chunk_ids=_field(e, "evidence_chunk_ids"),
            )
        )
        # Metric-typed entities are promoted to grounded KeyMetric rows.
        etype = (_field(e, "type") or _field(e, "entity_type") or "")
        if _METRIC_TYPE_HINT in str(etype).lower():
            session.add(
                KeyMetric(
                    ontology_id=onto.ontology_id,
                    tenant_id=tenant_id,
                    metric=_field(e, "name"),
                    definition=_field(e, "definition"),
                    evidence={"chunk_ids": _field(e, "evidence_chunk_ids", []) or []},
                    confidence=_field(e, "pagerank"),
                )
            )

    # ── relationship registry — graph edges are three-state "asserted" ──────
    for r in relationships:
        session.add(
            OntologyRelationship(
                ontology_id=onto.ontology_id,
                tenant_id=tenant_id,
                src_name=_field(r, "src_name"),
                dst_name=_field(r, "dst_name"),
                relation=_field(r, "relation"),
                state=_field(r, "state", "asserted") or "asserted",
                confidence=_field(r, "confidence"),
                evidence=_field(r, "evidence"),
            )
        )

    # ── open-vocab doc taxonomy (bulk SYNTHESIS LLM, gated by confidence) ────
    if documents:
        await _persist_doc_taxonomy(
            session, onto, tenant_id=tenant_id, container_id=container_id,
            documents=documents, llm=llm,
        )

    # ── temporal coverage (delegated to temporal.py), persisted per subject ──
    coverage = await compute_temporal_coverage(
        reader, tenant_id=tenant_id, container_id=container_id
    )
    for cov in coverage:
        session.add(
            TemporalCoverage(
                ontology_id=onto.ontology_id,
                tenant_id=tenant_id,
                subject_kind=cov.get("subject_kind"),
                subject=cov.get("subject"),
                min_date=cov.get("min_date"),
                max_date=cov.get("max_date"),
                density=cov.get("density"),
                last_mention_date=cov.get("last_mention_date"),
            )
        )

    await session.flush()
    return onto


async def _persist_doc_taxonomy(
    session, onto: TenantOntology, *, tenant_id: str, container_id: str,
    documents: list, llm,
) -> None:
    """Cluster documents into OPEN-VOCAB classes (bulk LLM) and persist the kept ones.

    Routing is bulk-only (``task=SYNTHESIS``, ``signals={}`` ⇒ escalation can
    never fire), so Opus is structurally unreachable. Each learned class is kept
    only when its clustering confidence clears ``ontology.doc_taxonomy_min_confidence``
    (``get_tunable`` + ``log_gate_decision`` — no inline literal).

    The LLM seam is AWAITED (async), matching the glossary miner's awaited seam so
    the production async ``gpt-4o-mini`` client wires cleanly behind one interface.
    """
    choice = select_model(
        task=TaskClass.SYNTHESIS, container_id=container_id, signals={}
    )
    assert choice.is_strong is False, "doc-taxonomy clustering must never escalate"

    payload = await llm.synthesize(
        _build_cluster_prompt(documents),
        model_id=choice.model_id,
        container_id=container_id,
    )
    classes = (payload or {}).get("classes", []) or []

    floor = float(get_tunable(container_id, TUN_DOC_TAXONOMY_MIN_CONFIDENCE))
    for cls in classes:
        conf = cls.get("confidence")
        rec = log_gate_decision(
            "ontology.doc_taxonomy_min_confidence",
            score=float(conf) if conf is not None else 0.0,
            threshold=floor,
            outcome="checked",
            container_id=container_id,
            doc_class=cls.get("doc_class"),
        )
        if not rec["passed"]:
            continue  # below the learned-class confidence floor → dropped
        session.add(
            DocTaxonomyClass(
                ontology_id=onto.ontology_id,
                tenant_id=tenant_id,
                doc_class=cls.get("doc_class"),
                confidence=conf,
                member_doc_ids=cls.get("member_doc_ids"),
            )
        )


__all__ = ["build_tenant_ontology"]

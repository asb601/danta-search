"""Build governed semantic memory from ingestion-time metadata.

This extractor reads existing facts produced by ingestion: file metadata,
semantic roles, semantic entities, approved/candidate relationships, and trust
scores. It does not call an LLM and it does not create runtime semantic truth.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import ingest_logger
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticEntity, SemanticRelationship
from app.models.semantic_memory import SemanticMemoryEvidence, SemanticMemoryRecord
from app.services.semantic_memory_governance import decide_governance_status
from app.services.semantic_memory_indexer import rebuild_memory_indexes
from app.services.semantic_memory_normalizer import (
    MemoryCandidate,
    MemoryEvidenceInput,
    canonical_key,
    normalise_candidate,
    split_role,
    terms_from_text,
)

_MAX_ROLE_MEMORIES = 30
_MAX_CAPABILITY_MEMORIES = 8
_MAX_RELATIONSHIP_MEMORIES = 20


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _file_terms(file: File | None, meta: FileMetadata) -> list[str]:
    return terms_from_text(
        file.name if file else None,
        meta.blob_path,
        meta.ai_description,
        meta.key_dimensions,
        meta.key_metrics,
    )


def _role_evidence(meta: FileMetadata, column_name: str, role: str) -> MemoryEvidenceInput:
    evidence_map = meta.column_role_evidence or {}
    evidence = evidence_map.get(column_name) if isinstance(evidence_map, dict) else None
    confidence = 0.55
    if isinstance(evidence, dict):
        try:
            confidence = float(evidence.get("confidence") or confidence)
        except (TypeError, ValueError):
            confidence = 0.55
    return MemoryEvidenceInput(
        source_type="column_role",
        source_id=column_name,
        evidence_key=role,
        evidence_value=evidence or {"column": column_name, "role": role},
        confidence_score=confidence,
        file_id=meta.file_id,
    )


def _entity_candidate(
    *,
    meta: FileMetadata,
    file: File | None,
    entity: SemanticEntity,
) -> MemoryCandidate:
    source = "semantic_layer"
    evidence = [MemoryEvidenceInput(
        source_type="semantic_entity",
        source_id=entity.id,
        evidence_key="entity",
        evidence_value={
            "entity_name": entity.entity_name,
            "primary_key": entity.primary_key,
            "grain": entity.grain,
        },
        confidence_score=entity.confidence_score,
        file_id=meta.file_id,
    )]
    decision = decide_governance_status(
        confidence_score=max(entity.confidence_score or 0.0, meta.ingestion_confidence_score or 0.0),
        evidence_count=len(evidence),
        source=source,
    )
    return MemoryCandidate(
        container_id=meta.container_id or "",
        memory_type="entity",
        canonical_key=canonical_key("entity", meta.container_id, entity.entity_name, meta.file_id),
        title=entity.entity_name,
        summary=entity.grain or meta.ai_description,
        normalized_terms=terms_from_text(entity.entity_name, entity.primary_key, _file_terms(file, meta)),
        behaviors=["entity_resolution", "retrieval_anchor"],
        dimensions={
            "primary_key": entity.primary_key,
            "metrics": entity.metrics or [],
            "dimensions": entity.dimensions or [],
        },
        confidence_score=entity.confidence_score or 0.0,
        authority_score=decision.authority_score,
        governance_status=decision.governance_status,
        source=source,
        source_file_id=meta.file_id,
        source_entity_id=entity.id,
        evidence=evidence,
    )


def _role_candidates(meta: FileMetadata, file: File | None) -> list[MemoryCandidate]:
    roles = meta.column_semantic_roles or {}
    candidates: list[MemoryCandidate] = []
    for column_name, role in list(roles.items())[:_MAX_ROLE_MEMORIES]:
        kind, label = split_role(str(role or ""))
        if not kind or not label:
            continue
        if kind in {"additive_measure", "non_additive_measure"}:
            memory_type = "metric"
            behaviors = ["aggregation", "kpi_contract"]
        elif kind == "date":
            memory_type = "temporal"
            behaviors = ["time_filter", "lifecycle_scope"]
        elif kind == "reference_key":
            memory_type = "dimension"
            behaviors = ["join_guidance", "ambiguity_check"]
        else:
            memory_type = "dimension"
            behaviors = ["filter_guidance", "entity_resolution"]
        evidence = [_role_evidence(meta, column_name, role)]
        confidence = max((ev.confidence_score for ev in evidence), default=0.5)
        decision = decide_governance_status(
            confidence_score=confidence,
            evidence_count=len(evidence),
            source="column_role",
            risk_reason="reference_key_requires_join_validation" if kind == "reference_key" else None,
        )
        candidates.append(MemoryCandidate(
            container_id=meta.container_id or "",
            memory_type=memory_type,
            canonical_key=canonical_key(memory_type, meta.container_id, meta.file_id, kind, label, column_name),
            title=label.replace("_", " "),
            summary=f"{column_name} is a {kind} role on this dataset.",
            normalized_terms=terms_from_text(label, kind, column_name, role, _file_terms(file, meta)),
            behaviors=behaviors,
            dimensions={"role_kind": kind, "role": role, "column": column_name},
            confidence_score=confidence,
            authority_score=decision.authority_score,
            governance_status=decision.governance_status,
            source="column_role",
            source_file_id=meta.file_id,
            evidence=evidence,
        ))
        if kind == "reference_key":
            null_decision = decide_governance_status(
                confidence_score=confidence * 0.9,
                evidence_count=len(evidence),
                source="column_role",
                risk_reason="null_semantic_requires_data_validation",
            )
            candidates.append(MemoryCandidate(
                container_id=meta.container_id or "",
                memory_type="null_semantic",
                canonical_key=canonical_key("null_semantic", meta.container_id, meta.file_id, label, column_name),
                title=f"missing {label.replace('_', ' ')}",
                summary=f"NULL in {column_name} indicates no referenced {label.replace('_', ' ')} value is present.",
                normalized_terms=terms_from_text(label, column_name, "missing", "null"),
                behaviors=["filter_guidance", "validation_required"],
                dimensions={"column": column_name, "role": role},
                confidence_score=confidence * 0.9,
                authority_score=null_decision.authority_score,
                governance_status=null_decision.governance_status,
                source="column_role",
                source_file_id=meta.file_id,
                evidence=evidence,
            ))
    return candidates


def _capability_candidates(meta: FileMetadata, file: File | None) -> list[MemoryCandidate]:
    phrases = [str(item).strip() for item in (meta.good_for or []) if str(item).strip()]
    candidates: list[MemoryCandidate] = []
    for idx, phrase in enumerate(phrases[:_MAX_CAPABILITY_MEMORIES]):
        evidence = [MemoryEvidenceInput(
            source_type="file_metadata",
            source_id=meta.id,
            evidence_key=f"good_for:{idx}",
            evidence_value=phrase,
            confidence_score=max(0.45, min(0.8, meta.ingestion_confidence_score or 0.55)),
            file_id=meta.file_id,
        )]
        decision = decide_governance_status(
            confidence_score=evidence[0].confidence_score,
            evidence_count=len(evidence),
            source="file_metadata",
        )
        candidates.append(MemoryCandidate(
            container_id=meta.container_id or "",
            memory_type="capability",
            canonical_key=canonical_key("capability", meta.container_id, meta.file_id, phrase),
            title=phrase[:120],
            summary=meta.ai_description,
            normalized_terms=terms_from_text(phrase, _file_terms(file, meta)),
            behaviors=["retrieval_guidance"],
            dimensions={"source": "good_for"},
            confidence_score=evidence[0].confidence_score,
            authority_score=decision.authority_score,
            governance_status=decision.governance_status,
            source="file_metadata",
            source_file_id=meta.file_id,
            evidence=evidence,
        ))
    return candidates


def _grain_candidate(meta: FileMetadata, entity: SemanticEntity | None) -> MemoryCandidate | None:
    if not entity or not entity.grain:
        return None
    evidence = [MemoryEvidenceInput(
        source_type="semantic_entity",
        source_id=entity.id,
        evidence_key="grain",
        evidence_value=entity.grain,
        confidence_score=entity.confidence_score or 0.0,
        file_id=meta.file_id,
    )]
    decision = decide_governance_status(
        confidence_score=entity.confidence_score or 0.0,
        evidence_count=len(evidence),
        source="semantic_layer",
    )
    return MemoryCandidate(
        container_id=meta.container_id or "",
        memory_type="grain",
        canonical_key=canonical_key("grain", meta.container_id, meta.file_id, entity.entity_name),
        title=f"grain: {entity.entity_name}",
        summary=entity.grain,
        normalized_terms=terms_from_text(entity.entity_name, entity.grain, entity.primary_key),
        behaviors=["fanout_guard", "plan_validation"],
        dimensions={"entity": entity.entity_name, "primary_key": entity.primary_key},
        confidence_score=entity.confidence_score or 0.0,
        authority_score=decision.authority_score,
        governance_status=decision.governance_status,
        source="semantic_layer",
        source_file_id=meta.file_id,
        source_entity_id=entity.id,
        evidence=evidence,
    )


def _relationship_candidate(meta: FileMetadata, rel: SemanticRelationship) -> MemoryCandidate:
    other_entity = rel.to_entity if rel.file_a_id == meta.file_id else rel.from_entity
    join_rule = rel.join_rule or {}
    evidence = [MemoryEvidenceInput(
        source_type="semantic_relationship",
        source_id=rel.id,
        evidence_key="join_rule",
        evidence_value=join_rule,
        confidence_score=rel.confidence_score or 0.0,
        file_id=meta.file_id,
    )]
    decision = decide_governance_status(
        confidence_score=rel.confidence_score or 0.0,
        evidence_count=len(evidence),
        source="relationship_graph",
        approval_status=rel.approval_status,
        risk_reason=rel.risk_reason,
    )
    return MemoryCandidate(
        container_id=meta.container_id or "",
        memory_type="relationship",
        canonical_key=canonical_key("relationship", meta.container_id, rel.id, meta.file_id),
        title=f"relationship to {other_entity}",
        summary=rel.risk_reason or f"{rel.from_entity}.{rel.from_column} joins {rel.to_entity}.{rel.to_column}",
        normalized_terms=terms_from_text(
            rel.from_entity,
            rel.to_entity,
            rel.from_column,
            rel.to_column,
            rel.relationship_type,
            join_rule,
        ),
        behaviors=["join_guidance", "plan_validation"],
        dimensions={
            "relationship_type": rel.relationship_type,
            "approval_status": rel.approval_status,
            "from_column": rel.from_column,
            "to_column": rel.to_column,
        },
        constraints={"requires_approved_relationship": rel.approval_status != "approved"},
        confidence_score=rel.confidence_score or 0.0,
        authority_score=decision.authority_score,
        governance_status=decision.governance_status,
        source="relationship_graph",
        source_file_id=meta.file_id,
        source_relationship_id=rel.id,
        evidence=evidence,
    )


async def extract_semantic_memory_candidates(file_id: str, db: AsyncSession) -> list[MemoryCandidate]:
    meta = (
        await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
    ).scalar_one_or_none()
    if not meta or not meta.container_id:
        return []

    file = await db.get(File, file_id)
    entity = (
        await db.execute(select(SemanticEntity).where(SemanticEntity.file_id == file_id))
    ).scalar_one_or_none()
    rels = (
        await db.execute(
            select(SemanticRelationship)
            .where(
                (SemanticRelationship.file_a_id == file_id) | (SemanticRelationship.file_b_id == file_id),
                SemanticRelationship.status == "active",
            )
            .order_by(SemanticRelationship.confidence_score.desc())
            .limit(_MAX_RELATIONSHIP_MEMORIES)
        )
    ).scalars().all()

    candidates: list[MemoryCandidate] = []
    if entity:
        candidates.append(_entity_candidate(meta=meta, file=file, entity=entity))
    grain = _grain_candidate(meta, entity)
    if grain:
        candidates.append(grain)
    candidates.extend(_role_candidates(meta, file))
    candidates.extend(_capability_candidates(meta, file))
    candidates.extend(_relationship_candidate(meta, rel) for rel in rels)
    return [c for c in (normalise_candidate(candidate) for candidate in candidates) if c]


async def upsert_semantic_memory_for_file(file_id: str, db: AsyncSession) -> dict[str, Any]:
    start = time.perf_counter()
    candidates = await extract_semantic_memory_candidates(file_id, db)
    if not candidates:
        return {"file_id": file_id, "records": 0, "duration_ms": _ms(start)}

    seen_keys = {candidate.canonical_key for candidate in candidates}
    existing_rows = (
        await db.execute(select(SemanticMemoryRecord).where(SemanticMemoryRecord.source_file_id == file_id))
    ).scalars().all()
    existing_by_key = {row.canonical_key: row for row in existing_rows}
    now = datetime.now(timezone.utc)

    upserted = 0
    for candidate in candidates:
        record = existing_by_key.get(candidate.canonical_key)
        if not record:
            record = SemanticMemoryRecord(id=str(uuid.uuid4()))
            db.add(record)
        record.container_id = candidate.container_id
        record.memory_type = candidate.memory_type
        record.canonical_key = candidate.canonical_key
        record.title = candidate.title
        record.summary = candidate.summary
        record.normalized_terms = candidate.normalized_terms
        record.behaviors = candidate.behaviors
        record.dimensions = candidate.dimensions
        record.constraints = candidate.constraints
        record.confidence_score = candidate.confidence_score
        record.authority_score = candidate.authority_score
        record.governance_status = candidate.governance_status
        record.status = candidate.status
        record.source = candidate.source
        record.source_file_id = candidate.source_file_id
        record.source_entity_id = candidate.source_entity_id
        record.source_relationship_id = candidate.source_relationship_id
        record.updated_at = now
        await db.flush()

        await db.execute(delete(SemanticMemoryEvidence).where(SemanticMemoryEvidence.memory_id == record.id))
        for evidence in candidate.evidence:
            db.add(SemanticMemoryEvidence(
                id=str(uuid.uuid4()),
                memory_id=record.id,
                file_id=evidence.file_id or candidate.source_file_id,
                source_type=evidence.source_type[:40],
                source_id=evidence.source_id[:80] if evidence.source_id else None,
                evidence_key=evidence.evidence_key[:120],
                evidence_value=evidence.evidence_value,
                confidence_score=evidence.confidence_score,
            ))
        await rebuild_memory_indexes(db, record)
        upserted += 1

    deprecated = 0
    for row in existing_rows:
        if row.canonical_key in seen_keys:
            continue
        row.status = "deprecated"
        row.governance_status = "deprecated"
        row.updated_at = now
        await rebuild_memory_indexes(db, row)
        deprecated += 1

    await db.commit()
    duration = _ms(start)
    ingest_logger.info(
        "semantic_memory_upserted",
        file_id=file_id,
        records=upserted,
        deprecated=deprecated,
        duration_ms=duration,
    )
    return {"file_id": file_id, "records": upserted, "deprecated": deprecated, "duration_ms": duration}
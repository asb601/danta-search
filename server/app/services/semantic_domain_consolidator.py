"""Governed cross-file semantic domain consolidation.

SemanticMemoryRecord rows are file-level evidence. This service consolidates
that evidence into sparse tenant-specific domains that can guide retrieval,
workflow routing, lifecycle reasoning, KPI consistency, and ambiguity handling.

It never authorizes execution, creates joins, or emits SQL.
"""
from __future__ import annotations

import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import ingest_logger
from app.models.semantic_memory import (
    SemanticDomainCluster,
    SemanticDomainConflict,
    SemanticDomainEvidence,
    SemanticDomainFileIndex,
    SemanticDomainTermIndex,
    SemanticMemoryRecord,
)
from app.services.semantic_memory_normalizer import bounded_score, canonical_key, slugify, terms_from_text

DOMAIN_TYPES = frozenset({
    "workflow_domain",
    "lifecycle_domain",
    "kpi_domain",
    "business_capability",
    "semantic_tag",
})

_WORKFLOW_MEMORY_TYPES = frozenset({"relationship", "capability", "grain", "null_semantic"})
_LIFECYCLE_MEMORY_TYPES = frozenset({"temporal", "grain", "null_semantic", "relationship"})
_KPI_MEMORY_TYPES = frozenset({"metric"})
_CAPABILITY_MEMORY_TYPES = frozenset({"capability"})
_WORKFLOW_BEHAVIORS = frozenset({"join_guidance", "retrieval_guidance", "plan_validation", "entity_resolution"})
_LIFECYCLE_BEHAVIORS = frozenset({"lifecycle_scope", "time_filter", "fanout_guard", "validation_required"})
_KPI_BEHAVIORS = frozenset({"aggregation", "kpi_contract"})
_GENERIC_TERMS = frozenset({
    "custom",
    "column",
    "dataset",
    "field",
    "file",
    "grain",
    "guidance",
    "join",
    "key",
    "memory",
    "record",
    "relationship",
    "role",
    "source",
    "table",
    "validation",
})


@dataclass(frozen=True)
class MemorySignal:
    memory_id: str
    file_id: str
    memory_type: str
    title: str
    terms: list[str]
    behaviors: list[str]
    role_kind: str | None
    confidence_score: float
    authority_score: float
    governance_status: str
    updated_at: datetime | None


@dataclass
class DomainAccumulator:
    domain_type: str
    domain_key: str
    term: str
    signals: list[MemorySignal] = field(default_factory=list)

    def add(self, signal: MemorySignal) -> None:
        self.signals.append(signal)


@dataclass(frozen=True)
class DomainCandidate:
    domain_type: str
    domain_key: str
    title: str
    summary: str
    terms: list[str]
    workflow_terms: list[str]
    lifecycle_terms: list[str]
    kpi_terms: list[str]
    synonym_terms: list[str]
    file_ids: list[str]
    memory_ids: list[str]
    evidence_count: int
    confidence_score: float
    authority_score: float
    drift_score: float
    governance_status: str
    conflict_summary: dict[str, Any]
    file_scores: dict[str, float]
    file_terms: dict[str, list[str]]
    memory_file_ids: dict[str, str]
    file_evidence_counts: dict[str, int]
    evidence_weights: dict[str, float]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _signal_from_record(record: SemanticMemoryRecord) -> MemorySignal | None:
    if not record.source_file_id:
        return None
    terms = _domain_terms(record)
    if not terms:
        return None
    dimensions = record.dimensions or {}
    return MemorySignal(
        memory_id=record.id,
        file_id=record.source_file_id,
        memory_type=record.memory_type,
        title=record.title,
        terms=terms,
        behaviors=terms_from_text(record.behaviors or []),
        role_kind=str(dimensions.get("role_kind") or "") or None,
        confidence_score=float(record.confidence_score or 0.0),
        authority_score=float(record.authority_score or 0.0),
        governance_status=record.governance_status,
        updated_at=record.updated_at,
    )


def _domain_terms(record: SemanticMemoryRecord) -> list[str]:
    raw_terms = terms_from_text(record.normalized_terms, record.title, record.summary, record.dimensions)
    terms: list[str] = []
    for term in raw_terms:
        if term in _GENERIC_TERMS or len(term) < 3:
            continue
        terms.append(term)
        if len(terms) >= int(get_settings().SEMANTIC_DOMAIN_MAX_TERMS):
            break
    return terms


def _domain_types_for_signal(signal: MemorySignal) -> list[str]:
    behaviors = set(signal.behaviors)
    types: list[str] = ["semantic_tag"]
    if signal.memory_type in _WORKFLOW_MEMORY_TYPES or behaviors & _WORKFLOW_BEHAVIORS:
        types.append("workflow_domain")
    if signal.memory_type in _LIFECYCLE_MEMORY_TYPES or behaviors & _LIFECYCLE_BEHAVIORS:
        types.append("lifecycle_domain")
    if signal.memory_type in _KPI_MEMORY_TYPES or behaviors & _KPI_BEHAVIORS:
        types.append("kpi_domain")
    if signal.memory_type in _CAPABILITY_MEMORY_TYPES:
        types.append("business_capability")
    return list(dict.fromkeys(types))


def _add_to_accumulators(accumulators: dict[tuple[str, str], DomainAccumulator], signal: MemorySignal) -> None:
    for domain_type in _domain_types_for_signal(signal):
        for term in signal.terms[:8]:
            key = canonical_key(domain_type, term)
            accumulators.setdefault((domain_type, key), DomainAccumulator(domain_type, key, term)).add(signal)


def build_domain_candidates_from_records(
    records: list[SemanticMemoryRecord],
    *,
    min_files: int | None = None,
    max_clusters: int | None = None,
) -> list[DomainCandidate]:
    settings = get_settings()
    min_files = max(1, int(min_files or settings.SEMANTIC_DOMAIN_MIN_FILES))
    max_clusters = max(1, int(max_clusters or settings.SEMANTIC_DOMAIN_MAX_CLUSTERS))
    signals = [signal for signal in (_signal_from_record(record) for record in records) if signal]
    accumulators: dict[tuple[str, str], DomainAccumulator] = {}
    for signal in signals:
        _add_to_accumulators(accumulators, signal)

    candidates: list[DomainCandidate] = []
    for accumulator in accumulators.values():
        file_ids = sorted({signal.file_id for signal in accumulator.signals})
        if len(file_ids) < min_files:
            continue
        candidates.append(_candidate_from_accumulator(accumulator))

    candidates.sort(
        key=lambda item: (
            item.authority_score,
            item.confidence_score,
            len(item.file_ids),
            item.evidence_count,
        ),
        reverse=True,
    )
    return candidates[:max_clusters]


def _candidate_from_accumulator(accumulator: DomainAccumulator) -> DomainCandidate:
    settings = get_settings()
    signals = accumulator.signals
    memory_ids = list(dict.fromkeys(signal.memory_id for signal in signals))
    file_ids = sorted({signal.file_id for signal in signals})
    term_counts = Counter(term for signal in signals for term in signal.terms)
    terms = [term for term, _ in term_counts.most_common(int(settings.SEMANTIC_DOMAIN_MAX_TERMS))]
    workflow_terms = _terms_for(signals, _WORKFLOW_MEMORY_TYPES, _WORKFLOW_BEHAVIORS)
    lifecycle_terms = _terms_for(signals, _LIFECYCLE_MEMORY_TYPES, _LIFECYCLE_BEHAVIORS)
    kpi_terms = _terms_for(signals, _KPI_MEMORY_TYPES, _KPI_BEHAVIORS)
    synonym_terms = [term for term in terms if term != accumulator.term][:12]

    conflict_summary = _conflict_summary(signals)
    conflict_count = int(conflict_summary.get("count", 0))
    drift_score = bounded_score(conflict_count * float(settings.SEMANTIC_DOMAIN_DECAY_PER_CONFLICT))
    authority_values = [signal.authority_score for signal in signals]
    confidence_values = [signal.confidence_score for signal in signals]
    diversity_bonus = min(0.16, math.log1p(len(file_ids)) * 0.05)
    evidence_bonus = min(0.12, math.log1p(len(memory_ids)) * 0.04)
    confidence = bounded_score(_avg(confidence_values) + evidence_bonus - drift_score)
    authority = bounded_score(_avg(authority_values) + diversity_bonus + evidence_bonus - drift_score)
    governance_status = "active" if authority >= 0.72 and confidence >= 0.55 and conflict_count == 0 else "candidate"
    if authority >= 0.82 and conflict_count <= 1:
        governance_status = "active"

    file_scores, file_terms, file_evidence_counts = _file_scores(signals)
    title_term = accumulator.term.replace("_", " ")
    return DomainCandidate(
        domain_type=accumulator.domain_type,
        domain_key=accumulator.domain_key,
        title=f"{accumulator.domain_type.replace('_', ' ')}: {title_term}",
        summary=f"Cross-file {accumulator.domain_type.replace('_', ' ')} evidence for {title_term} across {len(file_ids)} files.",
        terms=terms,
        workflow_terms=workflow_terms,
        lifecycle_terms=lifecycle_terms,
        kpi_terms=kpi_terms,
        synonym_terms=synonym_terms,
        file_ids=file_ids,
        memory_ids=memory_ids,
        evidence_count=len(signals),
        confidence_score=confidence,
        authority_score=authority,
        drift_score=drift_score,
        governance_status=governance_status,
        conflict_summary=conflict_summary,
        file_scores=file_scores,
        file_terms=file_terms,
        memory_file_ids={signal.memory_id: signal.file_id for signal in signals},
        file_evidence_counts=file_evidence_counts,
        evidence_weights={signal.memory_id: _contribution_weight(signal, drift_score) for signal in signals},
    )


def _terms_for(signals: list[MemorySignal], memory_types: frozenset[str], behaviors: frozenset[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for signal in signals:
        if signal.memory_type in memory_types or set(signal.behaviors) & behaviors:
            counts.update(signal.terms)
    return [term for term, _ in counts.most_common(12)]


def _conflict_summary(signals: list[MemorySignal]) -> dict[str, Any]:
    role_kind_by_term: dict[str, set[str]] = defaultdict(set)
    type_by_term: dict[str, set[str]] = defaultdict(set)
    files_by_term: dict[str, set[str]] = defaultdict(set)
    for signal in signals:
        for term in signal.terms[:10]:
            type_by_term[term].add(signal.memory_type)
            files_by_term[term].add(signal.file_id)
            if signal.role_kind:
                role_kind_by_term[term].add(signal.role_kind)

    conflicts: list[dict[str, Any]] = []
    for term, kinds in role_kind_by_term.items():
        if len(kinds) > 1:
            conflicts.append({"type": "role_kind_conflict", "term": term, "role_kinds": sorted(kinds)})
    for term, memory_types in type_by_term.items():
        if "metric" in memory_types and ({"dimension", "relationship", "temporal"} & memory_types):
            conflicts.append({"type": "metric_dimension_ambiguity", "term": term, "memory_types": sorted(memory_types)})

    file_distribution = {term: len(files) for term, files in files_by_term.items()}
    if file_distribution:
        max_files = max(file_distribution.values())
        broad_terms = [term for term, count in file_distribution.items() if count >= max_files and count > 8]
        if broad_terms:
            conflicts.append({"type": "overbroad_domain_term", "terms": broad_terms[:5]})

    return {
        "count": len(conflicts),
        "conflicts": conflicts[:8],
    }


def _file_scores(signals: list[MemorySignal]) -> tuple[dict[str, float], dict[str, list[str]], dict[str, int]]:
    by_file: dict[str, list[MemorySignal]] = defaultdict(list)
    for signal in signals:
        by_file[signal.file_id].append(signal)
    scores: dict[str, float] = {}
    file_terms: dict[str, list[str]] = {}
    file_evidence_counts: dict[str, int] = {}
    for file_id, file_signals in by_file.items():
        file_evidence_counts[file_id] = len(file_signals)
        scores[file_id] = bounded_score(
            _avg([signal.authority_score for signal in file_signals]) * 0.55
            + _avg([signal.confidence_score for signal in file_signals]) * 0.35
            + min(0.1, len(file_signals) * 0.02)
        )
        counts = Counter(term for signal in file_signals for term in signal.terms)
        file_terms[file_id] = [term for term, _ in counts.most_common(12)]
    return scores, file_terms, file_evidence_counts


def _contribution_weight(signal: MemorySignal, drift_score: float) -> float:
    governance_bonus = 0.08 if signal.governance_status == "active" else 0.0
    return bounded_score(signal.authority_score * 0.5 + signal.confidence_score * 0.42 + governance_bonus - drift_score)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


async def consolidate_semantic_domains_for_container(container_id: str, db: AsyncSession) -> dict[str, Any]:
    start = time.perf_counter()
    settings = get_settings()
    records = (
        await db.execute(
            select(SemanticMemoryRecord)
            .where(
                SemanticMemoryRecord.container_id == container_id,
                SemanticMemoryRecord.status == "active",
                SemanticMemoryRecord.governance_status.in_(["active", "candidate"]),
                SemanticMemoryRecord.source_file_id.isnot(None),
            )
            .order_by(SemanticMemoryRecord.authority_score.desc(), SemanticMemoryRecord.confidence_score.desc())
            .limit(int(settings.SEMANTIC_DOMAIN_MAX_SOURCE_MEMORIES))
        )
    ).scalars().all()

    candidates = build_domain_candidates_from_records(list(records))
    existing = (
        await db.execute(select(SemanticDomainCluster).where(SemanticDomainCluster.container_id == container_id))
    ).scalars().all()
    existing_by_key = {(row.domain_type, row.domain_key): row for row in existing}
    seen_keys: set[tuple[str, str]] = set()
    upserted = 0
    conflicts = 0
    now = datetime.now(timezone.utc)

    for candidate in candidates:
        seen_keys.add((candidate.domain_type, candidate.domain_key))
        cluster = existing_by_key.get((candidate.domain_type, candidate.domain_key))
        if not cluster:
            cluster = SemanticDomainCluster(id=str(uuid.uuid4()), container_id=container_id)
            db.add(cluster)
        _assign_cluster(cluster, candidate, now)
        await db.flush()
        await _replace_cluster_indexes(db, cluster, candidate)
        upserted += 1
        conflicts += int(candidate.conflict_summary.get("count", 0))

    stale = 0
    for cluster in existing:
        if (cluster.domain_type, cluster.domain_key) in seen_keys:
            continue
        cluster.status = "deprecated"
        cluster.governance_status = "deprecated"
        cluster.updated_at = now
        await _clear_cluster_indexes(db, cluster.id)
        stale += 1

    await db.commit()
    duration = _ms(start)
    ingest_logger.info(
        "semantic_domains_consolidated",
        container_id=container_id,
        source_memories=len(records),
        clusters=upserted,
        deprecated=stale,
        conflicts=conflicts,
        duration_ms=duration,
    )
    return {
        "container_id": container_id,
        "source_memories": len(records),
        "clusters": upserted,
        "deprecated": stale,
        "conflicts": conflicts,
        "duration_ms": duration,
    }


def _assign_cluster(cluster: SemanticDomainCluster, candidate: DomainCandidate, now: datetime) -> None:
    cluster.domain_type = candidate.domain_type
    cluster.domain_key = candidate.domain_key
    cluster.title = candidate.title
    cluster.summary = candidate.summary
    cluster.normalized_terms = candidate.terms
    cluster.workflow_terms = candidate.workflow_terms
    cluster.lifecycle_terms = candidate.lifecycle_terms
    cluster.kpi_terms = candidate.kpi_terms
    cluster.synonym_terms = candidate.synonym_terms
    cluster.contributor_file_ids = candidate.file_ids
    cluster.contributor_memory_ids = candidate.memory_ids
    cluster.evidence_count = candidate.evidence_count
    cluster.conflict_count = int(candidate.conflict_summary.get("count", 0))
    cluster.conflict_summary = candidate.conflict_summary
    cluster.confidence_score = candidate.confidence_score
    cluster.authority_score = candidate.authority_score
    cluster.drift_score = candidate.drift_score
    cluster.governance_status = candidate.governance_status
    cluster.status = "active"
    cluster.updated_at = now


async def _clear_cluster_indexes(db: AsyncSession, domain_id: str) -> None:
    await db.execute(delete(SemanticDomainEvidence).where(SemanticDomainEvidence.domain_id == domain_id))
    await db.execute(delete(SemanticDomainFileIndex).where(SemanticDomainFileIndex.domain_id == domain_id))
    await db.execute(delete(SemanticDomainTermIndex).where(SemanticDomainTermIndex.domain_id == domain_id))
    await db.execute(delete(SemanticDomainConflict).where(SemanticDomainConflict.domain_id == domain_id))


async def _replace_cluster_indexes(db: AsyncSession, cluster: SemanticDomainCluster, candidate: DomainCandidate) -> None:
    await _clear_cluster_indexes(db, cluster.id)
    for memory_id in candidate.memory_ids:
        file_id = candidate.memory_file_ids.get(memory_id)
        db.add(SemanticDomainEvidence(
            id=str(uuid.uuid4()),
            domain_id=cluster.id,
            memory_id=memory_id,
            file_id=file_id,
            evidence_type=candidate.domain_type,
            evidence_key=candidate.domain_key,
            evidence_terms=candidate.terms[:16],
            contribution_weight=candidate.evidence_weights.get(memory_id, 0.0),
            confidence_score=candidate.confidence_score,
            authority_score=candidate.authority_score,
            decay_factor=max(0.0, 1.0 - candidate.drift_score),
        ))
    for file_id, score in candidate.file_scores.items():
        db.add(SemanticDomainFileIndex(
            id=str(uuid.uuid4()),
            container_id=cluster.container_id,
            domain_id=cluster.id,
            file_id=file_id,
            domain_type=candidate.domain_type,
            score=score,
            terms=candidate.file_terms.get(file_id, [])[:12],
            evidence_count=candidate.file_evidence_counts.get(file_id, 0),
        ))
    for term in candidate.terms[: int(get_settings().SEMANTIC_DOMAIN_MAX_TERMS)]:
        db.add(SemanticDomainTermIndex(
            id=str(uuid.uuid4()),
            container_id=cluster.container_id,
            domain_id=cluster.id,
            term=term,
            domain_type=candidate.domain_type,
            weight=1.0 + min(1.0, candidate.evidence_count * 0.05),
            status="active" if cluster.status == "active" else "inactive",
        ))
    for conflict in (candidate.conflict_summary.get("conflicts") or [])[:8]:
        key = slugify(str(conflict.get("term") or conflict.get("terms") or conflict.get("type") or "conflict"))
        db.add(SemanticDomainConflict(
            id=str(uuid.uuid4()),
            container_id=cluster.container_id,
            domain_id=cluster.id,
            conflict_type=str(conflict.get("type") or "semantic_conflict")[:40],
            conflict_key=key[:255],
            severity="warning",
            file_ids=candidate.file_ids[:20],
            memory_ids=candidate.memory_ids[:20],
            details=conflict,
            resolution_status="open",
        ))
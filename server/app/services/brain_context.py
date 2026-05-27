"""Bounded BrainContext objects used by retrieval, prompt, and validators."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class BrainMemoryBrief:
    id: str
    memory_type: str
    title: str
    summary: str | None
    terms: list[str]
    behaviors: list[str]
    confidence_score: float
    authority_score: float
    source_file_id: str | None
    score: float

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "id": self.id[:8],
            "type": self.memory_type,
            "title": self.title[:120],
            "confidence": round(self.confidence_score, 3),
            "authority": round(self.authority_score, 3),
            "score": round(self.score, 3),
            "file_id": self.source_file_id[:8] if self.source_file_id else None,
        }


@dataclass(frozen=True)
class BrainDomainBrief:
    id: str
    domain_type: str
    domain_key: str
    title: str
    terms: list[str]
    workflow_terms: list[str]
    lifecycle_terms: list[str]
    kpi_terms: list[str]
    synonym_terms: list[str]
    contributor_file_ids: list[str]
    confidence_score: float
    authority_score: float
    drift_score: float
    conflict_count: int
    score: float
    file_scores: dict[str, float] = field(default_factory=dict)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "id": self.id[:8],
            "type": self.domain_type,
            "title": self.title[:120],
            "files": len(self.contributor_file_ids),
            "confidence": round(self.confidence_score, 3),
            "authority": round(self.authority_score, 3),
            "drift": round(self.drift_score, 3),
            "conflicts": self.conflict_count,
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class RetrievalGuidance:
    anchor_file_ids: list[str] = field(default_factory=list)
    domain_anchor_file_ids: list[str] = field(default_factory=list)
    preferred_terms: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)
    workflow_terms: list[str] = field(default_factory=list)
    lifecycle_terms: list[str] = field(default_factory=list)
    kpi_terms: list[str] = field(default_factory=list)
    authority_by_file_id: dict[str, float] = field(default_factory=dict)
    domain_authority_by_file_id: dict[str, float] = field(default_factory=dict)
    ambiguity_flags: list[str] = field(default_factory=list)
    topology_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_file_ids": self.anchor_file_ids,
            "domain_anchor_file_ids": self.domain_anchor_file_ids,
            "preferred_terms": self.preferred_terms,
            "domain_terms": self.domain_terms,
            "workflow_terms": self.workflow_terms,
            "lifecycle_terms": self.lifecycle_terms,
            "kpi_terms": self.kpi_terms,
            "authority_by_file_id": self.authority_by_file_id,
            "domain_authority_by_file_id": self.domain_authority_by_file_id,
            "ambiguity_flags": self.ambiguity_flags,
            "topology_hints": self.topology_hints,
        }


@dataclass(frozen=True)
class ExecutionEnvelope:
    memory_ids: list[str] = field(default_factory=list)
    domain_ids: list[str] = field(default_factory=list)
    anchor_file_ids: list[str] = field(default_factory=list)
    shortlist_file_ids: list[str] = field(default_factory=list)
    approved_join_count: int = 0
    execution_mode: str | None = None
    ambiguity_flags: list[str] = field(default_factory=list)
    authority_floor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_ids": self.memory_ids,
            "domain_ids": self.domain_ids,
            "anchor_file_ids": self.anchor_file_ids,
            "shortlist_file_ids": self.shortlist_file_ids,
            "approved_join_count": self.approved_join_count,
            "execution_mode": self.execution_mode,
            "ambiguity_flags": self.ambiguity_flags,
            "authority_floor": round(self.authority_floor, 3),
        }


@dataclass(frozen=True)
class BrainContext:
    records: list[BrainMemoryBrief] = field(default_factory=list)
    domains: list[BrainDomainBrief] = field(default_factory=list)
    retrieval_guidance: RetrievalGuidance = field(default_factory=RetrievalGuidance)
    execution_envelope: ExecutionEnvelope = field(default_factory=ExecutionEnvelope)
    token_estimate: int = 0
    caps: dict[str, int] = field(default_factory=dict)

    def with_execution_scope(
        self,
        *,
        shortlist_file_ids: list[str],
        approved_join_count: int,
        execution_mode: str | None,
    ) -> "BrainContext":
        envelope = replace(
            self.execution_envelope,
            shortlist_file_ids=shortlist_file_ids,
            approved_join_count=approved_join_count,
            execution_mode=execution_mode,
        )
        return replace(self, execution_envelope=envelope)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "record_count": len(self.records),
            "domain_count": len(self.domains),
            "records": [record.to_trace_dict() for record in self.records[:10]],
            "domains": [domain.to_trace_dict() for domain in self.domains[:10]],
            "retrieval_guidance": self.retrieval_guidance.to_dict(),
            "execution_envelope": self.execution_envelope.to_dict(),
            "token_estimate": self.token_estimate,
            "caps": self.caps,
        }
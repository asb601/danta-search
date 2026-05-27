"""Governance scoring for semantic memory records."""
from __future__ import annotations

from dataclasses import dataclass

from app.services.semantic_memory_normalizer import bounded_score


@dataclass(frozen=True)
class GovernanceDecision:
    governance_status: str
    authority_score: float
    reason: str


def decide_governance_status(
    *,
    confidence_score: float,
    evidence_count: int,
    source: str,
    approval_status: str | None = None,
    risk_reason: str | None = None,
) -> GovernanceDecision:
    confidence = bounded_score(confidence_score)
    approved = approval_status == "approved"
    source_weight = 0.1 if source in {"semantic_layer", "relationship_graph"} else 0.0
    evidence_weight = min(0.2, max(0, evidence_count) * 0.05)
    approval_weight = 0.15 if approved else 0.0
    risk_penalty = 0.2 if risk_reason else 0.0
    authority = bounded_score(confidence + source_weight + evidence_weight + approval_weight - risk_penalty)

    if risk_reason and not approved:
        return GovernanceDecision("candidate", authority, "risk_requires_review")
    if authority >= 0.78 and evidence_count > 0:
        return GovernanceDecision("active", authority, "sufficient_evidence")
    return GovernanceDecision("candidate", authority, "needs_more_evidence")
"""SemanticMemory / BrainContext regression checks.

Run with:
    PYTHONPATH=server python3 -m testing._semantic_memory_check
"""
from __future__ import annotations

from app.models.semantic_memory import SemanticMemoryRecord
from app.services.brain_context import BrainContext, BrainDomainBrief, BrainMemoryBrief, ExecutionEnvelope, RetrievalGuidance
from app.services.brain_prompt_slice import render_brain_context_prompt
from app.services.plan_ir import PlanIR, PlanStage, validate_plan_ir
from app.services.semantic_domain_consolidator import build_domain_candidates_from_records
from app.services.semantic_memory_governance import decide_governance_status
from app.services.semantic_memory_normalizer import MemoryCandidate, canonical_key, normalise_candidate


def test_memory_normalization_and_governance() -> None:
    decision = decide_governance_status(
        confidence_score=0.82,
        evidence_count=2,
        source="semantic_layer",
    )
    assert decision.governance_status == "active"
    candidate = MemoryCandidate(
        container_id="container-1",
        memory_type="metric",
        canonical_key=canonical_key("metric", "container-1", "Amount", "file-1"),
        title="Amount",
        summary="An additive measure available on the dataset.",
        normalized_terms=["Amount", "amount", "additive-measure"],
        behaviors=["aggregation", "kpi_contract"],
        confidence_score=0.8,
        authority_score=decision.authority_score,
        governance_status=decision.governance_status,
        source_file_id="file-1",
    )
    normalized = normalise_candidate(candidate)
    assert normalized is not None
    assert normalized.governance_status == "active"
    assert "amount" in normalized.normalized_terms
    assert len(normalized.normalized_terms) == len(set(normalized.normalized_terms))


def test_brain_prompt_is_bounded_guidance() -> None:
    context = BrainContext(
        domains=[
            BrainDomainBrief(
                id="domain-1",
                domain_type="workflow_domain",
                domain_key="workflow_domain:invoice",
                title="workflow domain: invoice",
                terms=["invoice", "approval", "payment"],
                workflow_terms=["invoice", "approval"],
                lifecycle_terms=["payment"],
                kpi_terms=[],
                synonym_terms=["settlement"],
                contributor_file_ids=["file-1", "file-2"],
                confidence_score=0.78,
                authority_score=0.82,
                drift_score=0.0,
                conflict_count=0,
                score=0.72,
                file_scores={"file-1": 0.8, "file-2": 0.7},
            )
        ],
        records=[
            BrainMemoryBrief(
                id="memory-1",
                memory_type="metric",
                title="Amount",
                summary="Additive amount measure for aggregation validation.",
                terms=["amount", "additive", "measure"],
                behaviors=["aggregation", "kpi_contract"],
                confidence_score=0.84,
                authority_score=0.9,
                source_file_id="file-1",
                score=0.7,
            )
        ],
        retrieval_guidance=RetrievalGuidance(
            anchor_file_ids=["file-1"],
            domain_anchor_file_ids=["file-2"],
            preferred_terms=["amount"],
            domain_terms=["invoice"],
            workflow_terms=["approval"],
            lifecycle_terms=["payment"],
            authority_by_file_id={"file-1": 0.9},
            domain_authority_by_file_id={"file-2": 0.8},
            ambiguity_flags=["ambiguous_memory:amount"],
            topology_hints=["workflow_domain:invoice:files=2"],
        ),
        execution_envelope=ExecutionEnvelope(
            memory_ids=["memory-1"],
            domain_ids=["domain-1"],
            anchor_file_ids=["file-1"],
            shortlist_file_ids=["file-1"],
            approved_join_count=0,
            execution_mode="independent_analyses",
            ambiguity_flags=["ambiguous_memory:amount"],
            authority_floor=0.9,
        ),
        token_estimate=40,
        caps={"max_records": 4},
    )
    prompt = render_brain_context_prompt(context)
    assert "GOVERNED SEMANTIC MEMORY" in prompt
    assert "does not authorize" in prompt
    assert "Amount" in prompt
    assert "workflow domain" in prompt
    assert "workflow topology hints" in prompt
    assert "ambiguous_memory:amount" in prompt


def _record(
    memory_id: str,
    file_id: str,
    memory_type: str,
    title: str,
    terms: list[str],
    behaviors: list[str],
    role_kind: str | None = None,
) -> SemanticMemoryRecord:
    return SemanticMemoryRecord(
        id=memory_id,
        container_id="container-1",
        memory_type=memory_type,
        canonical_key=f"{memory_type}:{memory_id}",
        title=title,
        normalized_terms=terms,
        behaviors=behaviors,
        dimensions={"role_kind": role_kind} if role_kind else {},
        confidence_score=0.82,
        authority_score=0.86,
        governance_status="active",
        status="active",
        source_file_id=file_id,
    )


def test_cross_file_domain_consolidation() -> None:
    records = [
        _record("m1", "file-po", "relationship", "purchase order invoice approval", ["purchase_order", "invoice", "approval"], ["join_guidance"], "reference_key"),
        _record("m2", "file-ap", "capability", "invoice approval and settlement", ["invoice", "approval", "settlement"], ["retrieval_guidance"]),
        _record("m3", "file-pay", "temporal", "payment lifecycle date", ["payment", "invoice", "lifecycle"], ["lifecycle_scope"]),
        _record("m4", "file-ap", "metric", "invoice amount", ["invoice", "amount"], ["aggregation", "kpi_contract"]),
        _record("m5", "file-pay", "metric", "payment amount", ["payment", "amount"], ["aggregation", "kpi_contract"]),
    ]
    domains = build_domain_candidates_from_records(records, min_files=2, max_clusters=20)
    assert any(domain.domain_type == "workflow_domain" and "invoice" in domain.terms for domain in domains)
    assert any(domain.domain_type == "lifecycle_domain" and "invoice" in domain.terms and len(domain.file_ids) >= 2 for domain in domains)
    assert any(domain.domain_type == "kpi_domain" and "amount" in domain.terms for domain in domains)
    assert all(len(domain.file_ids) >= 2 for domain in domains)


def test_plan_ir_validator_blocks_outside_envelope() -> None:
    plan = PlanIR(
        id="plan-1",
        stages=[PlanStage(id="stage_1", operation="detail", file_ids=["file-2"])],
    )
    result = validate_plan_ir(
        plan,
        ExecutionEnvelope(shortlist_file_ids=["file-1"]),
    )
    assert result.ok is False
    assert any(issue.code == "file_outside_envelope" for issue in result.issues)


def main() -> None:
    test_memory_normalization_and_governance()
    test_brain_prompt_is_bounded_guidance()
    test_cross_file_domain_consolidation()
    test_plan_ir_validator_blocks_outside_envelope()
    print("PASS semantic memory, BrainContext, and Plan IR checks")


if __name__ == "__main__":
    main()
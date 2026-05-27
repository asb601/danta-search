"""SemanticMemory / BrainContext regression checks.

Run with:
    PYTHONPATH=server python3 -m testing._semantic_memory_check
"""
from __future__ import annotations

from app.services.brain_context import BrainContext, BrainMemoryBrief, ExecutionEnvelope, RetrievalGuidance
from app.services.brain_prompt_slice import render_brain_context_prompt
from app.services.plan_ir import PlanIR, PlanStage, validate_plan_ir
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
            preferred_terms=["amount"],
            authority_by_file_id={"file-1": 0.9},
            ambiguity_flags=["ambiguous_memory:amount"],
        ),
        execution_envelope=ExecutionEnvelope(
            memory_ids=["memory-1"],
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
    assert "ambiguous_memory:amount" in prompt


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
    test_plan_ir_validator_blocks_outside_envelope()
    print("PASS semantic memory, BrainContext, and Plan IR checks")


if __name__ == "__main__":
    main()
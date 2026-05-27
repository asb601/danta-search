"""Prompt compaction regression checks.

Run with:
    PYTHONPATH=server python3 -m testing._prompt_compaction_check
"""
from __future__ import annotations

from app.agent.prompts.prompt_builder import (
    _SYSTEM_PROMPT_TOKEN_BUDGET,
    build_system_prompt,
)
from app.core.token_counter import count_tokens
from app.services.execution_strategy import ClusterPlan, ExecutionStrategy
from app.services.sql_context_builder import ApprovedJoin, SQLContext


def _synthetic_catalog(size: int = 18) -> tuple[list[dict], dict[str, str]]:
    catalog: list[dict] = []
    parquet_paths: dict[str, str] = {}
    for idx in range(1, size + 1):
        blob = f"erp/domain/file_{idx:02d}_purchase_invoice_receiving_workflow.csv"
        parquet_paths[blob] = f"parquet/file_{idx:02d}.parquet"
        catalog.append({
            "file_id": f"fid-{idx:02d}",
            "blob_path": blob,
            "ai_description": (
                "This file is the PRIMARY source for a verbose workflow diagnostic description "
                "covering purchase order, invoice, receiving, approval, payment, reconciliation, "
                "temporal eligibility, authority, and exception handling. " * 3
            ),
            "key_dimensions": [f"dimension_{j}" for j in range(12)],
            "key_metrics": [f"metric_{j}" for j in range(12)],
            "column_names": [f"COL_{j:03d}" for j in range(90)],
            "column_stats": {f"FISCAL_YEAR_{j}": {"dtype": "numeric", "min": 2018, "max": 2026} for j in range(8)},
            "date_range_start": "2018-01-01",
            "date_range_end": "2026-05-27",
        })
    return catalog, parquet_paths


def _synthetic_sql_context_note() -> str:
    joins = [
        ApprovedJoin(
            left_file_id=f"fid-{idx:02d}",
            right_file_id=f"fid-{idx + 1:02d}",
            left_table=f"FILE_{idx}",
            right_table=f"FILE_{idx + 1}",
            left_col="PO_ID",
            right_col="PO_ID",
            relationship_type="approved_workflow_join",
            confidence=0.92,
        )
        for idx in range(1, 18)
    ]
    ctx = SQLContext(
        approved_joins=joins,
        column_bindings={
            f"business_role_{idx:02d}": [f"FILE_{(idx % 18) + 1}.COL_{idx:03d}", f"FILE_{((idx + 3) % 18) + 1}.COL_{idx + 1:03d}"]
            for idx in range(60)
        },
        date_columns={
            f"fiscal_period_{idx:02d}": [f"FILE_{(idx % 18) + 1}.DATE_{idx:03d}"]
            for idx in range(20)
        },
    )
    strategy = ExecutionStrategy(
        mode="multi_cluster",
        clusters=[
            ClusterPlan(files=["FILE_1", "FILE_2"], file_ids=["fid-01", "fid-02"], strategy="joined_sql"),
            ClusterPlan(files=["FILE_3"], file_ids=["fid-03"], strategy="standalone"),
        ],
    )
    return "\n\n".join([ctx.to_prompt_section(), strategy.to_prompt_section()])


def test_prompt_compaction_budget_and_observability_sections() -> None:
    catalog, parquet_paths = _synthetic_catalog()
    legacy_observability_note = "\n".join([
        "--- QUERY-TIME WORKFLOW ASSEMBLY ---",
        "workflow_query: true; tasks: 9; selected_candidates: 120; rejected_candidates: 80",
        "CANDIDATE WORKFLOW DECISIONS:",
        *[
            f"  - file_{idx}.csv: selected; score=0.9; task=task_{idx}; temporal=in_window; authority=transaction:0.90; reasons=diagnostic"
            for idx in range(120)
        ],
        "WORKFLOW EXPANSION CANDIDATES:",
        *[
            f"  - candidate_{idx}.csv covers transaction:domain_{idx} [confidence:0.8; evidence:diagnostic]"
            for idx in range(80)
        ],
        "REACHABLE JOIN PATHS (require an intermediate table not yet in context):",
        *[
            f"  FILE_{idx} -> FILE_{idx + 1} via BRIDGE_{idx} [confidence:0.8]"
            for idx in range(80)
        ],
        "---",
    ])
    prompt = build_system_prompt(
        catalog=catalog,
        parquet_paths_all=parquet_paths,
        parquet_blob_path=None,
        container_name="test",
        sample_rows_by_blob={blob: [{"COL_001": "value"}] for blob in list(parquet_paths)[:8]},
        conversation_context=("Prior turn with long diagnostic context. " * 120),
        total_file_count=42,
        sql_context_note=_synthetic_sql_context_note(),
        workflow_topology_note=legacy_observability_note,
    )

    tokens = count_tokens(prompt, "gpt-4o-mini")
    assert tokens <= _SYSTEM_PROMPT_TOKEN_BUDGET, tokens
    assert "CANDIDATE WORKFLOW DECISIONS" not in prompt
    assert "selected; score" not in prompt
    assert "WORKFLOW EXPANSION CANDIDATES" not in prompt
    assert "REACHABLE JOIN PATHS" not in prompt
    assert "Required logical tables in the current shortlist" in prompt
    assert "additional approved join(s) omitted" in prompt
    assert "semantic role group(s) omitted" in prompt


def main() -> None:
    test_prompt_compaction_budget_and_observability_sections()
    print("PASS prompt compaction budget and observability isolation")


if __name__ == "__main__":
    main()

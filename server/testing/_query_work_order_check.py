"""Query work-order and discovery evidence regression checks.

Usage:
  cd server && source .venv/bin/activate && python3 -m testing._query_work_order_check
"""
from __future__ import annotations

import os
import sys

_server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

from app.services.business_intent_planner import BusinessIntentPlan, split_entity_terms
from app.services.query_work_order import build_query_work_order
from app.services.source_evidence_selector import build_discovery_candidate_evidence


def _entry(file_id: str, blob_path: str, columns: list[str], description: str) -> dict:
    return {
        "file_id": file_id,
        "blob_path": blob_path,
        "column_names": columns,
        "columns_info": [{"name": col} for col in columns],
        "column_semantic_roles": {},
        "ai_description": description,
        "good_for": [],
        "key_metrics": [],
        "key_dimensions": [],
    }


def _po_plan(query: str) -> BusinessIntentPlan:
    entities = [
        "po_details",
        "year",
        "current_status",
        "pending_approvals",
        "delivery_status",
        "invoice_matching_issues",
        "recommended_next_actions",
    ]
    constraints = {"date_range": "2025"}
    source_terms, output_terms, filter_terms = split_entity_terms(query, entities, constraints)
    return BusinessIntentPlan(
        intent="complex_multi_step",
        entities=entities,
        behaviors=["time_filtered", "multi_step"],
        constraints=constraints,
        confidence=0.7,
        source_anchor_terms=source_terms,
        output_terms=output_terms,
        filter_terms=filter_terms,
    )


def test_po_work_order_splits_sources_outputs_and_filters() -> None:
    query = (
        "Analyze PO details for year 2025 and summarize: current status; "
        "pending approvals; delivery status; invoice matching issues; recommended next actions"
    )
    plan = _po_plan(query)
    work_order = build_query_work_order(query, plan)

    assert plan.source_anchor_terms == ["po_details"]
    assert "year" in plan.filter_terms
    assert "year" not in plan.source_anchor_terms
    assert set(plan.output_terms) >= {
        "current_status",
        "pending_approvals",
        "delivery_status",
        "invoice_matching_issues",
        "recommended_next_actions",
    }
    assert work_order.must_inspect_before_sql is True
    assert any("po" in variant and "invoice" in variant for variant in work_order.candidate_search_queries)


def test_po_discovery_evidence_finds_related_tables_without_year_pin() -> None:
    query = (
        "Analyze PO details for year 2025 and summarize: current status; "
        "pending approvals; delivery status; invoice matching issues; recommended next actions"
    )
    plan = _po_plan(query)
    work_order = build_query_work_order(query, plan)
    catalog = [
        _entry(
            "po_lines",
            "PO_LINES_ALL",
            ["PO_HEADER_ID", "PO_LINE_ID", "LINE_STATUS", "ITEM_DESCRIPTION"],
            "Purchase order line details and current status",
        ),
        _entry(
            "po_distributions",
            "PO_DISTRIBUTIONS_ALL",
            ["PO_HEADER_ID", "PO_DISTRIBUTION_ID", "DISTRIBUTION_STATUS", "QUANTITY_ORDERED"],
            "Purchase order accounting distributions and approval progress",
        ),
        _entry(
            "ap_invoice_lines",
            "AP_INVOICE_LINES_ALL",
            ["INVOICE_ID", "PO_LINE_ID", "MATCH_STATUS", "LINE_AMOUNT"],
            "Accounts payable invoice line matching issues for purchase orders",
        ),
        _entry(
            "bseg",
            "BSEG",
            ["BELNR", "BUKRS", "GJAHR", "HKONT", "DMBTR"],
            "Accounting document line items by fiscal year",
        ),
    ]

    evidence = build_discovery_candidate_evidence(work_order=work_order, catalog=catalog)
    evidence_by_file = {item.file_id: item for item in evidence}

    assert "po_lines" in evidence_by_file
    assert "po_distributions" in evidence_by_file
    assert "ap_invoice_lines" in evidence_by_file
    assert evidence_by_file["po_lines"].source_anchor_match_count > 0
    assert evidence_by_file["po_distributions"].source_anchor_match_count > 0
    assert evidence_by_file["ap_invoice_lines"].output_match_count > 0
    assert evidence_by_file.get("bseg") is None or evidence_by_file["bseg"].source_anchor_match_count == 0


if __name__ == "__main__":
    checks = [
        test_po_work_order_splits_sources_outputs_and_filters,
        test_po_discovery_evidence_finds_related_tables_without_year_pin,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
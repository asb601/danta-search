"""Execution retrieval gate regression checks.

Usage:
  cd server && python3 -m testing._execution_gate_check
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

_server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

from app.services.execution_retrieval_gate import narrow_catalog_for_execution


@dataclass
class _IntentPlan:
    behaviors: list[str] = field(default_factory=list)


def _entry(
    file_id: str,
    blob_path: str,
    *,
    columns: list[str],
    roles: dict[str, str],
    description: str,
) -> dict:
    return {
        "file_id": file_id,
        "blob_path": blob_path,
        "column_names": columns,
        "column_semantic_roles": roles,
        "ai_description": description,
        "good_for": [],
        "key_metrics": [],
        "key_dimensions": [],
    }


def test_operational_query_suppresses_transformed_extracts() -> None:
    catalog = [
        _entry(
            "derived",
            "po_status_summary.csv",
            columns=["PO_HEADER_ID", "STATUS", "SUMMARY_BUCKET"],
            roles={"STATUS": "custom:attribute:status"},
            description="Curated transformed analytics summary of purchase order status",
        ),
        _entry(
            "po_headers",
            "PO_HEADERS_ALL.csv",
            columns=["PO_HEADER_ID", "APPROVAL_STATUS", "VENDOR_ID"],
            roles={
                "PO_HEADER_ID": "custom:entity_key:purchase_order",
                "APPROVAL_STATUS": "custom:attribute:approval_status",
                "VENDOR_ID": "custom:reference_key:supplier",
            },
            description="Purchase order header workflow approval status",
        ),
        _entry(
            "po_lines",
            "PO_LINES_ALL.csv",
            columns=["PO_LINE_ID", "PO_HEADER_ID", "LINE_STATUS"],
            roles={
                "PO_LINE_ID": "custom:entity_key:purchase_order_line",
                "PO_HEADER_ID": "custom:reference_key:purchase_order",
                "LINE_STATUS": "custom:attribute:status",
            },
            description="Purchase order line operational detail and status",
        ),
    ]

    result = narrow_catalog_for_execution(
        query="Show open purchase order approval status by supplier",
        intent_plan=_IntentPlan(behaviors=["open_items", "detail_rows"]),
        catalog=catalog,
        max_tables=2,
    )

    selected_ids = {entry["file_id"] for entry in result.selected_catalog}
    assert selected_ids == {"po_headers", "po_lines"}
    assert result.transformed_suppressed_count == 1


def test_explicit_reporting_query_can_keep_summary() -> None:
    catalog = [
        _entry(
            "summary",
            "po_status_summary.csv",
            columns=["STATUS", "TOTAL_AMOUNT"],
            roles={"TOTAL_AMOUNT": "custom:additive_measure:amount"},
            description="Curated transformed analytics summary dashboard for procurement KPIs",
        ),
        _entry(
            "po_headers",
            "PO_HEADERS_ALL.csv",
            columns=["PO_HEADER_ID", "APPROVAL_STATUS"],
            roles={"PO_HEADER_ID": "custom:entity_key:purchase_order"},
            description="Purchase order header workflow approval status",
        ),
    ]

    result = narrow_catalog_for_execution(
        query="Use the PO status summary report for procurement KPI trend",
        intent_plan=_IntentPlan(behaviors=[]),
        catalog=catalog,
        max_tables=2,
    )

    selected_ids = {entry["file_id"] for entry in result.selected_catalog}
    assert "summary" in selected_ids
    assert result.transformed_suppressed_count == 0


if __name__ == "__main__":
    checks = [
        test_operational_query_suppresses_transformed_extracts,
        test_explicit_reporting_query_can_keep_summary,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
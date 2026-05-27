"""Workflow cognition regression checks.

Usage:
  cd server && source .venv/bin/activate && python3 -m testing._workflow_cognition_check
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

_server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

from app.services.workflow_cognition import assemble_workflow_cognition, infer_workflow_primitives


@dataclass
class _IntentPlan:
    behaviors: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def _entry(
    file_id: str,
    blob_path: str,
    *,
    columns: list[str],
    roles: dict[str, str],
    description: str = "",
    good_for: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> dict:
    return {
        "file_id": file_id,
        "blob_path": blob_path,
        "column_names": columns,
        "column_semantic_roles": roles,
        "ai_description": description,
        "good_for": good_for or [],
        "key_metrics": [],
        "key_dimensions": [],
        "date_range_start": date_start,
        "date_range_end": date_end,
    }


def _decision(result, name: str):
    return next(decision for decision in result.decisions if decision.blob_path == name)


def test_po_temporal_authority_and_boundary_scoring() -> None:
    catalog = [
        _entry(
            "po_lines",
            "PO_LINES_ALL",
            columns=["PO_HEADER_ID", "PO_LINE_ID", "ITEM_DESCRIPTION", "LINE_STATUS"],
            roles={
                "PO_HEADER_ID": "custom:reference_key:purchase_order",
                "PO_LINE_ID": "custom:entity_key:purchase_order_line",
                "LINE_STATUS": "custom:attribute:po_line_status",
            },
            description="Purchase order line workflow detail and status",
        ),
        _entry(
            "ap_invoices",
            "AP_INVOICES_ALL",
            columns=["INVOICE_ID", "INVOICE_DATE", "APPROVAL_STATUS", "PO_HEADER_ID"],
            roles={
                "INVOICE_ID": "custom:entity_key:invoice",
                "PO_HEADER_ID": "custom:reference_key:purchase_order",
                "INVOICE_DATE": "custom:date:invoice_date",
                "APPROVAL_STATUS": "custom:attribute:approval_status",
            },
            description="Accounts payable invoice approval and invoice matching status",
            date_start="2025-01-01",
            date_end="2025-12-31",
        ),
        _entry(
            "outbound_delivery",
            "05_outbound_delivery",
            columns=["DELIVERY", "DELIVERY_STATUS", "ACTUAL_GOODS_ISSUE_DATE"],
            roles={
                "DELIVERY": "custom:entity_key:delivery",
                "DELIVERY_STATUS": "custom:attribute:delivery_status",
                "ACTUAL_GOODS_ISSUE_DATE": "custom:date:goods_issue_date",
            },
            description="Curated outbound delivery shipment status for sales fulfillment",
            date_start="2026-01-01",
            date_end="2026-12-31",
        ),
    ]
    result = assemble_workflow_cognition(
        query=(
            "Analyze PO details for year 2025 and summarize current status, "
            "pending approvals, delivery status, invoice matching issues, and recommended next actions."
        ),
        intent_plan=_IntentPlan(behaviors=["time_filtered", "open_items", "detail_rows"]),
        current_shortlist=catalog,
        full_catalog=catalog,
        grounding_quality="aggregated:keyword_degraded+semantic_bridge",
    )

    assert result.workflow_query is True
    assert {task.task_id for task in result.tasks} >= {
        "authorization_state",
        "fulfillment_state",
        "invoice_reconciliation",
    }
    outbound = _decision(result, "05_outbound_delivery")
    invoices = _decision(result, "AP_INVOICES_ALL")
    po_lines = _decision(result, "PO_LINES_ALL")

    assert outbound.temporal_eligibility.status == "outside_window"
    assert "temporal_scope_mismatch" in outbound.rejection_reasons
    assert outbound.transactional_authority.source_type == "transformed_analytics"
    assert invoices.score > outbound.score
    assert po_lines.score > outbound.score


def test_cash_receipt_does_not_satisfy_procurement_receiving() -> None:
    catalog = [
        _entry(
            "cash_receipts",
            "AR_CASH_RECEIPTS_ALL",
            columns=["CASH_RECEIPT_ID", "RECEIPT_DATE", "AMOUNT", "CONFIRMED_FLAG"],
            roles={
                "CASH_RECEIPT_ID": "custom:entity_key:cash_receipt",
                "RECEIPT_DATE": "custom:date:receipt_date",
                "AMOUNT": "custom:additive_measure:amount",
                "CONFIRMED_FLAG": "custom:attribute:receipt_status",
            },
            description="Accounts receivable cash receipt payment confirmation",
        ),
        _entry(
            "po_locations",
            "PO_LINE_LOCATIONS_ALL",
            columns=["LINE_LOCATION_ID", "PO_LINE_ID", "NEED_BY_DATE", "QUANTITY_RECEIVED"],
            roles={
                "LINE_LOCATION_ID": "custom:entity_key:purchase_order_schedule",
                "PO_LINE_ID": "custom:reference_key:purchase_order_line",
                "NEED_BY_DATE": "custom:date:need_by_date",
                "QUANTITY_RECEIVED": "custom:additive_measure:quantity_received",
            },
            description="Purchase order receiving progress and schedule status",
        ),
        _entry(
            "invoice_lines",
            "AP_INVOICE_LINES_ALL",
            columns=["INVOICE_ID", "PO_LINE_ID", "AMOUNT", "MATCH_STATUS"],
            roles={
                "INVOICE_ID": "custom:reference_key:invoice",
                "PO_LINE_ID": "custom:reference_key:purchase_order_line",
                "AMOUNT": "custom:additive_measure:amount",
                "MATCH_STATUS": "custom:attribute:matching_status",
            },
            description="Accounts payable invoice line matching and reconciliation",
        ),
    ]
    result = assemble_workflow_cognition(
        query="Analyze delayed fulfillment and uninvoiced receipts",
        intent_plan=_IntentPlan(behaviors=[]),
        current_shortlist=catalog,
        full_catalog=catalog,
        grounding_quality="aggregated:keyword_degraded",
    )

    cash_receipts = _decision(result, "AR_CASH_RECEIPTS_ALL")
    po_locations = _decision(result, "PO_LINE_LOCATIONS_ALL")
    invoice_lines = _decision(result, "AP_INVOICE_LINES_ALL")

    assert "process_boundary_mismatch" in cash_receipts.rejection_reasons
    assert po_locations.score > cash_receipts.score
    assert invoice_lines.score > cash_receipts.score


def test_primitives_distinguish_invoice_approval_from_generic_status() -> None:
    primitive = infer_workflow_primitives(_entry(
        "ap_invoices",
        "AP_INVOICES_ALL",
        columns=["INVOICE_ID", "APPROVAL_STATUS", "INVOICE_DATE"],
        roles={
            "INVOICE_ID": "custom:entity_key:invoice",
            "APPROVAL_STATUS": "custom:attribute:approval_status",
            "INVOICE_DATE": "custom:date:invoice_date",
        },
        description="Accounts payable invoice approval status at invoice header grain",
    ))

    assert "invoice" in primitive.business_objects
    assert "approval" in primitive.process_signals
    assert "accounts_payable" in primitive.operational_domains
    assert primitive.workflow_grain in {"header", "record"}


if __name__ == "__main__":
    checks = [
        test_po_temporal_authority_and_boundary_scoring,
        test_cash_receipt_does_not_satisfy_procurement_receiving,
        test_primitives_distinguish_invoice_approval_from_generic_status,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
"""Schema-grounded SQL validator regression checks.

Usage:
  cd server && source .venv/bin/activate && python3 -m testing._schema_sql_validator_check
"""
from __future__ import annotations

import json
import os
import sys

_server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

import app.agent.tools.sql as sql_tools
from app.services.file_identity import build_file_identity_map
from app.services.promotion_state import build_initial_promotion_state
from app.services.schema_sql_validator import build_schema_index, validate_logical_sql_schema
from app.services.sql_context_builder import ApprovedJoin, SQLContext


def _catalog() -> list[dict]:
    return [
        {
            "file_id": "po-lines",
            "blob_path": "1a7ca399_PO_LINES_ALL.csv",
            "columns_info": [
                {"name": "po_line_id", "type": "int64"},
                {"name": "po_header_id", "type": "int64"},
                {"name": "quantity", "type": "int64"},
                {"name": "unit_price", "type": "float64"},
            ],
        },
        {
            "file_id": "invoice-lines",
            "blob_path": "b03a3fd7_AP_INVOICE_LINES_ALL.csv",
            "columns_info": [
                {"name": "invoice_id", "type": "int64"},
                {"name": "po_header_id", "type": "int64"},
                {"name": "po_line_id", "type": "int64"},
                {"name": "amount", "type": "float64"},
            ],
        },
        {
            "file_id": "gl-headers",
            "blob_path": "e52d207d_GL_JE_HEADERS.csv",
            "columns_info": [
                {"name": "je_header_id", "type": "int64"},
                {"name": "period_name", "type": "string", "top_values": ["JAN-25", "FEB-25"], "distinct_count": 6},
                {"name": "status", "type": "string", "top_values": ["P"], "distinct_count": 1},
            ],
        },
    ]


def _validator_inputs():
    catalog = _catalog()
    identities = build_file_identity_map(catalog, {}, "container")
    schema_index = build_schema_index(catalog, identities)
    return identities, schema_index


def test_scoped_unknown_column_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT status, COUNT(*) AS count
        FROM PO_LINES_ALL
        WHERE po_line_id IN (SELECT po_line_id FROM AP_INVOICE_LINES_ALL)
        GROUP BY status
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "unknown_column"
    assert report.errors[0].table == "PO_LINES_ALL"


def test_nested_scope_valid_columns_pass() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT quantity
        FROM PO_LINES_ALL
        WHERE po_line_id IN (SELECT po_line_id FROM AP_INVOICE_LINES_ALL)
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert report.ok, report.to_error_payload()


def test_exhaustive_unknown_filter_value_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        "SELECT COUNT(*) AS count FROM GL_JE_HEADERS WHERE status = 'Pending Approval'",
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "unknown_filter_value"
    assert report.errors[0].details["known_values"] == ["P"]


def test_partially_unknown_filter_values_warn_only() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        "SELECT COUNT(*) AS count FROM GL_JE_HEADERS WHERE status IN ('P', 'X')",
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert report.ok
    assert report.warnings
    assert report.warnings[0].code == "partially_unknown_filter_values"


def test_having_count_star_zero_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT po_line_id, COUNT(*) AS count
        FROM AP_INVOICE_LINES_ALL
        GROUP BY po_line_id
        HAVING COUNT(*) = 0
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "unsatisfiable_having_count_star_zero"


def test_unverified_in_subquery_key_swap_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT COUNT(*) AS count
        FROM AP_INVOICE_LINES_ALL
        WHERE invoice_id IN (SELECT je_header_id FROM GL_JE_HEADERS)
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "unverified_cross_table_relation"


def test_strong_same_name_in_subquery_key_is_allowed() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT quantity
        FROM PO_LINES_ALL
        WHERE po_line_id IN (SELECT po_line_id FROM AP_INVOICE_LINES_ALL)
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert report.ok, report.to_error_payload()


def test_approved_join_allows_different_column_names() -> None:
    identities, schema_index = _validator_inputs()
    sql_ctx = SQLContext(approved_joins=[ApprovedJoin(
        left_file_id="invoice-lines",
        right_file_id="gl-headers",
        left_table="AP_INVOICE_LINES_ALL",
        right_table="GL_JE_HEADERS",
        left_col="invoice_id",
        right_col="je_header_id",
        relationship_type="approved_test_join",
        confidence=0.99,
    )])
    report = validate_logical_sql_schema(
        """
        SELECT COUNT(*) AS count
        FROM AP_INVOICE_LINES_ALL
        WHERE invoice_id IN (SELECT je_header_id FROM GL_JE_HEADERS)
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
        sql_ctx=sql_ctx,
    )
    assert report.ok, report.to_error_payload()


def test_literal_label_projection_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        "SELECT 'Pending Delivery' AS delivery_status, COUNT(*) AS count FROM PO_LINES_ALL",
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "literal_label_projection"


def test_case_literal_label_projection_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT CASE WHEN status = 'P' THEN 'Pending Approval' ELSE 'Other' END AS status_label,
               COUNT(*) AS count
        FROM GL_JE_HEADERS
        GROUP BY status_label
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "literal_label_projection"


def test_unsupported_aggregate_business_alias_is_blocked() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT po_header_id, COUNT(*) AS pending_invoices
        FROM AP_INVOICE_LINES_ALL
        WHERE po_header_id IS NOT NULL AND amount > 0
        GROUP BY po_header_id
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert not report.ok
    assert report.errors[0].code == "unsupported_metric_alias"
    assert "pending" in report.errors[0].details["missing_tokens"]


def test_supported_neutral_aggregate_alias_is_allowed() -> None:
    identities, schema_index = _validator_inputs()
    report = validate_logical_sql_schema(
        """
        SELECT po_header_id, SUM(quantity) AS total_quantity
        FROM PO_LINES_ALL
        WHERE po_header_id IS NOT NULL
        GROUP BY po_header_id
        """,
        identities,
        schema_index,
        allowed_file_ids=identities.allowed_file_ids(),
    )
    assert report.ok, report.to_error_payload()


def test_run_sql_uses_schema_validation_before_engine() -> None:
    catalog = _catalog()
    identities = build_file_identity_map(catalog, {}, "container")
    schema_index = build_schema_index(catalog, identities)
    store: dict = {
        "promotion_state": build_initial_promotion_state(
            discovery_file_ids=[entry["file_id"] for entry in catalog],
            execution_file_ids=[entry["file_id"] for entry in catalog],
            must_inspect_before_sql=False,
        )
    }
    run_sql = next(tool for tool in sql_tools.build_sql_tools(
        "UseDevelopmentStorage=true",
        "container",
        None,
        store,
        allowed_blob_paths=identities.allowed_physical_uris(),
        file_identities=identities,
        allowed_file_ids=identities.allowed_file_ids(),
        schema_index=schema_index,
    ) if tool.name == "run_sql")

    result = json.loads(run_sql.invoke({"sql": "SELECT status FROM PO_LINES_ALL"}))
    assert result["schema_validation_error"] is True
    assert result["issues"][0]["code"] == "unknown_column"


def main() -> None:
    checks = [
        test_scoped_unknown_column_is_blocked,
        test_nested_scope_valid_columns_pass,
        test_exhaustive_unknown_filter_value_is_blocked,
        test_partially_unknown_filter_values_warn_only,
        test_having_count_star_zero_is_blocked,
        test_unverified_in_subquery_key_swap_is_blocked,
        test_strong_same_name_in_subquery_key_is_allowed,
        test_approved_join_allows_different_column_names,
        test_literal_label_projection_is_blocked,
        test_case_literal_label_projection_is_blocked,
        test_unsupported_aggregate_business_alias_is_blocked,
        test_supported_neutral_aggregate_alias_is_allowed,
        test_run_sql_uses_schema_validation_before_engine,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    main()

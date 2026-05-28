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
                {"name": "po_line_id", "type": "int64"},
                {"name": "amount", "type": "float64"},
            ],
        },
        {
            "file_id": "gl-headers",
            "blob_path": "e52d207d_GL_JE_HEADERS.csv",
            "columns_info": [
                {"name": "invoice_id", "type": "int64"},
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
        test_run_sql_uses_schema_validation_before_engine,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    main()

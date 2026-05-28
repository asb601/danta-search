"""Promotion-state regression checks.

Usage:
  cd server && source .venv/bin/activate && python3 -m testing._promotion_state_check
"""
from __future__ import annotations

import json
import os
import sys
import asyncio

_server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

from app.agent.tools.catalog import build_catalog_tools
import app.agent.tools.sql as sql_tools
from app.services.file_identity import build_file_identity_map
from app.services.promotion_state import build_initial_promotion_state


def _entry() -> dict:
    return {
        "file_id": "file-1",
        "blob_path": "orders.csv",
        "columns_info": [
            {"name": "ORDER_ID", "type": "int64", "sample_values": [1], "unique_values": [1]},
            {"name": "STATUS", "type": "string", "sample_values": ["OPEN"], "unique_values": ["OPEN"]},
        ],
        "column_names": ["ORDER_ID", "STATUS"],
        "ai_description": "Order status records",
        "good_for": [],
        "key_metrics": [],
        "key_dimensions": [],
    }


def _tool(tools: list, name: str):
    return next(tool for tool in tools if tool.name == name)


async def test_sql_requires_schema_promotion_for_workflow_queries() -> None:
    catalog = [_entry()]
    identities = build_file_identity_map(catalog, {}, "container")
    store: dict = {
        "promotion_state": build_initial_promotion_state(
            discovery_file_ids=["file-1"],
            execution_file_ids=[],
            must_inspect_before_sql=True,
        ),
        "_scratchpad": {"discovery_candidates": []},
    }
    store["_scratchpad"]["promotion_state"] = store["promotion_state"]

    run_sql = _tool(sql_tools.build_sql_tools(
        "UseDevelopmentStorage=true",
        "container",
        None,
        store,
        allowed_blob_paths=identities.allowed_physical_uris(),
        file_identities=identities,
        allowed_file_ids=identities.allowed_file_ids(),
    ), "run_sql")

    blocked = json.loads(run_sql.invoke({"sql": "SELECT * FROM ORDERS"}))
    assert blocked["promotion_required"] is True
    assert store["promotion_state"]["promoted_file_ids"] == []

    get_file_schema = _tool(build_catalog_tools(
        catalog,
        {},
        "container",
        file_identities=identities,
        state_store=store,
    ), "get_file_schema")
    schema = json.loads(await get_file_schema.ainvoke({"file_ref": "ORDERS"}))
    assert schema["logical_table"] == "ORDERS"
    assert store["promotion_state"]["schema_inspected_file_ids"] == ["file-1"]
    assert store["promotion_state"]["promoted_file_ids"] == ["file-1"]

    original_execute = sql_tools._execute
    try:
        sql_tools._execute = lambda *args, **kwargs: ([{"ORDER_ID": 1, "STATUS": "OPEN"}], 1)
        allowed = json.loads(run_sql.invoke({"sql": "SELECT ORDER_ID, STATUS FROM ORDERS"}))
    finally:
        sql_tools._execute = original_execute

    assert allowed["row_count"] == 1
    assert store["sql_results"] == [{"ORDER_ID": 1, "STATUS": "OPEN"}]


async def main() -> None:
    checks = [test_sql_requires_schema_promotion_for_workflow_queries]
    for check in checks:
        await check()
        print(f"[PASS] {check.__name__}")


if __name__ == "__main__":
    asyncio.run(main())
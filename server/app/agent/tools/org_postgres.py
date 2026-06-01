"""Live read-only org Postgres tools — list_org_database + run_org_sql.

These tools expose an organization's OWN live Postgres database (the encrypted
OrgAISettings.postgres_url, resolved per-request) to the agent, ALONGSIDE the
ingested Parquet catalog. They are only registered when:

  * settings.ORG_LIVE_DB_ENABLED is True, AND
  * the resolved org AI settings carry a non-empty postgres_url.

Read-only enforcement is defence in depth (see core.org_postgres_client):
  1. SQL validation rejects anything that is not a single read-only SELECT.
  2. Execution runs inside a `READ ONLY` Postgres transaction with a timeout.

Both tools are async coroutines (the underlying client is asyncpg); LangGraph's
ToolNode awaits them on the graph's event loop.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logger import chat_logger
from app.core.org_postgres_client import (
    OrgDBError,
    execute_readonly,
)


def build_org_postgres_tools(dsn: str, introspection: dict[str, list[dict]]) -> list:
    """Return the live org-DB tools bound to a DSN + pre-fetched introspection.

    `introspection` is the {"schema.table": [{column, type}, ...]} mapping
    produced by org_postgres_client.introspect(); it is snapshotted at build
    time so list_org_database is a zero-I/O lookup.
    """

    @tool
    async def list_org_database() -> str:
        """List the tables and columns available in the organization's LIVE database.
        Use this BEFORE writing run_org_sql so you reference only real
        schema-qualified tables and exact column names. Returns a JSON mapping of
        "schema.table" -> list of {column, type}. This live database is SEPARATE
        from the ingested file catalog (search_catalog / run_sql); use it only when
        the question needs current operational data from the org's own database."""
        return json.dumps({
            "tables": introspection,
            "table_count": len(introspection),
            "hint": (
                "Reference tables as schema.table in run_org_sql. "
                "Only read-only SELECT statements are permitted."
            ),
        }, default=str)

    @tool
    async def run_org_sql(sql: str) -> str:
        """Execute a single read-only SELECT against the organization's LIVE Postgres database.
        Use schema-qualified table names exactly as shown by list_org_database.
        Only a single read-only SELECT is allowed — no DDL/DML, no multiple
        statements (the query runs inside a READ ONLY transaction). Results are
        capped server-side. Returns row_count, column names, and the result rows.
        This is the org's live operational database, NOT the ingested Parquet
        catalog — use run_sql for ingested files."""
        max_rows = int(get_settings().ORG_DB_MAX_ROWS)
        try:
            rows, row_count = await execute_readonly(dsn, sql, max_rows=max_rows)
        except OrgDBError as exc:
            chat_logger.info("run_org_sql_rejected", error=str(exc)[:300])
            return json.dumps({"error": str(exc), "fatal_execution_error": True})
        except Exception as exc:  # noqa: BLE001 — never break the agent loop
            chat_logger.warning("run_org_sql_error", error=str(exc)[:300])
            return json.dumps({
                "error": "An unexpected error occurred querying the live database.",
                "fatal_execution_error": True,
            })

        resp = {
            "row_count": row_count,
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        }
        if row_count >= max_rows:
            resp["warning"] = (
                f"Results capped at {max_rows} rows. Add WHERE/GROUP BY/LIMIT "
                "to narrow the result set."
            )
        return json.dumps(resp, default=str)

    return [list_org_database, run_org_sql]

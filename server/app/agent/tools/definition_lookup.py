"""
lookup_field_definition tool — returns business meaning for a column name
from any schema / data-dictionary file registered for this container.

Design
------
At graph build time, `load_schema_registry(db, container_id, ...)` is called
once.  It queries all SchemaDictionary rows for the container and reads each
parquet into a flat Python dict:

    {field_name_lower → {"description": str, "notes": str | None}}

This dict is handed to `build_definition_lookup_tool` which closes over it.
All lookups at agent runtime are pure in-memory O(1) — zero network calls.

The same dict is also passed to `build_column_tool` so that `inspect_column`
automatically enriches its output with business meaning when available.

Loading cost (one-time per request):
  - Typically 1–3 schema dict files, each a few hundred rows at most.
  - DataFusion parquet read over Azure: ~100–300 ms per file.
  - Entire registry load: <1 second in the normal case.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from app.core.logger import pipeline_logger


def build_definition_lookup_tool(
    field_definitions: dict[str, dict],
) -> list:
    """
    Return a list containing the lookup_field_definition tool.

    Args:
        field_definitions: {field_name_lower → {"description": str, "notes": str|None}}
            Pre-loaded from all schema dictionaries for this container.
            Pass {} to get a tool that always says "no definition found".
    """

    @tool
    def lookup_field_definition(field_name: str) -> str:
        """Look up the business / semantic meaning of a column name.

        Use this when you see an unfamiliar column (e.g. SHKZG, BLART,
        OPEN_ITEM_FLAG, MONAT) and need to understand what it represents
        before writing SQL or interpreting results.

        Returns the description and any extended notes from the organisation's
        uploaded data dictionary.  If no definition is registered, says so —
        do NOT hallucinate a definition in that case; inspect sample values
        instead.
        """
        if not field_definitions:
            pipeline_logger.info(
                "lookup_field_definition",
                field=field_name,
                result="no_schema_dicts_registered",
            )
            return json.dumps({
                "field_name": field_name,
                "found": False,
                "message": (
                    "No schema dictionary has been uploaded for this workspace. "
                    "To enable semantic definitions, upload a data dictionary file "
                    "containing columns like FIELD_NAME and DESCRIPTION."
                ),
            })

        key = field_name.strip().upper()
        entry = field_definitions.get(key)

        if not entry:
            pipeline_logger.info(
                "lookup_field_definition",
                field=field_name,
                result="not_found",
            )
            return json.dumps({
                "field_name": field_name,
                "found": False,
                "message": (
                    f"No definition found for '{field_name}' in the registered "
                    f"schema dictionaries ({len(field_definitions)} fields indexed). "
                    "Try inspecting sample values with inspect_column instead."
                ),
            })

        result = {
            "field_name": field_name,
            "found": True,
            "description": entry["description"],
        }
        if entry.get("notes"):
            result["notes"] = entry["notes"]

        pipeline_logger.info(
            "lookup_field_definition",
            field=field_name,
            result="found",
            description_preview=entry["description"][:80],
        )
        return json.dumps(result)

    return [lookup_field_definition]


async def load_schema_registry(
    db,
    container_id: str | None,
    connection_string: str,
    container_name: str,
) -> dict[str, dict]:
    """
    Load all schema dictionaries for a container and return a flat
    {FIELD_NAME_UPPER → {"description": str, "notes": str|None}} dict.

    Called once per request in _build_agent_context before the graph runs.
    Returns {} if no schema dicts are registered or on any error.
    """
    if not container_id:
        return {}

    try:
        from sqlalchemy import select
        from app.models.schema_dictionary import SchemaDictionary
        from app.core.datafusion_client import execute_query_sync as _df_exec
        from app.core.duckdb_client import execute_query_sync as _duckdb_exec
        from app.core.config import get_settings
        import asyncio

        rows = (
            await db.execute(
                select(
                    SchemaDictionary.parquet_blob_path,
                    SchemaDictionary.source_blob_path,
                    SchemaDictionary.field_name_col,
                    SchemaDictionary.description_col,
                    SchemaDictionary.notes_col,
                )
                .where(SchemaDictionary.container_id == container_id)
                # Newest dictionary wins on key collision (see merge below).
                .order_by(SchemaDictionary.created_at.desc())
            )
        ).all()

        if not rows:
            return {}

        settings = get_settings()
        merged: dict[str, dict] = {}

        for row in rows:
            parquet_path = row.parquet_blob_path
            source_path = row.source_blob_path
            fn_col = row.field_name_col
            desc_col = row.description_col
            notes_col = row.notes_col

            # Build SELECT — include notes_col only if it exists.
            select_cols = f'"{fn_col}", "{desc_col}"'
            if notes_col:
                select_cols += f', "{notes_col}"'

            # Try parquet first (faster, columnar).  Fall back to the source
            # CSV if parquet hasn't been produced yet or the read fails — this
            # keeps the agent's definitions available even when parquet
            # conversion is queued, missing, or stale.
            attempts: list[tuple[str, str]] = []
            if parquet_path:
                attempts.append(("parquet", f"read_parquet('az://{container_name}/{parquet_path}')"))
            if source_path:
                attempts.append(("csv", f"read_csv_auto('az://{container_name}/{source_path}')"))

            if not attempts:
                pipeline_logger.warning(
                    "schema_registry_no_source",
                    field_name_col=fn_col,
                )
                continue

            load_rows: list[dict] = []
            total = 0
            last_err: str | None = None
            for source_kind, table_expr in attempts:
                sql = (
                    f"SELECT {select_cols} FROM {table_expr} "
                    f"WHERE \"{fn_col}\" IS NOT NULL"
                )
                try:
                    # Cap is intentionally large; SAP DDIC dumps may be ~40k
                    # fields. If a real dictionary is larger we want to know.
                    _MAX_DICT_ROWS = 50_000
                    if settings.QUERY_ENGINE == "datafusion":
                        load_rows, total = await asyncio.to_thread(
                            _df_exec, sql, connection_string,
                            max_rows=_MAX_DICT_ROWS, container_name=container_name
                        )
                    else:
                        load_rows, total = await asyncio.to_thread(
                            _duckdb_exec, sql, connection_string, max_rows=_MAX_DICT_ROWS
                        )
                    last_err = None
                    if load_rows:
                        if source_kind == "csv" and parquet_path:
                            pipeline_logger.info(
                                "schema_registry_csv_fallback",
                                source_path=source_path,
                                parquet_path=parquet_path,
                            )
                        break
                except Exception as exc:
                    last_err = str(exc)[:300]
                    continue

            if last_err and not load_rows:
                pipeline_logger.warning(
                    "schema_registry_load_failed",
                    parquet_path=parquet_path,
                    source_path=source_path,
                    error=last_err,
                )
                continue

            if total > len(load_rows):
                pipeline_logger.warning(
                    "schema_registry_truncated",
                    parquet_path=parquet_path,
                    loaded=len(load_rows),
                    total=total,
                    cap=50_000,
                )

            for r in load_rows:
                field = str(r.get(fn_col) or "").strip().upper()
                description = str(r.get(desc_col) or "").strip()
                notes = str(r.get(notes_col) or "").strip() if notes_col else None
                if field and description and field not in merged:
                    merged[field] = {
                        "description": description,
                        "notes": notes if notes else None,
                    }

            pipeline_logger.info(
                "schema_registry_loaded",
                parquet_path=parquet_path,
                source_path=source_path,
                fields_loaded=len(load_rows),
                container_id=container_id,
            )

        pipeline_logger.info(
            "schema_registry_ready",
            container_id=container_id,
            total_fields=len(merged),
            dict_count=len(rows),
        )
        return merged

    except Exception as exc:
        pipeline_logger.warning(
            "schema_registry_error",
            container_id=container_id,
            error=str(exc)[:300],
        )
        return {}

"""Request-local discovery-to-execution promotion state.

The state is a plain dict because it lives inside the per-request LangGraph tool
store. Helpers in this module keep the schema consistent across tools without
introducing persistence or cross-request coupling.
"""
from __future__ import annotations

from typing import Any


class PromotionRequiredError(ValueError):
    """Raised when SQL references files that have not been inspected/promoted."""


def get_promotion_state(state_store: dict | None) -> dict | None:
    if not state_store:
        return None
    state = state_store.get("promotion_state")
    if isinstance(state, dict):
        return state
    scratchpad = state_store.get("_scratchpad")
    if isinstance(scratchpad, dict):
        state = scratchpad.get("promotion_state")
        if isinstance(state, dict):
            return state
    return None


def build_initial_promotion_state(
    *,
    discovery_file_ids: list[str],
    execution_file_ids: list[str],
    must_inspect_before_sql: bool,
) -> dict[str, Any]:
    discovery_ids = list(dict.fromkeys(fid for fid in discovery_file_ids if fid))
    execution_ids = list(dict.fromkeys(fid for fid in execution_file_ids if fid))
    return {
        "discovery_file_ids": discovery_ids,
        "execution_file_ids": execution_ids,
        "schema_inspected_file_ids": [],
        "data_inspected_file_ids": [],
        "promoted_file_ids": [] if must_inspect_before_sql else execution_ids,
        "promotion_audit": [],
        "must_inspect_before_sql": bool(must_inspect_before_sql),
        "sql_execution_requires_promotion": True,
        "auto_promote_on_inspection": True,
    }


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _eligible_for_promotion(state: dict, file_id: str) -> bool:
    discovery_ids = set(state.get("discovery_file_ids") or [])
    execution_ids = set(state.get("execution_file_ids") or [])
    if not discovery_ids and not execution_ids:
        return True
    return file_id in discovery_ids or file_id in execution_ids


def mark_schema_inspected(
    state_store: dict | None,
    *,
    file_id: str | None,
    logical_table: str | None = None,
    tool: str = "get_file_schema",
) -> None:
    state = get_promotion_state(state_store)
    if not state or not file_id:
        return
    inspected = state.setdefault("schema_inspected_file_ids", [])
    _append_unique(inspected, file_id)
    state.setdefault("promotion_audit", []).append({
        "event": "schema_inspected",
        "file_id": file_id,
        "logical_table": logical_table,
        "tool": tool,
    })
    if state.get("auto_promote_on_inspection") and _eligible_for_promotion(state, file_id):
        promote_file(
            state_store,
            file_id=file_id,
            logical_table=logical_table,
            reason="schema_inspection",
            tool=tool,
        )


def mark_data_inspected(
    state_store: dict | None,
    *,
    file_id: str | None,
    logical_table: str | None = None,
    tool: str = "inspect_data_format",
) -> None:
    state = get_promotion_state(state_store)
    if not state or not file_id:
        return
    inspected = state.setdefault("data_inspected_file_ids", [])
    _append_unique(inspected, file_id)
    state.setdefault("promotion_audit", []).append({
        "event": "data_inspected",
        "file_id": file_id,
        "logical_table": logical_table,
        "tool": tool,
    })


def promote_file(
    state_store: dict | None,
    *,
    file_id: str | None,
    logical_table: str | None = None,
    reason: str = "runtime_promotion",
    tool: str = "runtime",
) -> None:
    state = get_promotion_state(state_store)
    if not state or not file_id:
        return
    if not _eligible_for_promotion(state, file_id):
        state.setdefault("promotion_audit", []).append({
            "event": "promotion_rejected",
            "file_id": file_id,
            "logical_table": logical_table,
            "reason": "not_in_discovery_or_execution_scope",
            "tool": tool,
        })
        return
    promoted = state.setdefault("promoted_file_ids", [])
    before = set(promoted)
    _append_unique(promoted, file_id)
    if file_id not in before:
        state.setdefault("promotion_audit", []).append({
            "event": "promoted",
            "file_id": file_id,
            "logical_table": logical_table,
            "reason": reason,
            "tool": tool,
        })


def require_sql_promotion(state_store: dict | None, referenced_file_ids: list[str]) -> None:
    state = get_promotion_state(state_store)
    if not state or not state.get("sql_execution_requires_promotion"):
        return
    promoted = set(state.get("promoted_file_ids") or [])
    missing = [fid for fid in referenced_file_ids if fid and fid not in promoted]
    if not missing:
        return
    state.setdefault("promotion_audit", []).append({
        "event": "sql_blocked_unpromoted_files",
        "file_ids": missing,
    })
    raise PromotionRequiredError(
        "SQL blocked because one or more referenced logical tables have not been inspected/promoted. "
        "Call get_file_schema on each target logical table first, then run SQL using those logical table names. "
        f"Unpromoted file ids: {', '.join(missing[:6])}."
    )


def promoted_physical_uris(state_store: dict | None, file_identities: Any) -> set[str] | None:
    state = get_promotion_state(state_store)
    if not state or not file_identities:
        return None
    promoted = set(state.get("promoted_file_ids") or [])
    if not promoted:
        return set()
    uris: set[str] = set()
    for file_id in promoted:
        identity = file_identities.by_id.get(file_id)
        if not identity:
            continue
        uris.add(identity.source_uri)
        if identity.parquet_uri:
            uris.add(identity.parquet_uri)
    return uris
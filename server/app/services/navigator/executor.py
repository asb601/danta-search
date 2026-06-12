"""[4] EXECUTE — run a verified step's logical SQL through the engine (I11).

The renderer ([3e]) emits fully-quoted logical SQL; this stage turns that into
numbers the SAME way the coordinator/graph seams do:

    canonicalize_logical_sql(logical → physical)  →  _execute(physical, engine)

So every number a step produces comes from DataFusion/DuckDB on Parquet, never
from the LLM and never invented (I11). The two reused primitives are imported
once at module scope (``canonicalize_logical_sql`` and ``_execute``) — the tests
monkeypatch THESE module-level names, which is also the seam the driver (P5)
relies on for store cleanup around the call.

``execute`` is the loop's executor: it NEVER raises. A canonicalize failure, an
authorization failure, or an engine error all collapse to an empty-rows
``StepResult`` with ``scalar=None`` and an ``error_marker`` so the driver can
abstain (I12) instead of crashing the request.

This module is store-lifecycle-FREE on purpose: the driver (P5) owns the
per-request store (pop on success/clarify, keep on abstain). The executor just
runs the SQL the renderer handed it.

The navigator is self-contained — this module imports nothing from
``app.services.resolve.*``. ``_execute`` and ``canonicalize_logical_sql`` are the
agent/services reuse primitives (they survive the P5 cutover).
"""
from __future__ import annotations

import asyncio
import numbers
from typing import Any, Optional

from app.agent.tools.sql import _execute
from app.services.logical_sql import canonicalize_logical_sql
from app.services.navigator.types import StepResult


def _coerce_scalar(value: Any) -> Optional[float]:
    """Return ``value`` as a float when it is a genuine number, else None.

    Booleans are explicitly rejected — a True/False cell is a flag, not a measure.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _derive_scalar(rows: tuple[dict, ...]) -> Optional[float]:
    """Deterministically extract the single-number answer from the result SHAPE.

    A scalar exists ONLY when the step reduces to exactly one value:
      * exactly ONE row, and
      * within that row, exactly ONE numeric column.

    The single-numeric-column rule is what disambiguates the two single-row cases:
      * ``{"amount": 100.0}``                 -> 100.0   (pure aggregate)
      * ``{"VENDOR_ID": "V1", "amount": 250}``-> 250.0   (one grain value + one measure)
      * ``{"a": 1.0, "b": 2.0}``              -> None    (two measures: ambiguous)
      * ``{"name": "VendorCo"}``              -> None    (no numeric measure)
      * many rows                              -> None    (a distribution, not a value)

    PURE — reads the shape, never guesses a column by name.
    """
    if len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        return None
    numeric_values = [
        coerced
        for coerced in (_coerce_scalar(v) for v in row.values())
        if coerced is not None
    ]
    if len(numeric_values) == 1:
        return numeric_values[0]
    return None


async def execute(
    logical_sql: str,
    *,
    identity_map,
    allowed_file_ids,
    connection_string: str,
    container_name: str,
    step_id: str,
    table: str,
    measure_label: str,
    grain: str,
    max_rows: int = 20,
) -> StepResult:
    """Canonicalize then execute a verified step's logical SQL → ``StepResult``.

    Numbers come from the engine (I11). NEVER raises: any canonicalize/auth/engine
    failure returns an empty-rows ``StepResult`` (scalar=None) carrying an
    ``error_marker`` so the driver can abstain (I12).
    """
    try:
        canonical = canonicalize_logical_sql(
            logical_sql, identity_map, allowed_file_ids=allowed_file_ids
        )
    except Exception as exc:  # noqa: BLE001 — executor never raises; driver abstains
        return StepResult(
            step_id=step_id,
            sql=logical_sql,
            rows=(),
            total=0,
            table=table,
            measure_label=measure_label,
            grain=grain,
            scalar=None,
            error_marker=f"canonicalize_failed: {str(exc)[:200]}",
        )

    try:
        rows, total = await asyncio.to_thread(
            _execute,
            canonical.executable_sql,
            connection_string,
            container_name,
            max_rows,
        )
    except Exception as exc:  # noqa: BLE001 — executor never raises; driver abstains
        return StepResult(
            step_id=step_id,
            sql=logical_sql,
            rows=(),
            total=0,
            table=table,
            measure_label=measure_label,
            grain=grain,
            scalar=None,
            error_marker=f"execute_failed: {str(exc)[:200]}",
        )

    rows = list(rows or [])
    total = int(total) if total is not None else len(rows)
    capped = tuple(rows[:max_rows])
    scalar = _derive_scalar(capped)

    return StepResult(
        step_id=step_id,
        sql=logical_sql,
        rows=capped,
        total=total,
        table=table,
        measure_label=measure_label,
        grain=grain,
        scalar=scalar,
    )

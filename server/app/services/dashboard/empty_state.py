"""Phase 4 — honest empty-state classification (G5).

A zero-row widget is one of three things, and saying which one honestly is the
point (invariant #4: a confident wrong claim is worse than an explained blank):
  ERROR   — the query errored or the SQL failed
  MISSING — no table was resolved (nothing to be empty ABOUT)
  EMPTY   — a table resolved but 0 rows matched (shown with the real loaded range)

Pure and fail-closed: an unknown/malformed shape never raises and never fabricates
an out-of-coverage claim. We SHOW the loaded coverage range; we do not assert the
requested window is outside it (that needs the feasibility gate — deferred).
"""
from __future__ import annotations

EMPTY = "empty"
MISSING = "missing"
ERROR = "error"


def classify_empty(result: dict, table, intent) -> tuple[str, str]:
    """Classify WHY a widget has no rows. Returns (state, human_message)."""
    result = result or {}
    # ERROR first: both the dashboard wrapper's `error` AND the agent's SQL
    # `execution_error` count — the latter is the common real failure key, NOT
    # `error`, so checking only `error` would miss most execution failures.
    if result.get("error") or result.get("execution_error"):
        return ERROR, "This question could not be answered."
    files_used = result.get("files_used") or []
    # MISSING only when NO table resolved. `files_used` alone is NOT a resolution
    # signal (it is populated from SQL-tool blobs, not retrieval), so a resolved
    # table with empty files_used is EMPTY, not MISSING.
    if table is None and not files_used:
        return MISSING, "Data not available for this question."
    # EMPTY: a table was involved but nothing matched. Surface the real loaded
    # range so an analyst sees (e.g.) that the data is a single period.
    coverage: list = []
    if table is not None:
        try:
            coverage = table.date_coverage() or []
        except Exception:
            coverage = []
    if coverage:
        return EMPTY, "No matching data for this question. Loaded data covers: " + "; ".join(coverage) + "."
    return EMPTY, "No matching data for this question."

"""Stage 5 (part) — element-type router.

Pure logic. Decides how a retrieved chunk's content is materialized for the LLM
context based on its element type:

* ``text``  → ``immediate``        — text is already in the chunk; no extraction.
* ``table`` → ``on_demand_table``  — lazy hi-res table extraction (Stage 6).
* ``image`` → ``on_demand_vision`` — lazy GPT-4o vision description (Stage 6).

This encodes the "extract at ingest for text, lazy for tables/images" design
exception (CONTRACTS Hard rule #2). Unknown / formula types fall back to
``immediate`` so they pass straight through as text.
"""
from __future__ import annotations

from pdf_chat.ingestion.ton_schema import ElementType

ROUTE_IMMEDIATE = "immediate"
ROUTE_ON_DEMAND_TABLE = "on_demand_table"
ROUTE_ON_DEMAND_VISION = "on_demand_vision"

_ROUTES: dict[str, str] = {
    ElementType.TEXT.value: ROUTE_IMMEDIATE,
    ElementType.TABLE.value: ROUTE_ON_DEMAND_TABLE,
    ElementType.IMAGE.value: ROUTE_ON_DEMAND_VISION,
    ElementType.FORMULA.value: ROUTE_IMMEDIATE,
}


def route_by_element_type(element_type: str | ElementType) -> str:
    """Map an element type to its materialization route.

    Args:
        element_type: an ``ElementType`` or its string value (``"text"`` etc.).

    Returns:
        One of ``immediate`` / ``on_demand_table`` / ``on_demand_vision``.
        Unknown types route to ``immediate`` (treated as plain text).
    """
    if isinstance(element_type, ElementType):
        key = element_type.value
    else:
        key = str(element_type).lower()
    return _ROUTES.get(key, ROUTE_IMMEDIATE)

"""Phase 3 — Tool Protocol + registry + the read tools (contract C3).

This module owns the ``Tool`` Protocol, the ``TOOL_REGISTRY`` dict, and the
``register_tool`` registrar that the capped tool loop (``agent/loop.py``) drives.
It is the *only* place that knows how to map an abstract tool name to a concrete
call against the Phase-2 ``Neo4jSearcher`` surface (entry points cited in the
plan's "Entry Points" section). The loop never touches the searcher directly —
it dispatches through ``TOOL_REGISTRY`` so Phase-4/5 can register new capabilities
(``structured_query``, ``glossary_lookup``) with ZERO loop change.

Design rules honored here:

* **PRIMARY retrieval is ``multi_vector_search``** — the RRF-fused
  chunk/section-card/doc-card leg (``neo4j_searcher.py:384``). Plain
  ``vector_search`` is a single-representation fallback tool; the legacy
  ``hybrid_search`` is NEVER wrapped as a Phase-3 tool.
* **Per-hop tenant isolation (spec §3 inv 3)** — every tool threads
  ``state.tenant_id`` (and the optional ``doc_ids`` document subset) to every
  searcher leg. The per-hop tenant predicate lives in the searcher's Cypher; the
  tool's job is to never drop the ``tenant_id`` argument.
* **Card demotion is NOT done here.** A card hit's only legitimate downstream use
  is pulling its ``src_chunk_ids`` into context — that demotion happens in
  ``agent/synthesis.py`` (HARD ENTRY GATE), not in retrieval. Tools return the
  searcher's hits verbatim so the fusion/grounding signal is preserved.
* **No score-comparison literal** lives here — fan-out sizes / top_k come from
  the searcher's own tunable defaults (the tools pass ``None`` so the searcher
  resolves ``get_tunable``). Tools make no gate decisions, so no
  ``log_gate_decision`` call belongs in this module.

Phase-4 / Phase-5 SEAM (names reserved, NO impl here):
  * ``structured_query`` — Phase-4 value-evidenced CSV/DataFusion bridge
    (sequential, passes ``container_id``/``allowed_domains``/``user_id``).
  * ``glossary_lookup`` — Phase-5 definitional glossary lookup.
Both are reserved in ``RESERVED_TOOL_NAMES`` so integration can detect the seam,
but neither is registered in ``TOOL_REGISTRY`` — registering them later is a
``register_tool`` call, never a loop change (contract C3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime infra import
    from pdf_chat.agent.state import PdfChatState


# --------------------------------------------------------------------------- #
# Tool Protocol (contract C3)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Tool(Protocol):
    """A read/act capability the agent loop can dispatch by name.

    A tool is a thin, awaitable adapter over a deterministic backend call. It
    NEVER decides routing/gating (that is the loop's + gates' job) and NEVER
    mutates ``state`` — it returns the backend's ``list[dict]`` hits and lets the
    loop merge them (tracking ``seen_chunk_ids`` for the monotonic-progress
    guard). ``runtime_checkable`` so tests can assert ``isinstance(tool, Tool)``.
    """

    name: str

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        ...


# --------------------------------------------------------------------------- #
# Registry + registrar (contract C3)
# --------------------------------------------------------------------------- #
TOOL_REGISTRY: dict[str, Tool] = {}

# Phase-4/5 capability names are RESERVED by name only — discoverable by the
# loop/integration, but deliberately ABSENT from TOOL_REGISTRY until those phases
# register a concrete impl via register_tool (no loop change required).
RESERVED_TOOL_NAMES: frozenset[str] = frozenset({"structured_query", "glossary_lookup"})


def register_tool(tool: Tool) -> Tool:
    """Register ``tool`` in ``TOOL_REGISTRY`` keyed by ``tool.name``; return it.

    Re-registering an already-claimed name raises ``ValueError`` (no silent
    shadowing — a name collision is a wiring bug). ``register_tool`` wraps any
    callable that satisfies the ``Tool`` Protocol; a raw LangChain
    ``StructuredTool`` must first be adapted into a ``Tool`` (the loop only ever
    stores Protocol-conformant objects, never a raw StructuredTool).

    A name in ``RESERVED_TOOL_NAMES`` (the Phase-4/5 seams ``structured_query`` /
    ``glossary_lookup``) IS accepted — registering one means that seam is being
    filled deliberately by its owning phase (e.g. Phase-4
    ``build_structured_query_tool``). The reservation set exists so the loop /
    integration can DISCOVER the seam, not to forbid filling it; the
    already-registered guard above still prevents accidental double-registration.
    """
    name = getattr(tool, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError("tool must expose a non-empty string `name`")
    if name in TOOL_REGISTRY:
        raise ValueError(f"tool name already registered: {name!r}")
    TOOL_REGISTRY[name] = tool
    return tool


# --------------------------------------------------------------------------- #
# Searcher access helpers (per-hop tenant + doc-subset threading)
# --------------------------------------------------------------------------- #
def _searcher(deps):
    """Return the searcher adapter from ``deps`` (attr or mapping)."""
    if deps is None:
        raise ValueError("tool requires deps with a `searcher`")
    searcher = getattr(deps, "searcher", None)
    if searcher is None and isinstance(deps, dict):
        searcher = deps.get("searcher")
    if searcher is None:
        raise ValueError("deps.searcher is required to run a retrieval tool")
    return searcher


async def _maybe_await(value):
    """Await ``value`` only if it is awaitable.

    The real ``Neo4jSearcher`` methods are SYNC (the neo4j driver is sync), but a
    test/adapter may inject an async searcher. Supporting both keeps the tool
    contract (``async def run``) stable regardless of the backend's sync-ness.
    """
    if hasattr(value, "__await__"):
        return await value
    return value


# --------------------------------------------------------------------------- #
# Phase-3 read tools (each wraps exactly one searcher leg)
# --------------------------------------------------------------------------- #
class _MultiVectorSearchTool:
    """PRIMARY retrieval: RRF-fused chunk + section-card + doc-card ANN.

    Wraps ``Neo4jSearcher.multi_vector_search`` (entry point :384). This is the
    default first leg of the loop — NOT plain ``vector_search`` and NEVER the
    legacy ``hybrid_search``. Threads ``tenant_id`` + ``doc_ids`` (per-hop tenant
    isolation lives in the searcher Cypher). Returns the fused hit dicts verbatim
    (card demotion to ``src_chunk_ids`` is the synthesizer's job).
    """

    name = "multi_vector_search"

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        searcher = _searcher(deps)
        top_k = kw.get("top_k", state.top_k)
        return await _maybe_await(
            searcher.multi_vector_search(
                state.query_vector,
                state.tenant_id,
                top_k=top_k,
                doc_ids=state.doc_ids,
            )
        )


class _VectorSearchTool:
    """Single-representation chunk ANN fallback.

    Wraps ``Neo4jSearcher.vector_search`` (entry point :203). Available as a
    narrower probe when the fused primary is not desired; threads ``tenant_id`` +
    ``doc_ids`` exactly like the primary.
    """

    name = "vector_search"

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        searcher = _searcher(deps)
        top_k = kw.get("top_k", state.top_k)
        return await _maybe_await(
            searcher.vector_search(
                state.query_vector,
                state.tenant_id,
                top_k=top_k,
                doc_ids=state.doc_ids,
            )
        )


class _GraphTraverseTool:
    """Entity → RELATED_TO → MENTIONS walk to related chunks.

    Wraps ``Neo4jSearcher.graph_traversal`` (entry point :241). Reachable ONLY
    when an anchor entity has been linked into ``state.entity`` (the entity-linker
    is the gate); with no entity the tool is a no-op returning ``[]`` so the loop
    never issues an anchorless (and therefore meaningless) graph walk. Per-hop
    tenant isolation is enforced in the searcher Cypher; the tool threads the
    ``tenant_id`` on every call.
    """

    name = "graph_traverse"

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        entity = kw.get("entity", state.entity)
        if not entity:
            return []
        searcher = _searcher(deps)
        limit = kw.get("limit")
        return await _maybe_await(
            searcher.graph_traversal(
                entity,
                state.tenant_id,
                limit=limit,
                doc_ids=state.doc_ids,
            )
        )


class _GetEntityNeighborsTool:
    """Related-entity neighbourhood of an anchor entity.

    Wraps ``Neo4jSearcher.entity_neighbors`` (entry point :276) — contract C2
    explicitly names this tool as that method's wrapper. Per-hop tenant isolation
    in the Cypher; the tool threads ``tenant_id`` + ``doc_ids``. Returns neighbour
    dicts (``name``/``etype``/``normalized_value``), not chunks.
    """

    name = "get_entity_neighbors"

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        entity = kw.get("entity", state.entity)
        if not entity:
            return []
        searcher = _searcher(deps)
        limit = kw.get("limit")
        return await _maybe_await(
            searcher.entity_neighbors(
                entity,
                state.tenant_id,
                limit=limit,
                doc_ids=state.doc_ids,
            )
        )


class _CommunityReportLookupTool:
    """ANN over the CITED community-report vector space (persisted reports).

    Wraps ``Neo4jSearcher.community_report_lookup`` (entry point :317). Only
    cited reports are ever written, so this returns grounded community summaries
    with their ``citations`` (chunk ids). Threads ``tenant_id`` (the report node
    carries it, so the leg is tenant-isolated). Used for ``global`` intent.
    """

    name = "community_report_lookup"

    async def run(self, state: "PdfChatState", deps, **kw) -> list[dict]:
        searcher = _searcher(deps)
        limit = kw.get("limit")
        return await _maybe_await(
            searcher.community_report_lookup(
                state.query_vector,
                state.tenant_id,
                limit=limit,
            )
        )


# --------------------------------------------------------------------------- #
# Registration — Phase-3 read tools only. Phase-4/5 seams stay unregistered.
# --------------------------------------------------------------------------- #
# Ordered so multi_vector_search (PRIMARY) leads; the loop reads it first.
_PHASE3_TOOLS: tuple[Tool, ...] = (
    _MultiVectorSearchTool(),
    _VectorSearchTool(),
    _GraphTraverseTool(),
    _GetEntityNeighborsTool(),
    _CommunityReportLookupTool(),
)

for _t in _PHASE3_TOOLS:
    # Idempotent on re-import: skip if already registered (test runners that
    # re-import the module must not trip the no-shadow guard).
    if _t.name not in TOOL_REGISTRY:
        register_tool(_t)

PHASE3_TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in _PHASE3_TOOLS)


__all__ = [
    "Tool",
    "TOOL_REGISTRY",
    "register_tool",
    "PHASE3_TOOL_NAMES",
    "RESERVED_TOOL_NAMES",
]

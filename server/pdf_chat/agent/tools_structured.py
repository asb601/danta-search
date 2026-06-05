"""Phase 4 Task 4 — the ``structured_query`` Tool (value-evidenced CSV bridge leg).

This tool fills the Phase-4 ``structured_query`` SEAM reserved (by name only) in
``pdf_chat/agent/tools.py``. It does NOT re-implement any CSV/structured query
logic: it delegates to the READ-ONLY ``run_agent_query`` entry point under
``server/app/`` (imported, never modified), inheriting that path's feasibility +
negative-claim gates for free. The PDF agent loop dispatches this tool by name
exactly like a retrieval tool; the result is wrapped in the same one-element
``list[dict]`` shape the Phase-3 tools return so the loop's merge logic is
unchanged.

⚠️ SEQUENTIAL CONTRACT (read before wiring this into the live loop):
    ``run_agent_query`` runs against an async SQLAlchemy session
    (``StructuredQueryDeps.db``). **That async DB session is NOT
    concurrency-safe.** This tool therefore runs STRICTLY SEQUENTIALLY — the
    Phase-3 tool loop must NEVER dispatch it concurrently (no ``asyncio.gather``)
    with another DB-touching tool. The loop is already single-threaded /
    sequential, so this is a contract to preserve, not a new lock to add. There
    is deliberately no concurrency primitive in this module: introducing one
    would mask a loop bug rather than prevent it.

Why no tunables/gates here: this module makes NO score comparison and NO gate
decision (the value-evidence gate lives in ``bridge/reconcile.py``; the
feasibility/negative-claim gates live inside ``run_agent_query``). It is a pure
scope-passing adapter, so there is no ``get_tunable`` / ``log_gate_decision``
call and no magic literal to register.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pdf_chat.agent.tools import Tool

# Type of the READ-ONLY structured entry point
# (app/agent/graph/graph.py::run_agent_query). Imported lazily by the caller and
# injected via deps so this module never imports server/app/ at load time.
RunAgentQuery = Callable[..., Awaitable[dict]]


@dataclass
class StructuredQueryDeps:
    """Everything the structured leg needs to call ``run_agent_query`` in-scope.

    ``run_agent_query`` is injected (not imported here) so this module stays
    import-safe with zero ``server/app/`` coupling and is trivially fakeable in
    tests. ``db`` is the async SQLAlchemy session — see the module-level
    SEQUENTIAL CONTRACT: it is not concurrency-safe.
    """

    run_agent_query: RunAgentQuery
    db: Any
    container_id: str
    allowed_domains: list[str]
    user_id: str
    actor_email: str = ""
    actor_role: str = ""
    org_id: str | None = None


def _wrap_result(result: dict) -> list[dict]:
    """Shape a ``run_agent_query`` result like the other Phase-3 tool outputs.

    Returns a ONE-element list (the structured leg is a single deterministic
    answer, not a ranked hit list) carrying ``answer`` / ``data`` /
    ``files_used`` and a ``source="structured"`` marker so the synthesizer can
    tell a CSV-grounded answer from a PDF-graph hit.
    """
    result = result or {}
    return [
        {
            "answer": result.get("answer", ""),
            "data": result.get("data", []),
            "files_used": result.get("files_used", []),
            "row_count": result.get("row_count", 0),
            "source": "structured",
        }
    ]


async def structured_query(deps: StructuredQueryDeps, query: str) -> list[dict]:
    """Run ``query`` through the READ-ONLY structured agent and wrap the answer.

    Runs STRICTLY SEQUENTIALLY: it awaits a single ``run_agent_query`` call on
    the shared async DB session (``deps.db``), which is NOT concurrency-safe. No
    ``asyncio.gather`` / fan-out is performed here, and the caller (the Phase-3
    loop) must not dispatch this concurrently with another DB-touching tool.

    Scope (``container_id`` / ``allowed_domains`` / ``user_id`` + the optional
    actor/org) is threaded UNCHANGED so the structured side enforces the same
    RBAC + tenant isolation as the chat path.
    """
    result = await deps.run_agent_query(
        query,
        deps.db,
        container_id=deps.container_id,
        allowed_domains=deps.allowed_domains,
        user_id=deps.user_id,
        actor_email=deps.actor_email,
        actor_role=deps.actor_role,
        org_id=deps.org_id,
    )
    return _wrap_result(result)


class _StructuredQueryTool:
    """Phase-3 ``Tool``-Protocol adapter over :func:`structured_query`.

    ``name`` is the reserved seam name ``"structured_query"``. ``run`` matches
    the protocol (``async run(self, state, deps, **kw)``) but ignores the loop's
    retrieval ``deps`` (the searcher) — the structured leg uses the
    ``StructuredQueryDeps`` captured at build time. Like :func:`structured_query`
    it is STRICTLY SEQUENTIAL: a single awaited DB call, no concurrency, because
    the async DB session is not concurrency-safe.
    """

    name = "structured_query"

    def __init__(self, deps: StructuredQueryDeps) -> None:
        self._deps = deps

    async def run(self, state, deps, **kw) -> list[dict]:
        # The query may arrive via kw (loop convention) or fall back to the
        # state's query text; the loop-supplied retrieval ``deps`` is unused.
        query = kw.get("query")
        if query is None and state is not None:
            query = getattr(state, "query", None) or getattr(state, "query_text", None)
        return await structured_query(self._deps, query or "")


def build_structured_query_tool(deps: StructuredQueryDeps) -> Tool:
    """Build a Phase-3 ``Tool`` for the reserved ``structured_query`` seam.

    Register it via ``pdf_chat.agent.tools.register_tool`` (which accepts the
    reserved name). Activation (registering into the live agent deps) is the
    deferred turn-on step — building + testing it here is this phase's scope.
    """
    return _StructuredQueryTool(deps)


__all__ = [
    "StructuredQueryDeps",
    "structured_query",
    "build_structured_query_tool",
]

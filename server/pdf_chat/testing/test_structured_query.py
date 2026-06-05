"""Phase 4 Task 4 — tests for the sequential, scope-passing ``structured_query`` Tool.

The tool delegates to the READ-ONLY ``run_agent_query`` (CSV/structured side),
inheriting its feasibility + negative-claim gates. It MUST:

* thread ``container_id`` / ``allowed_domains`` / ``user_id`` (and the optional
  actor/org scope) unchanged into ``run_agent_query``,
* wrap the result dict in a one-element list shaped like the other Phase-3 tool
  outputs (``answer`` / ``data`` / ``files_used`` + a ``source="structured"``
  marker),
* register into the reserved Phase-4 seam (``RESERVED_TOOL_NAMES`` reserves
  ``structured_query``; ``register_tool`` accepts a reserved name),
* run STRICTLY SEQUENTIALLY — the async DB session is not concurrency-safe.
"""
from __future__ import annotations

import inspect

import pytest

from pdf_chat.agent import tools_structured
from pdf_chat.agent.tools import RESERVED_TOOL_NAMES, TOOL_REGISTRY, Tool, register_tool
from pdf_chat.agent.tools_structured import (
    StructuredQueryDeps,
    build_structured_query_tool,
    structured_query,
)


class FakeSession:
    """Stand-in for the async SQLAlchemy session (never touched here)."""


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the reserved seam clear between tests (registration is global)."""
    TOOL_REGISTRY.pop("structured_query", None)
    yield
    TOOL_REGISTRY.pop("structured_query", None)


@pytest.mark.asyncio
async def test_structured_query_passes_scope_and_is_sequential():
    calls = []

    async def fake_run_agent_query(query, db, **kw):
        calls.append((query, db, kw))
        return {"answer": "42", "data": [], "row_count": 0, "files_used": []}

    deps = StructuredQueryDeps(
        run_agent_query=fake_run_agent_query,
        db=FakeSession(),
        container_id="c1",
        allowed_domains=["finance"],
        user_id="u1",
    )
    out = await structured_query(deps, "total spend for vendor V-100")

    assert len(calls) == 1
    query, db, kw = calls[0]
    assert query == "total spend for vendor V-100"
    assert isinstance(db, FakeSession)
    assert kw["container_id"] == "c1"
    assert kw["allowed_domains"] == ["finance"]
    assert kw["user_id"] == "u1"
    # one-element list, shaped like other tool outputs
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["answer"] == "42"
    assert out[0]["data"] == []
    assert out[0]["files_used"] == []
    assert out[0]["source"] == "structured"


@pytest.mark.asyncio
async def test_structured_query_threads_actor_and_org_scope():
    calls = []

    async def fake_run_agent_query(query, db, **kw):
        calls.append(kw)
        return {"answer": "ok", "data": [{"x": 1}], "files_used": ["f1.parquet"]}

    deps = StructuredQueryDeps(
        run_agent_query=fake_run_agent_query,
        db=FakeSession(),
        container_id="c2",
        allowed_domains=["hr", "ops"],
        user_id="u2",
        actor_email="a@b.com",
        actor_role="owner",
        org_id="org-9",
    )
    out = await structured_query(deps, "headcount by dept")

    kw = calls[0]
    assert kw["actor_email"] == "a@b.com"
    assert kw["actor_role"] == "owner"
    assert kw["org_id"] == "org-9"
    assert out[0]["data"] == [{"x": 1}]
    assert out[0]["files_used"] == ["f1.parquet"]


@pytest.mark.asyncio
async def test_tool_registers_into_reserved_seam():
    async def fake_run_agent_query(query, db, **kw):
        return {"answer": "x", "data": [], "files_used": []}

    deps = StructuredQueryDeps(
        run_agent_query=fake_run_agent_query,
        db=FakeSession(),
        container_id="c1",
        allowed_domains=["finance"],
        user_id="u1",
    )
    tool = build_structured_query_tool(deps)

    # name is the reserved seam name; the protocol is satisfied
    assert tool.name == "structured_query"
    assert "structured_query" in RESERVED_TOOL_NAMES
    assert isinstance(tool, Tool)

    register_tool(tool)  # reserved name accepted by register_tool
    assert TOOL_REGISTRY["structured_query"].name == "structured_query"


@pytest.mark.asyncio
async def test_tool_run_delegates_to_structured_query():
    calls = []

    async def fake_run_agent_query(query, db, **kw):
        calls.append((query, kw))
        return {"answer": "delegated", "data": [], "files_used": []}

    deps = StructuredQueryDeps(
        run_agent_query=fake_run_agent_query,
        db=FakeSession(),
        container_id="c3",
        allowed_domains=["finance"],
        user_id="u3",
    )
    tool = build_structured_query_tool(deps)
    # the Phase-3 loop calls run(state, deps2, **kw); the structured tool ignores
    # the loop's searcher-deps and uses its own StructuredQueryDeps.
    out = await tool.run(state=None, deps=None, query="spend by vendor")

    assert calls[0][0] == "spend by vendor"
    assert calls[0][1]["container_id"] == "c3"
    assert out[0]["answer"] == "delegated"
    assert out[0]["source"] == "structured"


def test_run_is_async_not_concurrent_dispatch():
    """The tool's run is a plain coroutine — no asyncio.gather/fan-out inside it.

    The Phase-3 loop is single-threaded sequential; the structured tool must not
    introduce its own concurrency around the shared async DB session.
    """
    assert inspect.iscoroutinefunction(structured_query)
    src = inspect.getsource(tools_structured)
    # no actual fan-out CALL (the docstring may *name* gather to forbid it).
    assert "asyncio.gather(" not in src
    # the sequential contract must be documented in the module
    lowered = src.lower()
    assert "sequential" in lowered
    assert "not concurrency-safe" in lowered

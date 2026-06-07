"""Phase 3 — parallel widget execution helper (pure, no DB/LLM/FastAPI).

`run_widgets` parallelizes ONLY the per-widget agent call and returns results IN
INPUT ORDER, so the route's sequential post-processing stays byte-identical whether
the flag is on or off. Tests prove: order preserved despite out-of-order completion,
the semaphore bound holds, parallel=False is strictly sequential, exceptions are
returned in place (one bad widget never cancels the others), empty input is safe.

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_parallel_widgets.py -q
"""
from __future__ import annotations

import asyncio

from app.services.dashboard.query_engine import run_widgets


def _run(coro):
    return asyncio.run(coro)


def test_order_preserved_despite_completion_order():
    async def run_one(intent):
        # later intents finish sooner -> completion order != input order
        await asyncio.sleep(0.005 * (10 - intent))
        return {"i": intent}

    res = _run(run_widgets([1, 2, 3, 4], run_one, concurrency=4, parallel=True))
    assert res == [{"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}]


def test_concurrency_bound_respected():
    state = {"inflight": 0, "max": 0}

    async def run_one(_intent):
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return _intent

    _run(run_widgets(list(range(8)), run_one, concurrency=2, parallel=True))
    assert state["max"] <= 2


def test_sequential_when_parallel_false():
    starts: list = []
    state = {"inflight": 0, "max": 0}

    async def run_one(intent):
        starts.append(intent)
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.005)
        state["inflight"] -= 1
        return intent

    res = _run(run_widgets([3, 1, 2], run_one, concurrency=4, parallel=False))
    assert res == [3, 1, 2]
    assert state["max"] == 1        # never more than one in flight
    assert starts == [3, 1, 2]      # strict input order, no overlap


def test_concurrency_one_equals_sequential():
    state = {"inflight": 0, "max": 0}

    async def run_one(intent):
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.005)
        state["inflight"] -= 1
        return intent

    res = _run(run_widgets([1, 2, 3], run_one, concurrency=1, parallel=True))
    assert res == [1, 2, 3]
    assert state["max"] == 1


def test_exception_returned_in_place_not_raised():
    async def run_one(intent):
        if intent == 2:
            raise ValueError("boom")
        return {"i": intent}

    res = _run(run_widgets([1, 2, 3], run_one, concurrency=2, parallel=True))
    assert res[0] == {"i": 1} and res[2] == {"i": 3}      # siblings unaffected
    assert isinstance(res[1], Exception)                  # returned in place, not raised


def test_exception_lands_at_input_index_under_out_of_order_completion():
    # The intersection of order-preservation + in-place exceptions: the raising
    # widget completes LAST, siblings finish out of order — the Exception must still
    # land at its INPUT index, not its completion position.
    async def run_one(intent):
        await asyncio.sleep(0.05 if intent == 2 else 0.005 * (5 - intent))
        if intent == 2:
            raise ValueError("boom")
        return {"i": intent}

    res = _run(run_widgets([1, 2, 3, 4], run_one, concurrency=4, parallel=True))
    assert res[0] == {"i": 1}
    assert isinstance(res[1], Exception)          # at input index 1, despite finishing last
    assert res[2] == {"i": 3} and res[3] == {"i": 4}


def test_empty_intents_is_safe_both_modes():
    assert _run(run_widgets([], None, concurrency=3, parallel=True)) == []
    assert _run(run_widgets([], None, concurrency=3, parallel=False)) == []

"""F4 — the tool-call budget must count ACTUAL tool calls, not iterations.

I17 fired 24 run_sql calls under MAX_TOOL_CALLS=8 because the counter advanced by
1 per agent iteration regardless of how many tool calls that iteration emitted.
"""
from app.agent.graph.graph_builder import next_tool_call_count
from app.agent.state import MAX_TOOL_CALLS


def test_counts_actual_calls_not_iterations():
    assert next_tool_call_count(0, 3) == 3
    assert next_tool_call_count(5, 8) == 13


def test_iteration_with_no_tool_calls_does_not_advance():
    assert next_tool_call_count(4, 0) == 4


def test_budget_bounds_thrash_with_headroom_over_legit():
    # Cut runaway well below the observed 24-call spin, with comfortable headroom
    # over legitimate queries (observed <= 6 tool calls in the run).
    assert 6 < MAX_TOOL_CALLS < 24

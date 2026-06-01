"""Pytest suite for the model deployment pool + embedding batcher.

These are PURE-function tests: no network, no openai. The modules soft-guard
the openai import, so they import and the pure selection/parse/plan layers run
even when openai is absent.

Covers:
  A. select_deployment — weighted pick deterministic given rng_token; cooling
     lane skipped; all-cooling picks soonest-to-recover; tier='high' prefers the
     gpt-4o (high) lane.
  B. load_deployments — empty/garbage config falls back to the legacy single
     deployment; clamps; skips unknown kinds; parses JSON-string and list forms.
  C. plan_batches — chunks by count AND token budget, flushes leftovers,
     oversized single input ships alone.
  D. Degenerate inputs — empty pool, 0/negative budgets never crash.

Run with:
    cd server && uv run pytest testing/test_model_pool.py -q
"""

from __future__ import annotations

import asyncio

from app.core.embed_batcher import BatchPlan, embed_all, plan_batches
from app.core.model_pool import (
    Deployment,
    HealthState,
    load_deployments,
    select_deployment,
)


def _dep(
    name: str,
    *,
    kind: str = "chat",
    weight: float = 1.0,
    tier: str = "standard",
    deployment_id: str = "",
) -> Deployment:
    return Deployment(
        name=name,
        kind=kind,
        endpoint="https://example.openai.azure.com",
        deployment_id=deployment_id or name,
        rpm=1000,
        tpm=1_000_000,
        weight=weight,
        region="eastus",
        tier=tier,
    )


# ---------------------------------------------------------------------------
# A. select_deployment — pure weighted, health-aware selection.
# ---------------------------------------------------------------------------


def test_weighted_selection_deterministic_given_token():
    """Same rng_token -> same lane, and tokens map onto the cumulative weight
    line: weights [3, 1] over [a, b] means token<0.75 -> a, else b."""
    deps = (_dep("a", weight=3.0), _dep("b", weight=1.0))
    health: dict[str, HealthState] = {}

    # total weight = 4; a covers [0, 3), b covers [3, 4) => token*4 boundary 0.75.
    assert select_deployment(deps, health, 0.0, now=0.0).name == "a"
    assert select_deployment(deps, health, 0.5, now=0.0).name == "a"
    assert select_deployment(deps, health, 0.74, now=0.0).name == "a"
    assert select_deployment(deps, health, 0.75, now=0.0).name == "b"
    assert select_deployment(deps, health, 0.99, now=0.0).name == "b"
    # Determinism: identical token yields identical lane every time.
    for _ in range(5):
        assert select_deployment(deps, health, 0.3, now=0.0).name == "a"


def test_cooling_lane_is_skipped():
    """A lane cooling past `now` is excluded; selection lands on the healthy one
    regardless of which side of the weight line the token falls."""
    deps = (_dep("a", weight=1.0), _dep("b", weight=1.0))
    # a is cooling until t=100; now=10 -> a unhealthy, only b is selectable.
    health = {"a": HealthState(cooling_until=100.0)}
    for token in (0.0, 0.49, 0.5, 0.99):
        chosen = select_deployment(deps, health, token, now=10.0)
        assert chosen.name == "b"


def test_all_cooling_picks_soonest_to_recover():
    """If every candidate lane is cooling, return the one with the smallest
    cooling_until (soonest to recover)."""
    deps = (_dep("a"), _dep("b"), _dep("c"))
    health = {
        "a": HealthState(cooling_until=300.0),
        "b": HealthState(cooling_until=150.0),  # soonest
        "c": HealthState(cooling_until=200.0),
    }
    chosen = select_deployment(deps, health, 0.9, now=10.0)
    assert chosen.name == "b"


def test_tier_high_prefers_gpt4o_lane():
    """tier='high' narrows to high lanes when any exist (the gpt-4o lane)."""
    deps = (
        _dep("mini-1", weight=1.0, tier="standard", deployment_id="gpt-4o-mini"),
        _dep("mini-2", weight=1.0, tier="standard", deployment_id="gpt-4o-mini"),
        _dep("gpt4o", weight=0.5, tier="high", deployment_id="gpt-4o"),
    )
    health: dict[str, HealthState] = {}
    # Across the whole token range, high tier collapses to the single high lane.
    for token in (0.0, 0.3, 0.6, 0.99):
        chosen = select_deployment(deps, health, token, kind="chat", tier="high", now=0.0)
        assert chosen.deployment_id == "gpt-4o"
        assert chosen.tier == "high"


def test_tier_high_falls_back_when_no_high_lane():
    """tier='high' with no high lane degrades to the standard candidates."""
    deps = (_dep("mini-1", tier="standard"), _dep("mini-2", tier="standard"))
    health: dict[str, HealthState] = {}
    chosen = select_deployment(deps, health, 0.0, tier="high", now=0.0)
    assert chosen is not None
    assert chosen.tier == "standard"


def test_no_lane_of_kind_returns_none():
    """Asking for an embedding lane when only chat lanes exist -> None."""
    deps = (_dep("a", kind="chat"),)
    assert select_deployment(deps, {}, 0.5, kind="embedding", now=0.0) is None


def test_all_zero_weights_uniform_pick():
    """All-zero weights must not divide by zero; degrade to uniform-by-token."""
    deps = (_dep("a", weight=0.0), _dep("b", weight=0.0), _dep("c", weight=0.0))
    assert select_deployment(deps, {}, 0.0, now=0.0).name == "a"
    assert select_deployment(deps, {}, 0.99, now=0.0).name == "c"


# ---------------------------------------------------------------------------
# B. load_deployments — parse + clamp + legacy fallback.
# ---------------------------------------------------------------------------


def test_empty_config_falls_back_to_legacy_single_deployment():
    legacy = _dep("legacy-chat")
    assert load_deployments(None, legacy=legacy) == (legacy,)
    assert load_deployments("", legacy=legacy) == (legacy,)
    assert load_deployments("   ", legacy=legacy) == (legacy,)
    assert load_deployments([], legacy=legacy) == (legacy,)


def test_garbage_config_falls_back_to_legacy():
    legacy = _dep("legacy-chat")
    # Malformed JSON, non-list JSON, and unknown-kind rows all degrade to legacy.
    assert load_deployments("{not json", legacy=legacy) == (legacy,)
    assert load_deployments('{"a": 1}', legacy=legacy) == (legacy,)  # JSON object, not array
    assert load_deployments([{"kind": "bogus"}], legacy=legacy) == (legacy,)


def test_empty_config_no_legacy_returns_empty_tuple():
    assert load_deployments(None) == tuple()
    assert load_deployments([]) == tuple()


def test_parse_json_string_and_clamps():
    raw = (
        '[{"name":"a","kind":"chat","endpoint":"https://e","deployment_id":"gpt-4o-mini",'
        '"rpm":-5,"tpm":0,"weight":-2.0,"region":"eastus","tier":"HIGH"}]'
    )
    deps = load_deployments(raw)
    assert len(deps) == 1
    d = deps[0]
    assert d.name == "a"
    assert d.rpm == 1  # clamped up from -5
    assert d.tpm == 1  # clamped up from 0
    assert d.weight == 0.0  # clamped up from -2.0
    assert d.tier == "high"  # normalized from "HIGH"


def test_parse_list_of_dicts_skips_unknown_kind():
    raw = [
        {"name": "chat-1", "kind": "chat", "deployment_id": "gpt-4o-mini"},
        {"name": "bad", "kind": "image"},
        {"name": "emb-1", "kind": "embedding", "deployment_id": "text-embedding-3-small"},
    ]
    deps = load_deployments(raw)
    names = [d.name for d in deps]
    assert names == ["chat-1", "emb-1"]


# ---------------------------------------------------------------------------
# C. plan_batches — pack by count AND token budget; flush leftovers.
# ---------------------------------------------------------------------------


def test_plan_batches_chunks_by_count():
    # 5 inputs, batch_max=2, generous token budget -> [2, 2, 1].
    plans = plan_batches([10, 10, 10, 10, 10], batch_max=2, token_budget=10_000)
    assert plans == [
        BatchPlan((0, 1)),
        BatchPlan((2, 3)),
        BatchPlan((4,)),
    ]


def test_plan_batches_chunks_by_token_budget():
    # batch_max generous, but token budget=100 forces a flush every ~2 items.
    plans = plan_batches([60, 60, 60, 60, 10], batch_max=100, token_budget=100)
    # 60 -> [0]; +60 would be 120>100 -> flush, [1]; etc; last 10 joins prior 60?
    # Walk: [0](60); next 60 -> 120>100 flush -> plan(0); [1](60); next 60 ->120>100
    # flush -> plan(1); [2](60); next 60 -> flush -> plan(2); [3](60); next 10 ->70<=100
    # so [3,4]. Final flush -> plan(3,4).
    assert plans == [
        BatchPlan((0,)),
        BatchPlan((1,)),
        BatchPlan((2,)),
        BatchPlan((3, 4)),
    ]


def test_plan_batches_oversized_input_ships_alone():
    # An input bigger than the whole budget must NOT be dropped; it ships alone.
    plans = plan_batches([5, 9999, 5], batch_max=100, token_budget=100)
    # [0](5); next 9999 -> 5+9999>100 flush -> plan(0); [1](9999); next 5 ->
    # 9999+5>100 flush -> plan(1); [2](5); final flush -> plan(2).
    assert plans == [BatchPlan((0,)), BatchPlan((1,)), BatchPlan((2,))]


def test_plan_batches_flushes_leftovers():
    plans = plan_batches([1, 1, 1], batch_max=10, token_budget=10_000)
    # Everything fits in one batch; the trailing partial is still emitted.
    assert plans == [BatchPlan((0, 1, 2))]


def test_plan_batches_clamps_count_token_negative():
    # batch_max<=0 and token_budget<=0 are clamped to 1; negative tokens -> 0.
    plans = plan_batches([-5, -5, -5], batch_max=0, token_budget=-10)
    # batch_max clamps to 1 -> each input alone; negative tokens clamp to 0.
    assert plans == [BatchPlan((0,)), BatchPlan((1,)), BatchPlan((2,))]


def test_plan_batches_empty_inputs():
    assert plan_batches([], batch_max=10, token_budget=100) == []


# ---------------------------------------------------------------------------
# D. Degenerate inputs never crash (and stay pure / network-free).
# ---------------------------------------------------------------------------


def test_select_on_empty_pool_returns_none():
    assert select_deployment(tuple(), {}, 0.5, now=0.0) is None
    assert select_deployment(tuple(), {}, 0.5, kind="embedding", tier="high", now=0.0) is None


def test_select_token_out_of_range_is_clamped():
    deps = (_dep("a", weight=1.0), _dep("b", weight=1.0))
    # token < 0 clamps to 0 (-> a); token > 1 clamps to 1 (-> last, b).
    assert select_deployment(deps, {}, -5.0, now=0.0).name == "a"
    assert select_deployment(deps, {}, 5.0, now=0.0).name == "b"


def test_embed_all_empty_inputs_makes_no_call():
    class _BoomPool:
        async def aembed(self, **_kw):  # pragma: no cover - must never be hit
            raise AssertionError("aembed should not be called for empty inputs")

    out = asyncio.run(embed_all(_BoomPool(), [], [], batch_max=10, token_budget=100))
    assert out == []


def test_embed_all_reassembles_in_input_order():
    class _Item:
        def __init__(self, vec):
            self.embedding = vec

    class _Resp:
        def __init__(self, data):
            self.data = data

    class _FakePool:
        """Returns a vector == [index_token] for each input, in batch order."""

        def __init__(self):
            self.calls: list[list[str]] = []

        async def aembed(self, *, inputs, tier="standard"):
            self.calls.append(list(inputs))
            return _Resp([_Item([float(len(s))]) for s in inputs])

    pool = _FakePool()
    inputs = ["a", "bb", "ccc", "dddd"]
    # batch_max=2 -> two calls of two; vectors must realign to input order.
    out = asyncio.run(embed_all(pool, inputs, [1, 1, 1, 1], batch_max=2, token_budget=10_000))
    assert out == [[1.0], [2.0], [3.0], [4.0]]
    assert pool.calls == [["a", "bb"], ["ccc", "dddd"]]

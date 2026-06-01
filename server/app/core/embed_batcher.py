"""Embedding batcher: a pure packing planner + a thin flush shell.

Embedding endpoints are bounded by BOTH a per-request item count and a
per-request token budget. Sending one item at a time wastes the per-call
overhead; sending too many (or too many tokens) gets the call rejected. This
module mirrors the ``resource_profile.py`` split:

1. ``plan_batches()`` — *pure* function. Given the per-input token counts and
   the two caps, it packs inputs into ``BatchPlan`` groups that respect both
   bounds, ships an oversized single input ALONE (never dropped), and flushes
   the trailing leftovers. No I/O — feed it a token list in a unit test and
   assert on the plan shape.

2. ``embed_all()`` — *thin IO shell*. It plans, issues one ``pool.aembed`` call
   per plan, and reassembles the returned vectors back into the original input
   order. The pool (and thus all network + failover) is injected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPlan:
    """One embedding request: the original input indices it covers, in order."""

    indices: tuple[int, ...]


def plan_batches(
    token_counts,
    batch_max: int,
    token_budget: int,
) -> list[BatchPlan]:
    """Pure: pack inputs into batches bounded by BOTH count and token budget.

    Walks the inputs in order and starts a new batch whenever adding the next
    input would exceed either the ``batch_max`` item count or the
    ``token_budget`` (the latter only once the current batch is non-empty, so an
    oversized single input still ships ALONE rather than being dropped). The
    trailing partial batch is always flushed.

    Both caps are clamped to >= 1 and each token count to >= 0 so a stray
    0/negative operator value can't produce an empty/degenerate or infinite plan.
    """
    batch_max = max(1, int(batch_max))
    token_budget = max(1, int(token_budget))

    plans: list[BatchPlan] = []
    current: list[int] = []
    current_tokens = 0

    for i, tok in enumerate(token_counts):
        tok = max(0, int(tok))
        # Flush before appending if the count cap is hit, or adding this input
        # would blow the token budget for an already-started batch.
        if len(current) >= batch_max or (current and current_tokens + tok > token_budget):
            plans.append(BatchPlan(tuple(current)))
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += tok

    if current:
        plans.append(BatchPlan(tuple(current)))
    return plans


async def embed_all(
    pool,
    inputs,
    token_counts,
    *,
    batch_max: int,
    token_budget: int,
    tier: str = "standard",
) -> list:
    """Thin shell: plan -> one ``pool.aembed`` per plan -> reassemble in order.

    Returns a list of vectors aligned to the original ``inputs`` order. Empty
    ``inputs`` short-circuits to an empty list (no network call). The pool owns
    all selection / failover; this function only routes data through it.
    """
    vectors: list = [None] * len(inputs)
    if not inputs:
        return vectors

    for plan in plan_batches(token_counts, batch_max, token_budget):
        resp = await pool.aembed(
            inputs=[inputs[i] for i in plan.indices],
            tier=tier,
        )
        for slot, item in zip(plan.indices, resp.data):
            vectors[slot] = item.embedding

    return vectors

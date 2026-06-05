"""Phase 6 (hardening) — rate-limit backoff + bounded concurrency tests.

Every test runs with ZERO infra: the executor takes injectable fakes (sleeper,
deterministic jitter, a duck-typed rate-error predicate) and the embeddings
wiring uses a monkeypatched ``embed_texts`` so no Azure OpenAI call is made.

Mirrors the existing pdf_chat test idiom: plain ``asyncio.run`` (no
pytest-asyncio marker needed) since each coroutine is self-contained.
"""
from __future__ import annotations

import asyncio

import pytest

from pdf_chat.ingestion.rate_limiter import (
    BoundedBackoffExecutor,
    RateLimitExhausted,
)


class _FakeRateError(Exception):
    """Stands in for a 429 from Azure OpenAI."""


def _is_rate_error(exc: Exception) -> bool:
    return isinstance(exc, _FakeRateError)


# --------------------------------------------------------------------------- #
# call_with_backoff — growth, exhaustion, non-rate passthrough
# --------------------------------------------------------------------------- #
def test_backoff_retries_with_growing_delay_then_succeeds():
    delays: list[float] = []

    async def _sleep(d: float) -> None:
        delays.append(d)

    calls = {"n": 0}

    async def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeRateError()
        return "ok"

    ex = BoundedBackoffExecutor(
        container_id="c1",
        is_rate_error=_is_rate_error,
        sleep=_sleep,
        jitter=lambda lo, hi: lo,  # deterministic: no random jitter
    )
    out = asyncio.run(ex.call_with_backoff(_flaky))
    assert out == "ok"
    assert calls["n"] == 3
    # two retries → two sleeps, each strictly larger (exponential growth)
    assert len(delays) == 2
    assert delays[1] > delays[0]


def test_backoff_raises_exhausted_after_max_attempts():
    attempts = {"n": 0}

    async def _sleep(d: float) -> None:
        return None

    async def _always_429() -> str:
        attempts["n"] += 1
        raise _FakeRateError()

    ex = BoundedBackoffExecutor(
        container_id="c1",
        is_rate_error=_is_rate_error,
        sleep=_sleep,
        jitter=lambda lo, hi: lo,
        max_attempts=3,  # explicit override for the test (still tunable-defaulted)
    )
    with pytest.raises(RateLimitExhausted):
        asyncio.run(ex.call_with_backoff(_always_429))
    # budget honored: the fn is called exactly max_attempts times, no more
    assert attempts["n"] == 3


def test_backoff_increments_backoff_and_dlq_metrics(monkeypatch):
    """Each 429 retry bumps pdf_embed_backoff_count and exhaustion bumps
    pdf_embed_dlq_count, both tenant-scoped on container_id (Fix 11)."""
    from pdf_chat.observability import metrics as pdf_metrics

    pdf_metrics.reset()

    async def _sleep(d: float) -> None:
        return None

    async def _always_429() -> str:
        raise _FakeRateError()

    ex = BoundedBackoffExecutor(
        container_id="tenant-A",
        is_rate_error=_is_rate_error,
        sleep=_sleep,
        jitter=lambda lo, hi: lo,
        max_attempts=3,
    )
    with pytest.raises(RateLimitExhausted):
        asyncio.run(ex.call_with_backoff(_always_429))

    snap = pdf_metrics.get_snapshot("tenant-A")
    # max_attempts=3 → two retries before the budget is spent → two backoff bumps.
    assert snap["pdf_embed_backoff_count"] == 2
    # exhaustion → exactly one DLQ bump.
    assert snap["pdf_embed_dlq_count"] == 1


def test_backoff_metrics_not_incremented_on_success(monkeypatch):
    """A call that never hits a 429 leaves both counters at zero (Fix 11)."""
    from pdf_chat.observability import metrics as pdf_metrics

    pdf_metrics.reset()

    async def _ok() -> str:
        return "ok"

    ex = BoundedBackoffExecutor(container_id="tenant-A", is_rate_error=_is_rate_error)
    assert asyncio.run(ex.call_with_backoff(_ok)) == "ok"
    snap = pdf_metrics.get_snapshot("tenant-A")
    assert snap["pdf_embed_backoff_count"] == 0
    assert snap["pdf_embed_dlq_count"] == 0


def test_backoff_does_not_retry_non_rate_errors():
    calls = {"n": 0}

    async def _sleep(d: float) -> None:
        return None

    async def _boom() -> str:
        calls["n"] += 1
        raise ValueError("not a rate limit")

    ex = BoundedBackoffExecutor(
        container_id="c1",
        is_rate_error=_is_rate_error,
        sleep=_sleep,
        jitter=lambda lo, hi: lo,
    )
    with pytest.raises(ValueError):
        asyncio.run(ex.call_with_backoff(_boom))
    # non-rate error re-raised immediately — fn called exactly once
    assert calls["n"] == 1


def test_default_is_rate_error_recognizes_429_and_503():
    ex = BoundedBackoffExecutor(container_id="c1")

    class _Err429(Exception):
        status_code = 429

    class _Err503(Exception):
        http_status = 503

    class _ErrOther(Exception):
        status_code = 400

    assert ex._is_rate_error(_Err429()) is True
    assert ex._is_rate_error(_Err503()) is True
    assert ex._is_rate_error(_ErrOther()) is False
    assert ex._is_rate_error(ValueError("x")) is False


# --------------------------------------------------------------------------- #
# gather_bounded — semaphore caps in-flight concurrency
# --------------------------------------------------------------------------- #
def test_gather_bounded_respects_max_concurrency():
    active = {"now": 0, "peak": 0}

    async def _task() -> int:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0)  # yield so others can start
        active["now"] -= 1
        return 1

    ex = BoundedBackoffExecutor(
        container_id="c1",
        is_rate_error=_is_rate_error,
        max_concurrency=2,  # explicit override (still tunable-defaulted)
    )
    results = asyncio.run(ex.gather_bounded([_task for _ in range(6)]))
    assert sum(results) == 6
    assert active["peak"] <= 2


def test_gather_bounded_preserves_order():
    def _factory(value: int):
        async def _call() -> int:
            await asyncio.sleep(0)
            return value

        return _call

    ex = BoundedBackoffExecutor(
        container_id="c1",
        is_rate_error=_is_rate_error,
        max_concurrency=3,
    )
    out = asyncio.run(ex.gather_bounded([_factory(i) for i in range(5)]))
    assert out == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# embed_texts_bounded — order-preserving fan-out over the limiter
# --------------------------------------------------------------------------- #
def test_embed_texts_bounded_preserves_order(monkeypatch):
    import pdf_chat.ingestion.embeddings as emb

    def _fake_embed(texts, *, model=None):
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    out = asyncio.run(
        emb.embed_texts_bounded([["a", "bb"], ["ccc"]], container_id="c1")
    )
    assert out == [[[1.0], [2.0]], [[3.0]]]

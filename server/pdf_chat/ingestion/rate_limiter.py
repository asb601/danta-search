"""Exponential-backoff + bounded-concurrency wrapper for Azure OpenAI calls.

At ingestion scale (thousands of embedding/extraction calls) Azure OpenAI returns
429/503. This executor retries rate-limited coroutines with exponential backoff
(base*2**attempt, capped at max_seconds, + jitter), bounds in-flight calls with a
semaphore, and raises :class:`RateLimitExhausted` when the attempt budget is spent
so the caller can DLQ the page (mirrors the retry/DLQ contract in the ingestion
tasks). Non-rate errors are re-raised immediately (never retried).

EVERY knob (base/max delay, max attempts, concurrency, jitter ratio) resolves from
``pdf_chat/tunables.py`` via ``get_tunable(container_id, key)`` with NO inline
numeric default — there is no delay/cap/score literal in this file (Spec §3 inv 4).
Every backoff retry and the exhaustion decision is emitted via ``log_gate_decision``
so a delay/budget is never compared-and-discarded silently.

Pure + import-safe: all infra (the sleeper, the rate-error predicate, the jitter
source) is injectable, so importing this module touches no network and tests run
with deterministic fakes.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

from ..observability import metrics as _metrics
from ..tunables import get_tunable, log_gate_decision

# Tunable keys (single source of truth lives in tunables.py::TUNABLE_DEFAULTS).
_TUN_BASE = "obs.backoff.base_seconds"
_TUN_MAX = "obs.backoff.max_seconds"
_TUN_ATTEMPTS = "obs.backoff.max_attempts"
_TUN_CONCURRENCY = "obs.embed.max_concurrency"
_TUN_JITTER = "obs.backoff.jitter_ratio"


class RateLimitExhausted(RuntimeError):
    """Raised when the backoff attempt budget is exhausted (caller should DLQ)."""


def _default_is_rate_error(exc: Exception) -> bool:
    """Duck-typed default: Azure OpenAI errors carry status_code / http_status."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "http_status", None)
    return status in (429, 503)


class BoundedBackoffExecutor:
    """Retries rate-limited coroutines with exponential backoff; bounds concurrency."""

    def __init__(
        self,
        container_id: str,
        *,
        is_rate_error: Callable[[Exception], bool] = _default_is_rate_error,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        base_seconds: float | None = None,
        max_seconds: float | None = None,
        max_attempts: int | None = None,
        max_concurrency: int | None = None,
        jitter_ratio: float | None = None,
    ) -> None:
        self.container_id = container_id
        self._is_rate_error = is_rate_error
        self._sleep = sleep
        self._jitter = jitter
        # Tunable defaults resolved per-container — never literals here (Spec §3.4).
        # An explicit override (None ⇒ resolve) keeps tests deterministic.
        self.base_seconds = (
            base_seconds
            if base_seconds is not None
            else float(get_tunable(container_id, _TUN_BASE))
        )
        self.max_seconds = (
            max_seconds
            if max_seconds is not None
            else float(get_tunable(container_id, _TUN_MAX))
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else int(get_tunable(container_id, _TUN_ATTEMPTS))
        )
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else int(get_tunable(container_id, _TUN_CONCURRENCY))
        )
        self.jitter_ratio = (
            jitter_ratio
            if jitter_ratio is not None
            else float(get_tunable(container_id, _TUN_JITTER))
        )
        self._sem = asyncio.Semaphore(self.max_concurrency)

    def _delay_for(self, attempt: int) -> float:
        """Exponential delay for ``attempt`` (1-based), capped, + bounded jitter."""
        raw = self.base_seconds * (2 ** attempt)
        capped = min(raw, self.max_seconds)
        spread = capped * self.jitter_ratio
        return self._jitter(capped, capped + spread)

    async def call_with_backoff(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Await ``fn`` retrying ONLY rate errors with exponential backoff.

        Non-rate errors propagate immediately. After ``max_attempts`` rate errors
        the budget is spent and :class:`RateLimitExhausted` is raised so the caller
        can DLQ the unit of work.
        """
        attempt = 0
        while True:
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 — re-raised below if non-rate
                if not self._is_rate_error(exc):
                    raise
                attempt += 1
                if attempt >= self.max_attempts:
                    # Score == attempts spent, threshold == budget → passed=True
                    # marks the budget as reached (exhausted); caller DLQs.
                    log_gate_decision(
                        "rate_limiter.backoff_exhausted",
                        score=attempt,
                        threshold=self.max_attempts,
                        outcome="dlq",
                        container_id=self.container_id,
                    )
                    # Make the advertised DLQ counter real (Fix 11): one bump per
                    # exhausted unit of work, tenant-scoped on container_id.
                    _metrics.inc(self.container_id, "pdf_embed_dlq_count")
                    raise RateLimitExhausted(
                        f"rate-limited after {attempt} attempts"
                    ) from exc
                delay = self._delay_for(attempt)
                log_gate_decision(
                    "rate_limiter.backoff_retry",
                    score=delay,
                    threshold=self.max_seconds,
                    outcome="retry",
                    container_id=self.container_id,
                    attempt=attempt,
                )
                # Make the advertised backoff counter real (Fix 11): one bump per
                # retry (429 absorbed), tenant-scoped on container_id.
                _metrics.inc(self.container_id, "pdf_embed_backoff_count")
                await self._sleep(delay)

    async def gather_bounded(
        self, factories: list[Callable[[], Awaitable[Any]]]
    ) -> list[Any]:
        """Run coroutine factories with at most ``max_concurrency`` in flight.

        Results preserve the input order. Each factory runs under the backoff
        retry path so a transient 429 in any one call is absorbed without aborting
        the whole fan-out.
        """

        async def _guarded(factory: Callable[[], Awaitable[Any]]) -> Any:
            async with self._sem:
                return await self.call_with_backoff(factory)

        return await asyncio.gather(*[_guarded(f) for f in factories])

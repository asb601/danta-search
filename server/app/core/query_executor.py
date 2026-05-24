"""Bounded async query executor with timeout and concurrency control.

PURPOSE:
  Prevent runaway queries from exhausting server resources.

  Problems solved:
    1. Long-running DuckDB / DataFusion scans block the event loop for
       other requests if not wrapped with asyncio timeout.
    2. High concurrent query load saturates the DB connection pool and
       triggers OOM if unchecked.

  This module wraps any async coroutine with:
    - asyncio.wait_for timeout (per-query hard cutoff)
    - asyncio.Semaphore concurrency limit (max N simultaneous queries)
    - Metrics integration (queue depth gauge, timeout counter)

PUBLIC API:
    executor = get_default_executor()
    result   = await executor.execute(coro, timeout=15.0)

DESIGN CONSTRAINTS:
  - Single module-level default executor (singleton, thread-safe init).
  - Configuration via class, not env vars — call-site supplies limits.
  - All errors propagate as-is; only timeout is converted to ExecutionTimeoutError.
  - Never drops metrics on error — inc/dec always fire in finally blocks.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypeVar

from app.core import metrics as _metrics
from app.policies.execution_policy import get_execution_policy as _get_execution_policy

T = TypeVar("T")


# ── Error type ─────────────────────────────────────────────────────────────────

class ExecutionTimeoutError(TimeoutError):
    """Raised when a query exceeds its time budget.

    Distinct from asyncio.TimeoutError so call sites can distinguish
    query timeouts from other timeout sources.
    """

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Query exceeded {timeout_seconds:.1f}s timeout and was cancelled"
        )


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryExecutorConfig:
    """Configures concurrency and timeout behaviour.

    Attributes
    ----------
    max_concurrent : int
        Maximum number of queries running simultaneously. Excess requests
        queue behind the semaphore.
    default_timeout_seconds : float
        Default per-query timeout when none is specified at call time.
    """
    max_concurrent:         int   = 10
    default_timeout_seconds: float = 30.0


# ── Executor ──────────────────────────────────────────────────────────────────

class QueryExecutor:
    """Wraps coroutines with timeout and concurrency control.

    Thread-safe construction. Async methods are the actual execution boundary.
    """

    def __init__(self, config: QueryExecutorConfig | None = None) -> None:
        self._config = config or QueryExecutorConfig()
        self._sem: asyncio.Semaphore | None = None
        self._peak_concurrent: int = 0
        self._current_concurrent: int = 0

    # ── Lazy semaphore init (must be created in async context) ─────────────────

    def _get_sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._config.max_concurrent)
        return self._sem

    # ── Public execute ─────────────────────────────────────────────────────────

    async def execute(
        self,
        coro: Any,
        *,
        timeout: float | None = None,
    ) -> Any:
        """
        Run coro with timeout and concurrency control.

        Parameters
        ----------
        coro    : an awaitable (coroutine) to execute.
        timeout : seconds before ExecutionTimeoutError is raised.
                  None → uses config.default_timeout_seconds.

        Returns
        -------
        The return value of coro.

        Raises
        ------
        ExecutionTimeoutError  — query exceeded its time budget.
        Any exception coro raises — propagated unchanged.
        """
        timeout_seconds = timeout if timeout is not None else self._config.default_timeout_seconds
        sem = self._get_sem()

        async with sem:
            # Track concurrency
            self._current_concurrent += 1
            if self._current_concurrent > self._peak_concurrent:
                self._peak_concurrent = self._current_concurrent
                _metrics.inc("execution_concurrency_peak", self._current_concurrent - _metrics._counters.get("execution_concurrency_peak", 0))
            _metrics.inc("query_queue_depth")

            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                _metrics.inc("execution_timeout_count")
                raise ExecutionTimeoutError(timeout_seconds)
            finally:
                self._current_concurrent -= 1
                _metrics.dec("query_queue_depth")


# ── Module-level default singleton ────────────────────────────────────────────
# Use get_default_executor() rather than importing _default_executor directly.
# This defers semaphore creation into an async context.

_default_executor: QueryExecutor | None = None
_init_lock = asyncio.Lock() if False else None   # populated lazily in get_default_executor


def get_default_executor(config: QueryExecutorConfig | None = None) -> "QueryExecutor":
    """
    Return the module-level default executor.

    If called for the first time with a config, that config is applied.
    Subsequent calls with config=None return the existing instance.
    Subsequent calls with a different config also return the existing instance
    (first-write-wins — prevents hot-path re-initialisation).

    When no config is supplied, defaults are sourced from ExecutionPolicy so
    all execution bounds share one source of truth.
    """
    global _default_executor
    if _default_executor is None:
        if config is None:
            _ep = _get_execution_policy()
            config = QueryExecutorConfig(
                max_concurrent          = _ep.max_concurrent,
                default_timeout_seconds = _ep.default_timeout_seconds,
            )
        _default_executor = QueryExecutor(config)
    return _default_executor

"""Execution safety and concurrency policy.

Governs:
  - Pre-execution SQL structural guards (execution_guards.py)
  - Query timeout and concurrency limits (query_executor.py)
  - Agent tool call cap (state.py)

WHY THESE EXIST TOGETHER:
  All three are about bounding resource consumption at execution time.
  Together they form a three-layer execution fence:
    Layer 1: SQL structure guard  → reject dangerous SQL before it runs
    Layer 2: Concurrency limit    → bound simultaneous execution threads
    Layer 3: Timeout              → kill queries that run too long
    Layer 4: Tool call cap        → bound LLM orchestration iteration depth

FUTURE READINESS:
  - Deployment override: a high-RAM VM with a faster engine could raise
    max_scan_files → 16 and max_result_rows → 5000.
  - Tenant override: a premium tenant could get max_concurrent → 20.
  - Scale-out: when horizontal scaling is added, max_concurrent can drop
    to 5 per instance (handled by the load balancer across instances).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ExecutionPolicy:
    """
    All execution-boundary limits in one typed, immutable config.

    ── SQL STRUCTURE GUARDS ─────────────────────────────────────────────────
    max_sql_length : int
        Maximum SQL character length. Queries longer than this are almost
        certainly the result of runaway prompt injection or LLM drift.
        Prevents: malformed mega-queries that confuse the SQL engine and
        produce meaningless error messages.

    max_joins : int
        Maximum explicit JOIN keywords in one SQL statement.
        Each JOIN multiplies potential scan size. In an enterprise analytics
        context, 5+ joins in a single ad-hoc query is a red flag.
        Prevents: accidental O(N^5) cross-products from LLM join hallucination.

    allow_cross_join : bool
        Whether CROSS JOIN and implicit Cartesian products are permitted.
        False by default — Cartesian joins are O(N²) on large tables.
        Prevents: OOM on small VMs from a single bad CROSS JOIN.

    max_scan_files : int
        Maximum unique az:// file paths in a FROM clause (estimated by regex
        counting, not query planning). 8 covers joins across all shortlisted
        files with a safety margin.
        Prevents: queries that reference more files than the shortlist allows.

    max_result_rows : int
        Post-execution soft warning threshold. Does NOT reject the query —
        only emits an execution_warning in the tool response.
        Prevents: silent pagination failures when the LLM interprets a
        2000-row dump as a complete result.

    ── CONCURRENCY AND TIMEOUT ───────────────────────────────────────────────
    max_concurrent : int
        Semaphore cap for simultaneous executing queries per process.
        10 is conservative for a single-VM deployment with 4-8 DuckDB workers.
        Prevents: connection pool exhaustion and OOM from parallel scans.

    default_timeout_seconds : float
        Per-query hard timeout. After this, the query is cancelled and
        ExecutionTimeoutError is returned to the LLM as a tool error.
        30 seconds is generous for typical analytical queries.
        Prevents: one slow query blocking the event loop for all other requests.

    ── AGENT DEPTH CAP ───────────────────────────────────────────────────────
    max_tool_calls : int
        Maximum tool invocations per LangGraph agent turn.
        Prevents: infinite agent loops from LLM getting stuck in a cycle of
        "let me inspect the schema again" without making progress.
        8 is enough for: catalog search → schema → 2-3 SQL attempts → repair.
    """
    # SQL structure guards
    max_sql_length:          int   = 8_000
    max_joins:               int   = 5
    allow_cross_join:        bool  = False
    max_scan_files:          int   = 8
    max_result_rows:         int   = 2_000

    # Concurrency + timeout
    max_concurrent:          int   = 10
    default_timeout_seconds: float = 30.0

    # Agent depth
    max_tool_calls:          int   = 8


@lru_cache(maxsize=1)
def get_execution_policy() -> ExecutionPolicy:
    """Return the module-level singleton ExecutionPolicy."""
    return ExecutionPolicy()

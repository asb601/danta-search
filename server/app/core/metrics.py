"""In-process metrics counters — lightweight observability without an external collector.

Tracks the key metrics from RND_IMPLEMENTATION_PLAN Phase 4:
  - query_duration          (p50/p95/p99 via rolling sample window)
  - azure_blob_bytes_read   (total bytes fetched from Azure per engine)
  - azure_blob_429_count    (throttle events)
  - catalog_miss_count      (catalog returned empty / fallback used)
  - llm_sql_failure_count   (run_sql returned an error to the LLM)
  - query_queue_depth       (active concurrent queries right now)
  - sql_forbidden_count     (SQL safety layer rejections)

Exposed via GET /api/metrics (see main.py).

All operations are thread-safe (threading.Lock + atomic int ops via threading module).
No external dependencies.
"""
from __future__ import annotations

import threading
import time
from collections import deque

# ── Latency sample window ─────────────────────────────────────────────────────
# Keep the last N query durations to compute approximate p50/p95/p99.
_LATENCY_WINDOW = 500

_lock = threading.Lock()

_latency_samples: deque[float] = deque(maxlen=_LATENCY_WINDOW)

_counters: dict[str, int | float] = {
    # Query engine
    "query_total":              0,
    "query_errors":             0,
    "query_queue_depth":        0,   # currently executing queries
    # Azure blob
    "azure_bytes_read":         0,
    "azure_bytes_written":      0,
    "azure_429_count":          0,
    # Catalog
    "catalog_miss_count":       0,   # catalog empty / fallback fired
    "catalog_fallback_count":   0,   # retrieval returned 0, used kw fallback
    # LLM / SQL
    "llm_sql_failure_count":    0,   # run_sql returned {"error": ...}
    "sql_forbidden_count":      0,   # sql_safety rejected a keyword
    "sql_blob_acl_denied":      0,   # sql_safety rejected an unauthorised az:// path
    # Ingestion
    "parquet_conversions":      0,
    "parquet_conversion_errors": 0,
    # SQL repair telemetry
    "sql_repair_tier1_count":      0,   # deterministic Tier-1 repair applied
    "sql_repair_tier2_count":      0,   # LLM Tier-2 repair applied
    "sql_repair_declined_count":   0,   # repair attempted, returned None (no fix found)
    "sql_repair_intent_rejected":  0,   # _validate_repair_intent check failed
    # Graph + join telemetry
    "weak_join_surfaced_count":    0,   # joins below soft floor but above hard floor
    "graph_health_degraded_count": 0,   # graph health computed as degraded or poor
    "orphan_entity_count":         0,   # orphan entities detected
    # Retrieval telemetry
    "retrieval_miss_count":        0,   # retrieval returned 0 results
    "resolver_miss_count":         0,   # entity resolver returned 0 candidates
    # Execution infrastructure
    "execution_timeout_count":     0,   # query_executor timeout fired
    "execution_guard_rejection":   0,   # execution guard hard pre-execution rejection
    "execution_concurrency_peak":  0,   # peak concurrent queries (gauge, not cumulative)
    # Confidence telemetry
    "low_confidence_query_count":  0,   # orchestration confidence < 0.50
    "ingestion_audit_error_count": 0,   # ingestion audit found error-level findings
    # Planner telemetry
    "planner_fallback_count":      0,   # deterministic planner declined → LangGraph agent path taken
    # Navigator telemetry
    "navigator_abstain_count":     0,   # navigator abstained (any reason) → agent fall-through
}


# ── Named increment helpers ───────────────────────────────────────────────────
# Thin wrappers over inc() so call-sites read intentionally and the counter name
# is referenced in exactly one place (typo-proof, grep-able).

def inc_planner_fallback(amount: int = 1) -> None:
    """Record that the deterministic semantic planner declined and the query
    fell back to the LangGraph agent path.

    CLAUDE.md names this the primary quality metric for the planner layer: the
    fallback rate drives ontology-coverage work. Increment it at the single
    site where a high-confidence deterministic plan would have bypassed the
    agent but did not (see TODO in app/agent/graph/graph.py — Dev-B does not
    own graph.py, so the increment site is wired by the lead).
    """
    inc("planner_fallback_count", amount)


def inc_navigator_abstain(reason: str, amount: int = 1) -> None:
    """Record that the navigator abstained (returned None → agent fall-through).

    Increments the total ``navigator_abstain_count`` AND a per-reason counter
    (``navigator_abstain.<reason>``) so a SYSTEMIC wiring regression (e.g. every
    step abstaining on the SAME verify reason) is distinguishable in /api/metrics
    from a healthy spread of honest "no plan fits" abstains. The reason is a
    data-driven label, not a hardcoded enum — new reason strings self-register.
    """
    inc("navigator_abstain_count", amount)
    if reason:
        inc(f"navigator_abstain.{reason}", amount)


# ── Public API ────────────────────────────────────────────────────────────────

def inc(key: str, amount: int | float = 1) -> None:
    """Increment a named counter by amount."""
    with _lock:
        _counters[key] = _counters.get(key, 0) + amount


def dec(key: str, amount: int | float = 1) -> None:
    """Decrement a named counter (for gauge-style metrics like queue_depth)."""
    with _lock:
        _counters[key] = max(0, _counters.get(key, 0) - amount)


def record_query_duration(duration_ms: float) -> None:
    """Add a query duration sample (milliseconds) to the rolling window."""
    with _lock:
        _latency_samples.append(duration_ms)


def get_snapshot() -> dict:
    """Return a point-in-time snapshot suitable for the /api/metrics response."""
    with _lock:
        snap = dict(_counters)
        samples = sorted(_latency_samples)

    n = len(samples)
    if n == 0:
        snap["latency_p50_ms"] = None
        snap["latency_p95_ms"] = None
        snap["latency_p99_ms"] = None
        snap["latency_sample_count"] = 0
    else:
        snap["latency_p50_ms"] = round(_percentile(samples, 50), 1)
        snap["latency_p95_ms"] = round(_percentile(samples, 95), 1)
        snap["latency_p99_ms"] = round(_percentile(samples, 99), 1)
        snap["latency_sample_count"] = n

    snap["snapshot_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return snap


# ── Internal helpers ──────────────────────────────────────────────────────────

def _percentile(sorted_samples: list[float], p: int) -> float:
    if not sorted_samples:
        return 0.0
    idx = (p / 100) * (len(sorted_samples) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = idx - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac

"""In-process per-tenant metrics for pdf_chat (Phase 6 hardening).

Mirrors server/app/core/metrics.py (counters + a rolling latency window,
thread-safe, no external collector) but is keyed by ``tenant_id`` so
/api/pdf/metrics reports a tenant's OWN numbers. The known counters cover query
volume, fallback rate, graph traversal, cross-domain bridge refusals, ingestion
rate-limit/backoff events, and cascading-delete events. Unknown counters may be
created lazily by ``inc`` and surface in the snapshot too.

Pure + import-safe: in-process accumulators only — importing this touches no
infra. The window size + percentiles are STRUCTURAL caps (how much history to
keep / how to summarize it), not score-comparison thresholds (spec §3.4).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# Structural cap: keep the last N latencies per tenant for approximate p50/p95/p99.
_LATENCY_WINDOW = 500

# Known counters (others may be created lazily by ``inc``). Documented so the
# /api/pdf/metrics surface is stable — every known key is always present (0 when
# never incremented) so a dashboard never sees a key appear/disappear.
_KNOWN = (
    "pdf_query_total",
    "pdf_query_errors",
    "pdf_fallback_count",
    "pdf_graph_traverse_count",
    "pdf_cross_domain_refused_count",
    "pdf_embed_backoff_count",
    "pdf_embed_dlq_count",
    "pdf_document_deleted_count",
    "pdf_orphan_entity_deleted_count",
)

_lock = threading.Lock()
_counters: dict[str, dict[str, int | float]] = defaultdict(lambda: defaultdict(int))
_latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=_LATENCY_WINDOW))


def inc(tenant_id: str, key: str, amount: int | float = 1) -> None:
    with _lock:
        _counters[tenant_id][key] = _counters[tenant_id].get(key, 0) + amount


def observe_latency(tenant_id: str, duration_ms: float) -> None:
    with _lock:
        _latency[tenant_id].append(duration_ms)


def reset() -> None:
    """Test seam: clear all tenant counters + latency windows."""
    with _lock:
        _counters.clear()
        _latency.clear()


def get_snapshot(tenant_id: str) -> dict:
    with _lock:
        counters = dict(_counters.get(tenant_id, {}))
        samples = sorted(_latency.get(tenant_id, ()))
    snap: dict = {k: counters.get(k, 0) for k in _KNOWN}
    snap.update(counters)  # include any lazily-created counters
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


def _percentile(sorted_samples: list[float], p: int) -> float:
    """Linear-interpolated percentile over an already-sorted sample list."""
    if not sorted_samples:
        return 0.0
    idx = (p / 100) * (len(sorted_samples) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = idx - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac

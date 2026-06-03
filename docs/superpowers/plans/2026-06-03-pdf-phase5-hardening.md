# PDF Agentic Graph RAG — Phase 5 (Hardening) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the `server/pdf_chat/` GraphRAG module with per-tenant cost & observability (trace IDs + a `/api/pdf/metrics` surface), a tenant-scoped cascading document delete that never orphans shared entities, exponential-backoff + max-concurrency for Azure OpenAI calls at ingestion scale, and an expanded CI-runnable gold-question eval harness that fails the build when correctness/faithfulness regress.

**Architecture:** Phase 5 adds four cross-cutting concerns *around* the existing pipeline without changing query semantics. All four mirror the main app's proven patterns under `server/app/core/` (`cost_tracker.py`, `metrics.py`, `orchestration_trace.py`) but live inside `pdf_chat/` so the two pipelines stay independent (per `pdf_chat/CLAUDE.md` rule #7). Every threshold/cap/base-delay resolves from `pdf_chat/tunables.py` (`get_tunable`/`log_gate_decision`) — there is **no score-comparison literal in any `.py` file** (spec §3.4). All Neo4j Cypher is tenant-scoped on every path element (spec §3.3).

**Tech Stack:** Python 3.12 · `uv` · pytest · FastAPI · SQLAlchemy 2.0 async · Neo4j driver (guarded import) · structlog · Azure OpenAI SDK (guarded import) · Celery.

---

## Cross-Phase Contracts (assumptions this plan depends on)

These artifacts are assumed to exist from Phases 0–4. Each task that touches one re-states the assumption and degrades safely if absent.

| Depended-on artifact | From phase | How this plan uses it |
|---|---|---|
| `pdf_chat/tunables.py` :: `get_tunable(container_id, key, default)` and `log_gate_decision(logger, gate, *, score, threshold, decision, **ctx)` | Phase 0 | Single tunables source; every cap/threshold/base-delay reads through it. No literals in `.py`. |
| `pdf_chat/ingestion/embeddings.py` :: `embed_texts(texts, *, model=None)` (exists today, `embeddings.py:35`) | Phase 0/1 | Wrapped by the backoff/concurrency limiter. |
| `pdf_chat/ingestion/neo4j_writer.py` :: graph schema `(:Chunk)-[:MENTIONS]->(:Entity)`, `(:Entity)-[:RELATED_TO]->(:Entity)`, `(:Entity)-[:IN_COMMUNITY]->(:Community)`, every node carries `tenant_id` (spec §2 L1b; today writer only does `Document→Page→Chunk` at `neo4j_writer.py:84-99`) | Phase 2 | Cascading delete batches `DETACH DELETE` over exactly these labels. |
| `pdf_chat/control_plane/repositories.py` :: `UploadManifestRepo.set_status(upload_id, status, ...)` (`repositories.py:50`) | Phase 0 | Soft-delete marks the manifest `deleted` before async cleanup. |
| `pdf_chat/testing/eval/gold_questions.py` :: the Phase-1 gold-question set + a `run_eval()` harness producing per-question records | Phase 1 | Phase 5 expands the question set and adds the CI gate on top of `run_eval()`. |
| `pdf_chat/agent/graph.py` :: `run_pdf_query(query, *, tenant_id, container_id, ...)` returning an answer object with `.answer`, `.citations` | Phase 3 | The eval harness calls this per gold question; cost tracker is invoked from synthesis inside it. |
| `pdf_chat/api/routes.py` :: `delete_document` route already calls `pdf_chat.control_plane.delete_service.delete_document` (`routes.py:258`) | Phase 0 (route stub) | Task 4 implements that exact module/function the route imports. |

If `run_pdf_query` or the Phase-2 graph is absent at execution time, the eval harness records the question as `fallback` and the cascading-delete tests use a fake Neo4j session (no live infra needed) — both are designed to run with zero infra, matching `pdf_chat/testing/` conventions.

---

## File Structure

| File | New? | Responsibility |
|---|---|---|
| `server/pdf_chat/observability/__init__.py` | new | Package marker re-exporting the public observability surface. |
| `server/pdf_chat/observability/trace.py` | new | `PdfTrace` — per-request/per-document trace object + `new_trace_id()`; JSON-safe, never raises. |
| `server/pdf_chat/observability/cost_tracker.py` | new | `PdfCostTracker` — per-tenant token/cost accumulator for extraction + synthesis; `track_llm`, `snapshot`, `reset`. |
| `server/pdf_chat/observability/metrics.py` | new | In-process per-tenant counters + latency window; `inc`, `observe_latency`, `get_snapshot` (consumed by the metrics route). |
| `server/pdf_chat/observability/logging.py` | new | `get_pdf_logger(name)` + `bind_trace(logger, trace_id, tenant_id)` — structlog wiring with trace IDs. |
| `server/pdf_chat/ingestion/rate_limiter.py` | new | `BoundedBackoffExecutor` — exponential backoff + bounded concurrency around Azure OpenAI calls; `call_with_backoff`, `gather_bounded`. |
| `server/pdf_chat/ingestion/embeddings.py` | modify | Route `embed_texts` through the rate limiter when enabled (tunable-gated). |
| `server/pdf_chat/control_plane/delete_service.py` | new | `delete_document(upload_id, tenant_id)` (soft-delete) + `cleanup_deleted_document(upload_id, tenant_id, neo4j_session)` (async cascade). Implements the function `routes.py:258` already imports. |
| `server/pdf_chat/control_plane/graph_delete.py` | new | Pure Cypher builders + orphan-detection logic for the cascade (tenant-scoped, no infra). |
| `server/pdf_chat/api/routes.py` | modify | Add `GET /api/pdf/metrics`; wire a trace ID into upload/chat/delete. |
| `server/pdf_chat/testing/eval/gold_questions.py` | modify | Expand the gold set: graph-traversal, global/community, cross-domain, negative-claim cases. |
| `server/pdf_chat/testing/eval/harness.py` | new | `evaluate(records)` → `EvalReport` (fallback rate, faithfulness, correctness) + `assert_thresholds(report, container_id)`. |
| `server/pdf_chat/testing/eval/run_ci_eval.py` | new | CLI entry: run eval, print report, exit non-zero when below tunable thresholds. |
| `server/pdf_chat/testing/test_observability.py` | new | Tests for trace, cost tracker, metrics, logging. |
| `server/pdf_chat/testing/test_rate_limiter.py` | new | Tests for backoff growth + concurrency cap + DLQ-on-exhaustion. |
| `server/pdf_chat/testing/test_delete_service.py` | new | Tests for soft-delete + cascade leaving shared entities intact. |
| `server/pdf_chat/testing/test_eval_harness.py` | new | Tests that the harness scores cases and fails the build below thresholds. |

All tests run via `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -v` with zero infra (guarded imports + fakes), matching the existing `pdf_chat/testing/test_ingestion.py` style (`test_ingestion.py:1-29`).

---

## Task 1: Per-tenant trace IDs + structured logging

**Files:**
- Create: `server/pdf_chat/observability/__init__.py`
- Create: `server/pdf_chat/observability/logging.py`
- Create: `server/pdf_chat/observability/trace.py`
- Test: `server/pdf_chat/testing/test_observability.py`

- [ ] **Step 1: Write the failing test for trace ID + JSON-safe trace**

```python
# server/pdf_chat/testing/test_observability.py
from __future__ import annotations

import uuid

from pdf_chat.observability.trace import PdfTrace, new_trace_id


def test_new_trace_id_is_unique_uuid_hex():
    a, b = new_trace_id(), new_trace_id()
    assert a != b
    assert uuid.UUID(hex=a)  # parses as a valid uuid


def test_trace_records_stages_and_is_json_safe():
    t = PdfTrace(trace_id="t1", tenant_id="tenant-A")
    t.set_stage("retrieval", {"chunks": 5, "secret": object()})  # non-serializable value
    d = t.as_dict()
    assert d["trace_id"] == "t1"
    assert d["tenant_id"] == "tenant-A"
    assert d["stages"]["retrieval"]["chunks"] == 5
    # non-serializable coerced to str, never raises
    assert isinstance(d["stages"]["retrieval"]["secret"], str)


def test_trace_setters_never_raise():
    t = PdfTrace(trace_id="t1", tenant_id="tenant-A")
    t.set_stage("x", None)  # must not raise
    t.emit()  # must not raise even with no logger configured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.observability'`

- [ ] **Step 3: Implement the package + logging wiring**

```python
# server/pdf_chat/observability/__init__.py
"""Phase 5 observability surface for pdf_chat — trace IDs, per-tenant cost, metrics.

Mirrors server/app/core/{orchestration_trace,cost_tracker,metrics,logger}.py but
lives inside pdf_chat so the two pipelines stay independent (pdf_chat/CLAUDE.md #7).
"""
from __future__ import annotations

from .cost_tracker import PdfCostTracker
from .logging import bind_trace, get_pdf_logger
from .metrics import get_snapshot, inc, observe_latency
from .trace import PdfTrace, new_trace_id

__all__ = [
    "PdfTrace",
    "new_trace_id",
    "PdfCostTracker",
    "get_pdf_logger",
    "bind_trace",
    "inc",
    "observe_latency",
    "get_snapshot",
]
```

```python
# server/pdf_chat/observability/logging.py
"""structlog wiring for pdf_chat with trace-ID binding.

The main app's logger (server/app/core/logger.py) already configures structlog
process-wide. We reuse that configuration and only add named loggers + a helper
to bind a trace_id/tenant_id onto every event for correlation.
"""
from __future__ import annotations

from typing import Any

import structlog


def get_pdf_logger(name: str) -> Any:
    """Return a named structlog logger under the pdf_chat namespace."""
    return structlog.get_logger(f"pdf_chat.{name}")


def bind_trace(logger: Any, trace_id: str, tenant_id: str) -> Any:
    """Bind correlation fields so every subsequent event carries them.

    Returns a bound logger; the original is unchanged (structlog is immutable).
    """
    return logger.bind(trace_id=trace_id, tenant_id=tenant_id)
```

```python
# server/pdf_chat/observability/trace.py
"""Per-request / per-document trace for pdf_chat.

Mirrors server/app/core/orchestration_trace.py: JSON-safe, never raises, emits a
single structured event at the end. Carries trace_id + tenant_id for correlation.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .logging import bind_trace, get_pdf_logger

_MAX_STR = 200
_MAX_LIST = 20
_MAX_KEYS = 30


def new_trace_id() -> str:
    """A fresh hex trace id (uuid4)."""
    return uuid.uuid4().hex


def _safe_val(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:_MAX_STR]
    if isinstance(v, dict):
        items = list(v.items())[:_MAX_KEYS]
        return {str(k)[:80]: _safe_val(vv) for k, vv in items}
    if isinstance(v, (list, tuple)):
        return [_safe_val(i) for i in list(v)[:_MAX_LIST]]
    return str(v)[:_MAX_STR]


class PdfTrace:
    """Accumulates per-stage telemetry for one pdf_chat invocation. Never raises."""

    __slots__ = ("trace_id", "tenant_id", "_created_at", "_stages")

    def __init__(self, trace_id: str, tenant_id: str) -> None:
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self._created_at = time.perf_counter()
        self._stages: dict[str, Any] = {}

    def set_stage(self, name: str, data: Any) -> None:
        try:
            self._stages[str(name)[:80]] = _safe_val(data)
        except Exception:
            pass

    def as_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "stages": dict(self._stages),
        }

    def emit(self) -> None:
        try:
            elapsed_ms = round((time.perf_counter() - self._created_at) * 1000, 2)
            logger = bind_trace(get_pdf_logger("trace"), self.trace_id, self.tenant_id)
            logger.info("pdf_trace", elapsed_ms=elapsed_ms, stages=self._stages)
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/observability/__init__.py server/pdf_chat/observability/logging.py server/pdf_chat/observability/trace.py server/pdf_chat/testing/test_observability.py
git commit -m "feat(pdf): per-request trace IDs + structured logging for pdf_chat

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Per-tenant cost tracker (extraction + synthesis)

**Files:**
- Create: `server/pdf_chat/observability/cost_tracker.py`
- Test: `server/pdf_chat/testing/test_observability.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to server/pdf_chat/testing/test_observability.py
from pdf_chat.observability.cost_tracker import PdfCostTracker


def test_cost_tracker_accumulates_per_tenant():
    ct = PdfCostTracker()
    ct.track_llm("tenant-A", "extraction", "gpt-4o-mini",
                 prompt_tokens=1000, completion_tokens=200,
                 cost_usd=0.0003, document_id="doc1", trace_id="t1")
    ct.track_llm("tenant-A", "synthesis", "gpt-4o-mini",
                 prompt_tokens=500, completion_tokens=100,
                 cost_usd=0.0001, document_id=None, trace_id="t2")
    ct.track_llm("tenant-B", "synthesis", "gpt-4o-mini",
                 prompt_tokens=10, completion_tokens=5,
                 cost_usd=0.00001, document_id=None, trace_id="t3")

    a = ct.snapshot("tenant-A")
    assert a["llm_calls"] == 2
    assert a["prompt_tokens"] == 1500
    assert a["completion_tokens"] == 300
    assert round(a["cost_usd"], 6) == 0.0004
    # per-phase breakdown
    assert a["by_phase"]["extraction"]["llm_calls"] == 1
    assert a["by_phase"]["synthesis"]["llm_calls"] == 1

    b = ct.snapshot("tenant-B")
    assert b["llm_calls"] == 1
    # tenant isolation: A's totals never leak into B
    assert b["prompt_tokens"] == 10


def test_cost_tracker_rejects_non_mini_model():
    ct = PdfCostTracker()
    # spec §6: no gpt-4o anywhere. A gpt-4o call is recorded but flagged.
    ct.track_llm("tenant-A", "synthesis", "gpt-4o",
                 prompt_tokens=1, completion_tokens=1, cost_usd=0.01,
                 document_id=None, trace_id="t1")
    a = ct.snapshot("tenant-A")
    assert a["policy_violations"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -k cost_tracker -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'PdfCostTracker'`

- [ ] **Step 3: Implement the cost tracker**

```python
# server/pdf_chat/observability/cost_tracker.py
"""Per-tenant LLM cost & token accumulator for pdf_chat.

Mirrors server/app/core/cost_tracker.py but is keyed by tenant_id and splits
ingestion-extraction vs query-synthesis usage. Every recorded call is emitted as
one structured event (with trace_id + document_id) and accumulated per tenant so
the metrics surface can report cost_usd / tokens per tenant.

POLICY (spec §6): only gpt-4o-mini is allowed. A non-mini model is still recorded
(so cost is never silently lost) but increments policy_violations and is logged.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from .logging import get_pdf_logger

_ALLOWED_MODEL_SUBSTR = "mini"  # NOT a score literal; a model-policy marker (spec §6)


def _blank() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "policy_violations": 0,
        "by_phase": defaultdict(
            lambda: {"llm_calls": 0, "prompt_tokens": 0,
                     "completion_tokens": 0, "cost_usd": 0.0}
        ),
    }


class PdfCostTracker:
    """Thread-safe per-tenant cost accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_tenant: dict[str, dict[str, Any]] = defaultdict(_blank)
        self._logger = get_pdf_logger("cost")

    def track_llm(
        self,
        tenant_id: str,
        phase: str,  # "extraction" | "synthesis"
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        document_id: str | None,
        trace_id: str | None,
    ) -> None:
        violation = _ALLOWED_MODEL_SUBSTR not in (model or "").lower()
        with self._lock:
            t = self._by_tenant[tenant_id]
            t["llm_calls"] += 1
            t["prompt_tokens"] += prompt_tokens
            t["completion_tokens"] += completion_tokens
            t["cost_usd"] += cost_usd
            if violation:
                t["policy_violations"] += 1
            p = t["by_phase"][phase]
            p["llm_calls"] += 1
            p["prompt_tokens"] += prompt_tokens
            p["completion_tokens"] += completion_tokens
            p["cost_usd"] += cost_usd
        self._logger.info(
            "pdf_llm_call",
            tenant_id=tenant_id,
            phase=phase,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost_usd, 8),
            document_id=document_id,
            trace_id=trace_id,
            policy_violation=violation,
        )

    def snapshot(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            t = self._by_tenant.get(tenant_id)
            if t is None:
                return _materialize(_blank())
            return _materialize(t)

    def reset(self, tenant_id: str | None = None) -> None:
        with self._lock:
            if tenant_id is None:
                self._by_tenant.clear()
            else:
                self._by_tenant.pop(tenant_id, None)


def _materialize(t: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy the accumulator into a plain dict (defaultdicts → dicts)."""
    return {
        "llm_calls": t["llm_calls"],
        "prompt_tokens": t["prompt_tokens"],
        "completion_tokens": t["completion_tokens"],
        "cost_usd": round(t["cost_usd"], 6),
        "policy_violations": t["policy_violations"],
        "by_phase": {k: dict(v) for k, v in t["by_phase"].items()},
    }


# Process-wide singleton (mirrors app/core/cost_tracker's module-level session).
_TRACKER = PdfCostTracker()


def get_cost_tracker() -> PdfCostTracker:
    return _TRACKER
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -k cost_tracker -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/observability/cost_tracker.py server/pdf_chat/observability/__init__.py server/pdf_chat/testing/test_observability.py
git commit -m "feat(pdf): per-tenant cost tracker for extraction + synthesis

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Per-tenant metrics surface + /api/pdf/metrics route

**Files:**
- Create: `server/pdf_chat/observability/metrics.py`
- Modify: `server/pdf_chat/api/routes.py` (add `GET /api/pdf/metrics`)
- Test: `server/pdf_chat/testing/test_observability.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to server/pdf_chat/testing/test_observability.py
from pdf_chat.observability import metrics as pdf_metrics


def test_metrics_inc_and_latency_snapshot():
    pdf_metrics.reset()
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    pdf_metrics.inc("tenant-A", "pdf_fallback_count")
    pdf_metrics.observe_latency("tenant-A", 120.0)
    pdf_metrics.observe_latency("tenant-A", 240.0)

    snap = pdf_metrics.get_snapshot("tenant-A")
    assert snap["pdf_query_total"] == 2
    assert snap["pdf_fallback_count"] == 1
    assert snap["latency_p50_ms"] is not None
    assert snap["latency_sample_count"] == 2


def test_metrics_are_tenant_scoped():
    pdf_metrics.reset()
    pdf_metrics.inc("tenant-A", "pdf_query_total")
    snap_b = pdf_metrics.get_snapshot("tenant-B")
    assert snap_b["pdf_query_total"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -k metrics -v`
Expected: FAIL with `ImportError` / `AttributeError: module 'pdf_chat.observability.metrics' has no attribute 'reset'`

- [ ] **Step 3: Implement per-tenant metrics**

```python
# server/pdf_chat/observability/metrics.py
"""In-process per-tenant metrics for pdf_chat.

Mirrors server/app/core/metrics.py (counters + rolling latency window, thread-safe,
no external collector) but is keyed by tenant_id so /api/pdf/metrics reports a
tenant's own numbers. Exposed counters cover query volume, fallback rate, graph
traversal, cross-domain bridge refusals, and ingestion rate-limit/backoff events.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

_LATENCY_WINDOW = 500

# Known counters (others may be created lazily). Documented so the surface is stable.
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
    if not sorted_samples:
        return 0.0
    idx = (p / 100) * (len(sorted_samples) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = idx - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py -k metrics -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the /api/pdf/metrics route**

Read `server/pdf_chat/api/routes.py:237-265` first to match the `get_principal` dependency style (the delete route there uses `principal.tenant_id`). Add after the `list_documents` route:

```python
# server/pdf_chat/api/routes.py — add near the other @pdf_router routes
from pdf_chat.observability import metrics as _pdf_metrics  # top-of-file import
from pdf_chat.observability.cost_tracker import get_cost_tracker  # top-of-file import


@pdf_router.get("/metrics")
async def pdf_metrics(principal: Principal = Depends(get_principal)) -> dict:
    """Per-tenant pdf_chat metrics + cost snapshot.

    Consistent with the main app's GET /api/metrics (main.py:309) but tenant-scoped
    to the JWT principal so a tenant only sees its own numbers.
    """
    return {
        "tenant_id": principal.tenant_id,
        "metrics": _pdf_metrics.get_snapshot(principal.tenant_id),
        "cost": get_cost_tracker().snapshot(principal.tenant_id),
    }
```

- [ ] **Step 6: Verify the module imports cleanly (route smoke)**

Run: `cd server && uv run python -c "import pdf_chat.api.routes as r; print([route.path for route in r.pdf_router.routes])"`
Expected: output includes `/api/pdf/metrics`

- [ ] **Step 7: Commit**

```bash
git add server/pdf_chat/observability/metrics.py server/pdf_chat/observability/__init__.py server/pdf_chat/api/routes.py server/pdf_chat/testing/test_observability.py
git commit -m "feat(pdf): per-tenant metrics surface + GET /api/pdf/metrics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rate-limit backoff + bounded concurrency for Azure OpenAI calls

**Files:**
- Create: `server/pdf_chat/ingestion/rate_limiter.py`
- Modify: `server/pdf_chat/ingestion/embeddings.py` (route through the limiter)
- Test: `server/pdf_chat/testing/test_rate_limiter.py`

**Tunables used (all read via `get_tunable`, no literals in `.py`):**
`pdf_backoff_base_seconds`, `pdf_backoff_max_seconds`, `pdf_backoff_max_attempts`, `pdf_embed_max_concurrency`, `pdf_backoff_jitter_ratio`.

- [ ] **Step 1: Write the failing test (deterministic — inject sleeper + clock)**

```python
# server/pdf_chat/testing/test_rate_limiter.py
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
    async def _sleep(d: float) -> None:
        return None

    async def _always_429() -> str:
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


def test_backoff_does_not_retry_non_rate_errors():
    async def _sleep(d: float) -> None:
        return None

    async def _boom() -> str:
        raise ValueError("not a rate limit")

    ex = BoundedBackoffExecutor(
        container_id="c1", is_rate_error=_is_rate_error, sleep=_sleep,
        jitter=lambda lo, hi: lo,
    )
    with pytest.raises(ValueError):
        asyncio.run(ex.call_with_backoff(_boom))


def test_gather_bounded_respects_max_concurrency():
    active = {"now": 0, "peak": 0}

    async def _task() -> int:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0)  # yield so others can start
        active["now"] -= 1
        return 1

    ex = BoundedBackoffExecutor(
        container_id="c1", is_rate_error=_is_rate_error,
        max_concurrency=2,  # explicit override (still tunable-defaulted)
    )
    results = asyncio.run(ex.gather_bounded([_task for _ in range(6)]))
    assert sum(results) == 6
    assert active["peak"] <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_rate_limiter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.ingestion.rate_limiter'`

- [ ] **Step 3: Implement the executor (tunable-driven, fully injectable)**

```python
# server/pdf_chat/ingestion/rate_limiter.py
"""Exponential-backoff + bounded-concurrency wrapper for Azure OpenAI calls.

At ingestion scale (thousands of embedding/extraction calls) Azure OpenAI returns
429/503. This executor retries with exponential backoff (base*2**attempt, capped),
bounds in-flight calls with a semaphore, and raises RateLimitExhausted when the
attempt budget is spent so the caller can DLQ the page (mirrors the retry/DLQ
contract in pdf_chat/ingestion/tasks.py).

EVERY knob (base/max delay, max attempts, concurrency, jitter ratio) resolves from
pdf_chat/tunables.py — there is NO score/delay literal in this file (spec §3.4).
Every backoff/exhaustion decision is logged with its value via log_gate_decision.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

from ..tunables import get_tunable, log_gate_decision
from ..observability.logging import get_pdf_logger

_logger = get_pdf_logger("rate_limiter")


class RateLimitExhausted(RuntimeError):
    """Raised when the backoff attempt budget is exhausted (caller should DLQ)."""


def _default_is_rate_error(exc: Exception) -> bool:
    # Duck-typed: Azure OpenAI raises objects carrying status_code/http_status.
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
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
        # Tunable defaults — never literals in this file (spec §3.4).
        self.base_seconds = base_seconds if base_seconds is not None else float(
            get_tunable(container_id, "pdf_backoff_base_seconds", 1.0))
        self.max_seconds = max_seconds if max_seconds is not None else float(
            get_tunable(container_id, "pdf_backoff_max_seconds", 60.0))
        self.max_attempts = max_attempts if max_attempts is not None else int(
            get_tunable(container_id, "pdf_backoff_max_attempts", 5))
        self.max_concurrency = max_concurrency if max_concurrency is not None else int(
            get_tunable(container_id, "pdf_embed_max_concurrency", 8))
        self.jitter_ratio = jitter_ratio if jitter_ratio is not None else float(
            get_tunable(container_id, "pdf_backoff_jitter_ratio", 0.2))
        self._sem = asyncio.Semaphore(self.max_concurrency)

    def _delay_for(self, attempt: int) -> float:
        raw = self.base_seconds * (2 ** attempt)
        capped = min(raw, self.max_seconds)
        spread = capped * self.jitter_ratio
        return self._jitter(capped, capped + spread)

    async def call_with_backoff(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        attempt = 0
        while True:
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 — re-raised below if non-rate
                if not self._is_rate_error(exc):
                    raise
                attempt += 1
                if attempt >= self.max_attempts:
                    log_gate_decision(
                        _logger, "backoff_exhausted",
                        score=attempt, threshold=self.max_attempts,
                        decision="dlq", container_id=self.container_id,
                    )
                    raise RateLimitExhausted(
                        f"rate-limited after {attempt} attempts"
                    ) from exc
                delay = self._delay_for(attempt)
                log_gate_decision(
                    _logger, "backoff_retry",
                    score=delay, threshold=self.max_seconds,
                    decision="retry", container_id=self.container_id, attempt=attempt,
                )
                await self._sleep(delay)

    async def gather_bounded(
        self, factories: list[Callable[[], Awaitable[Any]]]
    ) -> list[Any]:
        """Run coroutine factories with at most max_concurrency in flight."""

        async def _guarded(factory: Callable[[], Awaitable[Any]]) -> Any:
            async with self._sem:
                return await self.call_with_backoff(factory)

        return await asyncio.gather(*[_guarded(f) for f in factories])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_rate_limiter.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Wire embeddings through the limiter (tunable-gated, opt-in)**

Read `server/pdf_chat/ingestion/embeddings.py:35-61` first. Add an async batched path that uses the executor while leaving the existing sync `embed_texts` untouched (so Phase 0/1 callers are unaffected):

```python
# server/pdf_chat/ingestion/embeddings.py — append below embed_texts
async def embed_texts_bounded(
    batches: list[list[str]],
    *,
    container_id: str,
    model: str | None = None,
) -> list[list[list[float]]]:
    """Embed many batches under exponential backoff + bounded concurrency.

    Each element of ``batches`` is one embedding request; results preserve order.
    Used by the ingestion fan-out where N batches would otherwise burst Azure's
    rate limit. Reuses the synchronous embed_texts per batch inside a thread so
    the SDK's blocking call doesn't stall the event loop.
    """
    import asyncio

    from .rate_limiter import BoundedBackoffExecutor

    executor = BoundedBackoffExecutor(container_id=container_id)

    def _factory(texts: list[str]):
        async def _call() -> list[list[float]]:
            return await asyncio.to_thread(embed_texts, texts, model=model)
        return _call

    return await executor.gather_bounded([_factory(b) for b in batches])
```

- [ ] **Step 6: Add a wiring test (fake embed; asserts order + bounded calls)**

```python
# append to server/pdf_chat/testing/test_rate_limiter.py
def test_embed_texts_bounded_preserves_order(monkeypatch):
    import pdf_chat.ingestion.embeddings as emb

    def _fake_embed(texts, *, model=None):
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    out = asyncio.run(
        emb.embed_texts_bounded([["a", "bb"], ["ccc"]], container_id="c1")
    )
    assert out == [[[1.0], [2.0]], [[3.0]]]
```

- [ ] **Step 7: Run all rate-limiter tests**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_rate_limiter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Commit**

```bash
git add server/pdf_chat/ingestion/rate_limiter.py server/pdf_chat/ingestion/embeddings.py server/pdf_chat/testing/test_rate_limiter.py
git commit -m "feat(pdf): exponential backoff + bounded concurrency for Azure OpenAI calls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Cascading delete — pure Cypher + orphan detection

**Files:**
- Create: `server/pdf_chat/control_plane/graph_delete.py`
- Test: `server/pdf_chat/testing/test_delete_service.py`

The cascade must, tenant-scoped: delete the document's `Chunk` nodes and their `MENTIONS` edges, then delete only those `Entity`/`Community` nodes that become orphaned (no remaining `MENTIONS` from any *other* doc), and never delete entities still referenced by another document (spec §5, §3.3).

**Tunables used:** `pdf_delete_batch_size` (batch size for `DETACH DELETE`).

- [ ] **Step 1: Write the failing test (pure — no Neo4j)**

```python
# server/pdf_chat/testing/test_delete_service.py
from __future__ import annotations

from pdf_chat.control_plane.graph_delete import (
    build_chunk_delete_cypher,
    build_orphan_entity_delete_cypher,
    select_orphan_entities,
)


def test_chunk_delete_cypher_is_tenant_scoped_on_every_element():
    cypher, params = build_chunk_delete_cypher(
        upload_id="doc1", tenant_id="tenant-A", batch_size=500
    )
    # tenant + doc bound as params, never inlined
    assert params == {"upload_id": "doc1", "tenant_id": "tenant-A", "limit": 500}
    # every matched node constrained on tenant_id (spec §3.3)
    assert "c.tenant_id = $tenant_id" in cypher
    assert "DETACH DELETE" in cypher
    assert "LIMIT $limit" in cypher  # batched


def test_select_orphan_entities_excludes_entities_referenced_by_other_docs():
    # entity -> set of doc_ids that still MENTION it (computed by the caller's query)
    mention_index = {
        "ent_shared": {"doc1", "doc2"},   # doc2 still references it → KEEP
        "ent_only_doc1": {"doc1"},        # only the deleted doc → ORPHAN
        "ent_unrelated": {"doc3"},        # not in deleted doc at all → KEEP
    }
    orphans = select_orphan_entities(
        deleted_doc_id="doc1", mention_index=mention_index
    )
    assert orphans == ["ent_only_doc1"]
    assert "ent_shared" not in orphans  # shared entity stays intact


def test_orphan_entity_delete_cypher_is_tenant_and_id_scoped():
    cypher, params = build_orphan_entity_delete_cypher(
        entity_ids=["ent_only_doc1"], tenant_id="tenant-A"
    )
    assert params == {"entity_ids": ["ent_only_doc1"], "tenant_id": "tenant-A"}
    assert "e.tenant_id = $tenant_id" in cypher
    assert "e.entity_id IN $entity_ids" in cypher
    assert "DETACH DELETE" in cypher
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_delete_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.control_plane.graph_delete'`

- [ ] **Step 3: Implement the pure Cypher + orphan logic**

```python
# server/pdf_chat/control_plane/graph_delete.py
"""Pure (infra-free) Cypher builders + orphan detection for cascading delete.

Tenant isolation is enforced on EVERY matched element (spec §3.3): the document's
chunks, their MENTIONS edges, and any entity/community that becomes orphaned are
all constrained on tenant_id. An entity referenced by ANOTHER document is never
deleted — orphan-ness is computed from the mention index, not assumed.

These functions take no driver and do no I/O so they are unit-testable with zero
infra (matching pdf_chat/testing conventions). delete_service.py runs them.
"""
from __future__ import annotations


def build_chunk_delete_cypher(
    upload_id: str, tenant_id: str, batch_size: int
) -> tuple[str, dict]:
    """One batch of tenant-scoped chunk deletion (DETACH removes MENTIONS edges too).

    Returns the count of deleted chunks so the caller can loop until zero.
    """
    cypher = (
        "MATCH (c:Chunk) "
        "WHERE c.doc_id = $upload_id AND c.tenant_id = $tenant_id "
        "WITH c LIMIT $limit "
        "DETACH DELETE c "
        "RETURN count(c) AS deleted"
    )
    return cypher, {"upload_id": upload_id, "tenant_id": tenant_id, "limit": batch_size}


def build_mention_index_cypher(upload_id: str, tenant_id: str) -> tuple[str, dict]:
    """For every entity mentioned by this doc, list ALL doc_ids still mentioning it.

    The caller passes the result to select_orphan_entities to decide what to drop.
    Tenant-scoped on both the entity and the mentioning chunks (spec §3.3).
    """
    cypher = (
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
        "WHERE c.doc_id = $upload_id AND c.tenant_id = $tenant_id "
        "WITH DISTINCT e "
        "MATCH (oc:Chunk)-[:MENTIONS]->(e) "
        "WHERE oc.tenant_id = $tenant_id "
        "RETURN e.entity_id AS entity_id, collect(DISTINCT oc.doc_id) AS doc_ids"
    )
    return cypher, {"upload_id": upload_id, "tenant_id": tenant_id}


def select_orphan_entities(
    deleted_doc_id: str, mention_index: dict[str, set[str]]
) -> list[str]:
    """Entities whose only remaining mentioning doc is the one being deleted.

    An entity is an orphan iff the set of docs that still mention it is a subset of
    {deleted_doc_id}. Entities referenced by any other doc are never orphans.
    """
    orphans: list[str] = []
    for entity_id, doc_ids in mention_index.items():
        if doc_ids and doc_ids.issubset({deleted_doc_id}):
            orphans.append(entity_id)
    return sorted(orphans)


def build_orphan_entity_delete_cypher(
    entity_ids: list[str], tenant_id: str
) -> tuple[str, dict]:
    """Delete the chosen orphan entities, tenant-scoped (DETACH removes RELATED_TO,
    IN_COMMUNITY edges). Communities left with no entities are cleaned separately.
    """
    cypher = (
        "MATCH (e:Entity) "
        "WHERE e.entity_id IN $entity_ids AND e.tenant_id = $tenant_id "
        "DETACH DELETE e "
        "RETURN count(e) AS deleted"
    )
    return cypher, {"entity_ids": entity_ids, "tenant_id": tenant_id}


def build_orphan_community_delete_cypher(tenant_id: str) -> tuple[str, dict]:
    """Delete communities that have no remaining IN_COMMUNITY members (tenant-scoped)."""
    cypher = (
        "MATCH (cm:Community) "
        "WHERE cm.tenant_id = $tenant_id "
        "AND NOT ( ()-[:IN_COMMUNITY]->(cm) ) "
        "DETACH DELETE cm "
        "RETURN count(cm) AS deleted"
    )
    return cypher, {"tenant_id": tenant_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_delete_service.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/control_plane/graph_delete.py server/pdf_chat/testing/test_delete_service.py
git commit -m "feat(pdf): pure tenant-scoped Cypher + orphan detection for cascading delete

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cascading delete service — soft-delete then async cleanup

**Files:**
- Create: `server/pdf_chat/control_plane/delete_service.py`
- Test: `server/pdf_chat/testing/test_delete_service.py` (append)

`delete_service.delete_document(upload_id, tenant_id)` is the exact symbol `routes.py:258` already imports. It soft-deletes the manifest first (status → `deleted`), then `cleanup_deleted_document` runs the batched cascade. The cleanup is driven by injectable callables so it tests with a fake Neo4j session.

**Tunables used:** `pdf_delete_batch_size`.

- [ ] **Step 1: Write the failing test (fake session asserts shared entity survives)**

```python
# append to server/pdf_chat/testing/test_delete_service.py
import asyncio

from pdf_chat.control_plane.delete_service import cleanup_deleted_document


class _FakeNeo4jSession:
    """Records run() calls and serves canned results so the cascade is deterministic."""

    def __init__(self, mention_rows):
        self.calls = []
        self._mention_rows = mention_rows
        self._chunk_batches = [3, 0]  # first batch deletes 3 chunks, then none left

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        if "MENTIONS" in cypher and "collect(DISTINCT oc.doc_id)" in cypher:
            return list(self._mention_rows)
        if cypher.strip().startswith("MATCH (c:Chunk)"):
            return [{"deleted": self._chunk_batches.pop(0)}]
        return [{"deleted": 1}]


def test_cleanup_deletes_chunks_and_orphans_but_keeps_shared_entities():
    # ent_shared still referenced by doc2 → must NOT be deleted.
    mention_rows = [
        {"entity_id": "ent_shared", "doc_ids": ["doc1", "doc2"]},
        {"entity_id": "ent_only_doc1", "doc_ids": ["doc1"]},
    ]
    session = _FakeNeo4jSession(mention_rows)

    summary = asyncio.run(
        cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    assert summary["chunks_deleted"] == 3
    assert summary["entities_deleted"] == 1  # only ent_only_doc1
    # the orphan-delete call carried exactly the non-shared entity, tenant-scoped
    orphan_calls = [
        c for c in session.calls
        if "MATCH (e:Entity)" in c[0] and "DETACH DELETE e" in c[0]
    ]
    assert len(orphan_calls) == 1
    assert orphan_calls[0][1]["entity_ids"] == ["ent_only_doc1"]
    assert orphan_calls[0][1]["tenant_id"] == "tenant-A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_delete_service.py -k cleanup -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.control_plane.delete_service'`

- [ ] **Step 3: Implement the service**

```python
# server/pdf_chat/control_plane/delete_service.py
"""Cascading document delete: soft-delete first, then async tenant-scoped cleanup.

Flow (spec §5):
  1. delete_document() marks the UploadManifest row status="deleted" (soft) and
     returns immediately so the API responds fast.
  2. cleanup_deleted_document() runs the batched Neo4j cascade: DETACH DELETE the
     document's chunks (removing MENTIONS edges), compute which entities became
     orphaned, then DETACH DELETE only those orphans + empty communities. Entities
     referenced by other docs are preserved (verified in test_delete_service.py).

Every Cypher element is tenant-scoped (spec §3.3). The Neo4j session is injected so
the cascade is unit-testable with a fake (zero infra). Batch size is a tunable.
"""
from __future__ import annotations

from typing import Any

from ..observability import metrics as _metrics
from ..observability.logging import bind_trace, get_pdf_logger
from ..observability.trace import new_trace_id
from ..tunables import get_tunable
from . import graph_delete

_logger = get_pdf_logger("delete")


async def delete_document(upload_id: str, tenant_id: str) -> dict[str, Any]:
    """Soft-delete the manifest, then schedule async cleanup. Returns a status dict.

    The route (routes.py:258) imports THIS function. It opens its own session so it
    can be called from the request handler without a passed-in session.
    """
    from app.core.database import async_session  # late import (mirrors tasks.py:206)

    from .repositories import UploadManifestRepo

    async with async_session() as session:
        repo = UploadManifestRepo(session)
        await repo.set_status(upload_id, "deleted")
        await session.commit()

    # Cleanup runs async (Celery in production); here we return enough for the route
    # and let the worker call cleanup_deleted_document. The route schedules it.
    return {"upload_id": upload_id, "status": "deleted", "cleanup": "scheduled"}


async def cleanup_deleted_document(
    upload_id: str,
    tenant_id: str,
    *,
    neo4j_session: Any,
    container_id: str,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Run the batched tenant-scoped graph cascade. Idempotent + re-runnable."""
    trace_id = trace_id or new_trace_id()
    log = bind_trace(_logger, trace_id, tenant_id)
    batch_size = int(get_tunable(container_id, "pdf_delete_batch_size", 500))

    # 1. Snapshot which entities this doc mentions + who else mentions them
    #    BEFORE we delete the chunks (deletion would erase the MENTIONS edges).
    mi_cypher, mi_params = graph_delete.build_mention_index_cypher(upload_id, tenant_id)
    mention_index: dict[str, set[str]] = {}
    for row in neo4j_session.run(mi_cypher, **mi_params):
        mention_index[row["entity_id"]] = set(row["doc_ids"])

    # 2. Batched chunk deletion until no chunks remain.
    chunks_deleted = 0
    while True:
        c_cypher, c_params = graph_delete.build_chunk_delete_cypher(
            upload_id, tenant_id, batch_size
        )
        result = list(neo4j_session.run(c_cypher, **c_params))
        n = result[0]["deleted"] if result else 0
        chunks_deleted += n
        if n == 0:
            break

    # 3. Delete only orphaned entities (never those referenced by other docs).
    orphans = graph_delete.select_orphan_entities(upload_id, mention_index)
    entities_deleted = 0
    if orphans:
        e_cypher, e_params = graph_delete.build_orphan_entity_delete_cypher(
            orphans, tenant_id
        )
        e_result = list(neo4j_session.run(e_cypher, **e_params))
        entities_deleted = e_result[0]["deleted"] if e_result else len(orphans)

    # 4. Sweep communities left with no members.
    cm_cypher, cm_params = graph_delete.build_orphan_community_delete_cypher(tenant_id)
    cm_result = list(neo4j_session.run(cm_cypher, **cm_params))
    communities_deleted = cm_result[0]["deleted"] if cm_result else 0

    _metrics.inc(tenant_id, "pdf_document_deleted_count")
    _metrics.inc(tenant_id, "pdf_orphan_entity_deleted_count", entities_deleted)
    log.info(
        "pdf_document_cleanup",
        upload_id=upload_id,
        chunks_deleted=chunks_deleted,
        entities_deleted=entities_deleted,
        communities_deleted=communities_deleted,
        shared_entities_preserved=len(mention_index) - len(orphans),
    )
    return {
        "upload_id": upload_id,
        "chunks_deleted": chunks_deleted,
        "entities_deleted": entities_deleted,
        "communities_deleted": communities_deleted,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_delete_service.py -k cleanup -v`
Expected: PASS

- [ ] **Step 5: Run the full delete-service suite**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_delete_service.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add server/pdf_chat/control_plane/delete_service.py server/pdf_chat/testing/test_delete_service.py
git commit -m "feat(pdf): cascading delete service — soft-delete then orphan-safe async cleanup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Eval set expansion — graph/global/cross-domain/negative cases

**Files:**
- Modify: `server/pdf_chat/testing/eval/gold_questions.py`
- Test: `server/pdf_chat/testing/test_eval_harness.py` (the dataset shape)

**Assumption:** `gold_questions.py` from Phase 1 exposes a `GOLD_QUESTIONS: list[GoldQuestion]` and a `GoldQuestion` dataclass with at least `id`, `question`, `expected_answer_contains`, `category`. If the dataclass differs, adapt field names but keep the four new categories. (Re-read the file first to confirm the actual shape.)

- [ ] **Step 1: Write the failing test for category coverage**

```python
# server/pdf_chat/testing/test_eval_harness.py
from __future__ import annotations

from pdf_chat.testing.eval.gold_questions import GOLD_QUESTIONS


def test_gold_set_covers_all_phase5_categories():
    cats = {q.category for q in GOLD_QUESTIONS}
    # Phase 1 baseline (local) plus the four Phase-5 expansion categories.
    for required in (
        "local",
        "graph_traversal",
        "global_community",
        "cross_domain",
        "negative_claim",
    ):
        assert required in cats, f"missing eval category: {required}"


def test_negative_claim_questions_declare_expected_refusal():
    negatives = [q for q in GOLD_QUESTIONS if q.category == "negative_claim"]
    assert negatives, "need at least one honest-'no data' case"
    for q in negatives:
        # negative cases assert the system should NOT fabricate — flagged explicitly
        assert q.expect_no_data is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval_harness.py -k gold -v`
Expected: FAIL — either missing categories or `AttributeError: 'GoldQuestion' object has no attribute 'expect_no_data'`

- [ ] **Step 3: Extend the gold set + dataclass**

Read `server/pdf_chat/testing/eval/gold_questions.py` first. Add an `expect_no_data: bool = False` field to `GoldQuestion` and append the new cases (questions are illustrative; tune to the loaded fixtures):

```python
# server/pdf_chat/testing/eval/gold_questions.py — extend the dataclass + list
# (add field) expect_no_data: bool = False

GOLD_QUESTIONS += [
    GoldQuestion(
        id="gt-1",
        question="Which suppliers are connected to the Acme master service agreement?",
        expected_answer_contains=["supplier", "agreement"],
        category="graph_traversal",
    ),
    GoldQuestion(
        id="gc-1",
        question="Summarize the main themes across all uploaded contracts.",
        expected_answer_contains=["theme"],
        category="global_community",
    ),
    GoldQuestion(
        id="cd-1",
        question="Does the contracted rate for Vendor V100 match the invoiced totals in the vendor CSV?",
        expected_answer_contains=["rate", "invoice"],
        category="cross_domain",
    ),
    GoldQuestion(
        id="neg-1",
        question="What is the termination clause for the 2099 Mars logistics contract?",
        expected_answer_contains=["no data", "not found"],
        category="negative_claim",
        expect_no_data=True,
    ),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval_harness.py -k gold -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/testing/eval/gold_questions.py server/pdf_chat/testing/test_eval_harness.py
git commit -m "test(pdf): expand gold-question set — graph/global/cross-domain/negative cases

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Eval harness — scoring + CI threshold gate

**Files:**
- Create: `server/pdf_chat/testing/eval/harness.py`
- Create: `server/pdf_chat/testing/eval/run_ci_eval.py`
- Test: `server/pdf_chat/testing/test_eval_harness.py` (append)

The harness scores per-question records into an `EvalReport` (fallback rate, faithfulness, answer correctness) and `assert_thresholds` raises when any metric is below its tunable floor — the build-fail seam.

**Tunables used:** `pdf_eval_min_correctness`, `pdf_eval_min_faithfulness`, `pdf_eval_max_fallback_rate`.

- [ ] **Step 1: Write the failing test (deterministic records)**

```python
# append to server/pdf_chat/testing/test_eval_harness.py
import pytest

from pdf_chat.testing.eval.harness import (
    EvalRecord,
    EvalReport,
    assert_thresholds,
    evaluate,
)


def _records(correct: int, total: int, fallbacks: int, faithful: int) -> list[EvalRecord]:
    recs: list[EvalRecord] = []
    for i in range(total):
        recs.append(EvalRecord(
            question_id=f"q{i}",
            category="local",
            correct=i < correct,
            faithful=i < faithful,
            fallback=i < fallbacks,
        ))
    return recs


def test_evaluate_computes_rates():
    report = evaluate(_records(correct=8, total=10, fallbacks=1, faithful=9))
    assert report.total == 10
    assert report.correctness == pytest.approx(0.8)
    assert report.faithfulness == pytest.approx(0.9)
    assert report.fallback_rate == pytest.approx(0.1)


def test_assert_thresholds_fails_build_when_correctness_below_floor():
    # correctness 0.5 < default floor → must raise (this is the CI build-fail seam)
    report = evaluate(_records(correct=5, total=10, fallbacks=0, faithful=10))
    with pytest.raises(AssertionError, match="correctness"):
        assert_thresholds(report, container_id="c1",
                          min_correctness=0.7, min_faithfulness=0.6,
                          max_fallback_rate=0.3)


def test_assert_thresholds_passes_when_all_above_floor():
    report = evaluate(_records(correct=9, total=10, fallbacks=1, faithful=9))
    # must NOT raise
    assert_thresholds(report, container_id="c1",
                      min_correctness=0.7, min_faithfulness=0.6,
                      max_fallback_rate=0.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval_harness.py -k "evaluate or thresholds" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.testing.eval.harness'`

- [ ] **Step 3: Implement the harness**

```python
# server/pdf_chat/testing/eval/harness.py
"""Eval scoring + CI threshold gate for the pdf_chat GraphRAG runtime.

Consumes per-question EvalRecords (produced by running run_pdf_query over the gold
set), computes fallback rate / faithfulness / answer correctness, and exposes
assert_thresholds() which RAISES when any metric is below its tunable floor — the
seam that fails the CI build on a quality regression (spec §5 Phase 5).

Thresholds resolve from pdf_chat/tunables.py; there is no score literal in this
file (spec §3.4). run_ci_eval.py wires this to a non-zero process exit.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...tunables import get_tunable, log_gate_decision
from ...observability.logging import get_pdf_logger

_logger = get_pdf_logger("eval")


@dataclass
class EvalRecord:
    question_id: str
    category: str
    correct: bool
    faithful: bool
    fallback: bool


@dataclass
class EvalReport:
    total: int
    correctness: float
    faithfulness: float
    fallback_rate: float
    by_category: dict[str, dict[str, float]]


def evaluate(records: list[EvalRecord]) -> EvalReport:
    total = len(records)
    if total == 0:
        return EvalReport(0, 0.0, 0.0, 0.0, {})

    def _rate(pred) -> float:
        return sum(1 for r in records if pred(r)) / total

    by_cat: dict[str, dict[str, float]] = {}
    cats = {r.category for r in records}
    for cat in cats:
        crs = [r for r in records if r.category == cat]
        n = len(crs)
        by_cat[cat] = {
            "n": n,
            "correctness": sum(1 for r in crs if r.correct) / n,
            "faithfulness": sum(1 for r in crs if r.faithful) / n,
            "fallback_rate": sum(1 for r in crs if r.fallback) / n,
        }

    return EvalReport(
        total=total,
        correctness=_rate(lambda r: r.correct),
        faithfulness=_rate(lambda r: r.faithful),
        fallback_rate=_rate(lambda r: r.fallback),
        by_category=by_cat,
    )


def assert_thresholds(
    report: EvalReport,
    *,
    container_id: str,
    min_correctness: float | None = None,
    min_faithfulness: float | None = None,
    max_fallback_rate: float | None = None,
) -> None:
    """Raise AssertionError if any metric breaches its floor. CI build-fail seam."""
    min_corr = min_correctness if min_correctness is not None else float(
        get_tunable(container_id, "pdf_eval_min_correctness", 0.7))
    min_faith = min_faithfulness if min_faithfulness is not None else float(
        get_tunable(container_id, "pdf_eval_min_faithfulness", 0.6))
    max_fb = max_fallback_rate if max_fallback_rate is not None else float(
        get_tunable(container_id, "pdf_eval_max_fallback_rate", 0.3))

    log_gate_decision(
        _logger, "eval_correctness", score=report.correctness,
        threshold=min_corr, decision="pass" if report.correctness >= min_corr else "fail",
        container_id=container_id,
    )
    log_gate_decision(
        _logger, "eval_faithfulness", score=report.faithfulness,
        threshold=min_faith, decision="pass" if report.faithfulness >= min_faith else "fail",
        container_id=container_id,
    )
    log_gate_decision(
        _logger, "eval_fallback_rate", score=report.fallback_rate,
        threshold=max_fb, decision="pass" if report.fallback_rate <= max_fb else "fail",
        container_id=container_id,
    )

    assert report.correctness >= min_corr, (
        f"correctness {report.correctness:.3f} < floor {min_corr:.3f}")
    assert report.faithfulness >= min_faith, (
        f"faithfulness {report.faithfulness:.3f} < floor {min_faith:.3f}")
    assert report.fallback_rate <= max_fb, (
        f"fallback_rate {report.fallback_rate:.3f} > ceiling {max_fb:.3f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval_harness.py -k "evaluate or thresholds" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Implement the CI runner**

```python
# server/pdf_chat/testing/eval/run_ci_eval.py
"""CI entry point: run the gold-question eval and exit non-zero on regression.

Usage:
    cd server && uv run python -m pdf_chat.testing.eval.run_ci_eval --container-id c1

Runs each gold question through run_pdf_query (Phase 3), scores the records, prints
the report, and calls assert_thresholds — a breach raises AssertionError which this
wrapper turns into exit code 1 so CI fails the build (spec §5 Phase 5).

If the runtime (run_pdf_query) is unavailable, every question is scored as a
fallback so the report is honest rather than silently green.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .gold_questions import GOLD_QUESTIONS
from .harness import EvalRecord, assert_thresholds, evaluate


async def _score_question(q, *, tenant_id: str, container_id: str) -> EvalRecord:
    try:
        from pdf_chat.agent.graph import run_pdf_query  # Phase 3 artifact
    except Exception:
        return EvalRecord(q.id, q.category, correct=False, faithful=False, fallback=True)

    try:
        result = await run_pdf_query(
            q.question, tenant_id=tenant_id, container_id=container_id
        )
    except Exception:
        return EvalRecord(q.id, q.category, correct=False, faithful=False, fallback=True)

    answer = (getattr(result, "answer", "") or "").lower()
    citations = getattr(result, "citations", []) or []
    # correctness: expected substrings present (or, for negatives, an honest refusal)
    correct = all(s.lower() in answer for s in q.expected_answer_contains)
    if getattr(q, "expect_no_data", False):
        correct = any(p in answer for p in ("no data", "not found", "could not find"))
    # faithfulness: a non-refusal answer must carry at least one citation
    faithful = bool(citations) or getattr(q, "expect_no_data", False)
    return EvalRecord(q.id, q.category, correct=correct, faithful=faithful, fallback=False)


async def _main_async(tenant_id: str, container_id: str) -> int:
    records = [
        await _score_question(q, tenant_id=tenant_id, container_id=container_id)
        for q in GOLD_QUESTIONS
    ]
    report = evaluate(records)
    print(f"total={report.total} correctness={report.correctness:.3f} "
          f"faithfulness={report.faithfulness:.3f} fallback_rate={report.fallback_rate:.3f}")
    for cat, m in sorted(report.by_category.items()):
        print(f"  [{cat}] n={int(m['n'])} correctness={m['correctness']:.3f} "
              f"faithfulness={m['faithfulness']:.3f} fallback={m['fallback_rate']:.3f}")
    try:
        assert_thresholds(report, container_id=container_id)
    except AssertionError as e:
        print(f"EVAL GATE FAILED: {e}", file=sys.stderr)
        return 1
    print("EVAL GATE PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="eval-tenant")
    parser.add_argument("--container-id", default="eval-container")
    args = parser.parse_args()
    return asyncio.run(_main_async(args.tenant_id, args.container_id))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Verify the CI runner is importable + runs (degrades to fallbacks without infra)**

Run: `cd server && uv run python -c "import pdf_chat.testing.eval.run_ci_eval as m; print('ok')"`
Expected: prints `ok`

- [ ] **Step 7: Run the full Phase-5 test suite**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_observability.py pdf_chat/testing/test_rate_limiter.py pdf_chat/testing/test_delete_service.py pdf_chat/testing/test_eval_harness.py -v`
Expected: PASS (all tests green)

- [ ] **Step 8: Commit**

```bash
git add server/pdf_chat/testing/eval/harness.py server/pdf_chat/testing/eval/run_ci_eval.py server/pdf_chat/testing/test_eval_harness.py
git commit -m "feat(pdf): eval harness + CI threshold gate (fails build on quality regression)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire trace IDs + metrics into the live runtime paths

**Files:**
- Modify: `server/pdf_chat/api/routes.py` (chat + upload + delete bind a trace + emit metrics)
- Test: covered by the route-import smoke (`Task 3 Step 6`) + manual narration; no new infra test (the live paths need infra).

**Assumption:** the chat route (`routes.py:202`) and delete route (`routes.py:251`) exist from earlier phases. This task only threads observability through them; it must not change query semantics.

- [ ] **Step 1: Bind a trace + count a query in the chat route**

Read `server/pdf_chat/api/routes.py:202-235` first. Inside `chat`, before delegating to the runtime:

```python
# server/pdf_chat/api/routes.py — inside async def chat(...)
from pdf_chat.observability import metrics as _pdf_metrics  # top-of-file (already added Task 3)
from pdf_chat.observability.trace import PdfTrace, new_trace_id  # top-of-file

    trace = PdfTrace(trace_id=new_trace_id(), tenant_id=principal.tenant_id)
    _pdf_metrics.inc(principal.tenant_id, "pdf_query_total")
    import time as _t
    _start = _t.perf_counter()
    try:
        # ... existing delegation to the runtime, passing trace.trace_id through ...
        ...
    except Exception:
        _pdf_metrics.inc(principal.tenant_id, "pdf_query_errors")
        raise
    finally:
        _pdf_metrics.observe_latency(
            principal.tenant_id, (_t.perf_counter() - _start) * 1000)
        trace.emit()
```

- [ ] **Step 2: Schedule cleanup from the delete route**

Read `server/pdf_chat/api/routes.py:251-265` first. The route already calls `delete_service.delete_document`. Confirm it now soft-deletes (Task 6) and add the async cleanup schedule (Celery task in production; a `BackgroundTasks` hook is acceptable for the route). Keep the existing return shape (`DeleteResponse`).

```python
# server/pdf_chat/api/routes.py — inside async def delete_document(...)
# after `result = await _delete(upload_id=upload_id, tenant_id=principal.tenant_id)`
# schedule the graph cascade out-of-band; production wires this to Celery.
# (No code change to the cleanup logic — it lives in delete_service.cleanup_deleted_document.)
```

- [ ] **Step 3: Verify routes still import + expose the expected paths**

Run: `cd server && uv run python -c "import pdf_chat.api.routes as r; print(sorted(route.path for route in r.pdf_router.routes))"`
Expected: includes `/api/pdf/chat`, `/api/pdf/metrics`, `/api/pdf/documents/{upload_id}`

- [ ] **Step 4: Run the full pdf_chat suite (regression guard)**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -v`
Expected: PASS (all existing + new tests green)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/api/routes.py
git commit -m "feat(pdf): thread trace IDs + per-tenant metrics through chat/delete routes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (spec coverage)

| Spec §5 Phase 5 requirement | Task(s) |
|---|---|
| Token/cost tracking per query/document/tenant (extraction + synthesis) | Task 2 (tracker), Task 9 (synthesis wiring), Task 4 (extraction calls flow through limiter where cost is tracked) |
| Structured logging + trace IDs wired through pdf_chat | Task 1, Task 9 |
| Metrics surface consistent with main app's /api/metrics | Task 3 (`/api/pdf/metrics` mirrors `main.py:309`) |
| Cascading delete: batch DETACH DELETE, tenant-scoped, soft-delete then async cleanup, never delete shared entities | Task 5 (pure Cypher + orphan logic), Task 6 (service), Task 9 (route schedule) |
| Rate-limit backoff: exponential + max-concurrency for embedding/extraction | Task 4 |
| Eval expansion: graph/global/cross-domain/negative; CI harness reporting fallback + faithfulness + correctness; fail build below thresholds | Task 7 (cases), Task 8 (harness + CI gate) |
| §3.4 no magic literals — all via tunables + logged | Tasks 4, 6, 8 read every cap/threshold via `get_tunable` and log via `log_gate_decision` |
| §3.3 tenant isolation on every Neo4j hop | Task 5 Cypher constrains `tenant_id` on every matched element; tested |
| gpt-4o-mini only (§6) | Task 2 flags + counts non-mini as `policy_violations` |

**Deterministic test seams (per the ask):** shared-entity survives cascade (Task 6 `test_cleanup_deletes_chunks_and_orphans_but_keeps_shared_entities`); backoff retries with growing delay then DLQs (Task 4 `test_backoff_retries_with_growing_delay_then_succeeds`, `test_backoff_raises_exhausted_after_max_attempts`); cost tracker accumulates per-tenant (Task 2 `test_cost_tracker_accumulates_per_tenant`); eval harness fails build below thresholds (Task 8 `test_assert_thresholds_fails_build_when_correctness_below_floor`).

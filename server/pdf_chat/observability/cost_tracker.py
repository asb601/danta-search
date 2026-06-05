"""Per-tenant LLM cost & token accumulator for pdf_chat (Phase 6 hardening).

Mirrors server/app/core/cost_tracker.py but is keyed by ``tenant_id`` and splits
ingestion-extraction vs query-synthesis usage. Every recorded call is emitted as
one structured event (with trace_id + document_id) AND accumulated per tenant so
the metrics surface (/api/pdf/metrics) can report cost_usd / tokens per tenant.

POLICY (spec §6): only gpt-4o-mini is allowed anywhere in this stack. A non-mini
model is still RECORDED (so cost is never silently lost) but increments
``policy_violations`` and is logged with ``policy_violation=True`` so an operator
can catch a regression. The allowed-model test is a substring marker, documented
below as a MODEL-POLICY marker — it is NOT a score-comparison literal (spec §3.4).

Pure + import-safe: an in-process, thread-safe accumulator with no infra
dependency. Importing this module touches no database / network.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from .logging import get_pdf_logger

# MODEL-POLICY markers (NOT score literals, spec §3.4). The REAL policy is
# "no gpt-4o": gpt-4o-mini IS allowed (bulk model) and the model_router LEGITIMATELY
# escalates to claude-sonnet-4-6 / claude-opus-4-8, which are also allowed. So a
# violation is ONLY a gpt-4o model that is NOT the mini variant. We test substrings
# (rather than importing get_settings().chat_deployment() at module load) because
# the cost tracker must stay import-safe with zero infra (settings reads .env and
# may be unconfigured in a unit-test environment).
_FORBIDDEN_MODEL_MARKER = "gpt-4o"   # the banned family
_ALLOWED_MODEL_MARKER = "mini"       # ...except its -mini variant


def _blank() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "policy_violations": 0,
        "by_phase": defaultdict(
            lambda: {
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }
        ),
    }


class PdfCostTracker:
    """Thread-safe per-tenant cost accumulator (extraction + synthesis)."""

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
        m = (model or "").lower()
        violation = _FORBIDDEN_MODEL_MARKER in m and _ALLOWED_MODEL_MARKER not in m
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
        # Emit OUTSIDE the lock — structlog I/O must not hold the accumulator lock.
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
        """Clear one tenant (or all when ``tenant_id`` is None)."""
        with self._lock:
            if tenant_id is None:
                self._by_tenant.clear()
            else:
                self._by_tenant.pop(tenant_id, None)


def _materialize(t: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy the accumulator into a plain JSON-safe dict (defaultdicts → dicts)."""
    return {
        "llm_calls": t["llm_calls"],
        "prompt_tokens": t["prompt_tokens"],
        "completion_tokens": t["completion_tokens"],
        "cost_usd": round(t["cost_usd"], 6),
        "policy_violations": t["policy_violations"],
        "by_phase": {
            k: {
                "llm_calls": v["llm_calls"],
                "prompt_tokens": v["prompt_tokens"],
                "completion_tokens": v["completion_tokens"],
                "cost_usd": round(v["cost_usd"], 6),
            }
            for k, v in t["by_phase"].items()
        },
    }


# Process-wide singleton (mirrors app/core/cost_tracker's module-level session).
_TRACKER = PdfCostTracker()


def get_cost_tracker() -> PdfCostTracker:
    return _TRACKER

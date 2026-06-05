"""Per-request / per-document trace for pdf_chat (Phase 6 hardening).

Mirrors server/app/core/orchestration_trace.py: JSON-safe, never raises, emits a
single structured event at the end. Carries ``trace_id`` + ``tenant_id`` for
correlation across the ingestion + query pipelines. Every value is run through
``_safe_val`` before storage so a non-serializable object (an open session, a
numpy array, a raw exception) is coerced to a bounded string rather than blowing
up the trace — telemetry must NEVER crash the pipeline it observes.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .logging import bind_trace, get_pdf_logger

# ── Truncation bounds (structural caps, not score thresholds) ────────────────
_MAX_STR = 200    # max chars in any string field
_MAX_LIST = 20    # max items in any list field
_MAX_KEYS = 30    # max keys in any dict field
_MAX_KEY_LEN = 80  # max chars in any dict key / stage name


def new_trace_id() -> str:
    """A fresh hex trace id (uuid4)."""
    return uuid.uuid4().hex


def _safe_val(v: Any) -> Any:
    """Recursively coerce a value into a JSON-safe, size-bounded form.

    Primitives pass through; strings/lists/dicts are bounded; anything else is
    stringified. Never raises (a misbehaving ``__str__`` is caught by callers).
    """
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:_MAX_STR]
    if isinstance(v, dict):
        items = list(v.items())[:_MAX_KEYS]
        return {str(k)[:_MAX_KEY_LEN]: _safe_val(vv) for k, vv in items}
    if isinstance(v, (list, tuple)):
        return [_safe_val(i) for i in list(v)[:_MAX_LIST]]
    try:
        return str(v)[:_MAX_STR]
    except Exception:
        return "<unprintable>"


class PdfTrace:
    """Accumulates per-stage telemetry for one pdf_chat invocation. Never raises.

    Usage::

        trace = PdfTrace(trace_id=new_trace_id(), tenant_id=tenant_id)
        trace.set_stage("retrieval", {"chunks": len(chunks)})
        ...
        trace.emit()  # single structured event at the end
    """

    __slots__ = ("trace_id", "tenant_id", "_created_at", "_stages")

    def __init__(self, trace_id: str, tenant_id: str) -> None:
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self._created_at = time.perf_counter()
        self._stages: dict[str, Any] = {}

    def set_stage(self, name: str, data: Any) -> None:
        """Record one stage's decision/evidence. Idempotent + non-raising."""
        try:
            self._stages[str(name)[:_MAX_KEY_LEN]] = _safe_val(data)
        except Exception:
            pass  # telemetry — never raise

    def as_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "stages": dict(self._stages),
        }

    def emit(self) -> None:
        """Emit the complete trace as a single structured event. Never raises."""
        try:
            elapsed_ms = round((time.perf_counter() - self._created_at) * 1000, 2)
            logger = bind_trace(get_pdf_logger("trace"), self.trace_id, self.tenant_id)
            logger.info("pdf_trace", elapsed_ms=elapsed_ms, stages=self._stages)
        except Exception:
            pass  # trace emission must never crash the pipeline

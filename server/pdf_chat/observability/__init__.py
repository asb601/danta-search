"""Phase 6 (hardening) observability surface for pdf_chat.

Trace IDs, per-tenant cost, and per-tenant metrics. Mirrors
server/app/core/{orchestration_trace,cost_tracker,metrics,logger}.py but lives
inside pdf_chat so the two pipelines stay independent (pdf_chat/CLAUDE.md #7).

Pure + import-safe: every submodule is an in-process accumulator with no infra
dependency, so importing this package touches no database / network.
"""
from __future__ import annotations

from . import metrics
from .cost_tracker import PdfCostTracker, get_cost_tracker
from .logging import bind_trace, get_pdf_logger
from .metrics import get_snapshot, inc, observe_latency
from .trace import PdfTrace, new_trace_id

__all__ = [
    "PdfTrace",
    "new_trace_id",
    "get_pdf_logger",
    "bind_trace",
    "PdfCostTracker",
    "get_cost_tracker",
    "metrics",
    "inc",
    "observe_latency",
    "get_snapshot",
]

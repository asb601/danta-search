"""DeepDoc ONNX micro-component enhancers (OPTIONAL, addendum B).

A vendored-CONCEPT layer of optional page enhancers — multi-column reading-order,
table-structure for spanning cells, and table-rotation — gated on hard pages only
and degrading gracefully to the Phase-1 PyMuPDF/OCR fast path when the optional
native deps (onnxruntime / opencv / xgboost / scikit-learn) are absent.

Importing this package NEVER raises when those deps are missing: all native
imports are guarded in :mod:`._deps`. Callers use :func:`deepdoc_available` to
decide whether enhancement can run, and :func:`enhance_page` always returns a
valid ``UnifiedElement[]`` (the input unchanged on any skip/degrade/failure).
"""
from __future__ import annotations


class DeepDocUnavailable(RuntimeError):
    """Raised only by callers that explicitly REQUIRE the ONNX enhancers.

    The default path never raises — :func:`enhance_page` degrades silently. This
    exception exists for code that wants to assert availability up front.
    """


from ._deps import deepdoc_available  # noqa: E402
from .enhancer import enhance_page, should_enhance  # noqa: E402
from .column_order import assign_columns  # noqa: E402
from .table_structure import recognize_table_structure  # noqa: E402
from .rotation import best_rotation  # noqa: E402

__all__ = [
    "DeepDocUnavailable",
    "deepdoc_available",
    "enhance_page",
    "should_enhance",
    "assign_columns",
    "recognize_table_structure",
    "best_rotation",
]

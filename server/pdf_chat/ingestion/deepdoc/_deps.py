"""Guarded optional-dependency capability flags for the DeepDoc enhancers.

DeepDoc is an OPTIONAL micro-component layer (addendum B). Its native deps
(``onnxruntime``, ``opencv``/``cv2``, ``xgboost``, ``scikit-learn``) are heavy and
may be absent in a slim image. Every import is guarded so importing this module —
and the whole ``deepdoc`` package — never raises when a dep is missing. Callers
inspect the ``HAS_*`` flags / :func:`deepdoc_available` and degrade to the
Phase-1 PyMuPDF/OCR fast path when capabilities are unavailable.

Pure module — safe to import with zero infra and zero optional deps installed.
"""
from __future__ import annotations

try:
    import onnxruntime  # noqa: F401
    HAS_ONNX = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_ONNX = False

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_CV2 = False

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_XGB = False

try:
    from sklearn.cluster import KMeans  # noqa: F401
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_SKLEARN = False


def deepdoc_available() -> bool:
    """Whether the ONNX-backed enhancers can run at all.

    Enhancers need ONNX + cv2 at minimum (table-structure recognition, rotation).
    KMeans-only multi-column reading-order additionally needs sklearn, which is
    checked independently inside :mod:`column_order` so it can run even when the
    ONNX/cv2 table path is unavailable.
    """
    return HAS_ONNX and HAS_CV2

"""Pure tests for the OPTIONAL DeepDoc ONNX micro-component enhancers (addendum B).

These tests run with NO optional native deps installed (onnxruntime / cv2 /
xgboost / sklearn are all absent in CI). They cover BOTH directions required by
the plan:

  * ONNX/cv2/sklearn ABSENT  → graceful degradation (input returned unchanged,
    a decision logged, no exception).
  * ONNX/cv2/sklearn MOCKED  → the enhance path runs (KMeans column ordering,
    ONNX TSR spanning cells, rotation), via monkeypatched ``_deps`` flags +
    injected session/scorer seams.

No live infra. The ONNX session and OCR scorer are injected, so the deterministic
seams are pure-testable. Matches ``test_ingestion.py`` style (plain functions).
"""
from __future__ import annotations

import sys
import types

from pdf_chat.ingestion.ton_schema import UnifiedElement, ElementType, BBox
from pdf_chat.ingestion.deepdoc import (
    deepdoc_available,
    enhance_page,
    should_enhance,
    assign_columns,
    recognize_table_structure,
    best_rotation,
    DeepDocUnavailable,
)
from pdf_chat.ingestion.deepdoc import _deps, enhancer, column_order


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _el(eid: str, x1: float, y1: float, x2: float = 0.0, y2: float = 0.0,
        reading_order: int = 0, etype: ElementType = ElementType.TEXT) -> UnifiedElement:
    return UnifiedElement(
        element_id=eid,
        doc_id="doc-1",
        page_num=1,
        element_type=etype,
        content=eid,
        reading_order=reading_order,
        tenant_id="t1",
        bbox=BBox(x1=x1, y1=y1, x2=(x2 or x1 + 10), y2=(y2 or y1 + 10)),
    )


class _FakeKMeans:
    """Minimal KMeans stand-in: clusters 1-D left edges by nearest of `k` seeds.

    Seeds the first `k` distinct sorted x-values so a clean 2-column page splits
    deterministically. Mirrors the sklearn KMeans API surface used by the code.
    """

    def __init__(self, n_clusters=1, n_init=10, random_state=0):
        self.n_clusters = n_clusters
        self.labels_ = []
        self.cluster_centers_ = []

    def fit(self, samples):
        xs = sorted({s[0] for s in samples})
        seeds = xs[: self.n_clusters] if xs else [0.0]
        while len(seeds) < self.n_clusters:
            seeds.append(seeds[-1])
        self.cluster_centers_ = [[s] for s in seeds]
        self.labels_ = [
            min(range(len(seeds)), key=lambda i: abs(seeds[i] - s[0]))
            for s in samples
        ]
        return self

    def fit_predict(self, samples):
        return self.fit(samples).labels_


def _install_fake_sklearn(monkeypatch):
    """Inject a fake `sklearn.cluster.KMeans` + `sklearn.metrics.silhouette_score`."""
    sk = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    metrics = types.ModuleType("sklearn.metrics")
    cluster.KMeans = _FakeKMeans

    def _silhouette(samples, labels):
        # Reward a clean split: more distinct labels → higher score.
        return float(len(set(labels)))

    metrics.silhouette_score = _silhouette
    sk.cluster = cluster
    sk.metrics = metrics
    monkeypatch.setitem(sys.modules, "sklearn", sk)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", cluster)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)
    monkeypatch.setattr(column_order, "HAS_SKLEARN", True)


# ──────────────────────────────────────────────────────────────────────────────
# B1 — guarded deps + graceful degradation
# ──────────────────────────────────────────────────────────────────────────────

def test_deepdoc_available_reflects_flags(monkeypatch):
    monkeypatch.setattr(_deps, "HAS_ONNX", True)
    monkeypatch.setattr(_deps, "HAS_CV2", True)
    assert _deps.deepdoc_available() is True
    monkeypatch.setattr(_deps, "HAS_ONNX", False)
    assert _deps.deepdoc_available() is False


def test_deepdoc_available_false_when_deps_absent():
    # In CI none of onnxruntime/cv2 are installed → unavailable, no exception.
    assert deepdoc_available() is False


def test_package_imports_without_optional_deps():
    # The whole package import must not raise even with every optional dep absent.
    assert callable(enhance_page)
    assert issubclass(DeepDocUnavailable, RuntimeError)


def test_degrades_when_onnxruntime_absent(monkeypatch):
    # Flagged (complex) page, but ONNX absent → fast path: input returned unchanged.
    monkeypatch.setattr(enhancer, "deepdoc_available", lambda: False)
    els = [_el("a", 0, 0), _el("b", 0, 20)]
    out = enhance_page(els, container_id="t1", complexity_score=0.99,
                       extract_confidence=0.10)
    assert out is els  # untouched fast path


# ──────────────────────────────────────────────────────────────────────────────
# B2 — KMeans multi-column reading-order
# ──────────────────────────────────────────────────────────────────────────────

def test_assign_columns_skips_when_sklearn_absent(monkeypatch):
    monkeypatch.setattr(column_order, "HAS_SKLEARN", False)
    els = [_el("a", 0, 0), _el("b", 100, 0)]
    out = assign_columns(els, container_id="t1")
    assert out is els  # degraded: unchanged


def test_assign_columns_orders_two_column_page(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    # Left column at x≈10 (two rows), right column at x≈300 (two rows). Raw y would
    # interleave L1,R1,L2,R2; correct reading order is L1,L2,R1,R2.
    L1 = _el("L1", 10, 0, x2=110)
    R1 = _el("R1", 300, 5, x2=400)
    L2 = _el("L2", 12, 50, x2=112)
    R2 = _el("R2", 302, 55, x2=402)
    out = assign_columns([L1, R1, L2, R2], container_id="t1")
    order = [e.element_id for e in sorted(out, key=lambda e: e.reading_order)]
    assert order == ["L1", "L2", "R1", "R2"]


def test_assign_columns_single_column_top_to_bottom(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    a = _el("a", 10, 100, x2=110)
    b = _el("b", 10, 0, x2=110)
    c = _el("c", 10, 50, x2=110)
    out = assign_columns([a, b, c], container_id="t1")
    order = [e.element_id for e in sorted(out, key=lambda e: e.reading_order)]
    assert order == ["b", "c", "a"]


# ──────────────────────────────────────────────────────────────────────────────
# B3 — ONNX table-structure (spanning cells) + rotation
# ──────────────────────────────────────────────────────────────────────────────

class _MockTSRSession:
    """Mock ONNX session: `.run(img)` returns fixed cells incl. a colspan=2 cell."""

    def __init__(self, cells):
        self._cells = cells

    def run(self, image):
        return self._cells


def test_table_structure_emits_spanning_cells(monkeypatch):
    monkeypatch.setattr("pdf_chat.ingestion.deepdoc.table_structure.deepdoc_available",
                        lambda: True)
    cells = [
        {"row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "Q1+Q2", "confidence": 0.9},
        {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "text": "10", "confidence": 0.8},
        {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "20", "confidence": 0.8},
    ]
    el = recognize_table_structure(
        object(), container_id="t1",
        session_factory=lambda: _MockTSRSession(cells),
        doc_id="d", page_num=2, tenant_id="t1", element_id="tbl-1",
    )
    assert el is not None
    assert el.element_type == ElementType.TABLE
    # colspan=2 header cell expanded across both grid columns.
    header = el.content.splitlines()[0]
    assert header.count("Q1+Q2") == 2
    # confidence = min cell confidence.
    assert el.confidence == 0.8


def test_table_structure_degrades_when_unavailable(monkeypatch):
    monkeypatch.setattr("pdf_chat.ingestion.deepdoc.table_structure.deepdoc_available",
                        lambda: False)
    el = recognize_table_structure(object(), container_id="t1",
                                   session_factory=lambda: _MockTSRSession([]))
    assert el is None


def test_table_structure_none_when_no_session():
    # No injected session → degrade to fast path regardless of deps.
    el = recognize_table_structure(object(), container_id="t1", session_factory=None)
    assert el is None


def test_rotation_picks_best_angle(monkeypatch):
    monkeypatch.setattr("pdf_chat.ingestion.deepdoc.rotation.HAS_CV2", True)

    def _scorer(img, angle):
        return {0: 0.20, 90: 0.95, 180: 0.10, 270: 0.30}[angle]

    angle, conf = best_rotation(object(), container_id="t1", ocr_scorer=_scorer)
    assert angle == 90
    assert conf == 0.95


def test_rotation_degrades_without_cv2(monkeypatch):
    monkeypatch.setattr("pdf_chat.ingestion.deepdoc.rotation.HAS_CV2", False)
    angle, conf = best_rotation(object(), container_id="t1",
                                ocr_scorer=lambda i, a: 1.0)
    assert angle == 0


# ──────────────────────────────────────────────────────────────────────────────
# B4 — hard-page gate + enhance_page orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def test_should_enhance_gate_fires_on_complex():
    assert should_enhance(container_id="t1", complexity_score=0.99,
                          extract_confidence=0.99) is True


def test_should_enhance_gate_fires_on_low_confidence():
    assert should_enhance(container_id="t1", complexity_score=0.0,
                          extract_confidence=0.10) is True


def test_should_enhance_gate_skips_simple_page():
    assert should_enhance(container_id="t1", complexity_score=0.0,
                          extract_confidence=0.99) is False


def test_enhancer_skipped_on_simple_page():
    calls = []
    els = [_el("a", 0, 0)]
    out = enhance_page(els, container_id="t1", complexity_score=0.0,
                       extract_confidence=0.99,
                       page_image=object(),
                       session_factory=lambda: calls.append(1))
    assert out is els           # untouched
    assert calls == []          # session_factory never invoked


def test_enhancer_used_on_flagged_page(monkeypatch):
    _install_fake_sklearn(monkeypatch)
    monkeypatch.setattr(enhancer, "deepdoc_available", lambda: True)
    monkeypatch.setattr("pdf_chat.ingestion.deepdoc.table_structure.deepdoc_available",
                        lambda: True)
    L1 = _el("L1", 10, 0, x2=110)
    R1 = _el("R1", 300, 5, x2=400)
    L2 = _el("L2", 12, 50, x2=112)
    R2 = _el("R2", 302, 55, x2=402)
    cells = [
        {"row": 0, "col": 0, "colspan": 2, "text": "H", "confidence": 0.9},
        {"row": 1, "col": 0, "text": "1", "confidence": 0.7},
        {"row": 1, "col": 1, "text": "2", "confidence": 0.7},
    ]
    out = enhance_page(
        [L1, R1, L2, R2], container_id="t1",
        complexity_score=0.99, extract_confidence=0.10,
        page_image=object(),
        session_factory=lambda: _MockTSRSession(cells),
    )
    # Reading order rewritten (columns) AND a TSR table element appended.
    text_order_sorted = [e.element_id for e in
                         sorted([e for e in out if e.element_type == ElementType.TEXT],
                                key=lambda e: e.reading_order)]
    assert text_order_sorted == ["L1", "L2", "R1", "R2"]
    assert any(e.element_type == ElementType.TABLE for e in out)


def test_enhancer_never_raises_on_internal_failure(monkeypatch):
    monkeypatch.setattr(enhancer, "deepdoc_available", lambda: True)

    def _boom(elements, *, container_id):
        raise RuntimeError("kmeans exploded")

    monkeypatch.setattr(enhancer, "assign_columns", _boom)
    els = [_el("a", 0, 0)]
    out = enhance_page(els, container_id="t1", complexity_score=0.99,
                       extract_confidence=0.10)
    assert out is els  # failure → fast path, no exception

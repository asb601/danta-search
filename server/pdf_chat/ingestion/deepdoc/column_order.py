"""KMeans multi-column / reading-order enhancer (DeepDoc, optional).

Concept ported from RAGFlow ``_assign_column`` (KMeans cluster on each element's
left edge ``x0`` with an indent tolerance, then read column-by-column). We port
the **concept only** — no RAGFlow code is copied. ``scikit-learn`` is a guarded
optional dep: when absent, this degrades to returning the input unchanged.

The output overwrites ``reading_order`` so the page reads column-1 top-to-bottom,
then column-2, etc. — instead of being interleaved by raw ``y`` (which is wrong on
multi-column pages). All thresholds resolve via :func:`get_tunable`; every
skip/decision is emitted via :func:`log_gate_decision`. No bare literals.
"""
from __future__ import annotations

from ..ton_schema import UnifiedElement
from ...tunables import get_tunable, log_gate_decision
from ._deps import HAS_SKLEARN

# Tunable keys (defaults are NAMED here and passed to get_tunable; the canonical
# default-table addition is listed for the integration step — not edited here).
TUN_DD_INDENT_TOL_FRAC = "deepdoc.indent_tol_frac"   # default 0.12
TUN_DD_COLUMN_MAX_K = "deepdoc.column_max_k"          # default 4 (max columns tried)

_DEFAULT_INDENT_TOL_FRAC = 0.12
_DEFAULT_COLUMN_MAX_K = 4


def _left_edge(el: UnifiedElement) -> float:
    """Left edge (x1) of an element's bbox; missing bbox sorts to the far left."""
    return el.bbox.x1 if el.bbox is not None else 0.0


def _top_edge(el: UnifiedElement) -> float:
    """Top edge (y1) of an element's bbox; missing bbox sorts to the top."""
    return el.bbox.y1 if el.bbox is not None else 0.0


def _page_width(elements: list[UnifiedElement]) -> float:
    """Best-effort page width = max right edge across elements (>= 1.0)."""
    rights = [el.bbox.x2 for el in elements if el.bbox is not None]
    return max(rights) if rights else 1.0


def _silhouette_best_k(xs: list[float], max_k: int) -> int:
    """Pick the column count by best silhouette over k in [1, max_k].

    Concept ported from RAGFlow's best-of-`max_try` loop. Falls back to k=1 when
    there are too few points or sklearn cannot score a partition.
    """
    n = len(xs)
    if n < 2:
        return 1
    from sklearn.cluster import KMeans  # guarded by HAS_SKLEARN at call site
    from sklearn.metrics import silhouette_score

    samples = [[x] for x in xs]
    best_k, best_score = 1, float("-inf")
    upper = min(max_k, n)
    for k in range(2, upper + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(samples)
        except Exception:
            continue
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(samples, labels)
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def assign_columns(elements: list[UnifiedElement], *, container_id: str) -> list[UnifiedElement]:
    """Re-order ``elements`` into human reading order via left-edge column clusters.

    Clusters element left edges into K columns (K chosen by best silhouette up to
    a tunable cap), orders columns left-to-right, then sorts within a column
    top-to-bottom and rewrites each element's ``reading_order``. Elements whose
    left edges fall within an indent tolerance of a column centroid are treated as
    the same column (so indented lines do not spuriously split a column).

    Degrades to the input unchanged when sklearn is absent or there is nothing to
    cluster. Mutates and returns the same ``UnifiedElement`` objects.
    """
    if not elements:
        return elements

    if not HAS_SKLEARN:
        log_gate_decision(
            "deepdoc.column_order.skipped", score=0.0, threshold=0.0,
            outcome="fastpath", container_id=container_id, reason="sklearn_absent",
        )
        return elements

    width = _page_width(elements)
    indent_frac = get_tunable(container_id, TUN_DD_INDENT_TOL_FRAC, _DEFAULT_INDENT_TOL_FRAC)
    max_k = get_tunable(container_id, TUN_DD_COLUMN_MAX_K, _DEFAULT_COLUMN_MAX_K)
    indent_tol = width * indent_frac

    xs = [_left_edge(el) for el in elements]
    k = _silhouette_best_k(xs, max_k)

    if k <= 1:
        # Single column: stable top-to-bottom order is the correct reading order.
        ordered = sorted(elements, key=_top_edge)
        for i, el in enumerate(ordered):
            el.reading_order = i
        log_gate_decision(
            "deepdoc.column_order", score=float(k), threshold=2.0,
            outcome="single_column", container_id=container_id, n_elements=len(elements),
        )
        return ordered

    from sklearn.cluster import KMeans  # guarded above

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit([[x] for x in xs])
    labels = km.labels_
    centroids = [float(c[0]) for c in km.cluster_centers_]

    # Order columns left-to-right by centroid.
    col_rank = {label: rank for rank, (label, _c) in
                enumerate(sorted(enumerate(centroids), key=lambda lc: lc[1]))}

    def _column_of(el: UnifiedElement, raw_label: int) -> int:
        """Snap to the nearest centroid within indent tolerance (concept port)."""
        x = _left_edge(el)
        nearest = min(range(len(centroids)), key=lambda c: abs(centroids[c] - x))
        chosen = nearest if abs(centroids[nearest] - x) <= indent_tol else raw_label
        return col_rank[chosen]

    keyed = [
        (_column_of(el, int(labels[i])), _top_edge(el), i, el)
        for i, el in enumerate(elements)
    ]
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))

    ordered = [t[3] for t in keyed]
    for i, el in enumerate(ordered):
        el.reading_order = i

    log_gate_decision(
        "deepdoc.column_order", score=float(k), threshold=2.0,
        outcome="multi_column", container_id=container_id,
        n_columns=k, n_elements=len(elements),
    )
    return ordered

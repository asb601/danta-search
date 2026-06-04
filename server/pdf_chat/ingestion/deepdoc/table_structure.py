"""ONNX table-structure recognition with spanning-cell support (DeepDoc, optional).

Concept ported from RAGFlow's ``TableStructureRecognizer`` (TSR) — an ONNX model
emits per-cell boxes plus span labels (rowspan/colspan), and ``__cal_spans``
folds spanning cells into a grid. We port the **concept only**; no RAGFlow code is
copied. The ONNX session is INJECTED via ``session_factory`` so tests stay pure
(a mock session returns fixed cell/span tensors) and so production wires a real
``onnxruntime.InferenceSession`` from the pre-downloaded model dir.

Degrades gracefully: when ONNX/cv2 are absent or no session is provided, returns
``None`` so the orchestrator keeps the Phase-1 PyMuPDF/OCR table untouched. All
thresholds resolve via :func:`get_tunable`; decisions log via
:func:`log_gate_decision`. No bare literals.
"""
from __future__ import annotations

from ..ton_schema import UnifiedElement, ElementType, BBox
from ...tunables import get_tunable, log_gate_decision
from ._deps import deepdoc_available

TUN_DD_TSR_MIN_CONF = "deepdoc.tsr_min_confidence"   # default 0.50
_DEFAULT_TSR_MIN_CONF = 0.50


def _cell_grid_to_markdown(cells: list[dict]) -> str:
    """Render TSR cells (with row/col + span) into GitHub-flavoured markdown.

    Each cell dict carries ``row``, ``col``, ``rowspan``, ``colspan`` and ``text``.
    Spanning cells (colspan/rowspan > 1) are expanded across the grid columns so
    the markdown table preserves the spanned content (concept: ``__cal_spans``).
    """
    if not cells:
        return ""

    n_rows = max((c.get("row", 0) + max(1, c.get("rowspan", 1)) for c in cells), default=0)
    n_cols = max((c.get("col", 0) + max(1, c.get("colspan", 1)) for c in cells), default=0)
    if n_rows == 0 or n_cols == 0:
        return ""

    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        r0, c0 = c.get("row", 0), c.get("col", 0)
        text = str(c.get("text", "")).strip()
        rspan = max(1, c.get("rowspan", 1))
        cspan = max(1, c.get("colspan", 1))
        for dr in range(rspan):
            for dc in range(cspan):
                r, col = r0 + dr, c0 + dc
                if 0 <= r < n_rows and 0 <= col < n_cols:
                    # Expand the spanning cell's content across the covered grid
                    # cells so colspan/rowspan content is not lost.
                    grid[r][col] = text

    lines = []
    header = grid[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def recognize_table_structure(
    image,
    *,
    container_id: str,
    session_factory,
    doc_id: str = "",
    page_num: int = 0,
    tenant_id: str = "",
    element_id: str = "",
    bbox: BBox | None = None,
) -> UnifiedElement | None:
    """Run ONNX TSR on a cropped table ``image`` → a TABLE :class:`UnifiedElement`.

    ``session_factory()`` must return an object exposing ``.run(...)`` that yields
    a list of cell dicts (``row``/``col``/``rowspan``/``colspan``/``text``/
    ``confidence``). The min cell confidence becomes the element ``confidence``;
    if it falls below the tunable floor the result is still returned (the caller's
    gate already decided to enhance) but the low-confidence decision is logged.

    Returns ``None`` (degrade to fast path) when deps are unavailable, no session
    is provided, or the model emits no cells.
    """
    if not deepdoc_available() or session_factory is None:
        log_gate_decision(
            "deepdoc.table_structure.unavailable", score=0.0, threshold=0.0,
            outcome="fastpath", container_id=container_id,
        )
        return None

    session = session_factory()
    raw = session.run(image)
    cells = raw if isinstance(raw, list) else (raw or {}).get("cells", [])
    if not cells:
        log_gate_decision(
            "deepdoc.table_structure.empty", score=0.0, threshold=0.0,
            outcome="fastpath", container_id=container_id,
        )
        return None

    confidences = [float(c.get("confidence", 1.0)) for c in cells]
    min_conf = min(confidences) if confidences else 1.0
    floor = get_tunable(container_id, TUN_DD_TSR_MIN_CONF, _DEFAULT_TSR_MIN_CONF)
    log_gate_decision(
        "deepdoc.table_structure", score=min_conf, threshold=floor,
        outcome="recognized" if min_conf >= floor else "low_confidence",
        container_id=container_id, n_cells=len(cells),
    )

    markdown = _cell_grid_to_markdown(cells)
    return UnifiedElement(
        element_id=element_id,
        doc_id=doc_id,
        page_num=page_num,
        element_type=ElementType.TABLE,
        content=markdown,
        reading_order=0,
        tenant_id=tenant_id,
        bbox=bbox,
        confidence=min_conf,
        parser_version="deepdoc-tsr",
    )

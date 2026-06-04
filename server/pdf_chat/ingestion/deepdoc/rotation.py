"""Table-rotation evaluation (DeepDoc, optional).

Concept ported from RAGFlow's ``_evaluate_table_orientation`` — OCR the table at
each of the 4 cardinal angles and pick the angle with the highest OCR confidence.
We port the **concept only**; no RAGFlow code is copied. The OCR scorer is
INJECTED via ``ocr_scorer`` so tests stay pure (a mock returns a fixed best
angle) and so production wires the Phase-1 OCR engine.

Degrades gracefully: when cv2 is absent or no scorer is provided, returns the
zero-rotation default. All angles/thresholds resolve via :func:`get_tunable`;
decisions log via :func:`log_gate_decision`. No bare literals.
"""
from __future__ import annotations

from ...tunables import get_tunable, log_gate_decision
from ._deps import HAS_CV2

TUN_DD_ROTATION_MIN_CONF = "deepdoc.rotation_min_confidence"  # default 0.50
_DEFAULT_ROTATION_MIN_CONF = 0.50

# The candidate cardinal angles. This is an INTENT-layer constant (geometry, not
# customer-domain meaning) — the 4 cardinal orientations a scanned table can take.
_CANDIDATE_ANGLES = (0, 90, 180, 270)


def best_rotation(table_img, *, container_id: str, ocr_scorer) -> tuple[int, float]:
    """Pick the cardinal rotation that maximises OCR confidence for a table image.

    ``ocr_scorer(image, angle) -> float`` returns an OCR confidence in ``[0, 1]``
    for the image rotated by ``angle`` degrees. The angle with the highest score
    wins. Returns ``(angle, confidence)``.

    Degrades to ``(0, 1.0)`` when cv2 is unavailable or no scorer is provided —
    i.e. keep the page as-is (the Phase-1 OCR path already handled it).
    """
    if not HAS_CV2 or ocr_scorer is None:
        log_gate_decision(
            "deepdoc.rotation.unavailable", score=0.0, threshold=0.0,
            outcome="no_rotation", container_id=container_id,
        )
        return (0, 1.0)

    floor = get_tunable(container_id, TUN_DD_ROTATION_MIN_CONF, _DEFAULT_ROTATION_MIN_CONF)
    best_angle, best_conf = 0, float("-inf")
    for angle in _CANDIDATE_ANGLES:
        try:
            conf = float(ocr_scorer(table_img, angle))
        except Exception:
            continue
        if conf > best_conf:
            best_angle, best_conf = angle, conf

    if best_conf == float("-inf"):
        best_angle, best_conf = 0, 1.0

    log_gate_decision(
        "deepdoc.rotation", score=best_conf, threshold=floor,
        outcome="rotated" if best_angle != 0 else "upright",
        container_id=container_id, angle=best_angle,
    )
    return (best_angle, best_conf)

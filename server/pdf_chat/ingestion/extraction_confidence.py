"""Propagate per-element extraction confidence onto chunks (Spec §2 L1a).

Low-confidence OCR/table cells are FLAGGED (``low_confidence=True``), never
silently asserted, so downstream synthesis can caveat them. The flag threshold
is a tunable and the decision is logged (Spec §3 invariant 4).
"""
from __future__ import annotations

from pdf_chat.ingestion.ton_schema import Chunk
from pdf_chat.tunables import get_tunable, log_gate_decision


def propagate_confidence(
    chunks: list[Chunk],
    element_confidence: dict[str, float],
    *,
    container_id: str,
) -> list[Chunk]:
    """Stamp each chunk with its source element's confidence + low-confidence flag."""
    flag_below = get_tunable(container_id, "low_confidence_flag_below")
    for chunk in chunks:
        conf = element_confidence.get(chunk.source_element_id or "", 1.0)
        chunk.confidence = conf
        decision = log_gate_decision(
            "extraction_confidence",
            score=conf,
            threshold=flag_below,
            outcome="ok" if conf >= flag_below else "flagged_low",
            container_id=container_id,
            chunk_id=chunk.chunk_id,
        )
        chunk.low_confidence = not decision["passed"]
    return chunks

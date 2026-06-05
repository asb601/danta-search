"""Phase 4 — value-evidenced cross-domain bridge.

Reconciles a PDF ``Entity`` against the CSV master-key registry using literal
value overlap (``fingerprint_value``) only. Name equality is never a join signal;
a sub-threshold overlap is REFUSED, never a silent top-match.
"""
from __future__ import annotations

from pdf_chat.bridge.grain import (
    GrainAggregate,
    GrainFact,
    GrainResult,
    reconcile_grain,
)
from pdf_chat.bridge.reconcile import (
    EntityValueSample,
    MasterKeyColumn,
    PdfEntityValuesReader,
    ReconcileVerdict,
    build_bridge_for_entity,
    reconcile_entity_to_master_keys,
)

__all__ = [
    "EntityValueSample",
    "MasterKeyColumn",
    "ReconcileVerdict",
    "PdfEntityValuesReader",
    "reconcile_entity_to_master_keys",
    "build_bridge_for_entity",
    "GrainFact",
    "GrainAggregate",
    "GrainResult",
    "reconcile_grain",
]

"""Provenance vocabulary — the faithfulness labels for every learned artifact.

Spec §4: every glossary entry / ontology relationship / synthesized definition
carries a provenance so the surface can SAY how it knows a thing, not just assert
it. We surface a human-facing LABEL, never a raw confidence number, to the user.

A ``str``-Enum so the value persists directly into an open-vocab ``Text`` column
(``GlossaryEntry.provenance``) and serializes to JSON as the bare string —
matching the open-vocabulary convention used for ``status``/``doc_class`` (no
SQLAlchemy ``Enum`` columns anywhere in this layer).
"""
from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    """How a learned fact is known (faithfulness state)."""

    STATED = "stated"            # explicit definition confirmed in a cited span
    INFERRED = "inferred"        # distributional / co-reference signal only
    CONFLICTING = "conflicting"  # >=2 incompatible definitions, both surfaced
    NOT_FOUND = "not_found"      # no grounded basis → refuse, never fabricate


# Human-facing labels (spec §4) — shown to the user instead of a raw score.
_LABELS: dict[Provenance, str] = {
    Provenance.STATED: "stated in docs",
    Provenance.INFERRED: "inferred from usage",
    Provenance.CONFLICTING: "conflicting sources",
    Provenance.NOT_FOUND: "not found",
}


def label_for(p: "Provenance | str") -> str:
    """Map a ``Provenance`` (or its persisted value string) to a human label.

    Accepts the raw value string too (a Text column round-trips through here),
    and degrades any unrecognised value to ``"not found"`` so a stale/unknown
    persisted value never raises in the read path.
    """
    if not isinstance(p, Provenance):
        try:
            p = Provenance(p)
        except ValueError:
            return _LABELS[Provenance.NOT_FOUND]
    return _LABELS[p]

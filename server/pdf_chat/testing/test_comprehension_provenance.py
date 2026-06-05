"""Phase 5 — provenance vocabulary (faithfulness labels, no literals).

Kept in its own file (deviation from the plan, which folds it into the glossary
test file) so the parallel glossary worker does not collide on that file.
"""
from __future__ import annotations

import pytest


def test_provenance_enum_values():
    from pdf_chat.comprehension.provenance import Provenance

    assert Provenance.STATED.value == "stated"
    assert Provenance.INFERRED.value == "inferred"
    assert Provenance.CONFLICTING.value == "conflicting"
    assert Provenance.NOT_FOUND.value == "not_found"


def test_provenance_labels():
    from pdf_chat.comprehension.provenance import Provenance, label_for

    # Human-facing labels, NOT raw confidence numbers (spec §4).
    assert label_for(Provenance.STATED) == "stated in docs"
    assert label_for(Provenance.INFERRED) == "inferred from usage"
    assert label_for(Provenance.CONFLICTING) == "conflicting sources"
    assert label_for(Provenance.NOT_FOUND) == "not found"


def test_provenance_is_str_enum():
    """A str-Enum so a Text column / JSON serialization carries the bare value."""
    from pdf_chat.comprehension.provenance import Provenance

    assert isinstance(Provenance.STATED, str)
    assert Provenance.STATED == "stated"


def test_label_for_accepts_raw_value_string():
    """A persisted Text column round-trips through label_for via the value string."""
    from pdf_chat.comprehension.provenance import Provenance, label_for

    assert label_for("inferred") == "inferred from usage"
    assert label_for(Provenance.INFERRED) == "inferred from usage"


def test_label_for_unknown_falls_back_to_not_found():
    """An unrecognised provenance value never raises — it degrades to 'not found'."""
    from pdf_chat.comprehension.provenance import label_for

    assert label_for("something_unknown") == "not found"

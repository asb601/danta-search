"""Phase 0 — P0b relationship-map provenance forwarding tests (no real DB).

Proves build_relationship_map FORWARDS the fan-out/cardinality provenance the
Phase-2 join gate needs (value_overlap_pct, evidence_count, edge_provenance,
role_source, semantic_role) FAITHFULLY — missing fields stay None (never coerced
to a passing default), real values are never clamped, the min_confidence filter
is unchanged, and the LLM grounding (_render_join_section) is byte-identical
(zero behavior change today).

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_relationship_map.py -q
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.services.dashboard.data_catalog import build_relationship_map, _render_join_section


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _q):
        return _Result(self._rows)


def _rel(**kw):
    base = dict(
        file_a_id="A",
        file_b_id="B",
        shared_column="Vendor_ID",
        related_column="Vendor_ID",
        confidence_score=0.8,
        join_type="INNER JOIN",
        value_overlap_pct=0.92,
        evidence_count=14,
        edge_provenance={"card_a": 100, "card_b": 80, "key_kind_a": "pk", "key_kind_b": "fk"},
        role_source="fingerprint_index",
        semantic_role="custom:entity_key:record",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _run(rows, **kw):
    db = _FakeDB(rows)
    return asyncio.run(build_relationship_map(["A", "B"], db, **kw))


def test_forwards_all_provenance_fields():
    out = _run([_rel()])
    assert len(out) == 1
    e = out[0]
    # Original six keys preserved.
    for k in ("file_a_id", "file_b_id", "shared_column", "related_column", "confidence", "join_type"):
        assert k in e
    # The five newly-forwarded provenance signals.
    assert e["value_overlap_pct"] == 0.92
    assert e["evidence_count"] == 14
    assert e["edge_provenance"] == {"card_a": 100, "card_b": 80, "key_kind_a": "pk", "key_kind_b": "fk"}
    assert e["role_source"] == "fingerprint_index"
    assert e["semantic_role"] == "custom:entity_key:record"


def test_missing_provenance_preserved_as_none_not_coerced():
    # A legacy/heuristic edge missing the provenance columns: fields must be None,
    # never silently coerced to a "safe to join" default. No exception.
    rel = SimpleNamespace(
        file_a_id="A", file_b_id="B", shared_column="id",
        related_column="id", confidence_score=0.7, join_type="LEFT JOIN",
    )
    e = _run([rel])[0]
    assert e["value_overlap_pct"] is None
    assert e["evidence_count"] is None
    assert e["edge_provenance"] is None
    assert e["role_source"] is None
    assert e["semantic_role"] is None


def test_low_overlap_value_preserved_not_clamped():
    # A real-but-weak referential signal must round-trip exactly (the join gate,
    # not the catalog loader, decides the threshold).
    e = _run([_rel(value_overlap_pct=0.03)])[0]
    assert e["value_overlap_pct"] == 0.03


def test_edge_provenance_forwarded_opaquely():
    # Whatever keys the writer populated are preserved verbatim (no destructuring).
    prov = {"card_a": 5, "card_b": 5, "role_a": "x", "role_b": "y", "key_kind_a": "candidate", "key_kind_b": "candidate"}
    e = _run([_rel(edge_provenance=prov)])[0]
    assert e["edge_provenance"] == prov


def test_min_confidence_filter_unchanged():
    # P0b must not loosen the existing gate: a 0.4-confidence edge is still dropped
    # at the default min_confidence=0.5.
    assert _run([_rel(confidence_score=0.4)]) == []
    # ...and still present when the caller lowers the threshold.
    assert len(_run([_rel(confidence_score=0.4)], min_confidence=0.3)) == 1


def test_render_join_section_renders_safe_edge_without_leaking_provenance():
    # P2 update: _render_join_section now filters to SAFE joins (the P0b byte-frozen
    # guarantee is intentionally superseded — see test_dashboard_join_gate.py). The
    # forwarded provenance still must NOT leak into the LLM grounding text; it only
    # gates inclusion.
    tables = [SimpleNamespace(file_id="A", table_name="orders"),
              SimpleNamespace(file_id="B", table_name="vendors")]
    safe = {"file_a_id": "A", "file_b_id": "B", "shared_column": "Vendor_ID",
            "related_column": "Vendor_ID", "confidence": 0.8, "join_type": "INNER JOIN",
            "value_overlap_pct": 0.92, "evidence_count": 14,
            "edge_provenance": {"card_a": 100, "card_b": 50, "key_kind_a": "pk", "key_kind_b": "fk"},
            "role_source": "fingerprint_index", "semantic_role": "custom:entity_key:record"}
    out = _render_join_section(tables, [safe])
    assert "Vendor_ID" in out                                  # the safe join is advertised
    for leak in ("0.92", "card_a", "fingerprint_index", "evidence"):
        assert leak not in out                                 # provenance never leaks into grounding

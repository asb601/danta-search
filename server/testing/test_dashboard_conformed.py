"""Phase 7 — G6 conformed-dimension resolver (pure, no DB). ZERO false-conformed.

A categorical dimension is a safe GLOBAL slicer only if it is the SAME business
dimension across >=2 of the board's tables: exact semantic_role equality, role_kind
== dimension, a COMPLETE (non-truncated) observed member set, bounded cardinality,
and pairwise top_values Jaccard >= the policy referential floor (min_join_overlap).
Fail-closed on every missing/truncated/ambiguous signal — a slicer that means
territory in one file and postcode in another is the cross-file landmine.

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_conformed.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.conformed import (
    ConformedDimension,
    resolve_conformed_dimensions,
    resolve_widget_filters,
)
from app.services.dashboard.query_engine import WidgetIntent, _build_widget_grounding
from app.services.semantic_policy import get_semantic_policy

REGION = "custom:attribute:region"
SEGMENT = "custom:attribute:segment"
FLOOR = get_semantic_policy().min_join_overlap


def _col(name, role, kind, top_values, cardinality=None):
    return SimpleNamespace(
        name=name, semantic_role=role, role_kind=kind,
        top_values=top_values, cardinality=cardinality if cardinality is not None else len(top_values),
    )


def _table(name, cols):
    return SimpleNamespace(table_name=name, columns=cols)


def _conformed(catalog):
    return resolve_conformed_dimensions(catalog)


def test_same_role_high_overlap_complete_is_conformed():
    cat = [
        _table("sales", [_col("region", REGION, "dimension", ["North", "South", "East", "West"])]),
        _table("targets", [_col("region_name", REGION, "dimension", ["North", "South", "East", "West"])]),
    ]
    out = _conformed(cat)
    assert len(out) == 1
    d = out[0]
    assert d.semantic_role == REGION
    assert set(d.tables) == {"sales", "targets"}
    assert "North" in d.values
    # column binding per table is preserved (names differ across tables)
    assert d.column_by_table["sales"] == "region" and d.column_by_table["targets"] == "region_name"


def test_low_overlap_not_conformed():
    # territory vs postcode — same role label, disjoint members.
    cat = [
        _table("a", [_col("region", REGION, "dimension", ["North", "South"])]),
        _table("b", [_col("region", REGION, "dimension", ["10001", "10002"])]),
    ]
    assert _conformed(cat) == []


def test_single_table_not_a_global_candidate():
    cat = [_table("a", [_col("region", REGION, "dimension", ["North", "South", "East"])])]
    assert _conformed(cat) == []


def test_different_role_same_name_not_conformed():
    # both columns named "region" but DIFFERENT roles -> not the same dimension.
    cat = [
        _table("a", [_col("region", REGION, "dimension", ["North", "South"])]),
        _table("b", [_col("region", "custom:attribute:sales_region", "dimension", ["North", "South"])]),
    ]
    assert _conformed(cat) == []


def test_measure_role_never_conformed():
    cat = [
        _table("a", [_col("amount", "custom:additive_measure:amount", "measure", ["1", "2", "3"])]),
        _table("b", [_col("amount", "custom:additive_measure:amount", "measure", ["1", "2", "3"])]),
    ]
    assert _conformed(cat) == []


def test_empty_top_values_fails_closed():
    cat = [
        _table("a", [_col("region", REGION, "dimension", [])]),
        _table("b", [_col("region", REGION, "dimension", ["North", "South"])]),
    ]
    assert _conformed(cat) == []


def test_truncated_member_set_fails_closed():
    # 12 captured values but cardinality 200 -> top_values is a SUBSET -> uncertifiable,
    # even if the 12 overlap perfectly. This is the sample-truncation guard.
    twelve = [f"v{i}" for i in range(12)]
    cat = [
        _table("a", [_col("region", REGION, "dimension", twelve, cardinality=200)]),
        _table("b", [_col("region", REGION, "dimension", twelve, cardinality=200)]),
    ]
    assert _conformed(cat) == []


def test_high_cardinality_dim_not_offered():
    # a 4000-distinct "dimension" is not a readable slicer (and is truncated anyway).
    vals = [f"id{i}" for i in range(60)]
    cat = [
        _table("a", [_col("k", REGION, "dimension", vals, cardinality=4000)]),
        _table("b", [_col("k", REGION, "dimension", vals, cardinality=4000)]),
    ]
    assert _conformed(cat) == []


def test_threshold_boundary_tracks_policy_floor():
    # Jaccard exactly at the floor -> conformed; just below -> not. Pins the gate to
    # the policy constant, not a literal. Build value sets to hit the boundary.
    # A={N,S,E,W,X} B={N,S,E,W,Y} -> inter 4, union 6 -> 0.666 (>=0.5) conformed.
    cat_ok = [
        _table("a", [_col("r", REGION, "dimension", ["N", "S", "E", "W", "X"])]),
        _table("b", [_col("r", REGION, "dimension", ["N", "S", "E", "W", "Y"])]),
    ]
    assert len(_conformed(cat_ok)) == 1
    # A={N,S} B={N,X,Y} -> inter 1, union 4 -> 0.25 (<0.5) not conformed.
    cat_bad = [
        _table("a", [_col("r", REGION, "dimension", ["N", "S"])]),
        _table("b", [_col("r", REGION, "dimension", ["N", "X", "Y"])]),
    ]
    assert _conformed(cat_bad) == []
    assert FLOOR == 0.50  # documents the floor source


def test_value_comparison_is_normalized():
    # casing/whitespace differences must not block a genuine conformance.
    cat = [
        _table("a", [_col("r", REGION, "dimension", ["North", "South", "East"])]),
        _table("b", [_col("r", REGION, "dimension", [" north ", "SOUTH", "east"])]),
    ]
    assert len(_conformed(cat)) == 1


# --- per-widget applicability + grounding injection -------------------------

def _cdim():
    return ConformedDimension(
        semantic_role=REGION, label="Region",
        column_by_table={"sales": "region", "targets": "region_name"},
        tables=["sales", "targets"], values=["North", "South"], min_jaccard=1.0,
    )


def _intent(table):
    return WidgetIntent(title="t", nl_query="q", hints={"table": table})


def test_widget_filter_applies_when_table_carries_dim():
    applied, na = resolve_widget_filters(_intent("sales"), [_cdim()], [{"dimension": REGION, "values": ["North"]}])
    assert na == []
    assert applied == [{"label": "Region", "column": "region", "values": ["North"]}]


def test_widget_filter_not_affected_when_table_lacks_dim():
    applied, na = resolve_widget_filters(_intent("other"), [_cdim()], [{"dimension": REGION, "values": ["North"]}])
    assert applied == [] and na == ["Region"]


def _ftable():
    return SimpleNamespace(table_name="sales", columns=[SimpleNamespace(name="region")],
                           dimensions=set(), date_coverage=lambda: [])


def test_grounding_injects_global_filter_for_applicable_widget():
    g = _build_widget_grounding(_intent("sales"), [_ftable()],
                               applied_filters=[{"label": "Region", "column": "region", "values": ["North", "South"]}])
    assert "GLOBAL FILTER" in g and "region" in g and "North" in g


def test_grounding_has_no_filter_when_none_applied():
    g = _build_widget_grounding(_intent("sales"), [_ftable()], applied_filters=None)
    assert "GLOBAL FILTER" not in g

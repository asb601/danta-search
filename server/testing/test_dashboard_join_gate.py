"""Phase 2 — JOIN gate + additivity grounding (pure, no DB/LLM).

G1 (fan-out): a cross-file join is only advertised/assumed safe when the
cardinality is not many-to-many AND value_overlap_pct clears the policy floor —
fail-closed on any missing provenance (the erp-flat document-key lesson encoded).
G2 (additivity): a non-additive measure (ratio/rate) gets a "do not SUM" grounding
directive, driven by the ingestion ROLE only (never the column name).
Layer 3: post-exec, a widget whose result spans >1 table with no validated safe
relationship is flagged — the honest catch, since grounding is only advisory.

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_join_gate.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.join_gate import (
    classify_cardinality,
    safe_join,
    widget_join_safety,
)
from app.services.dashboard.data_catalog import DataCatalogColumn, _render_join_section
from app.services.dashboard.query_engine import WidgetIntent, _build_widget_grounding

ADD = "custom:additive_measure:revenue"
NONADD = "custom:non_additive_measure:margin_rate"


def _edge(*, overlap=0.8, ka="pk", kb="fk", card_a=100, card_b=50, prov=True):
    e = {"file_a_id": "A", "file_b_id": "B", "shared_column": "Vendor_ID",
         "related_column": "Vendor_ID", "confidence": 0.9, "value_overlap_pct": overlap}
    e["edge_provenance"] = (
        {"card_a": card_a, "card_b": card_b, "key_kind_a": ka, "key_kind_b": kb}
        if prov else None
    )
    return e


# --- classify_cardinality (fail-closed; class from key_kind shape, NOT cards) ---

def test_classify_failsclosed_on_missing_provenance():
    assert classify_cardinality(_edge(prov=False)) == "many_to_many"      # legacy edge
    assert classify_cardinality({"edge_provenance": {}}) == "many_to_many"  # no key_kind
    assert classify_cardinality(_edge(ka=None)) == "many_to_many"          # partial key_kind
    assert classify_cardinality(_edge(card_a=None)) == "many_to_many"      # missing card


def test_classify_from_key_kind_shape():
    assert classify_cardinality(_edge(ka="pk", kb="fk")) == "one_to_many"
    assert classify_cardinality(_edge(ka="fk", kb="pk")) == "many_to_one"
    assert classify_cardinality(_edge(ka="pk", kb="pk")) == "one_to_one"
    assert classify_cardinality(_edge(ka="fk", kb="fk")) == "many_to_many"
    assert classify_cardinality(_edge(ka="candidate", kb="pk")) == "many_to_many"


def test_classify_ignores_card_equality():
    # card_a == card_b must NOT imply 1:1 (cards are sample-scoped lower bounds).
    assert classify_cardinality(_edge(ka="fk", kb="fk", card_a=900, card_b=900)) == "many_to_many"


# --- safe_join (cardinality + value_overlap floor from policy; fail-closed) ----

def test_safe_join_rejects_document_key_low_overlap():
    # pk/pk but values don't reconcile (PO_Number-style) -> rejected by overlap floor.
    assert safe_join(_edge(overlap=0.03, ka="pk", kb="pk")) is False


def test_safe_join_allows_master_key():
    assert safe_join(_edge(overlap=0.79, ka="pk", kb="fk")) is True
    assert safe_join(_edge(overlap=0.79, ka="fk", kb="pk")) is True  # many_to_one is safe


def test_safe_join_failsclosed_on_none_overlap():
    assert safe_join(_edge(overlap=None, ka="pk", kb="pk")) is False


def test_safe_join_cardinality_dominates_high_overlap():
    # high overlap but many_to_many fan-out -> still unsafe.
    assert safe_join(_edge(overlap=0.95, ka="fk", kb="fk")) is False


def test_safe_join_overlap_floor_is_policy_driven():
    from app.services.semantic_policy import get_semantic_policy
    floor = get_semantic_policy().min_join_overlap
    assert safe_join(_edge(overlap=floor, ka="pk", kb="fk")) is True
    assert safe_join(_edge(overlap=floor - 0.01, ka="pk", kb="fk")) is False


# --- KNOWN JOINS grounding filtered to safe joins only ----------------------

def _tables():
    return [SimpleNamespace(file_id="A", table_name="orders"),
            SimpleNamespace(file_id="B", table_name="vendors")]


def test_render_join_section_keeps_safe_drops_unsafe():
    safe = _edge(overlap=0.8, ka="pk", kb="fk")
    unsafe = {**_edge(overlap=0.03, ka="pk", kb="pk"), "shared_column": "PO_Number"}
    out = _render_join_section(_tables(), [safe, unsafe])
    assert "Vendor_ID" in out          # the safe master-key join is advertised
    assert "PO_Number" not in out      # the document-key join is filtered out


def test_render_join_section_all_unsafe_is_empty():
    assert _render_join_section(_tables(), [_edge(prov=False)]) == ""


# --- additivity directive in per-widget grounding (role-driven) -------------

def _table_with(measure_role):
    col = DataCatalogColumn(
        name="margin_rate", data_type="float", semantic_role=measure_role,
        role_kind="measure", cardinality=100, null_ratio=0.0,
    )
    return SimpleNamespace(
        table_name="sales", columns=[col], dimensions=set(), date_coverage=lambda: [],
    )


def _intent(measure="margin_rate"):
    return WidgetIntent(title="t", nl_query="q", hints={"table": "sales"},
                        spec={"planned": {"measure": measure}})


def test_additivity_directive_present_for_non_additive_measure():
    g = _build_widget_grounding(_intent(), [_table_with(NONADD)])
    assert "sum" in g.lower() and ("never" in g.lower() or "do not" in g.lower())


def test_additivity_directive_absent_for_additive_measure():
    g = _build_widget_grounding(_intent(), [_table_with(ADD)])
    assert "never sum" not in g.lower()


def test_additivity_directive_absent_when_no_role():
    g = _build_widget_grounding(_intent(), [_table_with(None)])
    assert "never sum" not in g.lower()


# --- Layer 3: post-exec multi-table-no-safe-join detection ------------------

def _catalog():
    return [SimpleNamespace(file_id="A", blob_path="az://c/orders.parquet", parquet_path="orders.parquet"),
            SimpleNamespace(file_id="B", blob_path="az://c/vendors.parquet", parquet_path="vendors.parquet")]


def test_join_safety_single_table_is_safe():
    js = widget_join_safety(["az://c/orders.parquet"], _catalog(), [])
    assert js["multi_table"] is False and js["safe"] is True


def test_join_safety_two_tables_with_safe_edge():
    rels = [_edge(overlap=0.8, ka="pk", kb="fk")]  # A<->B safe
    js = widget_join_safety(["az://c/orders.parquet", "az://c/vendors.parquet"], _catalog(), rels)
    assert js["multi_table"] is True and js["safe"] is True


def test_join_safety_two_tables_unsafe_edge_is_flagged():
    rels = [_edge(overlap=0.03, ka="pk", kb="pk")]  # A<->B document key
    js = widget_join_safety(["az://c/orders.parquet", "az://c/vendors.parquet"], _catalog(), rels)
    assert js["multi_table"] is True and js["safe"] is False


def test_join_safety_two_tables_no_relationship_is_flagged():
    js = widget_join_safety(["az://c/orders.parquet", "az://c/vendors.parquet"], _catalog(), [])
    assert js["safe"] is False


def test_join_safety_resolves_parquet_path_form():
    # files_used may carry the parquet_path form; it must still resolve to file_id.
    js = widget_join_safety(["orders.parquet", "vendors.parquet"], _catalog(),
                            [_edge(overlap=0.8, ka="pk", kb="fk")])
    assert js["multi_table"] is True and js["safe"] is True


def test_join_safety_same_basename_distinct_tables_not_collapsed():
    # Two DIFFERENT tables that share a basename ('sales.parquet' under different
    # folders) must be detected as multi-table via their distinct full paths — the
    # ambiguous basename must never collapse them into one (fail-closed).
    cat = [SimpleNamespace(file_id="A", blob_path="az://c/2024/sales.parquet", parquet_path=None),
           SimpleNamespace(file_id="C", blob_path="az://c/2025/sales.parquet", parquet_path=None)]
    js = widget_join_safety(
        ["az://c/2024/sales.parquet", "az://c/2025/sales.parquet"], cat, []  # no safe edge
    )
    assert js["multi_table"] is True and js["safe"] is False

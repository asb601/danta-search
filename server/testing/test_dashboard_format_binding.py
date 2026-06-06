"""Phase 1 — de-heuristic the profiler/recommender (pure, no DB/LLM).

H2: role kind via the canonical taxonomy (exact dispatch, not name-substring).
H1: number format from the semantic ROLE (role gates; name is only a currency-unit
    hint inside a role-proven additive measure; ratios/keys never currency; the
    buggy `percent` format is NEVER emitted).
H3: bind the planner-NAMED measure/dimension (from intent.spec.planned) reconciled
    to the result columns; when the named column is absent, fail-closed to positional
    AND surface a warning — never a silent wrong-column substitution.

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_format_binding.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.data_catalog import (
    DataCatalogColumn,
    _role_kind,
    role_map_for_table,
)
from app.services.dashboard.recommendation_engine import (
    _format_for_role,
    _resolve_dimension,
    _resolve_named,
    recommend,
)
from app.services.dashboard.query_engine import WidgetIntent, profile_dataset

ADD = "custom:additive_measure:total_revenue"      # summable money/quantity
NONADD = "custom:non_additive_measure:margin_rate"  # ratio/rate — NOT summable
DATE = "custom:date:posting_date"
EKEY = "custom:entity_key:vendor_id"
RKEY = "custom:reference_key:plant"
ATTR = "custom:attribute:region"


# --- H2: _role_kind via canonical taxonomy ----------------------------------

def test_role_kind_canonical_dispatch():
    assert _role_kind(ADD) == "measure"
    assert _role_kind(NONADD) == "measure"
    assert _role_kind(DATE) == "date"
    assert _role_kind(EKEY) == "key"
    assert _role_kind(RKEY) == "key"
    assert _role_kind(ATTR) == "dimension"


def test_role_kind_empty_is_unknown():
    assert _role_kind(None) == "unknown"
    assert _role_kind("") == "unknown"


def test_role_kind_non_canonical_is_failsafe_dimension_not_measure():
    # A malformed/legacy role string must NOT be substring-guessed into a measure
    # (old behavior: "amount" in "transaction_amount" -> measure). Fail-safe to
    # dimension (a mis-bucketed dimension is never summed; a false measure is).
    assert _role_kind("transaction_amount") == "dimension"


# --- H1: role-aware format (role gates; name only hints currency unit) -------

def _shape():
    return SimpleNamespace()  # _format_for_role does not depend on shape internals


def test_format_additive_measure_with_money_name_is_currency():
    assert _format_for_role("total_revenue", _shape(), ADD) == "currency"


def test_format_additive_measure_without_money_name_is_number():
    # An additive COUNT (not money) must not get a $; name has no money token.
    assert _format_for_role("order_count", _shape(), ADD) == "number"


def test_format_non_additive_measure_is_never_currency_or_percent():
    # Role beats name: even with a money-ish token, a ratio is never currency,
    # and the buggy percent format is never emitted.
    assert _format_for_role("amount_ratio", _shape(), NONADD) == "number"
    assert _format_for_role("margin_pct", _shape(), NONADD) != "percent"
    assert _format_for_role("margin_pct", _shape(), NONADD) == "number"


def test_format_failclosed_to_name_when_no_role():
    assert _format_for_role("net_amount", _shape(), None) == "currency"
    assert _format_for_role("qty", _shape(), None) == "number"


def test_format_never_emits_percent_anywhere():
    # No input combination yields "percent" (renderer multiplies %, so it's unsafe).
    for name in ("pct", "percent", "rate", "ratio", "share", "margin_pct"):
        assert _format_for_role(name, _shape(), None) != "percent"
        assert _format_for_role(name, _shape(), NONADD) != "percent"


# --- H3: _resolve_named reconciliation --------------------------------------

def test_resolve_named_exact_and_normalized_and_single_contains():
    assert _resolve_named("revenue", ["region", "revenue"]) == ("revenue", "exact")
    assert _resolve_named("Net_Amount", ["region", "net_amount"]) == ("net_amount", "normalized")
    # single-candidate substring containment (SQL alias like total_net_amount)
    assert _resolve_named("net_amount", ["region", "total_net_amount"]) == ("total_net_amount", "contains")


def test_resolve_named_ambiguous_contains_returns_none():
    # two result columns contain the planned name -> ambiguous -> NO match (never guess).
    assert _resolve_named("amount", ["net_amount", "tax_amount"]) == (None, None)


def test_resolve_named_absent_returns_none():
    assert _resolve_named("profit", ["region", "revenue"]) == (None, None)


# --- H3: recommend binds the planner-named column, warns when absent ---------

def _intent(planned=None):
    spec = {"schema_version": 1, "planned": planned} if planned is not None else None
    return WidgetIntent(title="Revenue by region", nl_query="q", spec=spec)


def _rows():
    # measures profile in column order -> measures[0] == "revenue", measures[1] == "amount_tax"
    return [{"region": "APAC", "revenue": 10, "amount_tax": 2},
            {"region": "EU", "revenue": 20, "amount_tax": 3}]


def test_recommend_binds_planner_named_measure_not_positional():
    shape = profile_dataset(_rows(), None)
    intent = _intent({"measure": "amount_tax", "dimension": "region"})  # NOT measures[0]
    w = recommend(shape, intent, _rows())
    # The bound y/value must be the planner-named 'amount_tax', not positional 'revenue'.
    bound = w.config.get("y") or w.config.get("value")
    assert bound == "amount_tax", f"bound {bound!r}, config={w.config}"


def test_recommend_warns_and_failscloses_when_named_measure_absent():
    shape = profile_dataset(_rows(), None)
    intent = _intent({"measure": "profit", "dimension": "region"})  # absent
    warnings: list[str] = []
    w = recommend(shape, intent, _rows(), warnings=warnings)
    # Fail-closed: still renders (positional), but a warning is surfaced — no silent swap.
    assert any("profit" in m for m in warnings), warnings
    bound = w.config.get("y") or w.config.get("value")
    assert bound in ("revenue", "amount_tax")  # positional fallback, not crash


def test_recommend_fallback_path_no_spec_no_warning():
    shape = profile_dataset(_rows(), None)
    warnings: list[str] = []
    w = recommend(shape, _intent(None), _rows(), warnings=warnings)
    assert warnings == []  # no planner intent -> today's positional behavior, silent


def test_recommend_format_is_role_driven_via_role_map():
    shape = profile_dataset(_rows(), None)
    intent = _intent({"measure": "revenue", "dimension": "region"})
    role_map = {"revenue": ADD}
    w = recommend(shape, intent, _rows(), role_map=role_map)
    assert w.config.get("format") == "currency"
    # A non-additive role on the bound measure must never be currency.
    intent2 = _intent({"measure": "revenue", "dimension": "region"})
    w2 = recommend(shape, intent2, _rows(), role_map={"revenue": NONADD})
    assert w2.config.get("format") == "number"


# --- role_map_for_table helper (the route's map-build seam) ------------------

def _col(name, role):
    return DataCatalogColumn(
        name=name, data_type="float", semantic_role=role, role_kind="measure",
        cardinality=10, null_ratio=0.0,
    )


def test_role_map_for_table_builds_name_to_role_and_drops_empty():
    table = SimpleNamespace(columns=[_col("revenue", ADD), _col("region", ATTR), _col("noisy", None)])
    rm = role_map_for_table(table)
    assert rm == {"revenue": ADD, "region": ATTR}   # column with no role is dropped
    assert role_map_for_table(None) == {}            # no table -> empty (fail-closed)


# --- _resolve_dimension (named, then positional fallback) -------------------

def test_resolve_dimension_prefers_named_then_positional():
    shape = profile_dataset(_rows(), None)  # dimensions == ["region"]
    assert _resolve_dimension({"dimension": "region"}, shape) == "region"
    # named dimension absent -> positional fallback (first dimension)
    assert _resolve_dimension({"dimension": "country"}, shape) == "region"
    # no planned dimension -> positional
    assert _resolve_dimension(None, shape) == "region"


# --- single-measure + named-absent: bind the lone measure, NO warning -------

def test_recommend_single_measure_absent_binds_lone_measure_no_warning():
    # Per the design: when the named measure is absent but exactly ONE measure
    # exists, it is the unambiguous (aliased-aggregate) bind — no warning noise.
    rows = [{"region": "APAC", "revenue": 10}, {"region": "EU", "revenue": 20}]
    shape = profile_dataset(rows, None)
    warnings: list[str] = []
    w = recommend(shape, _intent({"measure": "profit", "dimension": "region"}), rows, warnings=warnings)
    assert warnings == []
    bound = w.config.get("y") or w.config.get("value")
    assert bound == "revenue"

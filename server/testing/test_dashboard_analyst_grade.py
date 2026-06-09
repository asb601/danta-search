"""Analyst-grade board planner — DETERMINISTIC honesty guards + derived-metric flow.

The LLM prompt (derived metrics, narrative arc, cross-source on validated keys) is
non-unit-testable; these tests cover the DETERMINISTIC backstops that keep the board
HONEST regardless of what the LLM proposes:

  D.1  Fail-closed targets — a target-requiring component (gauge_ring/progress_kpi/
       bullet) with NO real target binds is DOWNGRADED to a kpi tile, EXCEPT for an
       intrinsic 0-100% ratio metric carrying a structural max=100 (collection rate,
       fill rate), which keeps the gauge.
  D.3  Vanity removal — feasibility_filter drops a widget whose measure column is
       all-zero / single-constant, or whose dimension is single-constant (ORG_ID=204).
  Arc  The deterministic densifier still backstops the analyst arc (headline KPI
       ribbon w/ deltas + trend + ranked driver + a breakdown) when data supports it.
  A    Derived-metric `query` (the formula) is preserved through specs_to_intents so
       the SQL agent computes it; `measure` stays a REAL column for binding.

Run: cd server && uv run python -m pytest testing/test_dashboard_analyst_grade.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.board_planner import (
    WidgetSpec,
    ensure_composition,
    feasibility_filter,
    specs_to_intents,
)
from app.services.dashboard.query_engine import WidgetIntent, profile_dataset
from app.services.dashboard.recommendation_engine import recommend


# --------------------------------------------------------------------------
# Catalog fixtures with REAL column stats (min/max/cardinality) so the vanity
# guard has structural signal — mirrors what ingestion's column_stats carry.
# --------------------------------------------------------------------------

def _col(name, *, card=8, min_value=None, max_value=None):
    return SimpleNamespace(
        name=name, cardinality=card, min_value=min_value, max_value=max_value
    )


def _oebs_like_table():
    # An OE-line-like table: a real $ measure with spread, an all-zero measure
    # (SHIPPED_QUANTITY), a single-constant dimension (ORG_ID=204), a healthy dim,
    # and a temporal column.
    return SimpleNamespace(
        table_name="oe_lines",
        measures=["ordered_amount", "shipped_quantity", "unit_price"],
        dimensions=["region", "org_id"],
        temporal=["ordered_date"],
        row_count=10000,
        columns=[
            _col("ordered_amount", min_value=1.5, max_value=99000.0),
            _col("shipped_quantity", min_value=0, max_value=0),     # all-zero vanity
            _col("unit_price", min_value=12.0, max_value=12.0),     # single-constant
            _col("region", card=5),
            _col("org_id", card=1),                                  # constant ORG_ID
            _col("ordered_date", card=600),
        ],
    )


# ==========================================================================
# D.3 — Vanity removal (deterministic, in feasibility_filter)
# ==========================================================================

def test_feasibility_drops_all_zero_measure_widget():
    # A KPI on an all-zero measure (min==max==0) is meaningless — drop it.
    specs = [
        WidgetSpec(title="Total shipped quantity", question_type="kpi",
                   table="oe_lines", measure="shipped_quantity"),
        WidgetSpec(title="Total ordered amount", question_type="kpi",
                   table="oe_lines", measure="ordered_amount"),
    ]
    kept, reasons = feasibility_filter(specs, [_oebs_like_table()])
    titles = [k.title for k in kept]
    assert "Total shipped quantity" not in titles, titles
    assert "Total ordered amount" in titles, titles
    assert any("shipped quantity" in r.lower() for r in reasons), reasons


def test_feasibility_drops_single_constant_measure_widget():
    # unit_price is a single constant (min==max==12) — a headline aggregate of it
    # tells the executive nothing; drop it.
    specs = [
        WidgetSpec(title="Total unit price", question_type="kpi",
                   table="oe_lines", measure="unit_price"),
    ]
    kept, reasons = feasibility_filter(specs, [_oebs_like_table()])
    assert not kept, [k.title for k in kept]
    assert any("unit price" in r.lower() for r in reasons), reasons


def test_feasibility_drops_single_constant_dimension_breakdown():
    # A breakdown by ORG_ID (cardinality 1, constant 204) is a one-bar chart —
    # not a breakdown. Drop it.
    specs = [
        WidgetSpec(title="Amount by org", question_type="breakdown",
                   table="oe_lines", measure="ordered_amount", dimension="org_id",
                   viz="bar_chart"),
    ]
    kept, reasons = feasibility_filter(specs, [_oebs_like_table()])
    # It must not survive as an org_id breakdown — either dropped or repaired to a
    # different (non-constant) dimension.
    survivors = [k for k in kept if k.dimension == "org_id"]
    assert not survivors, [(k.title, k.dimension) for k in kept]


def test_feasibility_keeps_healthy_measure_and_dimension():
    # The good $ measure with spread and the healthy region dim must survive.
    specs = [
        WidgetSpec(title="Amount by region", question_type="breakdown",
                   table="oe_lines", measure="ordered_amount", dimension="region",
                   viz="bar_chart"),
    ]
    kept, _ = feasibility_filter(specs, [_oebs_like_table()])
    assert len(kept) == 1
    assert kept[0].measure == "ordered_amount" and kept[0].dimension == "region"


def test_vanity_guard_no_op_without_stats():
    # When a catalog has no min/max stats (None), the guard cannot prove vanity and
    # must NOT drop anything (fail-open on absent evidence — documented limitation).
    t = SimpleNamespace(
        table_name="t", measures=["m"], dimensions=["d"], temporal=[], row_count=10,
        columns=[_col("m", min_value=None, max_value=None), _col("d", card=None)],
    )
    specs = [WidgetSpec(title="Total m", question_type="kpi", table="t", measure="m")]
    kept, _ = feasibility_filter(specs, [t])
    assert len(kept) == 1


# ==========================================================================
# D.1 — Fail-closed targets (deterministic, in recommend)
# ==========================================================================

def test_gauge_without_target_downgraded_to_kpi():
    # A single-value KPI shape (no second measure) requested as a gauge_ring has NO
    # real target to bind. It must DOWNGRADE to a kpi tile, never render a fabricated
    # gauge ring.
    rows = [{"ordered_amount": 123456}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Total ordered amount", nl_query="q", requested_viz="gauge_ring",
        spec={"schema_version": 1, "planned": {"measure": "ordered_amount"}},
    )
    w = recommend(shape, intent, rows)
    assert w.component_type == "kpi_card", (w.component_type, w.config)
    assert "target" not in w.config


def test_progress_kpi_without_target_downgraded_to_kpi():
    rows = [{"revenue": 50000}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Revenue", nl_query="q", requested_viz="progress_kpi",
        spec={"schema_version": 1, "planned": {"measure": "revenue"}},
    )
    w = recommend(shape, intent, rows)
    assert w.component_type == "kpi_card", (w.component_type, w.config)


def test_bullet_without_target_downgraded_to_kpi():
    rows = [{"spend": 9000}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Spend", nl_query="q", requested_viz="bullet",
        spec={"schema_version": 1, "planned": {"measure": "spend"}},
    )
    w = recommend(shape, intent, rows)
    assert w.component_type == "kpi_card", (w.component_type, w.config)


def test_gauge_allowed_for_intrinsic_0_100_ratio_with_structural_max():
    # An intrinsic 0-100% ratio metric (collection rate / fill rate) carries a
    # STRUCTURAL max=100 in the planned spec. The gauge is HONEST here (the target
    # is the structural 100% ceiling, not a fabricated column) — keep it.
    rows = [{"collection_rate": 87.5}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Collection rate", nl_query="q", requested_viz="gauge_ring",
        spec={"schema_version": 1,
              "planned": {"measure": "collection_rate", "metric_max": 100}},
    )
    w = recommend(shape, intent, rows)
    assert w.component_type == "gauge_ring", (w.component_type, w.config)
    # The structural ceiling is bound as target_value (the LITERAL key the renderer
    # reads) so the ring fills toward 100% — NOT `target`, which the renderer treats
    # as a result-column name (a literal there silently degrades to a plain KPI).
    assert w.config.get("target_value") == 100
    assert "target" not in w.config


def test_gauge_with_real_target_column_unchanged():
    # When the dataset genuinely carries a distinct second measure, that is a real
    # target — the gauge stands (no downgrade, no fabrication).
    rows = [{"actual": 80, "plan": 100}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Actual vs plan", nl_query="q", requested_viz="gauge_ring",
        spec={"schema_version": 1, "planned": {"measure": "actual"}},
    )
    w = recommend(shape, intent, rows)
    assert w.component_type == "gauge_ring", (w.component_type, w.config)
    assert w.config.get("target") == "plan"


# ==========================================================================
# A — Derived-metric query (the formula) is preserved end-to-end
# ==========================================================================

def test_derived_metric_query_preserved_through_specs_to_intents():
    # A derived metric: measure stays a REAL column (remaining_amount) for binding,
    # but the formula lives in `query` and MUST reach the agent verbatim.
    formula = ("Compute the collection rate as 1 - SUM(remaining_amount) / "
               "SUM(original_amount), returned as a single percentage 0-100.")
    s = WidgetSpec(
        title="Collection rate", question_type="kpi", table="ar_invoices",
        measure="remaining_amount", query=formula,
    )
    catalog = [SimpleNamespace(table_name="ar_invoices", temporal=[])]
    intents = specs_to_intents([s], catalog)
    assert len(intents) == 1
    # The agent runs nl_query — it must be the formula, not a generated template.
    assert intents[0].nl_query == formula
    # The planned spec still carries the real binding measure.
    assert intents[0].spec["planned"]["measure"] == "remaining_amount"
    # And the formula is pinned for re-run faithfulness.
    assert intents[0].spec["planned"]["nl_query"] == formula


def test_metric_max_carried_through_specs_to_intents():
    # The structural ratio ceiling must reach the recommender via planned.metric_max.
    s = WidgetSpec(
        title="Fill rate", question_type="kpi", table="oe_lines",
        measure="shipped_ratio", query="shipped/ordered as a 0-100 percentage",
        metric_max=100,
    )
    catalog = [SimpleNamespace(table_name="oe_lines", temporal=[])]
    intents = specs_to_intents([s], catalog)
    assert intents[0].spec["planned"]["metric_max"] == 100


# ==========================================================================
# Arc — the analyst narrative still holds after the vanity guard.
#   A representative board (one healthy seed) densifies to headline KPIs + trend
#   + a ranked driver + breakdowns, with NO all-zero vanity tile and NO fabricated
#   gauge. Deterministic smoke (no LLM, no DB).
# ==========================================================================

def _analyst_table():
    # Healthy multi-measure table with a high-card "who" entity (vendor) for the
    # ranked driver view and a temporal column for the trend.
    return SimpleNamespace(
        table_name="po_lines",
        measures=["po_amount", "approved_amount"],
        dimensions=["region", "vendor", "org_id"],
        temporal=["approved_date"],
        row_count=20000,
        columns=[
            _col("po_amount", min_value=10.0, max_value=500000.0),
            _col("approved_amount", min_value=5.0, max_value=480000.0),
            _col("region", card=6),
            _col("vendor", card=80),       # high-card driver entity
            _col("org_id", card=1),        # constant — must not anchor a breakdown
            _col("approved_date", card=700),
        ],
    )


def test_analyst_arc_smoke_no_vanity_no_fabricated_gauge():
    catalog = [_analyst_table()]
    # The "LLM" proposed a derived-ratio KPI + a junk ORG_ID breakdown + a detail
    # table. Feasibility drops the junk; the densifier builds the rest of the arc.
    seed = [
        WidgetSpec(title="Approval rate", question_type="kpi", table="po_lines",
                   measure="approved_amount",
                   query="approved_amount / po_amount as a 0-100 percentage",
                   metric_max=100),
        WidgetSpec(title="Amount by org", question_type="breakdown", table="po_lines",
                   measure="po_amount", dimension="org_id", viz="bar_chart"),
        WidgetSpec(title="Top POs", question_type="detail", table="po_lines", viz="table"),
    ]
    kept, _ = feasibility_filter(seed, catalog)
    # The constant ORG_ID breakdown must not survive ON org_id.
    assert not any(s.dimension == "org_id" for s in kept), [(s.title, s.dimension) for s in kept]

    dense = ensure_composition(kept, catalog, max_widgets=10)
    intents = specs_to_intents(dense, catalog)

    resolved = []
    for it in intents:
        planned = (it.spec or {}).get("planned") or {}
        qt = planned.get("question_type")
        if qt == "trend":
            rows = [{"approved_date": "2024-01", planned["measure"]: 10},
                    {"approved_date": "2024-02", planned["measure"]: 20}]
        elif qt in ("breakdown", "share"):
            dim = planned["dimension"]
            n = 80 if dim == "vendor" else 5
            rows = [{dim: f"{dim[:2]}{i}", planned["measure"]: 1000 - i} for i in range(n)]
        elif qt == "kpi":
            rows = [{planned["measure"]: 87.5}]
        else:
            rows = [{"approved_date": "2024-01", "po_amount": 10, "region": "A"}]
        shape = profile_dataset(rows, None)
        resolved.append(recommend(shape, it, rows))

    from app.services.dashboard import assembly_engine
    config = assembly_engine.assemble(
        resolved, title="PO board", description=None, prompt="PO dashboard",
        generated_at="2026-06-08T00:00:00Z",
    )
    types = [w["type"] for w in config["widgets"]]
    board_visuals = [w for w in config["widgets"]
                     if w["type"] != "table" and not w.get("summary_only")]
    # Analyst arc: headline KPIs + trend + a ranked driver + breakdown(s).
    assert len(board_visuals) >= 5, types
    assert "line_chart" in types, types               # the trend
    assert "ranked_bar" in types, types               # the "who's driving it" driver
    # No fabricated-target gauge anywhere on the board.
    assert "gauge_ring" not in types or all(
        w["config"].get("target") == 100 for w in config["widgets"]
        if w["type"] == "gauge_ring"
    ), types
    import json
    json.dumps(config)  # JSONB-safe

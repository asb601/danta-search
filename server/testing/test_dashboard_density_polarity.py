"""Phase B — planner density + honesty polish (pure, no DB/LLM).

Covers the DETERMINISTIC parts of the dense-composition upgrade:

  1. Composition density (board_planner.ensure_composition): a feasible spec set
     is shaped into a PowerBI-style collage — a KPI ribbon + a trend + breakdowns —
     reaching a minimum VISUAL (non-table) count WHEN the data supports it, WITHOUT
     fabricating widgets (honest > padded). Structural only; the LLM still proposes
     the metrics/dimensions.
  2. Summary-only tagging (assembly_engine.assemble): a table intent is carried as
     a board widget tagged summary_only=True so the board frees its slot for a
     visual; non-table visuals are never tagged summary-only.
  3. Delta polarity by SEMANTIC ROLE/PLANNER HINT (recommendation_engine): an
     inverse metric (cost/aging/outstanding) rising is BAD → polarity 'inverse';
     a growth metric → 'positive'. Driven by the planner-emitted hint (the LLM
     proposes the metric semantics); fail-safe to 'positive', never fabricated.

Run: cd server && uv run python -m pytest testing/test_dashboard_density_polarity.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.dashboard.board_planner import WidgetSpec, ensure_composition
from app.services.dashboard.query_engine import WidgetIntent, profile_dataset
from app.services.dashboard.recommendation_engine import _resolve_polarity, recommend
from app.services.dashboard import assembly_engine


# --------------------------------------------------------------------------
# A catalog table rich enough to support a dense board: several measures, a
# temporal column, and two low-cardinality dimensions.
# --------------------------------------------------------------------------

def _col(name, kind, card=8):
    return SimpleNamespace(name=name, cardinality=card, min_value=None, max_value=None)


def _rich_table():
    return SimpleNamespace(
        table_name="sales",
        measures=["revenue", "cost", "orders"],
        dimensions=["region", "category"],
        temporal=["order_date"],
        row_count=5000,
        columns=[
            _col("revenue", "measure"),
            _col("cost", "measure"),
            _col("orders", "measure"),
            _col("region", "dimension", card=5),
            _col("category", "dimension", card=6),
            _col("order_date", "temporal", card=400),
        ],
    )


def _thin_table():
    # Only one measure, no temporal, one tiny dimension — the board CANNOT honestly
    # be made dense. ensure_composition must NOT pad it up to the minimum.
    return SimpleNamespace(
        table_name="tiny",
        measures=["headcount"],
        dimensions=["team"],
        temporal=[],
        row_count=12,
        columns=[_col("headcount", "measure"), _col("team", "dimension", card=3)],
    )


# ==========================================================================
# 1. Composition density
# ==========================================================================

def test_ensure_composition_reaches_min_visuals_when_data_supports():
    # The LLM proposed a single breakdown. The structural densifier should add a
    # KPI ribbon + a trend + extra breakdowns from EXISTING catalog columns so the
    # board has >= 5 visual (non-table) widgets.
    seed = [WidgetSpec(title="Revenue by region", question_type="breakdown",
                       table="sales", measure="revenue", dimension="region", viz="bar_chart")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=8)
    visuals = [s for s in out if s.question_type != "detail"]
    assert len(visuals) >= 5, [(s.question_type, s.measure, s.dimension) for s in out]


def test_ensure_composition_builds_a_kpi_ribbon():
    seed = [WidgetSpec(title="Trend", question_type="trend", table="sales",
                       measure="revenue", dimension="order_date", viz="line_chart")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=8)
    kpis = [s for s in out if s.question_type == "kpi"]
    # A headline ribbon of >= 3 KPIs drawn from the catalog's measures.
    assert len(kpis) >= 3, [(s.question_type, s.measure) for s in out]
    # Every KPI binds a REAL catalog measure (never fabricated).
    assert all(s.measure in {"revenue", "cost", "orders"} for s in kpis)


def test_ensure_composition_includes_a_trend_when_temporal_exists():
    seed = [WidgetSpec(title="Revenue by region", question_type="breakdown",
                       table="sales", measure="revenue", dimension="region", viz="bar_chart")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=8)
    assert any(s.question_type == "trend" for s in out), "a temporal column exists; expect a trend"


def test_ensure_composition_does_not_pad_thin_data():
    # Honesty: a thin table (1 measure, no temporal) yields FEW visuals, never 5.
    seed = [WidgetSpec(title="Headcount by team", question_type="breakdown",
                       table="tiny", measure="headcount", dimension="team", viz="bar_chart")]
    out = ensure_composition(seed, [_thin_table()], max_widgets=8)
    visuals = [s for s in out if s.question_type != "detail"]
    # No temporal -> no trend; one measure -> at most one KPI; one tiny dim ->
    # the single breakdown. Must NOT be inflated to the 5-visual minimum.
    assert len(visuals) < 5, [(s.question_type, s.measure, s.dimension) for s in out]


def test_ensure_composition_never_fabricates_columns():
    seed = [WidgetSpec(title="Revenue", question_type="kpi", table="sales", measure="revenue")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=8)
    cat = _rich_table()
    valid = set(cat.measures) | set(cat.dimensions) | set(cat.temporal)
    for s in out:
        for c in (s.measure, s.dimension, s.dimension2):
            assert c is None or c in valid, f"fabricated column {c!r} in {s}"


def test_ensure_composition_respects_max_widgets():
    seed = [WidgetSpec(title="Revenue by region", question_type="breakdown",
                       table="sales", measure="revenue", dimension="region", viz="bar_chart")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=4)
    assert len(out) <= 4


def test_ensure_composition_no_duplicate_lattice_tuples():
    seed = [WidgetSpec(title="Revenue by region", question_type="breakdown",
                       table="sales", measure="revenue", dimension="region", viz="bar_chart")]
    out = ensure_composition(seed, [_rich_table()], max_widgets=8)
    keys = [(s.question_type, s.table, s.measure, s.dimension, s.dimension2) for s in out]
    assert len(keys) == len(set(keys)), keys


# ==========================================================================
# 2. Summary-only tagging
# ==========================================================================

def _resolved(component_type, *, score=5.0):
    from app.services.dashboard.recommendation_engine import ResolvedWidget
    return ResolvedWidget(
        widget_id=component_type[:8],
        component_id=f"{component_type}.v1",
        component_type=component_type,
        title=component_type,
        dataset=[{"a": 1, "b": 2}],
        config={"columns": ["a", "b"]} if component_type == "table" else {"value": "a"},
        score=score,
        rationale="r",
        provenance={},
    )


def test_assemble_tags_table_widget_summary_only():
    widgets = [_resolved("kpi_card"), _resolved("bar_chart"), _resolved("table")]
    config = assembly_engine.assemble(
        widgets, title="T", description=None, prompt="p", generated_at="g"
    )
    by_type = {w["type"]: w for w in config["widgets"]}
    assert by_type["table"]["summary_only"] is True
    # Visuals are NOT tagged summary-only (so the board renders them).
    assert by_type["kpi_card"].get("summary_only", False) is False
    assert by_type["bar_chart"].get("summary_only", False) is False


def test_assemble_board_has_min_5_visuals_with_table_summary_only():
    widgets = [
        _resolved("kpi_card"), _resolved("metric_tile"), _resolved("delta_kpi"),
        _resolved("line_chart"), _resolved("bar_chart"), _resolved("pie_chart"),
        _resolved("table"),
    ]
    config = assembly_engine.assemble(
        widgets, title="T", description=None, prompt="p", generated_at="g"
    )
    board_visuals = [w for w in config["widgets"]
                     if w["type"] != "table" and not w.get("summary_only")]
    assert len(board_visuals) >= 5
    summary = [w for w in config["widgets"] if w.get("summary_only")]
    assert all(w["type"] == "table" for w in summary)


# ==========================================================================
# 3. Delta polarity by role / planner hint
# ==========================================================================

def test_resolve_polarity_inverse_from_planner_hint():
    # The LLM (which proposes the metric semantics) tagged the measure as an inverse
    # metric (cost/aging/outstanding rising is bad).
    assert _resolve_polarity({"polarity": "inverse"}, role=None) == "inverse"


def test_resolve_polarity_positive_from_planner_hint():
    assert _resolve_polarity({"polarity": "positive"}, role=None) == "positive"


def test_resolve_polarity_failsafe_positive_when_absent_or_invalid():
    # Never fabricate an inverse claim: missing/garbage hint => growth framing.
    assert _resolve_polarity(None, role=None) == "positive"
    assert _resolve_polarity({}, role=None) == "positive"
    assert _resolve_polarity({"polarity": "sideways"}, role=None) == "positive"


def test_recommend_carries_polarity_into_delta_config():
    rows = [{"order_date": "2026-01", "cost": 100}, {"order_date": "2026-02", "cost": 200}]
    shape = profile_dataset(rows, {"type": "line"})
    intent = WidgetIntent(
        title="Cost trend", nl_query="q",
        requested_viz="delta_kpi",
        spec={"schema_version": 1,
              "planned": {"measure": "cost", "dimension": "order_date", "polarity": "inverse"}},
    )
    w = recommend(shape, intent, rows)
    assert w.config.get("polarity") == "inverse", w.config


def test_recommend_kpi_default_polarity_positive_when_no_hint():
    rows = [{"region": "APAC", "revenue": 10}, {"region": "EU", "revenue": 20}]
    shape = profile_dataset(rows, None)
    intent = WidgetIntent(
        title="Revenue by region", nl_query="q", requested_viz="bar_chart",
        spec={"schema_version": 1, "planned": {"measure": "revenue", "dimension": "region"}},
    )
    w = recommend(shape, intent, rows)
    # bar_chart has no delta; polarity is only emitted for delta-bearing tiles.
    assert w.config.get("polarity", "positive") == "positive"


# ==========================================================================
# 4. End-to-end DETERMINISTIC smoke: ensure_composition → specs_to_intents →
#    recommend → assemble produces a dense board (>= 5 visuals) with the table
#    tagged summary-only. No LLM, no DB.
# ==========================================================================

def test_smoke_dense_board_5plus_visuals_table_summary_only():
    from app.services.dashboard.board_planner import specs_to_intents

    catalog = [_rich_table()]
    # The "LLM" proposed just a breakdown + a detail table; densify the rest.
    seed = [
        WidgetSpec(title="Revenue by region", question_type="breakdown",
                   table="sales", measure="revenue", dimension="region", viz="bar_chart"),
        WidgetSpec(title="Top orders", question_type="detail", table="sales", viz="table"),
    ]
    dense = ensure_composition(seed, catalog, max_widgets=8)
    intents = specs_to_intents(dense, catalog)

    # Profile/recommend each intent against a representative non-empty dataset so a
    # component binds (one row per dim for breakdowns, a 2-point series for trends).
    resolved = []
    for it in intents:
        planned = (it.spec or {}).get("planned") or {}
        qt = planned.get("question_type")
        if qt == "trend":
            rows = [{"order_date": "2026-01", planned["measure"]: 10},
                    {"order_date": "2026-02", planned["measure"]: 20}]
        elif qt in ("breakdown", "share"):
            rows = [{planned["dimension"]: "A", planned["measure"]: 10},
                    {planned["dimension"]: "B", planned["measure"]: 20}]
        elif qt == "kpi":
            rows = [{planned["measure"]: 999999}]
        else:  # detail
            rows = [{"order_date": "2026-01", "revenue": 10, "region": "A"}]
        shape = profile_dataset(rows, None)
        resolved.append(recommend(shape, it, rows))

    config = assembly_engine.assemble(
        resolved, title="Sales board", description=None, prompt="sales dashboard",
        generated_at="2026-06-08T00:00:00Z",
    )

    board_visuals = [w for w in config["widgets"]
                     if w["type"] != "table" and not w.get("summary_only")]
    assert len(board_visuals) >= 5, [(w["type"], w.get("summary_only")) for w in config["widgets"]]
    tables = [w for w in config["widgets"] if w["type"] == "table"]
    assert tables and all(w["summary_only"] is True for w in tables)
    import json
    json.dumps(config)  # JSONB-safe


# ==========================================================================
# 5. R1 — LLM polarity reaches densifier-injected widgets.
#    ensure_composition builds a measure->polarity map from the LLM-authored
#    specs; a KPI/trend it injects on an inverse measure must carry
#    polarity='inverse'. A measure the LLM never tagged stays None (no Python
#    inference — honest).
# ==========================================================================

def _inverse_rich_table():
    # 'overdue_balance' is the inverse measure; 'revenue'/'orders' are positive.
    return SimpleNamespace(
        table_name="ar",
        measures=["revenue", "overdue_balance", "orders"],
        dimensions=["region", "category", "vendor"],
        temporal=["invoice_date"],
        row_count=8000,
        columns=[
            _col("revenue", "measure"),
            _col("overdue_balance", "measure"),
            _col("orders", "measure"),
            _col("region", "dimension", card=5),
            _col("category", "dimension", card=6),
            _col("vendor", "dimension", card=40),   # high-card "who" entity
            _col("invoice_date", "temporal", card=600),
        ],
    )


def test_densifier_kpi_carries_llm_inverse_polarity():
    # The LLM authored ONE breakdown on the inverse measure 'overdue_balance' and
    # tagged its polarity. The densifier injects a KPI ribbon; the KPI on
    # 'overdue_balance' must inherit polarity='inverse' from the LLM-authored map.
    seed = [WidgetSpec(title="Overdue by region", question_type="breakdown",
                       table="ar", measure="overdue_balance", dimension="region",
                       viz="bar_chart", polarity="inverse")]
    out = ensure_composition(seed, [_inverse_rich_table()], max_widgets=8)
    kpis_on_inverse = [s for s in out
                       if s.question_type == "kpi" and s.measure == "overdue_balance"]
    assert kpis_on_inverse, [(s.question_type, s.measure) for s in out]
    assert all(s.polarity == "inverse" for s in kpis_on_inverse), \
        [(s.measure, s.polarity) for s in out]


def test_densifier_measure_without_llm_polarity_stays_none():
    # The LLM tagged ONLY 'overdue_balance'. 'revenue' / 'orders' were never
    # tagged, so the densifier KPIs on them keep polarity=None (no inference).
    seed = [WidgetSpec(title="Overdue by region", question_type="breakdown",
                       table="ar", measure="overdue_balance", dimension="region",
                       viz="bar_chart", polarity="inverse")]
    out = ensure_composition(seed, [_inverse_rich_table()], max_widgets=8)
    untagged = [s for s in out
                if s.question_type == "kpi" and s.measure in ("revenue", "orders")]
    assert untagged, [(s.question_type, s.measure) for s in out]
    assert all(s.polarity is None for s in untagged), \
        [(s.measure, s.polarity) for s in untagged]


def test_densifier_trend_carries_llm_inverse_polarity():
    # A trend the densifier injects on the inverse headline measure also inherits
    # the LLM polarity (so the trend-derived delta colors correctly).
    seed = [WidgetSpec(title="Overdue total", question_type="kpi",
                       table="ar", measure="overdue_balance", polarity="inverse")]
    out = ensure_composition(seed, [_inverse_rich_table()], max_widgets=8)
    trends = [s for s in out if s.question_type == "trend"]
    # The headline measure is the table's first measure ('revenue'); but if a
    # trend lands on 'overdue_balance' it must carry inverse. Assert per-measure.
    for s in trends:
        if s.measure == "overdue_balance":
            assert s.polarity == "inverse", (s.measure, s.polarity)
        elif s.measure in ("revenue", "orders"):
            assert s.polarity is None, (s.measure, s.polarity)


# ==========================================================================
# 6. R2 — ranked_bar is reachable.
#    'ranked_bar' is a valid viz; a high-cardinality "who" breakdown the
#    densifier emits resolves to a ranked_bar end-to-end through recommend().
# ==========================================================================

def test_ranked_bar_in_valid_viz():
    from app.services.dashboard.board_planner import _VALID_VIZ
    assert "ranked_bar" in _VALID_VIZ


def test_densifier_emits_ranked_bar_for_high_card_entity_dim():
    # 'vendor' (card=40) is the "who's driving it" entity dimension — moderate-
    # to-high cardinality. The densifier must route it to a top-N ranked view,
    # not a plain bar. (Either viz='ranked_bar' directly, or viz=None so
    # recommend() scores it — both must yield a ranked_bar end-to-end.)
    seed = [WidgetSpec(title="Revenue total", question_type="kpi",
                       table="ar", measure="revenue")]
    out = ensure_composition(seed, [_inverse_rich_table()], max_widgets=10)
    vendor_breakdowns = [s for s in out
                         if s.question_type == "breakdown" and s.dimension == "vendor"]
    assert vendor_breakdowns, [(s.question_type, s.dimension) for s in out]
    # The vendor breakdown must be eligible for ranked_bar: viz is either the
    # explicit 'ranked_bar' or None (so scoring picks it) — never pinned to bar.
    assert all(s.viz in ("ranked_bar", None) for s in vendor_breakdowns), \
        [(s.dimension, s.viz) for s in vendor_breakdowns]


def test_ranked_bar_reachable_through_recommend_for_high_card_entity():
    # End-to-end: a high-card entity breakdown intent → recommend() resolves to a
    # ranked_bar component (bar_chart's card gate excludes a >50-card dimension;
    # ranked_bar's allows it). Use 60 distinct vendors.
    from app.services.dashboard.board_planner import specs_to_intents

    catalog = [_inverse_rich_table()]
    seed = [WidgetSpec(title="Revenue by vendor", question_type="breakdown",
                       table="ar", measure="revenue", dimension="vendor",
                       viz="ranked_bar")]
    intents = specs_to_intents(seed, catalog)
    assert intents
    it = intents[0]
    rows = [{"vendor": f"V{i}", "revenue": 1000 - i} for i in range(60)]
    shape = profile_dataset(rows, None)
    w = recommend(shape, it, rows)
    assert w.component_type == "ranked_bar", (w.component_type, w.rationale)
    # top_n is structural (from visualization_rules default), not a magic literal.
    assert isinstance(w.config.get("top_n"), int) and w.config["top_n"] > 0


# ==========================================================================
# 7. R1+R2 smoke — a representative rich board (inverse measure + high-card
#    entity dim) densifies to >= 5 visuals INCLUDING a ranked_bar AND with the
#    inverse measure's tile carrying polarity='inverse'. No LLM, no DB.
# ==========================================================================

def test_smoke_rich_board_has_ranked_bar_and_inverse_polarity():
    from app.services.dashboard.board_planner import specs_to_intents

    catalog = [_inverse_rich_table()]
    # The "LLM" proposed an inverse-measure KPI + a detail table; the densifier
    # fills out the ribbon, the trend, and the breakdowns (incl. the high-card
    # vendor DRIVER view, which becomes a ranked_bar).
    seed = [
        WidgetSpec(title="Total overdue", question_type="kpi",
                   table="ar", measure="overdue_balance", polarity="inverse"),
        WidgetSpec(title="Top invoices", question_type="detail", table="ar", viz="table"),
    ]
    dense = ensure_composition(seed, catalog, max_widgets=10)
    intents = specs_to_intents(dense, catalog)

    resolved = []
    for it in intents:
        planned = (it.spec or {}).get("planned") or {}
        qt = planned.get("question_type")
        if qt == "trend":
            rows = [{"invoice_date": "2026-01", planned["measure"]: 10},
                    {"invoice_date": "2026-02", planned["measure"]: 20}]
        elif qt in ("breakdown", "share"):
            dim = planned["dimension"]
            # Many distinct entities for the high-card vendor breakdown so the
            # ranked_bar gate is exercised; few for low-card dims.
            n = 60 if dim == "vendor" else 4
            rows = [{dim: f"{dim[:2]}{i}", planned["measure"]: 1000 - i} for i in range(n)]
        elif qt == "kpi":
            rows = [{planned["measure"]: 999999}]
        else:  # detail
            rows = [{"invoice_date": "2026-01", "revenue": 10, "region": "A"}]
        shape = profile_dataset(rows, None)
        resolved.append(recommend(shape, it, rows))

    config = assembly_engine.assemble(
        resolved, title="AR board", description=None, prompt="AR dashboard",
        generated_at="2026-06-08T00:00:00Z",
    )

    board_visuals = [w for w in config["widgets"]
                     if w["type"] != "table" and not w.get("summary_only")]
    assert len(board_visuals) >= 5, [(w["type"], w.get("summary_only")) for w in config["widgets"]]
    # A ranked_bar is on the board (the vendor driver view).
    assert any(w["type"] == "ranked_bar" for w in config["widgets"]), \
        [w["type"] for w in config["widgets"]]
    # The inverse measure's delta-bearing tile carries polarity='inverse'.
    inverse_tiles = [w for w in config["widgets"]
                     if w["config"].get("value") == "overdue_balance"
                     and "polarity" in w["config"]]
    assert inverse_tiles, [(w["type"], w["config"]) for w in config["widgets"]]
    assert all(w["config"]["polarity"] == "inverse" for w in inverse_tiles), \
        [(w["type"], w["config"].get("polarity")) for w in inverse_tiles]
    import json
    json.dumps(config)  # JSONB-safe

"""Phase 0 — P0 spec-pin tests (pure, no DB/LLM).

Proves the validated WidgetSpec is pinned into the persisted config as a
faithful planned+bound contract, WITHOUT changing what renders, and that the
fail-closed faithfulness rules hold (aggregation marked inferred, no fabricated
executed-SQL, empty widgets record no fabricated binding).

Run: cd server && uv run --with pytest python -m pytest testing/test_dashboard_spec_pin.py -q
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.dashboard.board_planner import WidgetSpec, specs_to_intents
from app.services.dashboard.query_engine import WidgetIntent
from app.services.dashboard.recommendation_engine import ResolvedWidget, build_pinned_spec
from app.services.dashboard import assembly_engine


def _catalog():
    # Minimal catalog table: name + no temporal cols (so no time window injected).
    return [SimpleNamespace(table_name="orders", temporal=[])]


def _widget(provenance=None, dataset=None, config=None):
    return ResolvedWidget(
        widget_id="w1",
        component_id="bar.basic.v1",
        component_type="bar_chart",
        title="Revenue by region",
        dataset=dataset if dataset is not None else [{"region": "APAC", "rev": 10}],
        config=config if config is not None else {"x": "region", "y": "rev", "format": "currency"},
        score=5.0,
        rationale="bar fits a breakdown",
        provenance=provenance if provenance is not None else {},
    )


# --- specs_to_intents attaches the planned spec -----------------------------

def test_specs_to_intents_attaches_planned_spec():
    s = WidgetSpec(
        title="Revenue by region",
        question_type="breakdown",
        table="orders",
        measure="rev",
        dimension="region",
        viz="bar_chart",
        query="Total revenue grouped by region",
        chart_rationale="bar shows per-region comparison",
    )
    intents = specs_to_intents([s], _catalog())
    assert len(intents) == 1
    spec = intents[0].spec
    assert spec is not None
    planned = spec["planned"]
    assert planned["question_type"] == "breakdown"
    assert planned["table"] == "orders"
    assert planned["measure"] == "rev"
    assert planned["dimension"] == "region"
    assert planned["viz"] == "bar_chart"
    assert planned["chart_rationale"] == "bar shows per-region comparison"
    # The pinned nl_query is the exact re-run handle that went to the agent.
    assert planned["nl_query"] == intents[0].nl_query


def test_fallback_intent_has_no_spec():
    # Intents built directly (the decompose_prompt fallback path) carry no spec.
    intent = WidgetIntent(title="x", nl_query="y")
    assert intent.spec is None


# --- build_pinned_spec merges planned + bound -------------------------------

def test_build_pinned_spec_merges_planned_and_bound():
    intent = WidgetIntent(
        title="Revenue by region",
        nl_query="Total revenue grouped by region",
        spec={"schema_version": 1, "planned": {"question_type": "breakdown", "measure": "rev"}},
    )
    shape = SimpleNamespace(aggregation="SUM")
    widget = _widget()
    spec = build_pinned_spec(intent, widget, shape)
    assert spec["planned"]["measure"] == "rev"
    assert spec["bound"]["component_type"] == "bar_chart"
    assert spec["bound"]["config"] == {"x": "region", "y": "rev", "format": "currency"}
    assert spec["empty"] is False


def test_build_pinned_spec_marks_aggregation_inferred_and_omits_fake_sql():
    # The agent does NOT return executed SQL; the pin must label aggregation as
    # INFERRED (from output shape) and must NOT carry a fabricated `sql` field.
    intent = WidgetIntent(title="t", nl_query="q", spec={"planned": {}})
    shape = SimpleNamespace(aggregation="SUM")
    spec = build_pinned_spec(intent, _widget(), shape)
    assert spec["bound"]["aggregation_inferred"] == "SUM"
    assert "aggregation_applied" not in spec["bound"]
    assert "sql" not in spec["bound"] and "sql" not in spec


def test_build_pinned_spec_fallback_planned_none_does_not_raise():
    intent = WidgetIntent(title="t", nl_query="q")  # spec=None (fallback path)
    spec = build_pinned_spec(intent, _widget(), SimpleNamespace(aggregation="RAW"))
    assert spec["planned"] is None
    assert spec["bound"]["component_type"] == "bar_chart"


def test_build_pinned_spec_empty_widget_records_no_fabricated_binding():
    empty = _widget(dataset=[], config={"columns": []}, provenance={"empty": True})
    spec = build_pinned_spec(WidgetIntent(title="t", nl_query="q"), empty, SimpleNamespace(aggregation="NONE"))
    assert spec["empty"] is True
    assert spec["bound"]["config"] == {"columns": []}


# --- assemble carries the spec through, bumps version, stays JSON-safe -------

def test_assemble_carries_spec_in_provenance_and_is_json_safe():
    w = _widget(provenance={"route": "agent", "spec": {"schema_version": 1, "planned": {"table": "orders"}}})
    config = assembly_engine.assemble(
        [w], title="T", description=None, prompt="p", generated_at="2026-06-07T00:00:00Z"
    )
    pinned = config["widgets"][0]["provenance"]["spec"]
    assert pinned["planned"]["table"] == "orders"
    # Must serialize to JSONB without raising.
    json.dumps(config)


def test_assemble_version_bumped_to_1_2():
    config = assembly_engine.assemble(
        [_widget()], title="T", description=None, prompt="p", generated_at="2026-06-07T00:00:00Z"
    )
    assert config["version"] == "1.2"


def test_assemble_spec_does_not_perturb_layout_or_config():
    # Zero-render-change guard: the ONLY difference between a widget with a pinned
    # spec and one without must be inside provenance. grid/type/config/data/order
    # must be byte-identical.
    base_prov = {"route": "agent"}
    w_no_spec = _widget(provenance=dict(base_prov))
    w_spec = _widget(provenance={**base_prov, "spec": {"schema_version": 1, "planned": {}, "bound": {}}})
    c0 = assembly_engine.assemble([w_no_spec], title="T", description=None, prompt="p", generated_at="g")
    c1 = assembly_engine.assemble([w_spec], title="T", description=None, prompt="p", generated_at="g")
    p0 = {k: v for k, v in c0["widgets"][0].items() if k != "provenance"}
    p1 = {k: v for k, v in c1["widgets"][0].items() if k != "provenance"}
    assert p0 == p1
    # And provenance differs ONLY by the additive spec key.
    assert {k: v for k, v in c1["widgets"][0]["provenance"].items() if k != "spec"} == c0["widgets"][0]["provenance"]

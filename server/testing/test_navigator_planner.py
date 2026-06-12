"""Phase-1 tests for the navigator package: types.py + planner.py.

DETERMINISTIC — no live LLM, no DB. The single gpt-4o-mini call inside
``planner.plan`` is mocked by monkeypatching ``planner.get_client`` to return a
fake client whose ``chat.completions.create`` yields canned JSON — exactly how
``decompose.py`` consumes the client (``client, _ = get_client()`` then
``client.chat.completions.create(...)``).

Covers (planner):
  * single-entity question        -> StepDAG with 1 LOOKUP step
  * "customer ... vs vendor ..."  -> 2 LOOKUP steps, no depends_on
  * "ratio of X to Y per vendor"  -> 2 LOOKUP + 1 COMPOSE (depends_on = the two
                                     LOOKUP ids, compose_expr.op == "ratio")
  * malformed JSON                -> None (never raises)
  * cycle in depends_on           -> None
  * dangling depends_on reference -> None

Covers (types):
  * the frozen dataclasses reject mutation
  * StepLedger is mutable and StepLedger.get_scalar works

Run: cd server && uv run --with pytest python -m pytest testing/test_navigator_planner.py -q
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app.services.navigator.planner as P
from app.services.navigator.types import (
    StepDAG,
    StepKind,
    StepLedger,
    StepResult,
    IntentStep,
)


# --------------------------------------------------------------------------
# Mini-call mock: a fake AzureOpenAI client returning canned JSON.
# --------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs):  # noqa: ANN003 — signature mirrors the SDK
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def _mock_mini(monkeypatch, payload) -> None:
    """Patch planner.get_client to return a client that yields ``payload``.

    ``payload`` may be a dict/list (json-encoded) or a raw string (to inject
    malformed JSON). The deployment name returned is irrelevant to the test.
    """
    content = payload if isinstance(payload, str) else json.dumps(payload)

    def _fake_get_client():
        return _FakeClient(content), "mini-deployment"

    monkeypatch.setattr(P, "get_client", _fake_get_client)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# planner.plan
# --------------------------------------------------------------------------
def test_single_entity_one_lookup_step(monkeypatch):
    _mock_mini(
        monkeypatch,
        {
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "LOOKUP",
                    "entity": "vendor",
                    "measure_concept": "total spend",
                    "grain": "entity",
                    "grain_entity": "vendor",
                    "time_grain": None,
                    "filters": [],
                    "threshold": None,
                    "depends_on": [],
                    "join_entities": [],
                    "compose_expr": None,
                }
            ]
        },
    )
    dag = _run(P.plan("total vendor spend"))
    assert isinstance(dag, StepDAG)
    assert len(dag.steps) == 1
    step = dag.steps[0]
    assert step.kind == StepKind.LOOKUP
    assert step.entity == "vendor"
    assert step.depends_on == ()


def test_two_distinct_entities_two_lookup_no_depends(monkeypatch):
    _mock_mini(
        monkeypatch,
        {
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "LOOKUP",
                    "entity": "customer",
                    "measure_concept": "receipts",
                    "depends_on": [],
                },
                {
                    "step_id": "s2",
                    "kind": "LOOKUP",
                    "entity": "vendor",
                    "measure_concept": "payments",
                    "depends_on": [],
                },
            ]
        },
    )
    dag = _run(P.plan("customer receipts vs vendor payments"))
    assert dag is not None
    assert len(dag.steps) == 2
    assert all(s.kind == StepKind.LOOKUP for s in dag.steps)
    assert all(s.depends_on == () for s in dag.steps)
    assert {s.entity for s in dag.steps} == {"customer", "vendor"}


def test_ratio_two_lookup_plus_compose(monkeypatch):
    _mock_mini(
        monkeypatch,
        {
            "steps": [
                {
                    "step_id": "s1",
                    "kind": "LOOKUP",
                    "entity": "invoice",
                    "measure_concept": "overdue invoices",
                    "grain": "entity",
                    "grain_entity": "vendor",
                    "depends_on": [],
                },
                {
                    "step_id": "s2",
                    "kind": "LOOKUP",
                    "entity": "invoice",
                    "measure_concept": "total invoices",
                    "grain": "entity",
                    "grain_entity": "vendor",
                    "depends_on": [],
                },
                {
                    "step_id": "s3",
                    "kind": "COMPOSE",
                    "entity": None,
                    "measure_concept": "overdue ratio",
                    "depends_on": ["s1", "s2"],
                    "compose_expr": {"op": "ratio", "left_step": "s1", "right_step": "s2"},
                },
            ]
        },
    )
    dag = _run(P.plan("ratio of overdue to total invoices per vendor"))
    assert dag is not None
    assert len(dag.steps) == 3
    lookups = [s for s in dag.steps if s.kind == StepKind.LOOKUP]
    composes = [s for s in dag.steps if s.kind == StepKind.COMPOSE]
    assert len(lookups) == 2
    assert len(composes) == 1
    compose = composes[0]
    assert set(compose.depends_on) == {"s1", "s2"}
    assert compose.compose_expr is not None
    assert compose.compose_expr.get("op") == "ratio"


def test_malformed_json_returns_none(monkeypatch):
    _mock_mini(monkeypatch, "{not valid json at all ]]")
    dag = _run(P.plan("total vendor spend"))
    assert dag is None


def test_cycle_in_depends_on_returns_none(monkeypatch):
    _mock_mini(
        monkeypatch,
        {
            "steps": [
                {"step_id": "s1", "kind": "COMPOSE", "depends_on": ["s2"],
                 "compose_expr": {"op": "ratio", "left_step": "s2", "right_step": "s2"}},
                {"step_id": "s2", "kind": "COMPOSE", "depends_on": ["s1"],
                 "compose_expr": {"op": "ratio", "left_step": "s1", "right_step": "s1"}},
            ]
        },
    )
    dag = _run(P.plan("a cyclic plan"))
    assert dag is None


def test_dangling_depends_on_returns_none(monkeypatch):
    _mock_mini(
        monkeypatch,
        {
            "steps": [
                {"step_id": "s1", "kind": "LOOKUP", "entity": "vendor", "depends_on": []},
                {"step_id": "s3", "kind": "COMPOSE", "depends_on": ["s1", "s2"],
                 "compose_expr": {"op": "ratio", "left_step": "s1", "right_step": "s2"}},
            ]
        },
    )
    dag = _run(P.plan("plan with a dangling reference"))
    assert dag is None


def test_empty_question_returns_none(monkeypatch):
    # no mock needed: must short-circuit before the LLM call
    dag = _run(P.plan("   "))
    assert dag is None


def test_no_steps_returns_none(monkeypatch):
    _mock_mini(monkeypatch, {"steps": []})
    dag = _run(P.plan("anything"))
    assert dag is None


def test_client_raises_returns_none(monkeypatch):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(P, "get_client", _boom)
    dag = _run(P.plan("total vendor spend"))
    assert dag is None


# --------------------------------------------------------------------------
# types.py — frozen-ness + StepLedger
# --------------------------------------------------------------------------
def test_intentstep_is_frozen():
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    with pytest.raises((AttributeError, Exception)):
        step.entity = "customer"  # type: ignore[misc]


def test_stepdag_is_frozen():
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    dag = StepDAG(question="q", steps=(step,))
    with pytest.raises((AttributeError, Exception)):
        dag.question = "other"  # type: ignore[misc]


def test_stepresult_is_frozen():
    res = StepResult(step_id="s1", scalar=42.0)
    with pytest.raises((AttributeError, Exception)):
        res.scalar = 99.0  # type: ignore[misc]


def test_step_collection_fields_are_tuples():
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    assert isinstance(step.depends_on, tuple)
    assert isinstance(step.join_entities, tuple)
    assert isinstance(step.filters, tuple)


def test_stepledger_is_mutable_and_get_scalar_works():
    ledger = StepLedger()
    assert ledger.get_scalar("missing") is None
    ledger.results["s1"] = StepResult(step_id="s1", scalar=12.5)
    assert ledger.get_scalar("s1") == 12.5
    # a result with no scalar yields None
    ledger.results["s2"] = StepResult(step_id="s2", scalar=None)
    assert ledger.get_scalar("s2") is None
    # clarify is mutable
    ledger.clarify = {"reason": "ambiguous"}
    assert ledger.clarify == {"reason": "ambiguous"}

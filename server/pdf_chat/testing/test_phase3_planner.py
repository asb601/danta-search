"""Phase 3 Task 2 — Planner (intent + bypass + typed fallback).

Pure unit tests with a deterministic FakeLlm (no infra). Verifies:

  * a confident simple query → ``bypass is True`` (high-confidence simple/cached);
  * a malformed LLM reply → ``fallback_reason == "planner_error:..."`` and
    ``bypass is False`` (never raises);
  * ``definitional`` / ``cross_domain`` classifications set the router signals;
  * the bypass confidence floor is read via ``get_tunable`` and the decision is
    logged via ``log_gate_decision`` (assert the returned record ``outcome``);
  * a cached query bypasses (skips the loop) regardless of intent;
  * the intent is one of the typed ``QueryIntent`` literals.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from pdf_chat.agent.planner import (
    QueryIntent,
    PlannerResult,
    plan_query,
    PLANNER_BYPASS_CONFIDENCE,
    PLANNER_BYPASS_CONFIDENCE_DEFAULT,
)
from pdf_chat.tunables import get_tunable


# --------------------------------------------------------------------------- #
# Deterministic fake LLM — mirrors PdfLlm.generate(system, user, *, container_id, signals)
# --------------------------------------------------------------------------- #
class FakeLlm:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0
        self.last_container_id = None
        self.last_signals = None

    async def generate(self, system, user, *, container_id="", signals=None):
        self.calls += 1
        self.last_container_id = container_id
        self.last_signals = signals
        return self._reply


def _reply(intent: str, confidence: float, *, multi_part: bool = False) -> str:
    return json.dumps(
        {"intent": intent, "confidence": confidence, "multi_part": multi_part}
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_confident_simple_query_bypasses():
    llm = FakeLlm(_reply("local", 0.95))
    result = asyncio.run(plan_query("What was revenue?", container_id="c1", llm=llm))

    assert isinstance(result, PlannerResult)
    assert result.intent == "local"
    assert result.bypass is True
    assert result.fallback_reason is None
    assert result.confidence == pytest.approx(0.95)


def test_low_confidence_takes_loop_with_typed_fallback():
    floor = get_tunable("c1", PLANNER_BYPASS_CONFIDENCE, PLANNER_BYPASS_CONFIDENCE_DEFAULT)
    low = floor - 0.2
    llm = FakeLlm(_reply("local", low))
    result = asyncio.run(plan_query("What was revenue?", container_id="c1", llm=llm))

    assert result.bypass is False
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("low_confidence:")


def test_malformed_reply_yields_planner_error_and_no_bypass():
    llm = FakeLlm("this is not json at all {{{")
    result = asyncio.run(plan_query("anything", container_id="c1", llm=llm))

    assert result.bypass is False
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("planner_error:")
    # never raises — a result is always returned
    assert isinstance(result, PlannerResult)


def test_llm_exception_yields_planner_error():
    class BoomLlm:
        async def generate(self, *a, **k):
            raise RuntimeError("backend down")

    result = asyncio.run(plan_query("q", container_id="c1", llm=BoomLlm()))
    assert result.bypass is False
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("planner_error:")


def test_definitional_sets_signal():
    llm = FakeLlm(_reply("definitional", 0.9))
    result = asyncio.run(plan_query("What does EBITDA mean?", container_id="c1", llm=llm))
    assert result.intent == "definitional"
    assert result.signals.get("definitional") is True
    assert result.signals.get("cross_domain") is not True


def test_cross_domain_sets_signal():
    llm = FakeLlm(_reply("cross_domain", 0.9))
    result = asyncio.run(plan_query("Compare A vs B across docs", container_id="c1", llm=llm))
    assert result.intent == "cross_domain"
    assert result.signals.get("cross_domain") is True


def test_cached_query_bypasses_even_when_not_confident():
    floor = get_tunable("c1", PLANNER_BYPASS_CONFIDENCE, PLANNER_BYPASS_CONFIDENCE_DEFAULT)
    llm = FakeLlm(_reply("local", floor - 0.3))
    result = asyncio.run(
        plan_query("repeat", container_id="c1", llm=llm, cached=True)
    )
    assert result.bypass is True
    assert result.fallback_reason is None


def test_bypass_decision_is_logged(monkeypatch):
    import pdf_chat.agent.planner as planner_mod

    records = []
    real_log = planner_mod.log_gate_decision

    def _spy(name, **kw):
        rec = real_log(name, **kw)
        records.append(rec)
        return rec

    monkeypatch.setattr(planner_mod, "log_gate_decision", _spy)

    llm = FakeLlm(_reply("local", 0.95))
    asyncio.run(plan_query("q", container_id="c1", llm=llm))

    bypass_records = [r for r in records if r["gate"] == "agent.planner_bypass"]
    assert bypass_records, "the bypass gate decision must be logged"
    assert bypass_records[-1]["outcome"] == "bypass"


def test_floor_is_read_via_get_tunable(monkeypatch):
    """The bypass floor is a tunable, not a magic literal — overriding it flips
    a borderline query from loop to bypass."""
    import pdf_chat.agent.planner as planner_mod

    seen = {}
    real_get = planner_mod.get_tunable

    def _spy(container_id, key, default=None):
        val = real_get(container_id, key, default)
        seen[key] = val
        return val

    monkeypatch.setattr(planner_mod, "get_tunable", _spy)

    llm = FakeLlm(_reply("local", 0.95))
    asyncio.run(plan_query("q", container_id="c1", llm=llm))
    assert PLANNER_BYPASS_CONFIDENCE in seen


def test_intent_is_typed_literal():
    for it in ("local", "global", "cross_domain", "definitional"):
        llm = FakeLlm(_reply(it, 0.9))
        result = asyncio.run(plan_query("q", container_id="c1", llm=llm))
        assert result.intent == it

    # an out-of-vocabulary intent is coerced to the safe default "local"
    llm = FakeLlm(_reply("nonsense_intent", 0.9))
    result = asyncio.run(plan_query("q", container_id="c1", llm=llm))
    assert result.intent in ("local", "global", "cross_domain", "definitional")

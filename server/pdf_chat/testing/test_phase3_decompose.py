"""Phase-3 — multi-part DECOMPOSITION (agent/decompose.py).

Pure unit tests, zero infra. Verifies the BLOCKER fix: a multi-part query is
split into its requested OUTPUT COMPONENTS (data-driven via the LLM model-router
seam, with a deterministic clause fallback), the split is cap-bounded via a
tunable + logged, and it never raises.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from pdf_chat.agent import decompose as decompose_mod
from pdf_chat.agent.decompose import (
    AGENT_DECOMP_MAX_COMPONENTS,
    decompose_query,
)


class FakeLlm:
    """Returns a canned component JSON array (mirrors PdfLlm.generate)."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0
        self.last_container_id = None

    async def generate(self, system, user, *, container_id="", signals=None):
        self.calls += 1
        self.last_container_id = container_id
        return self._reply


class BoomLlm:
    async def generate(self, system, user, *, container_id="", signals=None):
        raise RuntimeError("model backend down")


def _run(*a, **kw):
    return asyncio.run(decompose_query(*a, **kw))


# --------------------------------------------------------------------------- #
# LLM-driven split (no hardcoded component dictionary)
# --------------------------------------------------------------------------- #
def test_llm_split_returns_components():
    llm = FakeLlm(json.dumps(["revenue", "headcount", "market share"]))
    comps = _run(
        "How did revenue, headcount and market share change?",
        container_id="c1",
        llm=llm,
    )
    assert comps == ["revenue", "headcount", "market share"]
    assert llm.calls == 1


def test_llm_split_tolerates_code_fence_and_prose():
    llm = FakeLlm('Sure! ```json\n["a", "b"]\n``` done')
    comps = _run("a and b?", container_id="c1", llm=llm)
    assert comps == ["a", "b"]


def test_llm_split_dedupes_case_insensitively():
    llm = FakeLlm(json.dumps(["Revenue", "revenue", "Costs"]))
    comps = _run("q", container_id="c1", llm=llm)
    assert comps == ["Revenue", "Costs"]


# --------------------------------------------------------------------------- #
# Fallback split — deterministic clause boundaries (never raises)
# --------------------------------------------------------------------------- #
def test_malformed_llm_reply_degrades_to_clause_fallback():
    llm = FakeLlm("not json at all")
    comps = _run(
        "What is the revenue and what is the headcount?",
        container_id="c1",
        llm=llm,
    )
    # Falls back to splitting on " and " (clause boundary) — never raises.
    assert len(comps) >= 2
    assert any("revenue" in c.lower() for c in comps)
    assert any("headcount" in c.lower() for c in comps)


def test_backend_failure_degrades_silently():
    comps = _run(
        "compare A and B; also C", container_id="c1", llm=BoomLlm()
    )
    assert len(comps) >= 2  # split on ";" / " and "


def test_no_llm_uses_fallback():
    comps = _run("revenue and costs", container_id="c1", llm=None)
    assert len(comps) == 2


def test_single_part_query_returns_one_component():
    llm = FakeLlm(json.dumps(["revenue"]))
    comps = _run("What was the revenue?", container_id="c1", llm=llm)
    # <=1 component → caller treats it as effectively single-part.
    assert len(comps) <= 1


# --------------------------------------------------------------------------- #
# Cap via tunable + logged (no magic literal)
# --------------------------------------------------------------------------- #
def test_component_cap_via_tunable_and_logged(monkeypatch):
    logged = []
    monkeypatch.setattr(
        decompose_mod,
        "log_gate_decision",
        lambda name, **kw: logged.append({"gate": name, **kw}) or {"gate": name, **kw},
    )
    monkeypatch.setenv("PDF_TUNABLE_AGENT.DECOMP_MAX_COMPONENTS", "2")
    llm = FakeLlm(json.dumps(["a", "b", "c", "d"]))
    comps = _run("a, b, c and d?", container_id="c1", llm=llm)
    assert comps == ["a", "b"]  # capped at the tunable
    caps = [r for r in logged if r["gate"] == AGENT_DECOMP_MAX_COMPONENTS]
    assert caps, "decomposition cap was not logged"
    assert caps[-1]["outcome"] == "cap"
    assert caps[-1]["score"] >= caps[-1]["threshold"]


def test_decompose_threads_container_id_to_llm():
    llm = FakeLlm(json.dumps(["a", "b"]))
    _run("a and b", container_id="tenant-xyz", llm=llm)
    assert llm.last_container_id == "tenant-xyz"


def test_decompose_never_raises_on_garbage():
    # Even an empty query degrades to a (possibly empty) list without raising.
    comps = _run("", container_id="c1", llm=BoomLlm())
    assert isinstance(comps, list)

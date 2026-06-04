"""Pure tests for the gold-question eval set + harness (Spec §5 Phase 1)."""
from __future__ import annotations

import asyncio
import json

from pdf_chat.eval.gold_questions import GoldQuestion, load_gold_set
from pdf_chat.eval.harness import score_answer, run_eval


def test_load_gold_set_returns_gold_questions():
    gold = load_gold_set()
    assert len(gold) >= 3
    assert all(isinstance(g, GoldQuestion) for g in gold)
    g = gold[0]
    assert g.question
    assert g.expected_keywords          # at least one expected keyword
    assert g.must_cite is True or g.must_cite is False


def test_score_answer_keyword_and_citation():
    g = GoldQuestion(id="q", question="x", expected_keywords=["total", "value"],
                     must_cite=True, expect_refusal=False)
    s = score_answer(g, answer="The total value is 100.", citations=[{"n": 1}])
    assert s["keyword_recall"] == 1.0
    assert s["cited"] is True
    assert s["passed"] is True


def test_score_answer_refusal_expected():
    g = GoldQuestion(id="q", question="x", expected_keywords=[],
                     must_cite=False, expect_refusal=True)
    s = score_answer(g, answer="I don't have enough information.", citations=[])
    assert s["passed"] is True          # a refusal where refusal is expected passes


def test_run_eval_records_baseline(tmp_path):
    gold = [GoldQuestion(id="q1", question="x", expected_keywords=["a"],
                         must_cite=False, expect_refusal=False)]

    async def _answer(q):
        return ("answer a", [])         # contains the expected keyword

    out_path = tmp_path / "baseline.json"
    summary = asyncio.run(run_eval(gold, _answer, baseline_path=out_path))
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1.0
    recorded = json.loads(out_path.read_text())
    assert recorded["summary"]["pass_rate"] == 1.0
    assert len(recorded["results"]) == 1

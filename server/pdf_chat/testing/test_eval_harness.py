"""Phase 6 eval CI gate tests — gold-set expansion + threshold assertion gate.

Strict TDD: these tests pin the Phase-6 additions on top of the REAL eval module
(pdf_chat/eval/{gold_questions,harness,run_ci_eval} + gold_set.json):

* Task 7 — the gold set carries a ``category`` and covers all 5 categories
  (local + the four Phase-6: graph_traversal, global_community, cross_domain,
  negative_claim); negative-claim rows declare ``expect_refusal=True``.
* Task 8 — ``run_eval`` summary exposes correctness/faithfulness/fallback_rate,
  ``assert_thresholds`` passes above floors and raises (matching "correctness")
  below, and the ``run_ci_eval`` CLI imports cleanly.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest

from pdf_chat.eval.gold_questions import GoldQuestion, load_gold_set
from pdf_chat.eval.harness import run_eval, assert_thresholds, score_answer


# --------------------------------------------------------------------------- #
# Task 7 — gold set expansion (REAL shape: category field + JSON data rows)
# --------------------------------------------------------------------------- #
def test_gold_question_has_category_defaulting_to_local():
    g = GoldQuestion(id="x", question="q", expected_keywords=["a"])
    assert g.category == "local"


def test_gold_set_covers_all_five_categories():
    gold = load_gold_set()
    cats = {g.category for g in gold}
    assert cats >= {
        "local",
        "graph_traversal",
        "global_community",
        "cross_domain",
        "negative_claim",
    }


def test_negative_claim_rows_declare_expect_refusal():
    gold = load_gold_set()
    negatives = [g for g in gold if g.category == "negative_claim"]
    assert negatives, "expected at least one negative_claim gold row"
    assert all(g.expect_refusal is True for g in negatives)


# --------------------------------------------------------------------------- #
# Task 8 — run_eval summary metrics + assert_thresholds gate
# --------------------------------------------------------------------------- #
def test_run_eval_summary_exposes_ci_metrics():
    gold = [
        GoldQuestion(id="ok", question="x", expected_keywords=["a"],
                     must_cite=True, expect_refusal=False),
    ]

    async def _answer(q):
        return ("answer a", [{"n": 1}])

    summary = asyncio.run(run_eval(gold, _answer))
    assert summary["correctness"] == 1.0
    assert summary["faithfulness"] == 1.0
    assert summary["fallback_rate"] == 0.0


def test_run_eval_fallback_rate_counts_unexpected_refusals():
    gold = [
        GoldQuestion(id="a", question="x", expected_keywords=["a"],
                     must_cite=True, expect_refusal=False),
        GoldQuestion(id="b", question="y", expected_keywords=["b"],
                     must_cite=True, expect_refusal=False),
    ]

    async def _answer(q):
        # First answers, second refuses (unexpected → counts as a fallback).
        if q == "x":
            return ("answer a", [{"n": 1}])
        return ("I don't have enough information.", [])

    summary = asyncio.run(run_eval(gold, _answer))
    assert summary["fallback_rate"] == 0.5


def test_run_eval_fallback_rate_counts_empty_answers():
    """An empty / whitespace-only content answer (the honest-fallback when
    run_pdf_query is unavailable) MUST count toward fallback_rate, otherwise the
    rate understates breakage (Fix 5)."""
    gold = [
        GoldQuestion(id="a", question="x", expected_keywords=["a"],
                     must_cite=True, expect_refusal=False),
        GoldQuestion(id="b", question="y", expected_keywords=["b"],
                     must_cite=True, expect_refusal=False),
    ]

    async def _answer(q):
        # First answers, second returns an empty answer (runtime unavailable).
        if q == "x":
            return ("answer a", [{"n": 1}])
        return ("   ", [])  # whitespace-only ⇒ honest fallback

    summary = asyncio.run(run_eval(gold, _answer))
    assert summary["fallback_rate"] == 0.5


def test_assert_thresholds_passes_above_floors():
    summary = {"correctness": 0.9, "faithfulness": 0.9, "fallback_rate": 0.0}
    # No raise above floors.
    assert_thresholds(summary, container_id="c1")


def test_assert_thresholds_raises_below_correctness_floor():
    summary = {"correctness": 0.1, "faithfulness": 0.9, "fallback_rate": 0.0}
    with pytest.raises(AssertionError, match="correctness"):
        assert_thresholds(summary, container_id="c1")


def test_assert_thresholds_raises_below_faithfulness_floor():
    summary = {"correctness": 0.9, "faithfulness": 0.1, "fallback_rate": 0.0}
    with pytest.raises(AssertionError, match="faithfulness"):
        assert_thresholds(summary, container_id="c1")


def test_assert_thresholds_raises_above_fallback_ceiling():
    summary = {"correctness": 0.9, "faithfulness": 0.9, "fallback_rate": 0.9}
    with pytest.raises(AssertionError, match="fallback_rate"):
        assert_thresholds(summary, container_id="c1")


def test_run_ci_eval_imports_cleanly():
    mod = importlib.import_module("pdf_chat.eval.run_ci_eval")
    assert hasattr(mod, "main")


# --------------------------------------------------------------------------- #
# Fix 4 — eval refusal markers are a SUPERSET of the runtime's refusal vocabulary
# --------------------------------------------------------------------------- #
def test_score_answer_recognizes_runtime_honest_rewrite_as_refusal():
    """An honest in-corpus refusal emitted by pdf_honest_rewrite (a retrieval-miss
    rewrite) on an expect_refusal gold question must score refused=True — honesty
    must not be punished as a non-refusal."""
    from pdf_chat.agent.negative_claim import PdfNegativeVerdict, pdf_honest_rewrite

    g = GoldQuestion(id="neg", question="q", expect_refusal=True)

    # Real retrieval-miss rewrite ("I could not confirm... a retrieval miss...").
    miss = pdf_honest_rewrite(
        PdfNegativeVerdict(is_negative_claim=True, proven=False, coverage_complete=False)
    )
    assert score_answer(g, miss, [])["refused"] is True
    assert score_answer(g, miss, [])["passed"] is True

    # Real proven-absence rewrite ("...in context but I have not verified...").
    absence = pdf_honest_rewrite(
        PdfNegativeVerdict(is_negative_claim=True, proven=False, coverage_complete=True)
    )
    assert score_answer(g, absence, [])["refused"] is True
    assert score_answer(g, absence, [])["passed"] is True


def test_score_answer_recognizes_negative_phrase_refusal():
    """A negative-claim phrase from the runtime's own vocabulary (e.g. 'no mention',
    'the documents do not') scores as a refusal for an expect_refusal question."""
    g = GoldQuestion(id="neg", question="q", expect_refusal=True)
    assert score_answer(g, "There is no mention of that vendor.", [])["refused"] is True
    assert score_answer(g, "The documents do not state any such figure.", [])[
        "refused"
    ] is True

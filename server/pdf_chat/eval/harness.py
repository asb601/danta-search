"""Eval scoring + baseline recorder (Spec §5 Phase 1 — eval moved up).

Pure scoring over (gold, answer, citations). ``run_eval`` drives an injected
``answer_fn`` (in production: a bound ``run_pdf_chat`` over a fixed test corpus)
across the gold set, scores each, and writes a baseline JSON so regressions are
measurable. ``answer_fn`` is async returning ``(answer, citations)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from pdf_chat.agent.negative_claim import _NEGATIVE_PHRASES_DEFAULT
from pdf_chat.tunables import get_tunable, log_gate_decision

from .gold_questions import GoldQuestion

# Refusal markers (data, not a hardcoded business rule): a refusal is a non-answer
# whose text contains any of these stems. Overridable by passing ``refusal_markers``.
#
# The eval markers MUST be a SUPERSET of the vocabulary the runtime actually emits,
# otherwise an honest in-corpus refusal (pdf_honest_rewrite / proven-absence) on a
# negative_claim gold question would be scored as a NON-refusal and FAIL — punishing
# honesty. We therefore MERGE three sources:
#   1. the original generic markers (insufficient/not found/cannot answer/...);
#   2. the negative-claim gate's own phrase vocabulary
#      (negative_claim._NEGATIVE_PHRASES_DEFAULT — "no mention", "does not state",
#      "the documents do not", "not specified", ...);
#   3. the honest-rewrite + insufficient-context stems the runtime emits
#      (pdf_honest_rewrite text + prompts.INSUFFICIENT_CONTEXT_MESSAGE) —
#      "could not", "did not", "not verified", "re-scan", "retry", ...
_HONEST_REWRITE_STEMS = (
    "could not", "did not", "not proof", "not verified", "have not verified",
    "re-scan", "retry", "do not contain", "i don't have", "sufficient accessible",
)
_DEFAULT_REFUSAL_MARKERS = tuple(
    dict.fromkeys(  # de-dupe while preserving order
        (
            "don't have", "do not have", "insufficient", "not found",
            "cannot answer", "no relevant",
        )
        + _NEGATIVE_PHRASES_DEFAULT
        + _HONEST_REWRITE_STEMS
    )
)


def _is_refusal(answer: str, markers: tuple[str, ...]) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in markers)


def score_answer(
    gold: GoldQuestion,
    answer: str,
    citations: list[dict],
    *,
    refusal_markers: tuple[str, ...] = _DEFAULT_REFUSAL_MARKERS,
) -> dict:
    """Score one answer: keyword recall, citation presence, pass/fail."""
    refused = _is_refusal(answer, refusal_markers)
    # An empty / whitespace-only content answer is the honest-fallback the runtime
    # emits when run_pdf_query is unavailable — it is breakage, not a real answer.
    empty = not (answer or "").strip()
    if gold.expect_refusal:
        passed = refused
        return {"id": gold.id, "passed": passed, "refused": refused,
                "empty": empty, "keyword_recall": 0.0, "cited": bool(citations)}

    low = (answer or "").lower()
    kws = gold.expected_keywords or []
    hits = sum(1 for kw in kws if kw.lower() in low)
    recall = (hits / len(kws)) if kws else 1.0
    cited = bool(citations)
    passed = (recall == 1.0) and (cited if gold.must_cite else True) and not refused
    return {"id": gold.id, "passed": passed, "refused": refused, "empty": empty,
            "keyword_recall": recall, "cited": cited}


async def run_eval(
    gold: list[GoldQuestion],
    answer_fn: Callable[[str], Awaitable[tuple[str, list[dict]]]],
    *,
    baseline_path: "Path | None" = None,
) -> dict:
    """Run the gold set through ``answer_fn``, score, and record a baseline."""
    results = []
    for g in gold:
        answer, citations = await answer_fn(g.question)
        score = score_answer(g, answer, citations)
        # Carry the gold flags onto the record so the CI metrics can be derived
        # without re-reading the gold set (run_ci_eval prints these too).
        score["expect_refusal"] = bool(g.expect_refusal)
        score["must_cite"] = bool(g.must_cite)
        score["category"] = g.category
        results.append(score)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    summary = {"total": total, "passed": passed,
               "pass_rate": (passed / total) if total else 0.0}
    summary.update(_ci_metrics(results))
    if baseline_path is not None:
        Path(baseline_path).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
    return summary


def _ci_metrics(results: list[dict]) -> dict:
    """Derive the three CI gate metrics from scored ``results``.

    * ``correctness``   — keyword recall on content questions; for refusal-expected
      questions the pass/fail (``passed``) stands in (a correct refusal == 1.0).
    * ``faithfulness``  — citation grounding rate over content questions that must
      cite (a content answer that fails to cite is unfaithful).
    * ``fallback_rate`` — fraction of CONTENT questions that refused unexpectedly
      (a refusal where an answer was expected == an honest-absence fallback).
    """
    content = [r for r in results if not r.get("expect_refusal")]
    refusals = [r for r in results if r.get("expect_refusal")]

    # Correctness: mean recall on content + correct-refusal credit on negatives.
    corr_terms = [float(r.get("keyword_recall", 0.0)) for r in content]
    corr_terms += [1.0 if r.get("passed") else 0.0 for r in refusals]
    correctness = (sum(corr_terms) / len(corr_terms)) if corr_terms else 1.0

    # Faithfulness: of the content questions that must cite, how many cited.
    must_cite = [r for r in content if r.get("must_cite")]
    cited = sum(1 for r in must_cite if r.get("cited"))
    faithfulness = (cited / len(must_cite)) if must_cite else 1.0

    # Fallback rate: content questions that refused (unexpected absence claim) OR
    # returned an EMPTY answer (the honest-fallback when run_pdf_query is
    # unavailable). Counting empties means the rate never understates breakage.
    fallback_content = sum(
        1 for r in content if r.get("refused") or r.get("empty")
    )
    fallback_rate = (fallback_content / len(content)) if content else 0.0

    return {"correctness": correctness, "faithfulness": faithfulness,
            "fallback_rate": fallback_rate}


def assert_thresholds(
    summary: dict,
    *,
    container_id: str,
    min_correctness: "float | None" = None,
    min_faithfulness: "float | None" = None,
    max_fallback_rate: "float | None" = None,
) -> None:
    """Gate a ``run_eval`` summary against the per-container CI floors.

    Resolves each floor via ``get_tunable`` (registry single source — no inline
    numeric default), emits a ``log_gate_decision`` per metric, and RAISES
    ``AssertionError`` when any metric breaches its floor. This is the CI
    build-fail seam: ``run_ci_eval`` calls it and exits non-zero on a raise.

    An explicit ``min_*``/``max_*`` argument overrides the tunable (CI override),
    otherwise the per-container tunable wins. ``fallback_rate`` is a CEILING, so
    its gate compares the negated rate against the negated ceiling (passed ==
    ``-rate >= -ceiling`` == ``rate <= ceiling``).
    """
    if min_correctness is None:
        min_correctness = get_tunable(container_id, "eval.min_correctness")
    if min_faithfulness is None:
        min_faithfulness = get_tunable(container_id, "eval.min_faithfulness")
    if max_fallback_rate is None:
        max_fallback_rate = get_tunable(container_id, "eval.max_fallback_rate")

    correctness = float(summary.get("correctness", 0.0))
    faithfulness = float(summary.get("faithfulness", 0.0))
    fallback_rate = float(summary.get("fallback_rate", 0.0))

    corr_gate = log_gate_decision(
        "eval.correctness", score=correctness, threshold=float(min_correctness),
        outcome="pass" if correctness >= min_correctness else "fail",
        container_id=container_id,
    )
    faith_gate = log_gate_decision(
        "eval.faithfulness", score=faithfulness, threshold=float(min_faithfulness),
        outcome="pass" if faithfulness >= min_faithfulness else "fail",
        container_id=container_id,
    )
    # Ceiling gate: pass iff rate <= ceiling ⇔ -rate >= -ceiling.
    fb_gate = log_gate_decision(
        "eval.fallback_rate", score=-fallback_rate, threshold=-float(max_fallback_rate),
        outcome="pass" if fallback_rate <= max_fallback_rate else "fail",
        container_id=container_id, fallback_rate=fallback_rate,
        max_fallback_rate=float(max_fallback_rate),
    )

    assert corr_gate["passed"], (
        f"correctness {correctness:.3f} below floor {float(min_correctness):.3f}"
    )
    assert faith_gate["passed"], (
        f"faithfulness {faithfulness:.3f} below floor {float(min_faithfulness):.3f}"
    )
    assert fb_gate["passed"], (
        f"fallback_rate {fallback_rate:.3f} above ceiling {float(max_fallback_rate):.3f}"
    )

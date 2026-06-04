"""Eval scoring + baseline recorder (Spec §5 Phase 1 — eval moved up).

Pure scoring over (gold, answer, citations). ``run_eval`` drives an injected
``answer_fn`` (in production: a bound ``run_pdf_chat`` over a fixed test corpus)
across the gold set, scores each, and writes a baseline JSON so regressions are
measurable. ``answer_fn`` is async returning ``(answer, citations)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from .gold_questions import GoldQuestion

# Refusal markers (data, not a hardcoded business rule): a refusal is a non-answer
# whose text contains any of these stems. Overridable by passing ``refusal_markers``.
_DEFAULT_REFUSAL_MARKERS = ("don't have", "do not have", "insufficient", "not found",
                            "cannot answer", "no relevant")


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
    if gold.expect_refusal:
        passed = refused
        return {"id": gold.id, "passed": passed, "refused": refused,
                "keyword_recall": 0.0, "cited": bool(citations)}

    low = (answer or "").lower()
    kws = gold.expected_keywords or []
    hits = sum(1 for kw in kws if kw.lower() in low)
    recall = (hits / len(kws)) if kws else 1.0
    cited = bool(citations)
    passed = (recall == 1.0) and (cited if gold.must_cite else True) and not refused
    return {"id": gold.id, "passed": passed, "refused": refused,
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
        results.append(score_answer(g, answer, citations))
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    summary = {"total": total, "passed": passed,
               "pass_rate": (passed / total) if total else 0.0}
    if baseline_path is not None:
        Path(baseline_path).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
    return summary

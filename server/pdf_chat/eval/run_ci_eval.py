"""CI eval runner — score the gold set via the Phase-3 runtime and gate on floors.

This is the build-fail seam: it loads the gold set, scores every question by
driving ``run_pdf_query`` (the Phase-3 public entry), prints a report, then calls
``assert_thresholds`` — which RAISES (exit non-zero) when correctness/
faithfulness/fallback_rate breaches its per-container floor.

Honesty rule: if ``run_pdf_query`` cannot be imported/executed for a question,
that question is scored as an unanswered FALLBACK (empty answer, no citations)
rather than being skipped — the report can never be silently green when the
runtime is broken.

Usage:
    python -m pdf_chat.eval.run_ci_eval --tenant-id T --container-id C
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Awaitable, Callable

from .gold_questions import GoldQuestion, load_gold_set
from .harness import assert_thresholds, run_eval


def _build_answer_fn(tenant_id: str, container_id: str) -> Callable[[str], Awaitable[tuple]]:
    """Bind ``run_pdf_query`` into the ``answer_fn`` shape ``run_eval`` expects.

    A per-question import/exec failure degrades to an empty answer + no citations
    (an honest fallback) so the gate sees the breakage instead of skipping it.
    """
    async def _answer(question: str) -> tuple:
        try:
            from pdf_chat.agent.graph import run_pdf_query
        except Exception:
            # Runtime unavailable: every question is an unanswered fallback.
            return ("", [])
        try:
            result = await run_pdf_query(
                question, tenant_id=tenant_id, container_id=container_id,
            )
            return (result.answer or "", list(result.citations or []))
        except Exception:
            return ("", [])

    return _answer


async def _run(tenant_id: str, container_id: str) -> dict:
    gold: list[GoldQuestion] = load_gold_set()
    answer_fn = _build_answer_fn(tenant_id, container_id)
    summary = await run_eval(gold, answer_fn)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pdf_chat CI eval gate.")
    parser.add_argument("--tenant-id", required=True, help="Tenant for isolation.")
    parser.add_argument(
        "--container-id", required=True,
        help="Container scoping every tunable + gate-decision log.",
    )
    args = parser.parse_args(argv)

    summary = asyncio.run(_run(args.tenant_id, args.container_id))
    print(json.dumps(summary, indent=2))

    try:
        assert_thresholds(summary, container_id=args.container_id)
    except AssertionError as exc:
        print(f"CI EVAL GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print("CI EVAL GATE PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

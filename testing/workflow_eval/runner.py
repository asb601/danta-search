"""
Workflow Evaluation Runner — main entry point

PURPOSE:
  Tie all eval framework components together into a single run_benchmark()
  function that produces a full BenchmarkResult from a WorkflowEvalCapture.

TWO RUN MODES:
  1. Live (instrumented) mode:
     The pipeline populates a WorkflowEvalCapture during execution.
     run_benchmark(benchmark, capture) evaluates it immediately.

  2. Offline simulation mode:
     run_offline_simulation(benchmark, catalog_sample) generates a
     synthetic capture from a catalog sample and scores it.
     Useful for pre-deployment regression testing against new catalogs.

MAIN ENTRY POINTS:
  run_benchmark(benchmark, capture)        → BenchmarkResult
  run_all_benchmarks(captures)             → list[BenchmarkResult]
  run_offline_simulation(benchmark, catalog) → BenchmarkResult

COMPLETE BENCHMARK RESULT:
  BenchmarkResult contains:
    - score:      WorkflowCompletenessScore
    - world:      PlannerWorldState
    - failure:    FailureClassification
    - expansion:  ExpansionEffectivenessReport
    - topology:   TopologyEffectivenessReport
    - safety:     BusinessTruthReport
    - benchmark:  the WorkflowBenchmark definition

EXAMPLE USAGE:

  # Live mode — run from inside your test harness after a real query
  from testing.workflow_eval import run_benchmark, load_benchmarks
  from testing.workflow_eval import WorkflowEvalCapture

  capture = WorkflowEvalCapture(
      query="show me overdue invoices",
      shortlist=shortlist,
      full_catalog=full_catalog,
      workflow_reqs=workflow_reqs,
      wf_topology=wf_topology,
      expansion=expansion,
      sql_ctx=sql_ctx,
      exec_strategy=exec_strategy,
      grounding_quality=grounding_quality,
      pre_expansion_shortlist=pre_expansion_shortlist,
  )
  benchmark = get_benchmark("WF-005")  # overdue_invoices
  result = run_benchmark(benchmark, capture)
  print(result.score.summary_line())
  print(result.failure.summary_line())
  print(result.safety.summary_line())

  # Offline mode
  results = run_offline_simulation(benchmark, catalog_sample)

  # Run all benchmarks against a single capture
  all_results = run_all_benchmarks_against_capture(capture)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from testing.workflow_eval.benchmarks import (
    WorkflowBenchmark,
    load_benchmarks,
    get_benchmark,
    BENCHMARK_REGISTRY,
)
from testing.workflow_eval.completeness_scorer import (
    WorkflowCompletenessScore,
    score_workflow_completeness,
)
from testing.workflow_eval.world_state_inspector import (
    PlannerWorldState,
    WorkflowEvalCapture,
    inspect_world_state,
)
from testing.workflow_eval.failure_classifier import (
    FailureClassification,
    classify_failure,
)
from testing.workflow_eval.expansion_validator import (
    ExpansionEffectivenessReport,
    validate_expansion_effectiveness,
)
from testing.workflow_eval.topology_validator import (
    TopologyEffectivenessReport,
    validate_topology_effectiveness,
)
from testing.workflow_eval.safety_diagnostics import (
    BusinessTruthReport,
    run_safety_diagnostics,
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Complete evaluation result for one benchmark against one capture.

    Contains all evaluation dimensions:
      - completeness: what fraction of workflow domains are visible
      - world state:  what the planner sees vs. what's hidden
      - failure:      what failure type (if any) occurred
      - expansion:    how much adaptive expansion helped
      - topology:     whether topology hints changed the outcome
      - safety:       whether answer would have been business-safe
    """
    benchmark: WorkflowBenchmark
    capture_query: str
    score: WorkflowCompletenessScore
    world: PlannerWorldState
    failure: FailureClassification
    expansion: ExpansionEffectivenessReport
    topology: TopologyEffectivenessReport
    safety: BusinessTruthReport
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def passed(self) -> bool:
        return self.score.passed and self.safety.is_safe_to_answer

    def print_summary(self) -> None:
        """Print a compact multi-line summary to stdout."""
        lines = [
            f"\n{'='*70}",
            f"  Benchmark: {self.benchmark.id} — {self.benchmark.workflow_category}",
            f"  Query:     {self.capture_query[:80]}",
            f"{'='*70}",
            f"  {self.score.summary_line()}",
            f"  {self.failure.summary_line()}",
            f"  {self.expansion.summary_line()}",
            f"  {self.topology.summary_line()}",
            f"  {self.safety.summary_line()}",
            f"  Visibility: {self.world.visibility_ratio:.0%} "
            f"({self.world.shortlist_file_count} shown, "
            f"{self.world.hidden_count} hidden)",
            f"{'='*70}",
        ]
        print("\n".join(lines))

    def to_dict(self) -> dict:
        """Serialize to a flat dict suitable for JSON export."""
        return {
            "benchmark_id": self.benchmark.id,
            "workflow_category": self.benchmark.workflow_category,
            "capture_query": self.capture_query,
            "evaluated_at": self.evaluated_at,
            "passed": self.passed(),
            # Completeness
            "workflow_completeness": self.score.workflow_completeness,
            "planner_world_completeness": self.score.planner_world_completeness,
            "required_satisfied": self.score.satisfied_required_count,
            "required_total": self.score.required_domain_count,
            "missing_required": [r.short_name for r in self.score.missing_required],
            "grounding_quality": self.score.grounding_quality,
            "failure_flags": self.score.failure_flags,
            # Failure
            "primary_failure": self.failure.primary_failure.value,
            "secondary_failures": [f.value for f in self.failure.secondary_failures],
            "failure_severity": self.failure.severity,
            "root_cause": self.failure.root_cause,
            "recovery_suggestion": self.failure.recovery_suggestion,
            # World state
            "shortlist_count": self.world.shortlist_file_count,
            "hidden_count": self.world.hidden_count,
            "visibility_ratio": self.world.visibility_ratio,
            "execution_mode": self.world.execution_mode,
            "approved_joins": self.world.approved_join_count,
            "topology_paths": len(self.world.reachable_topology_paths),
            "orphaned_tables": len(self.world.orphaned_tables),
            # Expansion
            "expansion_triggered": self.expansion.was_triggered,
            "expansion_verdict": self.expansion.expansion_verdict,
            "expansion_completeness_delta": self.expansion.completeness_delta,
            "expansion_files_added": self.expansion.files_added,
            "false_expansion_rate": self.expansion.false_expansion_rate,
            # Topology
            "topology_verdict": self.topology.topology_verdict,
            "topology_prevented_degradation": self.topology.topology_prevented_degradation,
            "bridge_paths_found": self.topology.bridge_paths_found,
            "potential_joins_surfaced": self.topology.potential_joins_surfaced,
            # Safety
            "safety_score": self.safety.safety_score,
            "safety_is_safe": self.safety.is_safe_to_answer,
            "safety_violations": self.safety.violation_count,
            "critical_safety_violations": self.safety.critical_violation_count,
        }


# ── Core run function ─────────────────────────────────────────────────────────

def run_benchmark(
    benchmark: WorkflowBenchmark,
    capture: WorkflowEvalCapture,
) -> BenchmarkResult:
    """
    Run a single benchmark evaluation against a captured pipeline state.

    This is the main evaluation function. It orchestrates all sub-evaluators
    and returns a complete BenchmarkResult.

    Args:
        benchmark: the WorkflowBenchmark to evaluate against
        capture:   the captured pipeline state (from a real or simulated run)

    Returns:
        BenchmarkResult with completeness, world state, failure, expansion,
        topology, and safety assessments
    """
    # 1. Workflow completeness
    score = score_workflow_completeness(
        benchmark=benchmark,
        shortlist=capture.shortlist,
        topology=capture.wf_topology,
        expansion=capture.expansion,
        pre_expansion_shortlist=capture.pre_expansion_shortlist,
        grounding_quality=capture.grounding_quality,
    )

    # 2. Planner world state
    world = inspect_world_state(benchmark=benchmark, capture=capture)

    # 3. Failure classification
    failure = classify_failure(score=score, world=world, benchmark=benchmark)

    # 4. Expansion effectiveness
    expansion = validate_expansion_effectiveness(benchmark=benchmark, capture=capture)

    # 5. Topology effectiveness
    topology = validate_topology_effectiveness(benchmark=benchmark, capture=capture)

    # 6. Business-truth safety
    safety = run_safety_diagnostics(benchmark=benchmark, score=score, world=world)

    return BenchmarkResult(
        benchmark=benchmark,
        capture_query=capture.query,
        score=score,
        world=world,
        failure=failure,
        expansion=expansion,
        topology=topology,
        safety=safety,
    )


# ── Batch runners ─────────────────────────────────────────────────────────────

def run_all_benchmarks_against_capture(
    capture: WorkflowEvalCapture,
    categories: list[str] | None = None,
) -> list[BenchmarkResult]:
    """
    Run all benchmarks (or a filtered subset) against a single pipeline capture.

    Useful for understanding how one query performs across all expected
    workflow scenarios (e.g. does the overdue invoice query trigger correctly
    for payment_aging benchmark too?).

    Args:
        capture:    captured pipeline state
        categories: optional category filter

    Returns:
        list of BenchmarkResult
    """
    benchmarks = load_benchmarks(categories=categories)
    return [run_benchmark(b, capture) for b in benchmarks]


def run_all_benchmarks(
    captures: list[WorkflowEvalCapture],
    match_by_query: bool = False,
) -> list[BenchmarkResult]:
    """
    Run the full benchmark suite against a list of captures.

    Strategy: each benchmark is matched to the capture whose query is most
    semantically similar. If match_by_query is False (default), each capture
    is run against all benchmarks.

    Args:
        captures:        list of WorkflowEvalCapture objects
        match_by_query:  if True, try to match capture to benchmark by query tokens

    Returns:
        list of BenchmarkResult (may be len(benchmarks) * len(captures))
    """
    results: list[BenchmarkResult] = []

    if match_by_query:
        for benchmark in BENCHMARK_REGISTRY:
            best_capture = _find_best_matching_capture(benchmark, captures)
            if best_capture:
                results.append(run_benchmark(benchmark, best_capture))
    else:
        for capture in captures:
            for benchmark in BENCHMARK_REGISTRY:
                results.append(run_benchmark(benchmark, capture))

    return results


def _find_best_matching_capture(
    benchmark: WorkflowBenchmark,
    captures: list[WorkflowEvalCapture],
) -> WorkflowEvalCapture | None:
    """
    Find the capture whose query best matches a benchmark's expected query.
    Simple token overlap — no LLM calls.
    """
    if not captures:
        return None

    benchmark_tokens = set(benchmark.workflow_query.lower().split())
    best: WorkflowEvalCapture | None = None
    best_score = -1

    for capture in captures:
        query_tokens = set(capture.query.lower().split())
        overlap = len(benchmark_tokens & query_tokens)
        if overlap > best_score:
            best_score = overlap
            best = capture

    return best


# ── Offline simulation ────────────────────────────────────────────────────────

def run_offline_simulation(
    benchmark: WorkflowBenchmark,
    catalog_sample: list[dict],
    grounding_quality: str = "role_cluster",
    execution_mode: str = "multi_cluster",
) -> BenchmarkResult:
    """
    Run a benchmark evaluation with a synthetic capture built from a catalog sample.

    This allows offline evaluation without running the full pipeline.
    The entire catalog_sample is treated as both the shortlist AND the full catalog
    (i.e. assumes perfect retrieval — tests catalog completeness rather than retrieval).

    Useful for:
    - Checking whether the catalog has enough semantic roles for a workflow
    - Pre-deployment validation of a new dataset's semantic role coverage
    - CI/CD regression tests that don't need a live server

    Args:
        benchmark:        the benchmark to evaluate
        catalog_sample:   list of catalog entries with column_semantic_roles
        grounding_quality: simulated retrieval quality
        execution_mode:   simulated execution mode string

    Returns:
        BenchmarkResult (expansion and topology will be neutral/absent)
    """
    # Synthetic execution strategy that reports the given mode
    class _SyntheticExecStrategy:
        mode = execution_mode
        def __repr__(self):
            return f"SyntheticExecStrategy(mode={self.mode!r})"

    capture = WorkflowEvalCapture(
        query=benchmark.workflow_query,
        shortlist=catalog_sample,
        full_catalog=catalog_sample,
        entity_resolution={},
        workflow_reqs=None,
        wf_topology=None,
        expansion=None,
        sql_ctx=None,
        exec_strategy=_SyntheticExecStrategy(),
        grounding_quality=grounding_quality,
        pre_expansion_shortlist=catalog_sample,
    )

    return run_benchmark(benchmark, capture)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_suite_summary(results: list[BenchmarkResult]) -> None:
    """Print a suite-level summary table."""
    print("\n" + "=" * 80)
    print(f"  WORKFLOW EVALUATION SUITE SUMMARY ({len(results)} benchmarks)")
    print("=" * 80)
    print(
        f"  {'ID':<10} {'Category':<28} {'Complete':>9} "
        f"{'Safety':>7} {'Failure':<22} {'Pass':<6}"
    )
    print("  " + "-" * 76)

    passed = 0
    for r in results:
        p = "PASS" if r.passed() else "FAIL"
        if r.passed():
            passed += 1
        print(
            f"  {r.benchmark.id:<10} {r.benchmark.workflow_category:<28} "
            f"{r.score.workflow_completeness:>8.0%} "
            f"{r.safety.safety_score:>7.0%} "
            f"{r.failure.primary_failure.value:<22} "
            f"{p:<6}"
        )

    print("  " + "-" * 76)
    rate = passed / len(results) if results else 0.0
    print(f"  Overall pass rate: {passed}/{len(results)} ({rate:.0%})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example: run offline simulation with an empty catalog
    # This will show all benchmarks failing (no semantic roles found)
    # which is the expected baseline for a cold-start system.
    print("Running offline simulation with empty catalog (cold-start baseline)...")
    empty_results = [
        run_offline_simulation(b, catalog_sample=[], grounding_quality="keyword_degraded")
        for b in BENCHMARK_REGISTRY
    ]
    _print_suite_summary(empty_results)

    # Demonstrate what a populated catalog result looks like
    print("\nExample: WF-001 with mock catalog entry that has invoice roles:")
    mock_catalog = [
        {
            "blob_name": "invoices_2024.parquet",
            "column_semantic_roles": {
                "INVOICE_ID": "custom:reference_key:invoice_line",
                "VENDOR_ID": "custom:entity_key:vendor_master",
                "AMOUNT": "custom:additive_measure:invoice_amount",
            },
        },
        {
            "blob_name": "purchase_orders.parquet",
            "column_semantic_roles": {
                "PO_NUMBER": "custom:reference_key:purchase_order_header",
                "VENDOR_CODE": "custom:entity_key:vendor_master",
                "ORDERED_QTY": "custom:additive_measure:ordered_quantity",
            },
        },
        {
            "blob_name": "vendor_master.parquet",
            "column_semantic_roles": {
                "VENDOR_KEY": "custom:entity_key:vendor_master",
                "VENDOR_NAME": "custom:attribute:vendor_name",
            },
        },
    ]
    wf001 = get_benchmark("WF-001")
    result = run_offline_simulation(wf001, catalog_sample=mock_catalog)
    result.print_summary()

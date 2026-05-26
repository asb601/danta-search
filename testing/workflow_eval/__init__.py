"""
Workflow Eval Framework — Package Init

Exports the public surface of the evaluation framework.

Entry points:
    from testing.workflow_eval import run_benchmark, load_benchmarks
    from testing.workflow_eval import WorkflowBenchmark, WorkflowCompletenessScore
    from testing.workflow_eval import PlannerWorldState, FailureClassification
    from testing.workflow_eval import RegressionRunner

Designed for two use patterns:
  1. Offline analysis: pass pre-captured WorkflowEvalCapture objects.
  2. Live regression: run against the server pipeline with eval hooks enabled.
"""
from testing.workflow_eval.benchmarks import (
    WorkflowBenchmark,
    DomainRequirement,
    TopologyRequirement,
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
    FailureType,
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
    SafetyViolation,
    BusinessTruthReport,
    run_safety_diagnostics,
)
from testing.workflow_eval.regression_runner import (
    RegressionBaseline,
    RegressionDelta,
    RegressionRunner,
)
from testing.workflow_eval.runner import (
    BenchmarkResult,
    run_benchmark,
    run_all_benchmarks,
)

__all__ = [
    # Benchmark definitions
    "WorkflowBenchmark", "DomainRequirement", "TopologyRequirement",
    "load_benchmarks", "get_benchmark", "BENCHMARK_REGISTRY",
    # Completeness scoring
    "WorkflowCompletenessScore", "score_workflow_completeness",
    # World-state inspection
    "PlannerWorldState", "WorkflowEvalCapture", "inspect_world_state",
    # Failure classification
    "FailureType", "FailureClassification", "classify_failure",
    # Expansion validation
    "ExpansionEffectivenessReport", "validate_expansion_effectiveness",
    # Topology validation
    "TopologyEffectivenessReport", "validate_topology_effectiveness",
    # Safety diagnostics
    "SafetyViolation", "BusinessTruthReport", "run_safety_diagnostics",
    # Regression
    "RegressionBaseline", "RegressionDelta", "RegressionRunner",
    # Runner
    "BenchmarkResult", "run_benchmark", "run_all_benchmarks",
]

"""
Regression Runner

PURPOSE:
  Persist benchmark evaluation results as a JSON baseline, then detect
  regressions and improvements by comparing current results to that baseline.

WHAT IT DOES:
  1. save_baseline(results, path):   serialize results to JSON
  2. load_baseline(path):            deserialize a previously-saved baseline
  3. compare_to_baseline(current, baseline): compute per-benchmark deltas
  4. generate_regression_report(deltas): produce a Markdown summary

REGRESSION DETECTION:
  A benchmark is considered REGRESSED if any of:
    - workflow_completeness decreased by more than REGRESSION_TOLERANCE
    - primary_failure_type changed to a more severe category
    - safety score decreased by more than SAFETY_TOLERANCE
    - a new critical safety violation appeared

  A benchmark is IMPROVED if any of:
    - workflow_completeness increased by more than IMPROVEMENT_THRESHOLD
    - primary_failure_type changed to a less severe / none category
    - safety score improved

BASELINE STORAGE:
  Baselines are stored in testing/workflow_eval/baselines/ as JSON files
  named: baseline_{YYYYMMDD_HHMMSS}.json (latest symlink: baseline_latest.json)

USAGE:
  from testing.workflow_eval.regression_runner import RegressionRunner
  runner = RegressionRunner()
  runner.save_baseline(results)
  # ... later, after code changes ...
  deltas = runner.compare_to_baseline(new_results)
  print(runner.generate_regression_report(deltas))
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testing.workflow_eval.completeness_scorer import WorkflowCompletenessScore
from testing.workflow_eval.failure_classifier import FailureClassification, FailureType
from testing.workflow_eval.safety_diagnostics import BusinessTruthReport

# ── Constants ─────────────────────────────────────────────────────────────────

REGRESSION_TOLERANCE = 0.05      # completeness drop < 5% is tolerated
SAFETY_TOLERANCE = 0.10          # safety drop < 10% is tolerated
IMPROVEMENT_THRESHOLD = 0.05     # completeness gain > 5% counts as improvement

_SEVERITY_RANK: dict[str, int] = {
    "none": 0, "minor": 1, "major": 2, "critical": 3
}

_BASELINES_DIR = Path(__file__).parent / "baselines"


# ── Result snapshot (what gets persisted) ─────────────────────────────────────

@dataclass
class BenchmarkSnapshot:
    """
    Serializable snapshot of one benchmark result.
    Stores only the metrics needed for regression detection —
    not the full object graph.
    """
    benchmark_id: str
    workflow_category: str
    workflow_completeness: float
    planner_world_completeness: float
    required_satisfied: int
    required_total: int
    primary_failure: str
    secondary_failures: list[str]
    failure_severity: str
    safety_score: float
    safety_is_safe: bool
    critical_safety_violations: int
    grounding_quality: str
    execution_mode: str
    expansion_triggered: bool
    expansion_verdict: str
    topology_verdict: str
    passed: bool
    timestamp: str


@dataclass
class RegressionBaseline:
    """A complete baseline — all benchmark snapshots at a point in time."""
    created_at: str
    label: str
    snapshots: list[BenchmarkSnapshot]

    @property
    def snapshot_map(self) -> dict[str, BenchmarkSnapshot]:
        return {s.benchmark_id: s for s in self.snapshots}


# ── Delta ─────────────────────────────────────────────────────────────────────

@dataclass
class RegressionDelta:
    """
    Per-benchmark delta between current result and baseline.

    Attributes:
        benchmark_id:            identifier
        workflow_category:       category
        completeness_delta:      current - baseline completeness
        safety_delta:            current - baseline safety score
        failure_changed:         True if primary_failure changed
        failure_from:            baseline failure type
        failure_to:              current failure type
        severity_changed:        True if failure severity changed
        severity_from:           baseline severity
        severity_to:             current severity
        regressed:               True if this is a regression
        improved:                True if this is an improvement
        regression_reasons:      why it regressed
        improvement_reasons:     why it improved
        is_new:                  True if benchmark not in baseline (new addition)
        is_missing:              True if benchmark in baseline but not in current
    """
    benchmark_id: str
    workflow_category: str
    completeness_delta: float
    safety_delta: float
    failure_changed: bool
    failure_from: str
    failure_to: str
    severity_changed: bool
    severity_from: str
    severity_to: str
    regressed: bool
    improved: bool
    regression_reasons: list[str]
    improvement_reasons: list[str]
    is_new: bool = False
    is_missing: bool = False


# ── Runner ────────────────────────────────────────────────────────────────────

class RegressionRunner:
    """
    Manages baseline persistence and regression comparison.

    Usage:
        runner = RegressionRunner()
        runner.save_baseline(results, label="pre-expansion-redesign")
        deltas = runner.compare_to_baseline(new_results)
        print(runner.generate_regression_report(deltas))
    """

    def __init__(self, baselines_dir: str | Path | None = None):
        self.baselines_dir = Path(baselines_dir) if baselines_dir else _BASELINES_DIR
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    # ── Serialization helpers ─────────────────────────────────────────────────

    @staticmethod
    def _snapshot_from_result(result: "BenchmarkResult") -> BenchmarkSnapshot:
        """Convert a BenchmarkResult to a serializable BenchmarkSnapshot."""
        from testing.workflow_eval.runner import BenchmarkResult
        return BenchmarkSnapshot(
            benchmark_id=result.benchmark.id,
            workflow_category=result.benchmark.workflow_category,
            workflow_completeness=result.score.workflow_completeness,
            planner_world_completeness=result.score.planner_world_completeness,
            required_satisfied=result.score.satisfied_required_count,
            required_total=result.score.required_domain_count,
            primary_failure=result.failure.primary_failure.value,
            secondary_failures=[f.value for f in result.failure.secondary_failures],
            failure_severity=result.failure.severity,
            safety_score=result.safety.safety_score,
            safety_is_safe=result.safety.is_safe_to_answer,
            critical_safety_violations=result.safety.critical_violation_count,
            grounding_quality=result.score.grounding_quality,
            execution_mode=result.world.execution_mode,
            expansion_triggered=result.expansion.was_triggered,
            expansion_verdict=result.expansion.expansion_verdict,
            topology_verdict=result.topology.topology_verdict,
            passed=result.score.passed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── Save / Load baseline ──────────────────────────────────────────────────

    def save_baseline(
        self,
        results: list["BenchmarkResult"],
        label: str = "",
    ) -> Path:
        """
        Serialize benchmark results as a JSON baseline.

        Args:
            results: list of BenchmarkResult objects
            label:   optional human label for this baseline

        Returns:
            Path to the saved baseline file
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_{ts}.json"
        path = self.baselines_dir / filename

        snapshots = [self._snapshot_from_result(r) for r in results]
        baseline = RegressionBaseline(
            created_at=datetime.now(timezone.utc).isoformat(),
            label=label or ts,
            snapshots=snapshots,
        )

        with open(path, "w") as f:
            json.dump(asdict(baseline), f, indent=2)

        # Update latest symlink
        latest = self.baselines_dir / "baseline_latest.json"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        # Write a "latest" file as a copy (symlinks may not work on all systems)
        with open(latest, "w") as f:
            json.dump(asdict(baseline), f, indent=2)

        return path

    def load_baseline(self, path: str | Path | None = None) -> RegressionBaseline:
        """
        Load a baseline from JSON.
        If path is None, loads the latest baseline.

        Args:
            path: path to a specific baseline file, or None for latest

        Returns:
            RegressionBaseline

        Raises:
            FileNotFoundError: if no baseline exists
        """
        if path is None:
            path = self.baselines_dir / "baseline_latest.json"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No baseline found at {path}. "
                "Run save_baseline() first."
            )

        with open(path) as f:
            data = json.load(f)

        snapshots = [BenchmarkSnapshot(**s) for s in data["snapshots"]]
        return RegressionBaseline(
            created_at=data["created_at"],
            label=data["label"],
            snapshots=snapshots,
        )

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare_to_baseline(
        self,
        current_results: list["BenchmarkResult"],
        baseline: RegressionBaseline | None = None,
    ) -> list[RegressionDelta]:
        """
        Compare current results to baseline, returning per-benchmark deltas.

        Args:
            current_results: list of BenchmarkResult from current run
            baseline:        baseline to compare against (loads latest if None)

        Returns:
            list of RegressionDelta, one per benchmark
        """
        if baseline is None:
            baseline = self.load_baseline()

        baseline_map = baseline.snapshot_map
        current_map = {r.benchmark.id: r for r in current_results}

        deltas: list[RegressionDelta] = []

        # Check all current results
        for bid, result in current_map.items():
            if bid not in baseline_map:
                deltas.append(RegressionDelta(
                    benchmark_id=bid,
                    workflow_category=result.benchmark.workflow_category,
                    completeness_delta=0.0,
                    safety_delta=0.0,
                    failure_changed=False,
                    failure_from="none",
                    failure_to=result.failure.primary_failure.value,
                    severity_changed=False,
                    severity_from="none",
                    severity_to=result.failure.severity,
                    regressed=False,
                    improved=False,
                    regression_reasons=[],
                    improvement_reasons=["new benchmark added"],
                    is_new=True,
                ))
                continue

            snap = baseline_map[bid]
            current_snap = self._snapshot_from_result(result)
            delta = _compute_delta(snap, current_snap)
            deltas.append(delta)

        # Check for benchmarks in baseline but not in current
        for bid in baseline_map:
            if bid not in current_map:
                snap = baseline_map[bid]
                deltas.append(RegressionDelta(
                    benchmark_id=bid,
                    workflow_category=snap.workflow_category,
                    completeness_delta=0.0,
                    safety_delta=0.0,
                    failure_changed=False,
                    failure_from=snap.primary_failure,
                    failure_to="unknown",
                    severity_changed=False,
                    severity_from=snap.failure_severity,
                    severity_to="unknown",
                    regressed=False,
                    improved=False,
                    regression_reasons=[],
                    improvement_reasons=[],
                    is_missing=True,
                ))

        return sorted(deltas, key=lambda d: d.benchmark_id)

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_regression_report(
        self,
        deltas: list[RegressionDelta],
        include_improvements: bool = True,
    ) -> str:
        """
        Generate a Markdown regression report from a list of deltas.

        Args:
            deltas:               list from compare_to_baseline()
            include_improvements: whether to include improvement details

        Returns:
            Markdown string
        """
        regressions = [d for d in deltas if d.regressed]
        improvements = [d for d in deltas if d.improved and not d.regressed]
        unchanged = [d for d in deltas if not d.regressed and not d.improved
                     and not d.is_new and not d.is_missing]
        new_benchmarks = [d for d in deltas if d.is_new]
        missing = [d for d in deltas if d.is_missing]

        lines: list[str] = [
            "# Workflow Evaluation Regression Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Summary",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| 🔴 Regressions | {len(regressions)} |",
            f"| 🟢 Improvements | {len(improvements)} |",
            f"| ⚪ Unchanged | {len(unchanged)} |",
            f"| 🆕 New | {len(new_benchmarks)} |",
            f"| ❓ Missing | {len(missing)} |",
            f"| Total | {len(deltas)} |",
            "",
        ]

        if regressions:
            lines += [
                "## Regressions",
                "",
                "| Benchmark | Category | Completeness Δ | Safety Δ | Failure | Reasons |",
                "|-----------|----------|----------------|----------|---------|---------|",
            ]
            for d in regressions:
                reasons = "; ".join(d.regression_reasons[:2])
                failure_change = (
                    f"{d.failure_from} → {d.failure_to}"
                    if d.failure_changed else d.failure_to
                )
                lines.append(
                    f"| {d.benchmark_id} | {d.workflow_category} "
                    f"| {d.completeness_delta:+.0%} "
                    f"| {d.safety_delta:+.0%} "
                    f"| {failure_change} "
                    f"| {reasons} |"
                )
            lines.append("")

        if improvements and include_improvements:
            lines += [
                "## Improvements",
                "",
                "| Benchmark | Category | Completeness Δ | Failure | Reasons |",
                "|-----------|----------|----------------|---------|---------|",
            ]
            for d in improvements:
                reasons = "; ".join(d.improvement_reasons[:2])
                failure_change = (
                    f"{d.failure_from} → {d.failure_to}"
                    if d.failure_changed else "—"
                )
                lines.append(
                    f"| {d.benchmark_id} | {d.workflow_category} "
                    f"| {d.completeness_delta:+.0%} "
                    f"| {failure_change} "
                    f"| {reasons} |"
                )
            lines.append("")

        if unchanged:
            lines += [
                "## Unchanged Benchmarks",
                "",
                f"{', '.join(d.benchmark_id for d in unchanged)}",
                "",
            ]

        if missing:
            lines += [
                "## Missing Benchmarks (in baseline but not in current run)",
                "",
                f"{', '.join(d.benchmark_id for d in missing)}",
                "",
            ]

        verdict = "✅ NO REGRESSIONS" if not regressions else f"❌ {len(regressions)} REGRESSION(S)"
        lines += ["---", f"**Verdict: {verdict}**", ""]

        return "\n".join(lines)


# ── Delta computation ─────────────────────────────────────────────────────────

def _compute_delta(
    baseline: BenchmarkSnapshot,
    current: BenchmarkSnapshot,
) -> RegressionDelta:
    completeness_delta = current.workflow_completeness - baseline.workflow_completeness
    safety_delta = current.safety_score - baseline.safety_score

    failure_changed = current.primary_failure != baseline.primary_failure
    severity_changed = current.failure_severity != baseline.failure_severity

    baseline_sev = _SEVERITY_RANK.get(baseline.failure_severity, 0)
    current_sev = _SEVERITY_RANK.get(current.failure_severity, 0)

    regression_reasons: list[str] = []
    improvement_reasons: list[str] = []

    # Completeness regression
    if completeness_delta < -REGRESSION_TOLERANCE:
        regression_reasons.append(
            f"completeness dropped {completeness_delta:+.0%}"
        )
    elif completeness_delta > IMPROVEMENT_THRESHOLD:
        improvement_reasons.append(
            f"completeness gained {completeness_delta:+.0%}"
        )

    # Safety regression
    if safety_delta < -SAFETY_TOLERANCE:
        regression_reasons.append(
            f"safety score dropped {safety_delta:+.0%}"
        )
    elif safety_delta > SAFETY_TOLERANCE:
        improvement_reasons.append(
            f"safety score improved {safety_delta:+.0%}"
        )

    # Severity regression
    if current_sev > baseline_sev:
        regression_reasons.append(
            f"failure severity escalated: {baseline.failure_severity} → {current.failure_severity}"
        )
    elif current_sev < baseline_sev:
        improvement_reasons.append(
            f"failure severity reduced: {baseline.failure_severity} → {current.failure_severity}"
        )

    # New critical safety violations
    if current.critical_safety_violations > baseline.critical_safety_violations:
        regression_reasons.append(
            f"new critical safety violations: "
            f"{baseline.critical_safety_violations} → {current.critical_safety_violations}"
        )

    # Failure type change tracking
    if failure_changed and baseline.primary_failure != "none" and current.primary_failure == "none":
        improvement_reasons.append(
            f"failure resolved: {baseline.primary_failure} → none"
        )
    elif failure_changed and current.primary_failure != "none" and baseline.primary_failure == "none":
        regression_reasons.append(
            f"new failure introduced: {current.primary_failure}"
        )

    return RegressionDelta(
        benchmark_id=current.benchmark_id,
        workflow_category=current.workflow_category,
        completeness_delta=completeness_delta,
        safety_delta=safety_delta,
        failure_changed=failure_changed,
        failure_from=baseline.primary_failure,
        failure_to=current.primary_failure,
        severity_changed=severity_changed,
        severity_from=baseline.failure_severity,
        severity_to=current.failure_severity,
        regressed=len(regression_reasons) > 0,
        improved=len(improvement_reasons) > 0 and len(regression_reasons) == 0,
        regression_reasons=regression_reasons,
        improvement_reasons=improvement_reasons,
    )

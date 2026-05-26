"""
Topology Effectiveness Validator

PURPOSE:
  Measure whether workflow_topology.py's join path surfacing (STEP 2.75)
  changed — or could have changed — the execution outcome.

WHAT IT MEASURES:
  - would_independent_without_topology:  would execution_strategy have chosen
    independent_analyses if topology hints were absent?
  - bridge_paths_found:                  how many cross-shortlist bridge paths exist
  - orphaned_tables_flagged:             files with no reachable topology path
  - topology_prevented_degradation:      True if topology hint changed execution mode
  - potential_joins_surfaced:            number of join opportunities the topology note
    surfaced to the planner that wouldn't be in the shortlist otherwise

WHY IT MATTERS:
  The topology note is injected into the system prompt before execution.
  If the execution_strategy already has enough approved join paths, topology
  adds no value. But if the shortlist lacks direct join evidence and topology
  bridges that gap, this metric shows the lift.

HOW IT WORKS:
  1. Check how many approved joins sql_ctx has
  2. Check WorkflowTopology.reachable_paths for bridge paths
  3. Compare execution mode with and without topology (simulated by checking
     if bridge_paths > 0 and approved_joins == 0)
  4. Count topology-exclusive join opportunities (paths in topology that
     are NOT covered by approved shortlist joins)

USAGE:
  report = validate_topology_effectiveness(benchmark, capture)
  print(report.topology_prevented_degradation)   # True/False
  print(report.potential_joins_surfaced)         # int
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from testing.workflow_eval.benchmarks import WorkflowBenchmark
from testing.workflow_eval.world_state_inspector import (
    WorkflowEvalCapture,
    _extract_approved_join_count,
    _extract_topology_paths,
    _extract_orphaned_tables,
    _extract_execution_mode,
)

if TYPE_CHECKING:
    from server.app.services.workflow_topology import WorkflowTopology


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TopologyPathDetail:
    """Details of a single topology-surfaced path."""
    source: str
    target: str
    bridge: str
    confidence: float
    join_type: str
    is_exclusive: bool  # True if NOT already covered by approved shortlist join


@dataclass
class TopologyEffectivenessReport:
    """
    Topology effectiveness for one benchmark run.

    Attributes:
        was_topology_present:            True if WorkflowTopology was computed
        bridge_paths_found:              total reachable paths in WorkflowTopology
        orphaned_tables_flagged:         files flagged as unreachable
        approved_join_count:             approved joins in SQLContext
        execution_mode:                  actual execution mode
        topology_exclusive_paths:        paths that exist in topology but NOT in
                                         approved shortlist joins
        potential_joins_surfaced:        len(topology_exclusive_paths) —
                                         join opportunities only visible via topology
        would_be_independent_without:    heuristic: if approved_joins==0 and
                                         topology adds bridges, would default to
                                         independent_analyses without topology
        topology_prevented_degradation:  True if topology changed execution outcome
        topology_note_populated:         True if topology note was non-empty
        topology_verdict:                "essential" | "helpful" | "neutral" | "absent"
    """
    was_topology_present: bool
    bridge_paths_found: int
    orphaned_tables_flagged: int
    approved_join_count: int
    execution_mode: str
    topology_exclusive_paths: list[TopologyPathDetail]
    potential_joins_surfaced: int
    would_be_independent_without: bool
    topology_prevented_degradation: bool
    topology_note_populated: bool
    topology_verdict: str  # "essential" | "helpful" | "neutral" | "absent"

    def summary_line(self) -> str:
        return (
            f"topology={self.topology_verdict} | "
            f"bridges={self.bridge_paths_found} | "
            f"exclusive={self.potential_joins_surfaced} | "
            f"prevented_degradation={self.topology_prevented_degradation}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_topology_note(wf_topology: Any) -> str:
    """Extract the topology note string from WorkflowTopology."""
    if wf_topology is None:
        return ""
    try:
        return str(wf_topology.topology_note or "")
    except Exception:
        return ""


def _build_path_details(
    topology_paths: list[dict],
    approved_join_count: int,
) -> list[TopologyPathDetail]:
    """
    Build TopologyPathDetail list and flag paths that are exclusive to topology
    (not already represented by approved shortlist joins).

    Heuristic: if approved_join_count == 0, ALL topology paths are exclusive.
    If approved_join_count > 0, we can't trivially tell which paths overlap
    without deep inspection — conservatively mark them as non-exclusive.
    """
    details: list[TopologyPathDetail] = []
    for p in topology_paths:
        is_exclusive = approved_join_count == 0
        details.append(TopologyPathDetail(
            source=p.get("source", ""),
            target=p.get("target", ""),
            bridge=p.get("bridge", ""),
            confidence=p.get("confidence", 0.0),
            join_type=p.get("join_type", ""),
            is_exclusive=is_exclusive,
        ))
    return details


def _needs_topology_for_benchmark(benchmark: WorkflowBenchmark) -> bool:
    """Does this benchmark have topology requirements?"""
    return (
        len(benchmark.topology_requirements) > 0
        or benchmark.complexity in ("two_table", "multi_table")
    )


# ── Main validator ────────────────────────────────────────────────────────────

def validate_topology_effectiveness(
    benchmark: WorkflowBenchmark,
    capture: WorkflowEvalCapture,
) -> TopologyEffectivenessReport:
    """
    Evaluate whether workflow topology hints changed (or could change) the
    execution outcome for this benchmark.

    Args:
        benchmark: the benchmark being evaluated
        capture:   captured pipeline state

    Returns:
        TopologyEffectivenessReport
    """
    wf_topology = capture.wf_topology
    was_present = wf_topology is not None

    topology_paths = _extract_topology_paths(wf_topology)
    orphaned = _extract_orphaned_tables(wf_topology)
    approved_join_count = _extract_approved_join_count(capture.sql_ctx)
    execution_mode = _extract_execution_mode(capture.exec_strategy)
    topology_note = _extract_topology_note(wf_topology)
    topology_note_populated = len(topology_note.strip()) > 0

    bridge_paths_found = len(topology_paths)
    path_details = _build_path_details(topology_paths, approved_join_count)
    exclusive_paths = [p for p in path_details if p.is_exclusive]

    potential_joins_surfaced = len(exclusive_paths)

    # Would execution have been independent_analyses without topology?
    # Heuristic: if approved_join_count == 0 and benchmark needs joins,
    # execution_strategy would likely select independent_analyses.
    # If topology bridges that gap (has paths), it prevented degradation.
    needs_joins = _needs_topology_for_benchmark(benchmark)
    would_be_independent = (
        needs_joins
        and approved_join_count == 0
    )
    topology_prevented_degradation = (
        would_be_independent
        and bridge_paths_found > 0
        and "independent" not in execution_mode.lower()
    )

    # Determine verdict
    if not was_present:
        verdict = "absent"
    elif topology_prevented_degradation:
        verdict = "essential"
    elif potential_joins_surfaced > 0:
        verdict = "helpful"
    else:
        verdict = "neutral"

    return TopologyEffectivenessReport(
        was_topology_present=was_present,
        bridge_paths_found=bridge_paths_found,
        orphaned_tables_flagged=len(orphaned),
        approved_join_count=approved_join_count,
        execution_mode=execution_mode,
        topology_exclusive_paths=exclusive_paths,
        potential_joins_surfaced=potential_joins_surfaced,
        would_be_independent_without=would_be_independent,
        topology_prevented_degradation=topology_prevented_degradation,
        topology_note_populated=topology_note_populated,
        topology_verdict=verdict,
    )

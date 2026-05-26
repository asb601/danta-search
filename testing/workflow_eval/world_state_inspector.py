"""
Planner World-State Inspector

PURPOSE:
  Capture a complete snapshot of what the planner sees — and crucially,
  what it CANNOT see — at the moment the system prompt is built.

WHY THIS MATTERS:
  The planner only reasons about what's in the shortlist. Files that are
  workflow-relevant but not shortlisted are invisible. This inspector
  exposes that gap explicitly.

WHAT IT CAPTURES:
  - Final shortlist files (visible to planner)
  - Hidden workflow files (relevant but not shortlisted)
  - Approved join paths from SQLContext
  - Topology bridge paths from WorkflowTopology
  - Missing semantic clusters from WorkflowRequirements
  - Files added by adaptive expansion
  - Execution mode and grounding quality

USAGE:
  capture = WorkflowEvalCapture(
      query=..., shortlist=..., full_catalog=...,
      workflow_reqs=..., wf_topology=..., expansion=...,
      ...
  )
  state = inspect_world_state(benchmark, capture)
  print(state.visibility_ratio)  # what fraction of relevant catalog is visible
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from testing.workflow_eval.benchmarks import WorkflowBenchmark
from testing.workflow_eval.completeness_scorer import (
    _collect_roles_from_shortlist,
    _ROLE_KIND_TO_TYPE,
    _parse_role_string,
)

if TYPE_CHECKING:
    from server.app.services.workflow_capability_resolver import WorkflowRequirements
    from server.app.services.workflow_topology import WorkflowTopology
    from server.app.services.semantic_expansion import ExpansionDecision


# ── Capture container ─────────────────────────────────────────────────────────

@dataclass
class WorkflowEvalCapture:
    """
    A snapshot of all intermediate pipeline state, captured during a real
    pipeline run. This is the bridge between production execution and the
    offline evaluation framework.

    Populate this in graph.py AFTER STEP 2.75 (topology) and BEFORE
    returning the final response, then pass it to run_benchmark().

    All fields are optional to allow partial captures for debugging.
    """
    query: str
    shortlist: list[dict] = field(default_factory=list)
    full_catalog: list[dict] = field(default_factory=list)
    entity_resolution: dict = field(default_factory=dict)
    workflow_reqs: Any = None            # WorkflowRequirements
    wf_topology: Any = None              # WorkflowTopology
    expansion: Any = None                # ExpansionDecision
    sql_ctx: Any = None                  # SQLContext
    exec_strategy: Any = None            # ExecutionStrategy
    grounding_quality: str = "unknown"
    pre_expansion_shortlist: list[dict] = field(default_factory=list)
    # Optional: populated if response was evaluated against reference
    llm_response_text: str = ""


# ── World-state snapshot ──────────────────────────────────────────────────────

@dataclass
class HiddenFile:
    """A catalog file that's relevant to the workflow but not in the shortlist."""
    blob_name: str
    relevance_reason: str           # which domain requirement it would satisfy
    satisfies_requirements: list[str]  # short_names of requirements it could serve


@dataclass
class PlannerWorldState:
    """
    Complete snapshot of planner visibility and blind spots.

    Attributes:
        final_shortlist_blobs:       file blobs visible to the planner
        shortlist_file_count:        number of files in shortlist
        hidden_workflow_files:       files that are workflow-relevant but absent
        hidden_count:                number of hidden files
        approved_join_count:         number of approved joins visible to planner
        reachable_topology_paths:    join paths from WorkflowTopology
        orphaned_tables:             files with no join path to shortlist
        missing_semantic_clusters:   domains not represented in shortlist
        expansion_files_added:       blobs added by adaptive expansion
        grounding_quality:           retrieval grounding quality string
        execution_mode:              single_joined | multi_cluster | independent_analyses
        visibility_ratio:            shortlist_count / relevant_catalog_count
        benchmark_relevant_files:    catalog files relevant to THIS benchmark
    """
    final_shortlist_blobs: list[str]
    shortlist_file_count: int
    hidden_workflow_files: list[HiddenFile]
    hidden_count: int
    approved_join_count: int
    reachable_topology_paths: list[dict]
    orphaned_tables: list[str]
    missing_semantic_clusters: list[dict]
    expansion_files_added: list[str]
    grounding_quality: str
    execution_mode: str
    visibility_ratio: float
    benchmark_relevant_files: list[str]

    def brief_report(self) -> str:
        lines = [
            f"Shortlist: {self.shortlist_file_count} files "
            f"(visibility={self.visibility_ratio:.0%})",
            f"Hidden relevant files: {self.hidden_count}",
            f"Approved joins visible: {self.approved_join_count}",
            f"Topology paths: {len(self.reachable_topology_paths)}",
            f"Orphaned tables: {len(self.orphaned_tables)}",
            f"Missing clusters: {len(self.missing_semantic_clusters)}",
            f"Expansion added: {len(self.expansion_files_added)} files",
            f"Execution mode: {self.execution_mode}",
            f"Grounding: {self.grounding_quality}",
        ]
        return "\n".join(lines)


# ── Inspector ─────────────────────────────────────────────────────────────────

def _get_shortlist_blobs(shortlist: list[dict]) -> list[str]:
    return [
        e.get("blob_name") or e.get("file_name") or e.get("id", "")
        for e in shortlist
    ]


def _is_file_relevant_to_benchmark(
    entry: dict,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    """
    Check if a catalog entry is relevant to a benchmark's domain requirements.
    Returns (is_relevant, list_of_satisfied_short_names).
    """
    roles_dict: dict = entry.get("column_semantic_roles") or {}
    satisfied: list[str] = []

    for req in benchmark.expected_domains:
        for role_str in roles_dict.values():
            if not isinstance(role_str, str):
                continue
            parsed = _parse_role_string(role_str)
            if not parsed:
                continue
            role_kind, domain_label = parsed
            mapped_type = _ROLE_KIND_TO_TYPE.get(role_kind, "")
            if mapped_type != req.role_type:
                continue
            if any(h.lower() in domain_label.lower() for h in req.label_hints):
                satisfied.append(req.short_name)
                break

    return len(satisfied) > 0, satisfied


def _extract_approved_join_count(sql_ctx: Any) -> int:
    """Extract approved join count from SQLContext safely."""
    if sql_ctx is None:
        return 0
    try:
        if hasattr(sql_ctx, "join_paths"):
            return len(sql_ctx.join_paths)
        if hasattr(sql_ctx, "approved_relationships"):
            return len(sql_ctx.approved_relationships)
    except Exception:
        pass
    return 0


def _extract_execution_mode(exec_strategy: Any) -> str:
    """Extract execution mode string from ExecutionStrategy safely."""
    if exec_strategy is None:
        return "unknown"
    try:
        if hasattr(exec_strategy, "mode"):
            return str(exec_strategy.mode.value if hasattr(exec_strategy.mode, "value") else exec_strategy.mode)
        if hasattr(exec_strategy, "execution_mode"):
            return str(exec_strategy.execution_mode)
    except Exception:
        pass
    return "unknown"


def _extract_topology_paths(wf_topology: Any) -> list[dict]:
    """Extract serializable topology path info from WorkflowTopology."""
    if wf_topology is None:
        return []
    try:
        result = []
        for path in wf_topology.reachable_paths:
            result.append({
                "source": getattr(path, "source_blob", ""),
                "target": getattr(path, "target_blob", ""),
                "bridge": getattr(path, "bridge_node", ""),
                "confidence": getattr(path, "confidence", 0.0),
                "join_type": getattr(path, "join_type", ""),
            })
        return result
    except Exception:
        return []


def _extract_orphaned_tables(wf_topology: Any) -> list[str]:
    if wf_topology is None:
        return []
    try:
        return list(wf_topology.orphaned_tables or [])
    except Exception:
        return []


def _extract_missing_clusters(workflow_reqs: Any) -> list[dict]:
    """Extract missing semantic domain clusters from WorkflowRequirements."""
    if workflow_reqs is None:
        return []
    try:
        result = []
        for domain in (workflow_reqs.missing_domains or []):
            result.append({
                "name": getattr(domain, "domain_name", str(domain)),
                "role_type": getattr(domain, "primary_role_kind", ""),
            })
        return result
    except Exception:
        return []


def _extract_expansion_files(expansion: Any, shortlist: list[dict]) -> list[str]:
    """Identify files added by adaptive expansion."""
    if expansion is None:
        return []
    try:
        added: list[str] = []
        for candidate in (expansion.candidates_added or []):
            blob = getattr(candidate, "blob_name", "") or getattr(candidate, "file_id", "")
            if blob:
                added.append(blob)
        return added
    except Exception:
        return []


def inspect_world_state(
    benchmark: WorkflowBenchmark,
    capture: WorkflowEvalCapture,
) -> PlannerWorldState:
    """
    Build a PlannerWorldState from a WorkflowEvalCapture and benchmark definition.

    This reveals:
    - How much of the benchmark-relevant catalog is visible to the planner
    - Which relevant files are hidden (not shortlisted)
    - What join paths are available
    - What topology hints exist
    - What the planner is missing

    Args:
        benchmark: the benchmark being evaluated
        capture:   the captured pipeline state

    Returns:
        PlannerWorldState with full visibility analysis
    """
    shortlist_blobs = set(_get_shortlist_blobs(capture.shortlist))

    # Find benchmark-relevant files in full catalog
    relevant_files: list[str] = []
    hidden_files: list[HiddenFile] = []

    for entry in capture.full_catalog:
        blob = entry.get("blob_name") or entry.get("file_name") or entry.get("id", "")
        is_relevant, satisfied_reqs = _is_file_relevant_to_benchmark(entry, benchmark)

        if not is_relevant:
            continue

        relevant_files.append(blob)

        if blob not in shortlist_blobs:
            # This file is relevant but hidden from planner
            reason_parts = []
            for req_name in satisfied_reqs:
                reason_parts.append(f"satisfies domain={req_name}")
            hidden_files.append(HiddenFile(
                blob_name=blob,
                relevance_reason="; ".join(reason_parts) or "workflow-relevant roles detected",
                satisfies_requirements=satisfied_reqs,
            ))

    visibility_ratio = (
        len(shortlist_blobs) / len(relevant_files)
        if relevant_files else 1.0
    )

    return PlannerWorldState(
        final_shortlist_blobs=list(shortlist_blobs),
        shortlist_file_count=len(shortlist_blobs),
        hidden_workflow_files=hidden_files,
        hidden_count=len(hidden_files),
        approved_join_count=_extract_approved_join_count(capture.sql_ctx),
        reachable_topology_paths=_extract_topology_paths(capture.wf_topology),
        orphaned_tables=_extract_orphaned_tables(capture.wf_topology),
        missing_semantic_clusters=_extract_missing_clusters(capture.workflow_reqs),
        expansion_files_added=_extract_expansion_files(capture.expansion, capture.shortlist),
        grounding_quality=capture.grounding_quality,
        execution_mode=_extract_execution_mode(capture.exec_strategy),
        visibility_ratio=min(visibility_ratio, 1.0),
        benchmark_relevant_files=relevant_files,
    )

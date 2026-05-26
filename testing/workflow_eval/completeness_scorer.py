"""
Workflow Completeness Scorer

PURPOSE:
  Given a WorkflowBenchmark (what's expected) and the actual pipeline output
  (what was retrieved), compute a WorkflowCompletenessScore.

WHAT IT MEASURES:
  - workflow_completeness:       fraction of expected domains that are directly
                                 visible in the planner's shortlist
  - planner_world_completeness:  fraction visible in shortlist OR reachable via
                                 workflow topology hint paths
  - per-domain satisfaction:     which DomainRequirements are met/missing

HOW DOMAIN MATCHING WORKS:
  Each file in the shortlist contributes its column_semantic_roles.
  A role entry looks like: {"ColumnName": "custom:role_kind:domain_label"}
  For each DomainRequirement we check:
    1. Any column in any shortlisted file has role_kind that maps to role_type
    2. AND the domain_label contains at least one token from label_hints
  If both conditions are true, the requirement is satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from testing.workflow_eval.benchmarks import DomainRequirement, WorkflowBenchmark

if TYPE_CHECKING:
    from server.app.services.workflow_capability_resolver import WorkflowRequirements
    from server.app.services.workflow_topology import WorkflowTopology
    from server.app.services.semantic_expansion import ExpansionDecision


# ── Scoring result ─────────────────────────────────────────────────────────────

@dataclass
class DomainCoverageDetail:
    """Single-domain coverage result."""
    requirement: DomainRequirement
    satisfied: bool
    satisfied_by_topology: bool        # True only if shortlist miss but topology has it
    satisfying_files: list[str]        # file blobs that satisfied this requirement
    evidence_roles: list[str]          # role strings that matched
    notes: str = ""


@dataclass
class WorkflowCompletenessScore:
    """
    Full completeness evaluation for one benchmark run.

    Attributes:
        benchmark_id:               benchmark identifier
        workflow_query:             the original query
        workflow_completeness:      direct shortlist coverage (0.0–1.0)
        planner_world_completeness: shortlist + topology coverage (0.0–1.0)
        required_domain_count:      total required domains in benchmark
        satisfied_required_count:   required domains satisfied
        domain_coverage:            per-domain detail list
        missing_required:           DomainRequirements that are required but absent
        missing_optional:           DomainRequirements that are optional but absent
        expansion_contributed:      True if expansion added new domain coverage
        topology_contributed:       True if topology bridged a coverage gap
        grounding_quality:          retrieval grounding quality string
        failure_flags:              list of detected failure indicators
        passed:                     True if completeness >= benchmark threshold
    """
    benchmark_id: str
    workflow_query: str
    workflow_completeness: float
    planner_world_completeness: float
    required_domain_count: int
    satisfied_required_count: int
    domain_coverage: list[DomainCoverageDetail]
    missing_required: list[DomainRequirement]
    missing_optional: list[DomainRequirement]
    expansion_contributed: bool
    topology_contributed: bool
    grounding_quality: str
    failure_flags: list[str]
    passed: bool

    @property
    def optional_domain_count(self) -> int:
        return sum(1 for d in self.domain_coverage if not d.requirement.is_required)

    @property
    def satisfied_optional_count(self) -> int:
        return sum(1 for d in self.domain_coverage if not d.requirement.is_required and d.satisfied)

    def summary_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{self.benchmark_id}] {verdict} | "
            f"completeness={self.workflow_completeness:.0%} "
            f"(world={self.planner_world_completeness:.0%}) | "
            f"required={self.satisfied_required_count}/{self.required_domain_count} | "
            f"grounding={self.grounding_quality}"
        )


# ── Role parsing ───────────────────────────────────────────────────────────────

_ROLE_KIND_TO_TYPE: dict[str, str] = {
    "entity_key": "entity",
    "reference_key": "transaction",
    "additive_measure": "transaction",
    "non_additive_measure": "transaction",
    "date": "dimension",
    "attribute": "dimension",
}


def _parse_role_string(role_str: str) -> tuple[str, str] | None:
    """
    Parse 'custom:role_kind:domain_label' → (role_kind, domain_label).
    Returns None if format is unexpected.
    """
    parts = role_str.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


def _collect_roles_from_shortlist(
    shortlist: list[dict],
) -> dict[str, list[tuple[str, str, str]]]:
    """
    Build: {file_blob → [(column_name, role_kind, domain_label), ...]}
    from column_semantic_roles in each shortlisted file's catalog entry.
    """
    result: dict[str, list[tuple[str, str, str]]] = {}
    for entry in shortlist:
        blob = entry.get("blob_name") or entry.get("file_name") or entry.get("id", "")
        roles_dict: dict = entry.get("column_semantic_roles") or {}
        parsed: list[tuple[str, str, str]] = []
        for col_name, role_str in roles_dict.items():
            if not isinstance(role_str, str):
                continue
            parsed_role = _parse_role_string(role_str)
            if parsed_role:
                parsed.append((col_name, parsed_role[0], parsed_role[1]))
        if parsed:
            result[blob] = parsed
    return result


def _collect_roles_from_topology(topology: "WorkflowTopology") -> list[tuple[str, str]]:
    """
    Extract (role_kind, domain_label) from topology bridge path hints.

    WorkflowTopology.reachable_paths contains ReachablePath objects.
    We look for topology-hinted files that may have relevant role info.
    This is intentionally shallow — topology only expands awareness of
    files that are joinable, not deep semantic role analysis.
    """
    hints: list[tuple[str, str]] = []
    try:
        for path in topology.reachable_paths:
            # ReachablePath.bridge_node may carry role hints if populated
            if hasattr(path, "bridge_role_kind") and path.bridge_role_kind:
                hints.append((path.bridge_role_kind, path.bridge_role_label or ""))
    except Exception:
        pass
    return hints


# ── Core scoring function ──────────────────────────────────────────────────────

def _score_requirement_against_shortlist(
    req: DomainRequirement,
    roles_by_file: dict[str, list[tuple[str, str, str]]],
) -> DomainCoverageDetail:
    """
    Check if a single DomainRequirement is satisfied by any file in the shortlist.
    Returns a DomainCoverageDetail with full evidence.
    """
    satisfying_files: list[str] = []
    evidence_roles: list[str] = []

    for blob, col_roles in roles_by_file.items():
        file_satisfies = False
        for col_name, role_kind, domain_label in col_roles:
            mapped_type = _ROLE_KIND_TO_TYPE.get(role_kind, "")
            if mapped_type != req.role_type:
                continue
            label_lower = domain_label.lower()
            if any(hint.lower() in label_lower for hint in req.label_hints):
                evidence_roles.append(
                    f"{blob}::{col_name} → {role_kind}:{domain_label}"
                )
                file_satisfies = True
        if file_satisfies and blob not in satisfying_files:
            satisfying_files.append(blob)

    return DomainCoverageDetail(
        requirement=req,
        satisfied=len(satisfying_files) > 0,
        satisfied_by_topology=False,
        satisfying_files=satisfying_files,
        evidence_roles=evidence_roles[:5],  # cap evidence list
    )


def score_workflow_completeness(
    benchmark: WorkflowBenchmark,
    shortlist: list[dict],
    topology: "WorkflowTopology | None" = None,
    expansion: "ExpansionDecision | None" = None,
    pre_expansion_shortlist: list[dict] | None = None,
    grounding_quality: str = "unknown",
) -> WorkflowCompletenessScore:
    """
    Compute a WorkflowCompletenessScore for one benchmark run.

    Args:
        benchmark:               the benchmark definition
        shortlist:               the final shortlisted catalog entries
        topology:                optional WorkflowTopology for world_completeness
        expansion:               optional ExpansionDecision to assess contribution
        pre_expansion_shortlist: shortlist before adaptive expansion
        grounding_quality:       retrieval grounding quality string

    Returns:
        WorkflowCompletenessScore with full coverage analysis
    """
    # Build role index from current shortlist
    roles_by_file = _collect_roles_from_shortlist(shortlist)

    # Score each domain requirement
    domain_coverage: list[DomainCoverageDetail] = []
    for req in benchmark.expected_domains:
        detail = _score_requirement_against_shortlist(req, roles_by_file)
        domain_coverage.append(detail)

    # Check topology contributions for unsatisfied required domains
    topology_contributed = False
    if topology is not None:
        topology_role_hints = _collect_roles_from_topology(topology)
        for detail in domain_coverage:
            if detail.satisfied:
                continue
            # Check if topology hints could bridge this gap
            for t_kind, t_label in topology_role_hints:
                mapped_type = _ROLE_KIND_TO_TYPE.get(t_kind, "")
                if mapped_type != detail.requirement.role_type:
                    continue
                if any(h.lower() in t_label.lower() for h in detail.requirement.label_hints):
                    detail.satisfied_by_topology = True
                    topology_contributed = True
                    detail.notes = "satisfied via topology bridge hint"
                    break

        # Also check topology.orphaned_tables hinting at available but unreachable files
        # (we don't flip satisfied_by_topology for orphans — they're explicitly hidden)

    # Check expansion contribution
    expansion_contributed = False
    if expansion is not None and pre_expansion_shortlist is not None:
        pre_roles = _collect_roles_from_shortlist(pre_expansion_shortlist)
        for detail in domain_coverage:
            if not detail.satisfied:
                continue
            # Was this satisfied before expansion too?
            pre_detail = _score_requirement_against_shortlist(detail.requirement, pre_roles)
            if not pre_detail.satisfied:
                # This requirement was only satisfied after expansion
                expansion_contributed = True
                detail.notes = "satisfied by adaptive expansion"

    # Aggregate counts
    required_domains = [d for d in domain_coverage if d.requirement.is_required]
    optional_domains = [d for d in domain_coverage if not d.requirement.is_required]

    satisfied_required = [
        d for d in required_domains if d.satisfied
    ]
    satisfied_required_or_topology = [
        d for d in required_domains if d.satisfied or d.satisfied_by_topology
    ]

    missing_required = [
        d.requirement for d in required_domains if not d.satisfied
    ]
    missing_optional = [
        d.requirement for d in optional_domains if not d.satisfied
    ]

    total_expected = len(benchmark.expected_domains)
    total_satisfied = sum(1 for d in domain_coverage if d.satisfied)
    total_world = sum(1 for d in domain_coverage if d.satisfied or d.satisfied_by_topology)

    workflow_completeness = total_satisfied / total_expected if total_expected > 0 else 0.0
    planner_world_completeness = total_world / total_expected if total_expected > 0 else 0.0

    # Detect failure flags
    failure_flags = _detect_failure_flags(
        benchmark=benchmark,
        missing_required=missing_required,
        workflow_completeness=workflow_completeness,
        grounding_quality=grounding_quality,
        shortlist_count=len(shortlist),
    )

    passed = (
        workflow_completeness >= benchmark.min_workflow_completeness
        and len(missing_required) == 0
    )

    return WorkflowCompletenessScore(
        benchmark_id=benchmark.id,
        workflow_query=benchmark.workflow_query,
        workflow_completeness=workflow_completeness,
        planner_world_completeness=planner_world_completeness,
        required_domain_count=len(required_domains),
        satisfied_required_count=len(satisfied_required),
        domain_coverage=domain_coverage,
        missing_required=missing_required,
        missing_optional=missing_optional,
        expansion_contributed=expansion_contributed,
        topology_contributed=topology_contributed,
        grounding_quality=grounding_quality,
        failure_flags=failure_flags,
        passed=passed,
    )


def _detect_failure_flags(
    benchmark: WorkflowBenchmark,
    missing_required: list[DomainRequirement],
    workflow_completeness: float,
    grounding_quality: str,
    shortlist_count: int,
) -> list[str]:
    """Detect failure indicators from scoring metrics."""
    flags: list[str] = []

    if grounding_quality == "keyword_degraded":
        flags.append("GROUNDING_DEGRADED")

    if workflow_completeness < benchmark.min_workflow_completeness:
        flags.append("BELOW_COMPLETENESS_THRESHOLD")

    if missing_required:
        names = ", ".join(r.short_name for r in missing_required)
        flags.append(f"MISSING_REQUIRED_DOMAINS:{names}")

    if shortlist_count <= 2 and benchmark.complexity == "multi_table":
        flags.append("PLANNER_STARVATION")

    if workflow_completeness == 0.0:
        flags.append("TOTAL_WORKFLOW_MISS")

    return flags

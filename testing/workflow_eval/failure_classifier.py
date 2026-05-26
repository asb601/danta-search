"""
Failure Classification Engine

PURPOSE:
  Given a WorkflowCompletenessScore and PlannerWorldState, determine the
  PRIMARY failure type, contributing secondary failures, root cause, and
  a concrete recovery suggestion.

FAILURE TAXONOMY (ordered by severity):

  NONE                      — All checks pass
  TOTAL_WORKFLOW_MISS       — Zero domains from expected set were retrieved
  PLANNER_STARVATION        — Shortlist too small for multi-table workflow
  WORKFLOW_MISS             — Some domains found but completeness below threshold
  RETRIEVAL_MISS            — Retrieval fell back to keyword_degraded grounding
  TOPOLOGY_MISS             — Joins needed but no topology paths visible
  JOIN_MISS                 — Independent analyses when join was needed
  PARTIAL_WORKFLOW          — Core domain missing but peripherals found
  EXECUTION_DEGRADATION     — Execution mode weaker than workflow requires
  GROUNDING_MISS            — Planner references concepts without supporting files
  FALSE_BUSINESS_INFERENCE  — Business status claimed without domain evidence

CLASSIFICATION LOGIC:
  Failure is NOT determined by output text (no LLM scoring here).
  It's determined purely from: completeness score, world state, execution mode,
  missing domain list, and grounding quality.
  This makes classification deterministic and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from testing.workflow_eval.benchmarks import WorkflowBenchmark
from testing.workflow_eval.completeness_scorer import WorkflowCompletenessScore
from testing.workflow_eval.world_state_inspector import PlannerWorldState


# ── Failure taxonomy ──────────────────────────────────────────────────────────

class FailureType(Enum):
    NONE = "none"
    TOTAL_WORKFLOW_MISS = "total_workflow_miss"
    PLANNER_STARVATION = "planner_starvation"
    WORKFLOW_MISS = "workflow_miss"
    RETRIEVAL_MISS = "retrieval_miss"
    TOPOLOGY_MISS = "topology_miss"
    JOIN_MISS = "join_miss"
    PARTIAL_WORKFLOW = "partial_workflow"
    EXECUTION_DEGRADATION = "execution_degradation"
    GROUNDING_MISS = "grounding_miss"
    FALSE_BUSINESS_INFERENCE = "false_business_inference"


_SEVERITY: dict[FailureType, str] = {
    FailureType.NONE: "none",
    FailureType.TOTAL_WORKFLOW_MISS: "critical",
    FailureType.PLANNER_STARVATION: "critical",
    FailureType.WORKFLOW_MISS: "major",
    FailureType.RETRIEVAL_MISS: "major",
    FailureType.TOPOLOGY_MISS: "major",
    FailureType.JOIN_MISS: "major",
    FailureType.PARTIAL_WORKFLOW: "minor",
    FailureType.EXECUTION_DEGRADATION: "minor",
    FailureType.GROUNDING_MISS: "major",
    FailureType.FALSE_BUSINESS_INFERENCE: "critical",
}

_RECOVERY_SUGGESTIONS: dict[FailureType, str] = {
    FailureType.NONE: "No action needed.",
    FailureType.TOTAL_WORKFLOW_MISS: (
        "Full retrieval failure. Check: (1) semantic roles in catalog, "
        "(2) entity resolution extracted correct tokens, "
        "(3) retrieval top-k was not collapsed to zero."
    ),
    FailureType.PLANNER_STARVATION: (
        "Shortlist is too small for this multi-table workflow. "
        "Check adaptive expansion triggers — "
        "_COMPLETENESS_EXPANSION_THRESHOLD may need lowering, "
        "or expansion slots are exhausted (_HARD_MAX_SHORTLIST)."
    ),
    FailureType.WORKFLOW_MISS: (
        "Completeness below threshold. Check semantic_expansion.py — "
        "verify missing domains are being detected by workflow_capability_resolver "
        "and that expansion candidates with matching roles exist in the catalog."
    ),
    FailureType.RETRIEVAL_MISS: (
        "Retrieval degraded to keyword fallback. "
        "Check semantic_recovery.py — verify role_cluster and graph_topology "
        "stages have sufficient data to recover before degrading."
    ),
    FailureType.TOPOLOGY_MISS: (
        "Joins needed but no topology paths were surfaced. "
        "Check workflow_topology.py — verify relationships exist with "
        "status='active' AND approval_status='approved'."
    ),
    FailureType.JOIN_MISS: (
        "Planner was given independent_analyses mode but join was required. "
        "Check execution_strategy.py — verify the soft governance text is "
        "directing the LLM to call extract_relations before declaring no join."
    ),
    FailureType.PARTIAL_WORKFLOW: (
        "Peripheral domains found but core domain missing. "
        "Review entity extraction — check if query tokens align with "
        "the core domain's role label tokens in column_semantic_roles."
    ),
    FailureType.EXECUTION_DEGRADATION: (
        "Execution mode was weaker than the workflow demands. "
        "Review execution_strategy.py mode selection logic — "
        "check approved_join_count and confidence thresholds."
    ),
    FailureType.GROUNDING_MISS: (
        "Planner referenced concepts not in shortlist. "
        "Check world state hidden_files — if relevant files were present "
        "in catalog but not shortlisted, expansion or retrieval is under-firing."
    ),
    FailureType.FALSE_BUSINESS_INFERENCE: (
        "Business status was claimed without domain evidence. "
        "The answer asserts facts (e.g. 'approved', 'paid', 'delivered') "
        "without the relevant domain being in the shortlist. "
        "This is a safety-critical failure requiring domain presence enforcement."
    ),
}


# ── Classification result ─────────────────────────────────────────────────────

@dataclass
class FailureClassification:
    """
    Full failure classification for one benchmark evaluation.

    Attributes:
        primary_failure:      most severe single failure type
        secondary_failures:   additional failure types detected
        root_cause:           concise root cause statement
        evidence:             specific observable signals that triggered classification
        severity:             "critical" | "major" | "minor" | "none"
        recovery_suggestion:  actionable guidance for the developer
        all_failures:         all detected FailureTypes (primary + secondary)
    """
    primary_failure: FailureType
    secondary_failures: list[FailureType]
    root_cause: str
    evidence: list[str]
    severity: Literal["critical", "major", "minor", "none"]
    recovery_suggestion: str

    @property
    def all_failures(self) -> list[FailureType]:
        return [self.primary_failure] + self.secondary_failures

    @property
    def has_failure(self) -> bool:
        return self.primary_failure != FailureType.NONE

    def summary_line(self) -> str:
        secondary = (
            f" + {', '.join(f.value for f in self.secondary_failures)}"
            if self.secondary_failures else ""
        )
        return (
            f"[{self.severity.upper()}] {self.primary_failure.value}{secondary} "
            f"— {self.root_cause}"
        )


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_total_workflow_miss(
    score: WorkflowCompletenessScore,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if score.workflow_completeness == 0.0:
        evidence.append(f"workflow_completeness=0.0 (zero domains from expected set)")
        return True, evidence
    return False, []


def _detect_planner_starvation(
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if benchmark.complexity == "multi_table" and world.shortlist_file_count <= 2:
        evidence.append(
            f"shortlist_count={world.shortlist_file_count} for "
            f"complexity={benchmark.complexity}"
        )
        if world.hidden_count > 0:
            evidence.append(f"{world.hidden_count} relevant files hidden from planner")
        return True, evidence
    return False, []


def _detect_workflow_miss(
    score: WorkflowCompletenessScore,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if score.workflow_completeness < benchmark.min_workflow_completeness:
        evidence.append(
            f"workflow_completeness={score.workflow_completeness:.2f} < "
            f"threshold={benchmark.min_workflow_completeness:.2f}"
        )
        if score.missing_required:
            names = [r.short_name for r in score.missing_required]
            evidence.append(f"missing required: {', '.join(names)}")
        return True, evidence
    return False, []


def _detect_retrieval_miss(
    score: WorkflowCompletenessScore,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if score.grounding_quality == "keyword_degraded":
        evidence.append("retrieval fell to keyword_degraded grounding quality")
        return True, evidence
    if score.grounding_quality in ("semantic_bridge",):
        # Not a hard failure but worth noting
        evidence.append(f"retrieval used degraded stage: {score.grounding_quality}")
        return False, []
    return False, []


def _detect_topology_miss(
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    has_topology_req = len(benchmark.topology_requirements) > 0
    if not has_topology_req:
        return False, []
    if world.approved_join_count == 0 and len(world.reachable_topology_paths) == 0:
        evidence.append("no approved join paths visible to planner")
        if benchmark.complexity in ("two_table", "multi_table"):
            evidence.append(
                f"benchmark requires topology but none available "
                f"(complexity={benchmark.complexity})"
            )
        return True, evidence
    return False, []


def _detect_join_miss(
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    is_independent = "independent" in world.execution_mode.lower()
    needs_join = benchmark.complexity in ("two_table", "multi_table")

    if is_independent and needs_join:
        evidence.append(
            f"execution_mode={world.execution_mode} but "
            f"benchmark.complexity={benchmark.complexity}"
        )
        evidence.append(
            "planner received independent_analyses governance for a join-required workflow"
        )
        return True, evidence
    return False, []


def _detect_partial_workflow(
    score: WorkflowCompletenessScore,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    has_partial = (
        score.satisfied_required_count > 0
        and len(score.missing_required) > 0
    )
    if has_partial:
        found = [
            d.requirement.short_name for d in score.domain_coverage
            if d.satisfied and d.requirement.is_required
        ]
        missing = [r.short_name for r in score.missing_required]
        evidence.append(f"found required: {', '.join(found)}")
        evidence.append(f"missing required: {', '.join(missing)}")
        return True, evidence
    return False, []


def _detect_execution_degradation(
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> tuple[bool, list[str]]:
    """
    Execution mode is degraded relative to what the workflow requires.
    single_joined < multi_cluster < independent_analyses in terms of join power.
    """
    evidence: list[str] = []
    mode = world.execution_mode.lower()

    needs_strong_join = (
        benchmark.complexity == "multi_table"
        and len(benchmark.topology_requirements) > 1
    )
    if needs_strong_join and "independent" in mode:
        evidence.append(
            f"multi-join benchmark but mode={world.execution_mode}"
        )
        return True, evidence
    return False, []


def _detect_grounding_miss(
    world: PlannerWorldState,
) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    if world.hidden_count > 0 and world.visibility_ratio < 0.50:
        evidence.append(
            f"visibility_ratio={world.visibility_ratio:.0%} — "
            f"{world.hidden_count} relevant files hidden from planner"
        )
        return True, evidence
    return False, []


def _detect_false_business_inference(
    score: WorkflowCompletenessScore,
    benchmark: WorkflowBenchmark,
    world: PlannerWorldState,
) -> tuple[bool, list[str]]:
    """
    Business-truth safety check: if a benchmark has truth requirements
    that map to specific domain short_names, and those domains are missing
    from the shortlist, any answer about those facts is unsupported.
    """
    evidence: list[str] = []
    # Check if any required domain that was NOT satisfied is also a business-truth requirement
    truth_related_reqs = {
        r.short_name for r in score.missing_required
    }
    triggered_truths = [
        tr for tr in benchmark.business_truth_requirements
        if any(key in tr for key in truth_related_reqs)
    ]
    if triggered_truths:
        evidence.append(
            f"business truth requirements unsatisfied: {triggered_truths}"
        )
        evidence.append(
            "answer may claim business facts without domain evidence"
        )
        return True, evidence
    return False, []


# ── Priority ordering ─────────────────────────────────────────────────────────

_DETECTION_PIPELINE = [
    # (name, detector_fn, failure_type)
    ("total_miss",   _detect_total_workflow_miss,    FailureType.TOTAL_WORKFLOW_MISS),
    ("starvation",   None,                           FailureType.PLANNER_STARVATION),
    ("workflow",     None,                           FailureType.WORKFLOW_MISS),
    ("retrieval",    _detect_retrieval_miss,         FailureType.RETRIEVAL_MISS),
    ("topology",     None,                           FailureType.TOPOLOGY_MISS),
    ("join",         None,                           FailureType.JOIN_MISS),
    ("partial",      _detect_partial_workflow,       FailureType.PARTIAL_WORKFLOW),
    ("exec_degrade", None,                           FailureType.EXECUTION_DEGRADATION),
    ("grounding",    _detect_grounding_miss,         FailureType.GROUNDING_MISS),
    ("biz_infer",    None,                           FailureType.FALSE_BUSINESS_INFERENCE),
]


# ── Main classification function ──────────────────────────────────────────────

def classify_failure(
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> FailureClassification:
    """
    Classify failure type(s) from completeness score and world state.

    Classification is deterministic — no LLM calls.
    Evidence comes from scoring metrics, not output text.

    Args:
        score:     WorkflowCompletenessScore from score_workflow_completeness()
        world:     PlannerWorldState from inspect_world_state()
        benchmark: the benchmark definition

    Returns:
        FailureClassification with primary, secondary failures and root cause
    """
    detected: list[tuple[FailureType, list[str]]] = []

    # Run all detectors
    is_total, ev = _detect_total_workflow_miss(score)
    if is_total:
        detected.append((FailureType.TOTAL_WORKFLOW_MISS, ev))

    is_starved, ev = _detect_planner_starvation(score, world, benchmark)
    if is_starved:
        detected.append((FailureType.PLANNER_STARVATION, ev))

    is_workflow, ev = _detect_workflow_miss(score, benchmark)
    if is_workflow:
        detected.append((FailureType.WORKFLOW_MISS, ev))

    is_retrieval, ev = _detect_retrieval_miss(score)
    if is_retrieval:
        detected.append((FailureType.RETRIEVAL_MISS, ev))

    is_topology, ev = _detect_topology_miss(score, world, benchmark)
    if is_topology:
        detected.append((FailureType.TOPOLOGY_MISS, ev))

    is_join, ev = _detect_join_miss(score, world, benchmark)
    if is_join:
        detected.append((FailureType.JOIN_MISS, ev))

    is_partial, ev = _detect_partial_workflow(score)
    if is_partial and not is_workflow:  # don't double-report
        detected.append((FailureType.PARTIAL_WORKFLOW, ev))

    is_exec_deg, ev = _detect_execution_degradation(world, benchmark)
    if is_exec_deg:
        detected.append((FailureType.EXECUTION_DEGRADATION, ev))

    is_grounding, ev = _detect_grounding_miss(world)
    if is_grounding:
        detected.append((FailureType.GROUNDING_MISS, ev))

    is_biz_infer, ev = _detect_false_business_inference(score, benchmark, world)
    if is_biz_infer:
        detected.append((FailureType.FALSE_BUSINESS_INFERENCE, ev))

    if not detected:
        return FailureClassification(
            primary_failure=FailureType.NONE,
            secondary_failures=[],
            root_cause="All workflow completeness checks passed.",
            evidence=[
                f"completeness={score.workflow_completeness:.0%}",
                f"grounding={score.grounding_quality}",
                f"mode={world.execution_mode}",
            ],
            severity="none",
            recovery_suggestion=_RECOVERY_SUGGESTIONS[FailureType.NONE],
        )

    # Sort by severity: critical > major > minor
    severity_rank = {"critical": 3, "major": 2, "minor": 1, "none": 0}
    detected.sort(
        key=lambda x: severity_rank.get(_SEVERITY.get(x[0], "none"), 0),
        reverse=True,
    )

    primary_type, primary_ev = detected[0]
    secondary = [t for t, _ in detected[1:]]
    all_evidence = primary_ev + [
        item for _, ev_list in detected[1:] for item in ev_list
    ]

    root_cause = _build_root_cause(primary_type, primary_ev, score, world, benchmark)

    return FailureClassification(
        primary_failure=primary_type,
        secondary_failures=secondary,
        root_cause=root_cause,
        evidence=all_evidence[:10],  # cap evidence list
        severity=_SEVERITY.get(primary_type, "none"),
        recovery_suggestion=_RECOVERY_SUGGESTIONS.get(
            primary_type, "Review pipeline logs."
        ),
    )


def _build_root_cause(
    failure: FailureType,
    evidence: list[str],
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
    benchmark: WorkflowBenchmark,
) -> str:
    if failure == FailureType.TOTAL_WORKFLOW_MISS:
        return (
            f"Retrieval returned zero files with roles matching benchmark "
            f"'{benchmark.workflow_category}' domain requirements."
        )
    if failure == FailureType.PLANNER_STARVATION:
        return (
            f"Planner has only {world.shortlist_file_count} files for a "
            f"{benchmark.complexity} workflow — {world.hidden_count} relevant "
            f"files were not shortlisted."
        )
    if failure == FailureType.WORKFLOW_MISS:
        missing = [r.short_name for r in score.missing_required]
        return (
            f"Workflow completeness {score.workflow_completeness:.0%} below "
            f"{benchmark.min_workflow_completeness:.0%} threshold. "
            f"Missing domains: {', '.join(missing)}."
        )
    if failure == FailureType.RETRIEVAL_MISS:
        return (
            f"Semantic retrieval degraded to '{score.grounding_quality}' — "
            f"role-cluster and graph-topology recovery stages did not fire."
        )
    if failure == FailureType.TOPOLOGY_MISS:
        return (
            f"Workflow requires join topology but {world.approved_join_count} "
            f"approved joins and {len(world.reachable_topology_paths)} topology "
            f"paths are visible to the planner."
        )
    if failure == FailureType.JOIN_MISS:
        return (
            f"Planner received '{world.execution_mode}' governance but "
            f"benchmark '{benchmark.id}' requires join-capable execution "
            f"({benchmark.complexity})."
        )
    if failure == FailureType.FALSE_BUSINESS_INFERENCE:
        missing = [r.short_name for r in score.missing_required]
        return (
            f"Answer may claim business facts ({benchmark.business_truth_requirements}) "
            f"without domain evidence — required domains missing: {', '.join(missing)}."
        )
    return "; ".join(evidence[:2]) if evidence else "See evidence list."

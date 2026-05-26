"""
Expansion Effectiveness Validator

PURPOSE:
  Measure whether adaptive expansion (STEP 2.57) actually improved workflow
  completeness, and whether it expanded correctly without false positives.

WHAT IT MEASURES:
  - domains_before:        semantic domain coverage before expansion
  - domains_after:         coverage after expansion
  - completeness_delta:    how much workflow completeness improved
  - files_added:           how many files expansion added
  - false_expansion_rate:  fraction of added files that matched no benchmark domain
  - triggers_fired:        which expansion trigger conditions were active

HOW IT WORKS:
  1. Score the benchmark against the PRE-expansion shortlist
  2. Score the benchmark against the POST-expansion shortlist
  3. Compute delta
  4. Check each added file: does it satisfy at least one expected domain?
     If not, it's a false expansion (irrelevant file was added).

USAGE:
  # capture.pre_expansion_shortlist = shortlist before STEP 2.57
  # capture.shortlist               = final shortlist after STEP 2.57
  report = validate_expansion_effectiveness(benchmark, capture)
  print(report.completeness_delta)  # e.g. +0.20 (20% improvement)
  print(report.false_expansion_rate) # e.g. 0.0 (no irrelevant files added)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from testing.workflow_eval.benchmarks import WorkflowBenchmark
from testing.workflow_eval.completeness_scorer import (
    score_workflow_completeness,
    WorkflowCompletenessScore,
    _collect_roles_from_shortlist,
    _ROLE_KIND_TO_TYPE,
    _parse_role_string,
)
from testing.workflow_eval.world_state_inspector import WorkflowEvalCapture

if TYPE_CHECKING:
    from server.app.services.semantic_expansion import ExpansionDecision


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AddedFileEvaluation:
    """Whether a single expansion-added file contributed domain coverage."""
    blob_name: str
    contributed_domains: list[str]   # short_names of requirements it satisfied
    is_useful: bool                  # True if contributed_domains is non-empty
    is_false_expansion: bool         # True if added but matched no expected domain


@dataclass
class ExpansionEffectivenessReport:
    """
    Before/after comparison of adaptive expansion.

    Attributes:
        was_triggered:           True if expansion ran (shortlists differ)
        triggers_fired:          expansion trigger conditions that were active
        files_added:             number of files added by expansion
        domains_before:          number of domains satisfied before expansion
        domains_after:           number of domains satisfied after expansion
        completeness_before:     workflow_completeness before expansion (0.0–1.0)
        completeness_after:      workflow_completeness after expansion (0.0–1.0)
        completeness_delta:      improvement from expansion (can be 0 or negative)
        recovered_domains:       domain short_names that only became visible post-expansion
        false_expansions:        files that were added but matched no expected domain
        false_expansion_rate:    false_expansions / files_added (0.0 = clean expansion)
        net_useful_files:        files added that contributed at least one domain
        added_file_evaluations:  per-file breakdown
        expansion_verdict:       "effective" | "partial" | "neutral" | "harmful"
    """
    was_triggered: bool
    triggers_fired: list[str]
    files_added: int
    domains_before: int
    domains_after: int
    completeness_before: float
    completeness_after: float
    completeness_delta: float
    recovered_domains: list[str]
    false_expansions: list[str]
    false_expansion_rate: float
    net_useful_files: int
    added_file_evaluations: list[AddedFileEvaluation]
    expansion_verdict: str  # "effective" | "partial" | "neutral" | "harmful"

    def summary_line(self) -> str:
        return (
            f"expansion={self.expansion_verdict} | "
            f"Δcompleteness={self.completeness_delta:+.0%} | "
            f"files_added={self.files_added} "
            f"(false_rate={self.false_expansion_rate:.0%}) | "
            f"recovered={self.recovered_domains}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_blob_set(shortlist: list[dict]) -> set[str]:
    return {
        e.get("blob_name") or e.get("file_name") or e.get("id", "")
        for e in shortlist
    }


def _file_satisfies_any_domain(
    entry: dict,
    benchmark: WorkflowBenchmark,
) -> list[str]:
    """Return list of domain short_names satisfied by this file."""
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

    return list(set(satisfied))


def _extract_triggers_fired(expansion: Any) -> list[str]:
    """Extract which triggers fired from ExpansionDecision."""
    if expansion is None:
        return []
    try:
        return list(expansion.triggers_fired or [])
    except Exception:
        return []


# ── Main validator ────────────────────────────────────────────────────────────

def validate_expansion_effectiveness(
    benchmark: WorkflowBenchmark,
    capture: WorkflowEvalCapture,
) -> ExpansionEffectivenessReport:
    """
    Compare workflow completeness before and after adaptive expansion.

    Requires capture.pre_expansion_shortlist to be populated.
    If not populated, assumes no expansion occurred.

    Args:
        benchmark: the benchmark being evaluated
        capture:   captured pipeline state (needs pre_expansion_shortlist)

    Returns:
        ExpansionEffectivenessReport with full before/after analysis
    """
    pre_shortlist = capture.pre_expansion_shortlist or capture.shortlist
    post_shortlist = capture.shortlist

    pre_blobs = _get_blob_set(pre_shortlist)
    post_blobs = _get_blob_set(post_shortlist)
    added_blobs = post_blobs - pre_blobs

    was_triggered = len(added_blobs) > 0
    triggers_fired = _extract_triggers_fired(capture.expansion)

    # Score before and after
    score_before = score_workflow_completeness(
        benchmark=benchmark,
        shortlist=pre_shortlist,
        grounding_quality=capture.grounding_quality,
    )
    score_after = score_workflow_completeness(
        benchmark=benchmark,
        shortlist=post_shortlist,
        grounding_quality=capture.grounding_quality,
    )

    completeness_delta = score_after.workflow_completeness - score_before.workflow_completeness

    # Determine which domains were recovered by expansion
    before_satisfied = {
        d.requirement.short_name
        for d in score_before.domain_coverage
        if d.satisfied
    }
    after_satisfied = {
        d.requirement.short_name
        for d in score_after.domain_coverage
        if d.satisfied
    }
    recovered_domains = list(after_satisfied - before_satisfied)

    # Evaluate each added file
    added_file_evals: list[AddedFileEvaluation] = []
    false_expansions: list[str] = []

    # Look up added files in the full catalog
    catalog_lookup = {
        e.get("blob_name") or e.get("file_name") or e.get("id", ""): e
        for e in (capture.full_catalog or [])
    }

    for blob in added_blobs:
        entry = catalog_lookup.get(blob, {})
        contributed = _file_satisfies_any_domain(entry, benchmark)
        is_useful = len(contributed) > 0
        is_false = not is_useful

        if is_false:
            false_expansions.append(blob)

        added_file_evals.append(AddedFileEvaluation(
            blob_name=blob,
            contributed_domains=contributed,
            is_useful=is_useful,
            is_false_expansion=is_false,
        ))

    files_added = len(added_blobs)
    net_useful = sum(1 for e in added_file_evals if e.is_useful)
    false_rate = len(false_expansions) / files_added if files_added > 0 else 0.0

    # Determine verdict
    if not was_triggered:
        verdict = "neutral"
    elif completeness_delta > 0.10:
        verdict = "effective"
    elif completeness_delta > 0.0:
        verdict = "partial"
    elif completeness_delta < 0.0:
        verdict = "harmful"  # shouldn't happen but detect if it does
    else:
        verdict = "neutral"

    return ExpansionEffectivenessReport(
        was_triggered=was_triggered,
        triggers_fired=triggers_fired,
        files_added=files_added,
        domains_before=score_before.satisfied_required_count,
        domains_after=score_after.satisfied_required_count,
        completeness_before=score_before.workflow_completeness,
        completeness_after=score_after.workflow_completeness,
        completeness_delta=completeness_delta,
        recovered_domains=recovered_domains,
        false_expansions=false_expansions,
        false_expansion_rate=false_rate,
        net_useful_files=net_useful,
        added_file_evaluations=added_file_evals,
        expansion_verdict=verdict,
    )

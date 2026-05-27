"""Adaptive Semantic Expansion — targeted shortlist expansion to cover missing domains.

PURPOSE:
  Decide whether, and how much, to expand the current shortlist beyond the initial
  top-K retrieval result. This is NOT a global top-K increase — it is TARGETED
  domain-filling: add exactly the files needed to cover missing semantic domains.

EXPANSION TRIGGERS (evaluated in priority order):
  1. Missing semantic domains (from WorkflowRequirements.missing_domains)
  2. Retrieval result set too small to cover multi-step query behaviors
  3. Fallback retrieval was used (grounding_quality signals degraded retrieval)

EXPANSION BOUNDS:
  _MAX_EXPANSION_SLOTS = 5  additional files max per request
  _HARD_MAX_SHORTLIST  = 15 total shortlist ceiling (prevents prompt overflow)
  Each expansion file must have a traceable evidence chain (no silent additions).

FILE RANKING within a missing domain:
  Combined score = (role_type_weight * 0.5) + (kw_score * 0.3) + (label_bonus * 0.20)
  role_type_weight: entity=1.0, transaction=0.8, dimension=0.6, unknown=0.5
  kw_score: keyword match against blob_path + column_names + ai_description
  label_bonus: how well the file's blob_path/description matches the domain_label

DESIGN CONSTRAINTS:
  - Zero LLM calls. Zero DB queries. Pure in-memory computation.
  - Operates on already-fetched full_catalog.
  - Produces an ExpansionDecision — caller decides whether to apply it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logger import pipeline_logger

_MAX_EXPANSION_SLOTS = 5
_HARD_MAX_SHORTLIST = 15

_ROLE_TYPE_WEIGHT: dict[str, float] = {
    "entity": 1.0,
    "transaction": 0.8,
    "dimension": 0.6,
}


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class ExpansionCandidate:
    """One file candidate for shortlist expansion."""
    file_id: str
    blob_path: str
    domain_label: str          # which missing domain this covers
    role_type: str             # entity / transaction / dimension
    evidence: str              # why this file is being added
    confidence: float          # composite score [0.0, 1.0]


@dataclass
class ExpansionDecision:
    """
    Decision on whether and how to expand the shortlist.

    If should_expand is False, the shortlist is used as-is.
    If True, expansion_candidates should be appended after deduplication.
    """
    should_expand: bool
    expansion_candidates: list[ExpansionCandidate]
    triggers: list[str]
    pre_expansion_size: int
    post_expansion_size: int
    workflow_completeness_before: float
    workflow_completeness_after: float

    def to_dict(self) -> dict:
        return {
            "should_expand": self.should_expand,
            "candidates_count": len(self.expansion_candidates),
            "triggers": self.triggers,
            "pre_size": self.pre_expansion_size,
            "post_size": self.post_expansion_size,
            "completeness_before": round(self.workflow_completeness_before, 3),
            "completeness_after": round(self.workflow_completeness_after, 3),
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def decide_expansion(
    workflow_reqs: Any,                   # WorkflowRequirements from workflow_capability_resolver
    exec_strategy: Any,                   # ExecutionStrategy (may be None pre-computation)
    intent_plan: Any,                     # BusinessIntentPlan
    confidence: Any,                      # ConfidenceScore (may be None pre-computation)
    current_shortlist: list[dict],
    full_catalog: list[dict],
    query_words: list[str],
    retrieval_channels: dict[str, list[str]] | None = None,
) -> ExpansionDecision:
    """
    Decide whether to expand the shortlist and which files to add.

    Expansion is triggered when:
      1. WorkflowRequirements signals missing semantic domains that have candidates
      2. Multi-step behaviors in intent_plan AND shortlist is small
      3. Retrieval grounding was degraded (fallback used)

    Returns an ExpansionDecision — the caller applies it or ignores it.
    """
    triggers: list[str] = []
    candidates: list[ExpansionCandidate] = []
    current_size = len(current_shortlist)
    current_ids: set[str] = {e.get("file_id") for e in current_shortlist if e.get("file_id")}
    full_by_id: dict[str, dict] = {
        e["file_id"]: e for e in full_catalog if e.get("file_id")
    }
    retrieval_channels = retrieval_channels or {}

    # Hard max guard: if shortlist already at ceiling, no expansion
    if current_size >= _HARD_MAX_SHORTLIST:
        return _no_expansion(current_size, getattr(workflow_reqs, "workflow_completeness", 1.0))

    available_slots = min(_MAX_EXPANSION_SLOTS, _HARD_MAX_SHORTLIST - current_size)

    # ── Trigger 1: Missing semantic domains ───────────────────────────────────
    if getattr(workflow_reqs, "expansion_needed", False):
        coverage_state = getattr(workflow_reqs, "coverage_state", "unknown")
        if getattr(workflow_reqs, "missing_domains", None):
            triggers.append("missing_semantic_domains")
        elif coverage_state == "activation_failed":
            triggers.append("activation_failed_recovery")

        ranked_by_domain: list[tuple[Any, list[tuple[str, float]]]] = []
        for domain in (getattr(workflow_reqs, "missing_domains", None) or []):
            best = domain.best_candidates
            if not best:
                continue
            ranked = _rank_domain_candidates(
                best, domain.domain_label, domain.role_type,
                full_catalog, query_words, retrieval_channels=retrieval_channels,
            )
            if ranked:
                ranked_by_domain.append((domain, ranked))

        # Breadth first: one best candidate per missing domain before spending
        # redundant slots on second-best candidates.
        for rank_index in (0, 1):
            for domain, ranked in ranked_by_domain:
                if len(candidates) >= available_slots or rank_index >= len(ranked):
                    continue
                fid, score = ranked[rank_index]
                if fid in current_ids:
                    continue
                entry = full_by_id.get(fid, {})
                candidates.append(ExpansionCandidate(
                    file_id=fid,
                    blob_path=entry.get("blob_path", fid),
                    domain_label=domain.domain_label,
                    role_type=domain.role_type,
                    evidence=f"missing_domain:{domain.domain_label}(activated_by:{domain.activated_by})",
                    confidence=min(score, 0.95),
                ))
                current_ids.add(fid)

        # Activation failure is not allowed to become silent no-op health. If
        # no domains could be activated, recover a bounded set of semantically
        # surfaced, role-bearing files so the planner receives inspectable state.
        if coverage_state == "activation_failed" and len(candidates) < available_slots:
            ranked = _rank_activation_failure_candidates(
                full_catalog=full_catalog,
                current_ids=current_ids,
                query_words=query_words,
                retrieval_channels=retrieval_channels,
            )
            for fid, score in ranked[: available_slots - len(candidates)]:
                if fid in current_ids:
                    continue
                entry = full_by_id.get(fid, {})
                candidates.append(ExpansionCandidate(
                    file_id=fid,
                    blob_path=entry.get("blob_path", fid),
                    domain_label="activation_recovery",
                    role_type="unknown",
                    evidence="activation_failed_semantic_recovery",
                    confidence=min(score, 0.75),
                ))
                current_ids.add(fid)

    # ── Trigger 2: Multi-step behaviors with thin shortlist ───────────────────
    behaviors = getattr(intent_plan, "behaviors", []) or []
    is_workflow_query = any(b in behaviors for b in ("open_items", "multi_step", "time_filtered"))
    if is_workflow_query and current_size < 3 and len(candidates) < available_slots:
        triggers.append("multi_step_thin_shortlist")
        # Add entity-type files from any detected domain that aren't already in candidates
        for domain in (getattr(workflow_reqs, "all_detected_domains", None) or []):
            if domain.role_type != "entity":
                continue
            for fid in domain.best_candidates:
                if fid in current_ids or len(candidates) >= available_slots:
                    break
                entry = full_by_id.get(fid, {})
                candidates.append(ExpansionCandidate(
                    file_id=fid,
                    blob_path=entry.get("blob_path", fid),
                    domain_label=domain.domain_label,
                    role_type="entity",
                    evidence="multi_step_entity_bridge",
                    confidence=0.55,
                ))
                current_ids.add(fid)

    if not candidates or not triggers:
        if getattr(workflow_reqs, "coverage_state", "unknown") == "activation_failed":
            return ExpansionDecision(
                should_expand=False,
                expansion_candidates=[],
                triggers=triggers or ["activation_failed_no_candidates"],
                pre_expansion_size=current_size,
                post_expansion_size=current_size,
                workflow_completeness_before=getattr(workflow_reqs, "workflow_completeness", 0.0),
                workflow_completeness_after=getattr(workflow_reqs, "workflow_completeness", 0.0),
            )
        return _no_expansion(current_size, getattr(workflow_reqs, "workflow_completeness", 1.0))

    # ── Compute post-expansion workflow completeness ───────────────────────────
    expanded_ids = current_ids | {c.file_id for c in candidates}
    all_domains = getattr(workflow_reqs, "all_detected_domains", []) or []
    if all_domains:
        covered_after = sum(
            1 for d in all_domains
            if any(fid in expanded_ids for fid in d.all_file_ids)
        )
        completeness_after = covered_after / len(all_domains)
    else:
        completeness_after = getattr(workflow_reqs, "workflow_completeness", 1.0)

    pipeline_logger.info(
        "adaptive_expansion_decision",
        triggers=triggers,
        candidates=[c.file_id[:8] for c in candidates],
        pre_size=current_size,
        post_size=current_size + len(candidates),
        completeness_before=round(getattr(workflow_reqs, "workflow_completeness", 1.0), 3),
        completeness_after=round(completeness_after, 3),
    )

    return ExpansionDecision(
        should_expand=True,
        expansion_candidates=candidates,
        triggers=triggers,
        pre_expansion_size=current_size,
        post_expansion_size=current_size + len(candidates),
        workflow_completeness_before=getattr(workflow_reqs, "workflow_completeness", 1.0),
        workflow_completeness_after=completeness_after,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _no_expansion(current_size: int, completeness: float) -> ExpansionDecision:
    return ExpansionDecision(
        should_expand=False,
        expansion_candidates=[],
        triggers=["no_expansion_needed"],
        pre_expansion_size=current_size,
        post_expansion_size=current_size,
        workflow_completeness_before=completeness,
        workflow_completeness_after=completeness,
    )


def _rank_domain_candidates(
    file_ids: list[str],
    domain_label: str,
    role_type: str,
    full_catalog: list[dict],
    query_words: list[str],
    retrieval_channels: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    """
    Rank candidate files by composite score:
      role_type_weight * 0.5 + kw_score * 0.3 + label_bonus * 0.20
    """
    role_weight = _ROLE_TYPE_WEIGHT.get(role_type, 0.5)
    fid_set = set(file_ids)
    label_tokens = set(t for t in re.split(r"[^a-z0-9]+", domain_label.lower()) if t)
    retrieval_channels = retrieval_channels or {}

    results: list[tuple[str, float]] = []
    for entry in full_catalog:
        fid = entry.get("file_id")
        if fid not in fid_set:
            continue

        blob = (entry.get("blob_path") or "").lower()
        desc = (entry.get("ai_description") or "").lower()
        col_text = " ".join(
            (c.get("name", "") if isinstance(c, dict) else str(c))
            for c in (entry.get("column_names") or [])
        ).lower()

        # Keyword score
        kw_score = 0.0
        for w in query_words:
            if w in blob:
                kw_score += 0.3
            if w in col_text:
                kw_score += 0.2
            if w in desc:
                kw_score += 0.1
        kw_score = min(kw_score, 1.0)

        # Domain label match bonus
        blob_tokens = set(t for t in re.split(r"[^a-z0-9]+", blob) if t)
        label_hits = len(label_tokens & blob_tokens)
        label_bonus = min(label_hits * 0.15, 0.30)

        channels = set(retrieval_channels.get(fid, []) or [])
        retrieval_bonus = 0.0
        if "vector" in channels or "opensearch" in channels:
            retrieval_bonus += 0.12
        if "graph" in channels:
            retrieval_bonus += 0.08

        score = (role_weight * 0.5) + (kw_score * 0.3) + label_bonus + retrieval_bonus
        results.append((fid, min(score, 0.95)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _rank_activation_failure_candidates(
    full_catalog: list[dict],
    current_ids: set[str],
    query_words: list[str],
    retrieval_channels: dict[str, list[str]],
) -> list[tuple[str, float]]:
    """Bounded recovery ranking for activation_failed state."""
    results: list[tuple[str, float]] = []
    for entry in full_catalog:
        fid = entry.get("file_id")
        if not fid or fid in current_ids:
            continue
        roles = entry.get("column_semantic_roles") or {}
        if not roles:
            continue
        channels = set(retrieval_channels.get(fid, []) or [])
        semantic_bonus = 0.0
        if "vector" in channels or "opensearch" in channels:
            semantic_bonus += 0.45
        if "graph" in channels:
            semantic_bonus += 0.25

        blob = (entry.get("blob_path") or "").lower()
        desc = (entry.get("ai_description") or "").lower()
        role_text = " ".join(str(v).lower() for v in roles.values())
        kw_score = 0.0
        for w in query_words:
            if w in blob:
                kw_score += 0.20
            if w in desc:
                kw_score += 0.10
            if w in role_text:
                kw_score += 0.20
        role_density = min(len(roles) / 8.0, 0.30)
        score = min(semantic_bonus + min(kw_score, 0.50) + role_density, 0.95)
        if score > 0:
            results.append((fid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def render_workflow_continuity_note(
    workflow_reqs: Any,
    expansion: ExpansionDecision | None,
    full_catalog: list[dict],
    *,
    max_domains: int = 8,
    max_candidates: int = 8,
) -> str:
    """Render only prompt-critical workflow coverage state.

    Expansion candidates and domain-filling diagnostics are observability data;
    callers keep the full ExpansionDecision for trace/logs instead of injecting
    the candidate list into the LLM context.
    """
    state = getattr(workflow_reqs, "coverage_state", "unknown")
    if state == "complete" and not getattr(workflow_reqs, "missing_domains", None):
        return ""

    lines = ["--- WORKFLOW COVERAGE ---"]
    lines.append(
        f"coverage_state={state}; workflow_completeness={getattr(workflow_reqs, 'workflow_completeness', 0.0):.2f}"
    )

    missing = list(getattr(workflow_reqs, "missing_domains", None) or [])[:max_domains]
    if missing:
        lines.append("missing_domains:")
        for domain in missing:
            lines.append(
                f"  - {domain.role_type}:{domain.domain_label}; activated_by={domain.activated_by}"
            )
    elif state in {"activation_failed", "unknown"}:
        evidence = ", ".join(getattr(workflow_reqs, "expansion_evidence", [])[:3])
        lines.append(f"activation_state={evidence or state}")

    if expansion and expansion.expansion_candidates:
        lines.append(f"expansion_applied={len(expansion.expansion_candidates)} file(s); candidate details kept in trace.")

    lines.append("---")
    return "\n".join(lines)

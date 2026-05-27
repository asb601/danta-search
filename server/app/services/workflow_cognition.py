"""Query-time workflow cognition assembly.

This module builds lightweight workflow semantics from existing catalog
metadata at request time. It does not create an ERP ontology and it does not
persist enterprise intelligence at ingestion time. It composes primitives that
already exist in the catalog into workflow tasks, candidate scores, temporal
eligibility, authority signals, and decision-trace records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.retrieval.temporal import parse_temporal


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")
_NUMERIC_PREFIX_RE = re.compile(r"^(?:.*/)?\d{2}[_-]", re.IGNORECASE)

_OPERATIONAL_TERMS = frozenset({
    "approval", "approved", "authorization", "authorisation", "pending",
    "status", "state", "delivery", "delivered", "fulfillment", "fulfilment",
    "shipment", "shipping", "receiving", "receipt", "received", "match",
    "matching", "matched", "unmatched", "reconciliation", "reconcile",
    "discrepancy", "variance", "invoice", "invoiced", "uninvoiced",
    "payment", "paid", "liability", "liabilities", "exposure", "open",
    "bottleneck", "delay", "delayed", "lifecycle", "exception", "issue",
})

_ANALYTICAL_TERMS = frozenset({
    "summary", "summarize", "summarise", "trend", "dashboard", "report",
    "aggregate", "aggregation", "group", "grouped", "top", "kpi", "metric",
    "analytics", "analysis",
})

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "show", "give",
    "list", "get", "all", "what", "where", "how", "when", "who", "which",
    "are", "was", "were", "have", "has", "had", "can", "will", "would",
    "current", "next", "details", "detail", "year", "years", "analyze",
    "analyse", "analysis", "summarize", "summarise", "recommended",
})


@dataclass(frozen=True)
class TemporalEligibility:
    """Temporal fit between the query window and a candidate file."""

    status: str
    score: float
    query_start: str | None = None
    query_end: str | None = None
    candidate_start: str | None = None
    candidate_end: str | None = None
    allowed_primary: bool = True
    allowed_secondary: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": round(self.score, 3),
            "query_start": self.query_start,
            "query_end": self.query_end,
            "candidate_start": self.candidate_start,
            "candidate_end": self.candidate_end,
            "allowed_primary": self.allowed_primary,
            "allowed_secondary": self.allowed_secondary,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TransactionalAuthority:
    """Authority profile for operational workflow reasoning."""

    source_type: str
    score: float
    transformation_level: str
    workflow_grain: str
    evidence: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "score": round(self.score, 3),
            "transformation_level": self.transformation_level,
            "workflow_grain": self.workflow_grain,
            "evidence": self.evidence[:6],
            "penalties": self.penalties[:6],
        }


@dataclass(frozen=True)
class WorkflowSemanticPrimitive:
    """Workflow building blocks inferred from one catalog entry."""

    file_id: str
    blob_path: str
    business_objects: list[str]
    process_signals: list[str]
    workflow_hints: list[str]
    operational_domains: list[str]
    semantic_role_labels: list[str]
    role_kinds: list[str]
    temporal_columns: list[str]
    workflow_grain: str
    transactional_authority: TransactionalAuthority

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id[:8],
            "file": self.blob_path,
            "business_objects": self.business_objects[:8],
            "process_signals": self.process_signals[:8],
            "workflow_hints": self.workflow_hints[:8],
            "operational_domains": self.operational_domains[:6],
            "semantic_role_labels": self.semantic_role_labels[:8],
            "role_kinds": self.role_kinds[:8],
            "temporal_columns": self.temporal_columns[:8],
            "workflow_grain": self.workflow_grain,
            "transactional_authority": self.transactional_authority.to_dict(),
        }


@dataclass(frozen=True)
class WorkflowTask:
    """One query-time workflow capability to validate independently."""

    task_id: str
    label: str
    required_signals: list[str]
    target_business_objects: list[str]
    target_operational_domains: list[str]
    lifecycle_role: str
    weight: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "required_signals": self.required_signals,
            "target_business_objects": self.target_business_objects,
            "target_operational_domains": self.target_operational_domains,
            "lifecycle_role": self.lifecycle_role,
            "weight": round(self.weight, 3),
            "evidence": self.evidence[:6],
        }


@dataclass(frozen=True)
class WorkflowCandidateDecision:
    """Workflow-aware candidate decision for one file."""

    file_id: str
    blob_path: str
    selected: bool
    score: float
    best_task_id: str | None
    score_components: dict[str, float]
    temporal_eligibility: TemporalEligibility
    transactional_authority: TransactionalAuthority
    workflow_fit: float
    business_object_compatibility: float
    process_continuity: float
    lifecycle_relevance: float
    transformed_object_penalty: float
    rejection_reasons: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id[:8],
            "file": self.blob_path,
            "selected": self.selected,
            "score": round(self.score, 3),
            "best_task_id": self.best_task_id,
            "score_components": {
                key: round(float(value), 3)
                for key, value in self.score_components.items()
            },
            "temporal_eligibility": self.temporal_eligibility.to_dict(),
            "transactional_authority": self.transactional_authority.to_dict(),
            "workflow_fit": round(self.workflow_fit, 3),
            "business_object_compatibility": round(self.business_object_compatibility, 3),
            "process_continuity": round(self.process_continuity, 3),
            "lifecycle_relevance": round(self.lifecycle_relevance, 3),
            "transformed_object_penalty": round(self.transformed_object_penalty, 3),
            "rejection_reasons": self.rejection_reasons[:6],
            "rationale": self.rationale[:6],
        }


@dataclass(frozen=True)
class WorkflowAssemblyResult:
    """Output of query-time workflow cognition assembly."""

    workflow_query: bool
    tasks: list[WorkflowTask]
    query_business_objects: list[str]
    query_operational_domains: list[str]
    query_process_signals: list[str]
    query_temporal_window: tuple[str | None, str | None]
    decisions: list[WorkflowCandidateDecision]
    ranked_shortlist: list[dict]
    warnings: list[str]
    summary: dict[str, Any]

    def to_trace_dict(self) -> dict[str, Any]:
        selected = [decision.to_dict() for decision in self.decisions if decision.selected]
        rejected = [decision.to_dict() for decision in self.decisions if not decision.selected]
        return {
            "workflow_query": self.workflow_query,
            "tasks": [task.to_dict() for task in self.tasks[:10]],
            "query_business_objects": self.query_business_objects,
            "query_operational_domains": self.query_operational_domains,
            "query_process_signals": self.query_process_signals,
            "query_temporal_window": {
                "start": self.query_temporal_window[0],
                "end": self.query_temporal_window[1],
            },
            "selected_candidates": selected[:12],
            "rejected_candidates": rejected[:12],
            "warnings": self.warnings[:12],
            "summary": self.summary,
        }


def assemble_workflow_cognition(
    *,
    query: str,
    intent_plan: Any,
    current_shortlist: list[dict],
    full_catalog: list[dict],
    grounding_quality: str = "",
) -> WorkflowAssemblyResult:
    """Assemble workflow semantics at query time and rank the shortlist."""
    query_tokens = _tokens(query)
    query_process_signals = _infer_process_signals(" ".join(query_tokens))
    query_business_objects = _infer_business_objects(" ".join(query_tokens), query_tokens)
    query_operational_domains = _infer_operational_domains(" ".join(query_tokens), query_tokens)
    query_start, query_end = parse_temporal(query)
    tasks = decompose_workflow_tasks(
        query=query,
        intent_plan=intent_plan,
        query_tokens=query_tokens,
        query_business_objects=query_business_objects,
        query_operational_domains=query_operational_domains,
        query_process_signals=query_process_signals,
    )
    workflow_query = bool(tasks) or _is_workflow_like(query_tokens, intent_plan)

    if not workflow_query:
        return WorkflowAssemblyResult(
            workflow_query=False,
            tasks=[],
            query_business_objects=query_business_objects,
            query_operational_domains=query_operational_domains,
            query_process_signals=query_process_signals,
            query_temporal_window=(_date_str(query_start), _date_str(query_end)),
            decisions=[],
            ranked_shortlist=current_shortlist,
            warnings=[],
            summary={"reason": "no_workflow_signals"},
        )

    decisions: list[WorkflowCandidateDecision] = []
    for entry in current_shortlist:
        primitive = infer_workflow_primitives(entry)
        temporal = classify_temporal_eligibility(entry, query_start, query_end)
        decision = score_workflow_candidate(
            primitive=primitive,
            temporal=temporal,
            tasks=tasks,
            query_business_objects=query_business_objects,
            query_operational_domains=query_operational_domains,
            query_process_signals=query_process_signals,
            analytical_intent=_has_analytical_intent(query_tokens, intent_plan),
            degraded_retrieval=bool(grounding_quality and grounding_quality != "retrieved"),
        )
        decisions.append(decision)

    decision_by_id = {decision.file_id: decision for decision in decisions}
    ranked_shortlist = sorted(
        current_shortlist,
        key=lambda entry: (
            decision_by_id.get(str(entry.get("file_id") or ""), _empty_decision()).score,
            -current_shortlist.index(entry),
        ),
        reverse=True,
    )
    ranked_decisions = sorted(decisions, key=lambda decision: decision.score, reverse=True)
    warnings = _build_workflow_warnings(ranked_decisions)
    selected_count = sum(1 for decision in ranked_decisions if decision.selected)
    out_of_window_count = sum(
        1 for decision in ranked_decisions
        if decision.temporal_eligibility.status == "outside_window"
    )
    transformed_primary_count = sum(
        1 for decision in ranked_decisions
        if decision.selected
        and decision.transactional_authority.source_type == "transformed_analytics"
    )
    summary = {
        "task_count": len(tasks),
        "candidate_count": len(decisions),
        "selected_count": selected_count,
        "rejected_count": len(decisions) - selected_count,
        "out_of_window_count": out_of_window_count,
        "transformed_primary_count": transformed_primary_count,
        "grounding_quality": grounding_quality,
    }

    return WorkflowAssemblyResult(
        workflow_query=True,
        tasks=tasks,
        query_business_objects=query_business_objects,
        query_operational_domains=query_operational_domains,
        query_process_signals=query_process_signals,
        query_temporal_window=(_date_str(query_start), _date_str(query_end)),
        decisions=ranked_decisions,
        ranked_shortlist=ranked_shortlist,
        warnings=warnings,
        summary=summary,
    )


def infer_workflow_primitives(entry: dict) -> WorkflowSemanticPrimitive:
    """Infer lightweight workflow primitives from one catalog entry."""
    file_id = str(entry.get("file_id") or "")
    blob_path = str(entry.get("blob_path") or file_id)
    text = _catalog_text(entry)
    tokens = _tokens(text)
    role_kinds, role_labels = _semantic_role_parts(entry)
    role_text = " ".join(role_labels)
    combined_text = " ".join([text, role_text])
    combined_tokens = _tokens(combined_text)

    business_objects = _infer_business_objects(combined_text, combined_tokens)
    process_signals = _infer_process_signals(combined_text)
    operational_domains = _infer_operational_domains(combined_text, combined_tokens)
    temporal_columns = _infer_temporal_columns(entry)
    workflow_grain = _infer_workflow_grain(combined_text, combined_tokens, role_labels)
    authority = infer_transactional_authority(
        entry=entry,
        text=combined_text,
        tokens=combined_tokens,
        role_kinds=role_kinds,
        process_signals=process_signals,
        workflow_grain=workflow_grain,
    )
    hints = sorted(set(process_signals + business_objects + operational_domains))[:12]

    return WorkflowSemanticPrimitive(
        file_id=file_id,
        blob_path=blob_path,
        business_objects=business_objects,
        process_signals=process_signals,
        workflow_hints=hints,
        operational_domains=operational_domains,
        semantic_role_labels=role_labels,
        role_kinds=role_kinds,
        temporal_columns=temporal_columns,
        workflow_grain=workflow_grain,
        transactional_authority=authority,
    )


def classify_temporal_eligibility(
    entry: dict,
    query_start: date | None,
    query_end: date | None,
) -> TemporalEligibility:
    """Classify candidate temporal fit for a query window."""
    candidate_start = _parse_catalog_date(entry.get("date_range_start"))
    candidate_end = _parse_catalog_date(entry.get("date_range_end"))

    if not query_start or not query_end:
        return TemporalEligibility(
            status="no_query_window",
            score=1.0,
            candidate_start=_date_str(candidate_start),
            candidate_end=_date_str(candidate_end),
            reason="query_has_no_temporal_constraint",
        )

    query_start_s = _date_str(query_start)
    query_end_s = _date_str(query_end)
    if not candidate_start or not candidate_end:
        return TemporalEligibility(
            status="temporal_unknown",
            score=0.62,
            query_start=query_start_s,
            query_end=query_end_s,
            candidate_start=_date_str(candidate_start),
            candidate_end=_date_str(candidate_end),
            allowed_primary=True,
            allowed_secondary=True,
            reason="candidate_has_no_date_range_metadata",
        )

    if candidate_end < query_start or candidate_start > query_end:
        return TemporalEligibility(
            status="outside_window",
            score=0.05,
            query_start=query_start_s,
            query_end=query_end_s,
            candidate_start=_date_str(candidate_start),
            candidate_end=_date_str(candidate_end),
            allowed_primary=False,
            allowed_secondary=False,
            reason="candidate_date_range_does_not_overlap_query_window",
        )

    if candidate_start <= query_start and candidate_end >= query_end:
        return TemporalEligibility(
            status="contains_window",
            score=1.0,
            query_start=query_start_s,
            query_end=query_end_s,
            candidate_start=_date_str(candidate_start),
            candidate_end=_date_str(candidate_end),
            reason="candidate_range_contains_query_window",
        )

    if candidate_start >= query_start and candidate_end <= query_end:
        return TemporalEligibility(
            status="inside_window",
            score=0.95,
            query_start=query_start_s,
            query_end=query_end_s,
            candidate_start=_date_str(candidate_start),
            candidate_end=_date_str(candidate_end),
            reason="candidate_range_is_inside_query_window",
        )

    return TemporalEligibility(
        status="partial_overlap",
        score=0.58,
        query_start=query_start_s,
        query_end=query_end_s,
        candidate_start=_date_str(candidate_start),
        candidate_end=_date_str(candidate_end),
        allowed_primary=True,
        allowed_secondary=True,
        reason="candidate_range_partially_overlaps_query_window",
    )


def infer_transactional_authority(
    *,
    entry: dict,
    text: str,
    tokens: list[str],
    role_kinds: list[str],
    process_signals: list[str],
    workflow_grain: str,
) -> TransactionalAuthority:
    """Score whether a file is authoritative for operational workflow state."""
    token_set = set(tokens)
    blob_path = str(entry.get("blob_path") or "").lower()
    good_for_text = " ".join(str(value).lower() for value in (entry.get("good_for") or []))
    evidence: list[str] = []
    penalties: list[str] = []
    score = 0.50
    source_type = "unknown"
    transformation_level = "unknown"

    role_density = min(len(role_kinds) / 12.0, 1.0)
    if role_density >= 0.25:
        score += 0.12 * role_density
        evidence.append("semantic_role_density")

    transactional_role_count = sum(
        1 for kind in role_kinds
        if kind in {"reference_key", "additive_measure", "non_additive_measure", "date"}
    )
    if transactional_role_count:
        score += min(transactional_role_count / 20.0, 0.16)
        evidence.append("transactional_role_kinds")

    if set(process_signals) & {
        "approval", "receiving", "delivery", "invoice", "reconciliation",
        "payment", "fulfillment", "liability", "lifecycle",
    }:
        score += 0.12
        evidence.append("workflow_process_signals")

    if any(term in token_set for term in {"master", "lookup", "dimension", "reference"}) or any(
        term in good_for_text for term in ("master", "lookup", "dimension")
    ):
        source_type = "reference_master"
        transformation_level = "reference"
        score -= 0.18
        penalties.append("reference_or_master_data_not_primary_workflow_state")

    if any(term in token_set for term in {"summary", "aggregate", "analytics", "dashboard", "report", "snapshot"}):
        source_type = "transformed_analytics"
        transformation_level = "transformed"
        score -= 0.24
        penalties.append("aggregate_or_analytics_surface")

    if _NUMERIC_PREFIX_RE.search(blob_path):
        source_type = "transformed_analytics"
        transformation_level = "curated_extract"
        score -= 0.18
        penalties.append("curated_numeric_prefix_surface")

    if "sample" in token_set or blob_path.endswith(".sample"):
        if source_type == "unknown":
            source_type = "sample_extract"
        transformation_level = "sample"
        score -= 0.10
        penalties.append("sample_extract")

    if source_type == "unknown" and (transactional_role_count or workflow_grain in {"line", "header", "distribution", "schedule"}):
        source_type = "transactional_source"
        transformation_level = "source_like"
        score += 0.10
        evidence.append("source_like_transactional_grain")

    if source_type == "unknown":
        source_type = "source_like_extract" if role_density > 0 else "unknown"
        transformation_level = "source_like" if role_density > 0 else "unknown"

    if source_type in {"transactional_source", "source_like_extract"}:
        score += 0.08
        evidence.append("preferred_operational_authority")

    return TransactionalAuthority(
        source_type=source_type,
        score=_clamp(score),
        transformation_level=transformation_level,
        workflow_grain=workflow_grain,
        evidence=evidence,
        penalties=penalties,
    )


def decompose_workflow_tasks(
    *,
    query: str,
    intent_plan: Any,
    query_tokens: list[str],
    query_business_objects: list[str],
    query_operational_domains: list[str],
    query_process_signals: list[str],
) -> list[WorkflowTask]:
    """Decompose workflow-like query language into bounded capabilities."""
    token_set = set(query_tokens)
    tasks: list[WorkflowTask] = []

    def add_task(
        task_id: str,
        label: str,
        required_signals: list[str],
        lifecycle_role: str,
        weight: float,
        evidence: str,
        *,
        target_business_objects: list[str] | None = None,
        target_operational_domains: list[str] | None = None,
    ) -> None:
        if any(task.task_id == task_id for task in tasks):
            return
        tasks.append(WorkflowTask(
            task_id=task_id,
            label=label,
            required_signals=required_signals,
            target_business_objects=target_business_objects or query_business_objects,
            target_operational_domains=target_operational_domains or query_operational_domains,
            lifecycle_role=lifecycle_role,
            weight=weight,
            evidence=[evidence],
        ))

    if {"approval", "approved", "approvals", "authorization", "authorisation", "pending"} & token_set:
        add_task(
            "authorization_state",
            "authorization and approval state",
            ["approval", "status"],
            "approval_state",
            1.0,
            "query_mentions_approval_or_pending_state",
        )

    if {"receive", "receiving", "receipt", "receipts", "received", "grn"} & token_set:
        add_task(
            "receiving_progress",
            "receiving progress and receipt evidence",
            ["receiving", "status"],
            "receiving_state",
            1.0,
            "query_mentions_receiving_or_receipts",
            target_business_objects=_prefer_objects(query_business_objects, ["goods_receipt", "purchase_order"]),
            target_operational_domains=_prefer_domains(query_operational_domains, ["procurement", "accounts_payable"]),
        )

    if {"delivery", "delivered", "fulfillment", "fulfilment", "shipment", "shipping", "delayed", "delay"} & token_set:
        add_task(
            "fulfillment_state",
            "fulfillment and delivery state",
            ["delivery", "fulfillment", "status"],
            "fulfillment_state",
            0.9,
            "query_mentions_delivery_or_fulfillment",
        )

    if {"invoice", "invoices", "invoiced", "uninvoiced", "matching", "matched", "unmatched", "match", "reconciliation"} & token_set:
        add_task(
            "invoice_reconciliation",
            "invoice reconciliation and matching integrity",
            ["invoice", "reconciliation"],
            "matching_state",
            1.0,
            "query_mentions_invoice_matching_or_reconciliation",
            target_business_objects=_prefer_objects(query_business_objects, ["invoice", "purchase_order", "goods_receipt"]),
            target_operational_domains=_prefer_domains(query_operational_domains, ["accounts_payable", "procurement"]),
        )

    if {"liability", "liabilities", "exposure", "open", "unpaid", "accrual", "accrued"} & token_set:
        add_task(
            "liability_exposure",
            "open liability and exposure state",
            ["liability", "payment", "invoice", "status"],
            "financial_exposure",
            0.95,
            "query_mentions_open_liability_or_exposure",
            target_operational_domains=_prefer_domains(query_operational_domains, ["accounts_payable", "procurement", "general_ledger"]),
        )

    if {"bottleneck", "bottlenecks", "lifecycle", "delay", "delayed", "exception", "exceptions", "issue", "issues", "discrepancy", "discrepancies"} & token_set:
        add_task(
            "operational_exception",
            "lifecycle bottleneck and exception analysis",
            ["lifecycle", "status", "reconciliation", "delivery", "receiving"],
            "exception_state",
            0.9,
            "query_mentions_bottlenecks_exceptions_or_discrepancies",
        )

    behaviors = set(getattr(intent_plan, "behaviors", []) or []) if intent_plan else set()
    if not tasks and (behaviors & {"open_items", "time_filtered", "multi_step"}):
        add_task(
            "state_assessment",
            "workflow state assessment",
            ["status", "lifecycle"],
            "state_assessment",
            0.75,
            "planner_detected_workflow_like_behavior",
        )

    return tasks


def score_workflow_candidate(
    *,
    primitive: WorkflowSemanticPrimitive,
    temporal: TemporalEligibility,
    tasks: list[WorkflowTask],
    query_business_objects: list[str],
    query_operational_domains: list[str],
    query_process_signals: list[str],
    analytical_intent: bool,
    degraded_retrieval: bool,
) -> WorkflowCandidateDecision:
    """Score a candidate against decomposed workflow tasks."""
    best_task_id: str | None = None
    best_task_fit = 0.0
    business_object_fit = _compatibility(query_business_objects, primitive.business_objects, neutral=0.55)
    domain_fit = _compatibility(query_operational_domains, primitive.operational_domains, neutral=0.58)
    lifecycle_fit = _signal_overlap(query_process_signals, primitive.process_signals, neutral=0.45)
    for task in tasks:
        signal_fit = _signal_overlap(task.required_signals, primitive.process_signals, neutral=0.20)
        task_object_fit = _compatibility(task.target_business_objects, primitive.business_objects, neutral=0.55)
        task_domain_fit = _compatibility(task.target_operational_domains, primitive.operational_domains, neutral=0.58)
        task_fit = ((signal_fit * 0.45) + (task_object_fit * 0.30) + (task_domain_fit * 0.25)) * task.weight
        if task_fit > best_task_fit:
            best_task_fit = task_fit
            best_task_id = task.task_id

    workflow_fit = best_task_fit if tasks else lifecycle_fit
    process_continuity = (business_object_fit * 0.40) + (domain_fit * 0.35) + (lifecycle_fit * 0.25)
    boundary_conflicts = _process_boundary_conflicts(
        query_business_objects=query_business_objects,
        query_operational_domains=query_operational_domains,
        primitive=primitive,
    )
    if boundary_conflicts:
        workflow_fit = min(workflow_fit, 0.36)
        process_continuity = min(process_continuity, 0.30)
    transformed_penalty = 0.0
    if primitive.transactional_authority.source_type == "transformed_analytics" and not analytical_intent:
        transformed_penalty = 0.22
    if primitive.transactional_authority.source_type == "reference_master" and workflow_fit > 0.35:
        transformed_penalty += 0.08
    if degraded_retrieval and workflow_fit < 0.45:
        transformed_penalty += 0.05

    temporal_component = temporal.score
    if not temporal.allowed_primary:
        temporal_component = min(temporal_component, 0.15)
        if primitive.transactional_authority.source_type == "transformed_analytics":
            transformed_penalty += 0.18

    score = (
        (workflow_fit * 0.28)
        + (business_object_fit * 0.18)
        + (domain_fit * 0.14)
        + (process_continuity * 0.14)
        + (primitive.transactional_authority.score * 0.14)
        + (temporal_component * 0.12)
        - transformed_penalty
    )
    score = _clamp(score)
    rejection_reasons: list[str] = []
    rationale: list[str] = []
    selected = score >= 0.42

    if not temporal.allowed_primary:
        selected = False
        rejection_reasons.append("temporal_scope_mismatch")
    if workflow_fit < 0.30:
        selected = False
        rejection_reasons.append("weak_workflow_fit")
    if process_continuity < 0.35 and workflow_fit < 0.55:
        selected = False
        rejection_reasons.append("process_boundary_mismatch")
    if boundary_conflicts:
        selected = False
        if "process_boundary_mismatch" not in rejection_reasons:
            rejection_reasons.append("process_boundary_mismatch")
        rejection_reasons.extend(boundary_conflicts[:3])
    if transformed_penalty >= 0.30:
        rejection_reasons.append("transformed_object_penalized")

    if workflow_fit >= 0.60:
        rationale.append("strong_task_signal_alignment")
    if business_object_fit >= 0.80:
        rationale.append("business_object_compatible")
    if domain_fit >= 0.80:
        rationale.append("operational_domain_compatible")
    if primitive.transactional_authority.score >= 0.70:
        rationale.append("transactional_authority_preferred")
    if temporal.status in {"inside_window", "contains_window", "no_query_window"}:
        rationale.append(f"temporal_{temporal.status}")

    return WorkflowCandidateDecision(
        file_id=primitive.file_id,
        blob_path=primitive.blob_path,
        selected=selected,
        score=score,
        best_task_id=best_task_id,
        score_components={
            "workflow_fit": workflow_fit,
            "business_object_compatibility": business_object_fit,
            "operational_domain_compatibility": domain_fit,
            "process_continuity": process_continuity,
            "transactional_authority": primitive.transactional_authority.score,
            "temporal_validity": temporal_component,
            "transformed_object_penalty": transformed_penalty,
        },
        temporal_eligibility=temporal,
        transactional_authority=primitive.transactional_authority,
        workflow_fit=workflow_fit,
        business_object_compatibility=business_object_fit,
        process_continuity=process_continuity,
        lifecycle_relevance=lifecycle_fit,
        transformed_object_penalty=transformed_penalty,
        rejection_reasons=rejection_reasons,
        rationale=rationale,
    )


def render_workflow_assembly_note(result: WorkflowAssemblyResult, *, max_candidates: int = 0) -> str:
    """Render only execution-critical workflow constraints for the prompt.

    Full candidate decisions and scoring evidence stay in WorkflowAssemblyResult
    and orchestration trace. The LLM only needs task boundaries, temporal scope,
    and the runtime policy guardrails required to avoid invalid workflow joins.
    """
    if not result.workflow_query:
        return ""
    lines = ["--- WORKFLOW EXECUTION CONSTRAINTS ---"]
    lines.append(
        "workflow_query=true; "
        f"tasks={len(result.tasks)}; "
        f"selected_evidence_tables={result.summary.get('selected_count', 0)}"
    )
    if result.query_temporal_window[0] and result.query_temporal_window[1]:
        lines.append(
            "temporal_window="
            f"{result.query_temporal_window[0]} to {result.query_temporal_window[1]}"
        )
    if result.tasks:
        lines.append("tasks:")
        for task in result.tasks[:4]:
            targets = ",".join(task.target_business_objects[:4]) or "unspecified_object"
            domains = ",".join(task.target_operational_domains[:4]) or "unspecified_domain"
            lines.append(
                f"  - {task.task_id}: {task.label}; objects={targets}; domains={domains}; signals={','.join(task.required_signals[:4])}"
            )
    if result.warnings:
        lines.append("warnings:")
        for warning in result.warnings[:3]:
            lines.append(f"  - {warning}")
    lines.append(
        "policy: use transactional workflow evidence first; do not use transformed analytics surfaces "
        "as primary evidence unless explicitly requested."
    )
    lines.append("---")
    return "\n".join(lines)


def _catalog_text(entry: dict) -> str:
    parts: list[str] = [
        str(entry.get("blob_path") or ""),
        str(entry.get("ai_description") or ""),
        " ".join(str(value) for value in (entry.get("good_for") or [])),
        " ".join(str(value) for value in (entry.get("key_metrics") or [])),
        " ".join(str(value) for value in (entry.get("key_dimensions") or [])),
        " ".join(str(value) for value in (entry.get("column_names") or [])),
    ]
    for column in entry.get("columns_info") or []:
        if isinstance(column, dict):
            parts.append(str(column.get("name") or ""))
    return " ".join(parts).lower()


def _semantic_role_parts(entry: dict) -> tuple[list[str], list[str]]:
    role_kinds: list[str] = []
    role_labels: list[str] = []
    for role_value in (entry.get("column_semantic_roles") or {}).values():
        parsed = _parse_role(str(role_value) if role_value else None)
        if not parsed:
            continue
        kind, label = parsed
        role_kinds.append(kind)
        role_labels.append(label.lower())
    return sorted(set(role_kinds)), sorted(set(role_labels))


def _parse_role(role_str: str | None) -> tuple[str, str] | None:
    if not role_str:
        return None
    match = _ROLE_RE.match(str(role_str))
    if not match:
        return None
    return match.group(1), match.group(2)


def _tokens(text: str) -> list[str]:
    expanded = str(text or "").lower().replace("_", " ").replace("-", " ")
    return [token for token in _TOKEN_RE.findall(expanded) if token and token not in _STOPWORDS]


def _infer_business_objects(text: str, tokens: list[str]) -> list[str]:
    token_set = set(tokens)
    objects: set[str] = set()
    joined = " ".join(tokens)
    if "po" in token_set or "purchase order" in joined or {"purchase", "order"} <= token_set:
        objects.add("purchase_order")
    if "procurement" in token_set:
        objects.add("purchase_order")
    if "invoice" in token_set or "invoices" in token_set or "invoiced" in token_set or "uninvoiced" in token_set:
        objects.add("invoice")
    if "supplier" in token_set or "vendor" in token_set:
        objects.add("supplier")
    if "customer" in token_set or "cust" in token_set:
        objects.add("customer")
    if "payment" in token_set or "payments" in token_set or "paid" in token_set or "check" in token_set:
        objects.add("payment")
    if "cash" in token_set and ("receipt" in token_set or "receipts" in token_set):
        objects.add("cash_receipt")
    elif "receipt" in token_set or "receipts" in token_set or "receiving" in token_set or "received" in token_set:
        objects.add("goods_receipt")
    if "delivery" in token_set or "shipment" in token_set or "shipping" in token_set:
        objects.add("delivery")
    if "warehouse" in token_set or "ewm" in token_set:
        objects.add("warehouse_task")
    if "ledger" in token_set or "journal" in token_set or "gl" in token_set:
        objects.add("ledger_entry")
    if "order" in token_set and "sales" in token_set and "purchase_order" not in objects:
        objects.add("sales_order")
    return sorted(objects)


def _infer_process_signals(text: str) -> list[str]:
    tokens = set(_tokens(text))
    signals: set[str] = set()
    if tokens & {"approval", "approved", "approvals", "authorization", "authorisation", "authorize", "pending"}:
        signals.add("approval")
    if tokens & {"status", "state", "open", "closed", "current", "pending"}:
        signals.add("status")
    if tokens & {"delivery", "delivered", "shipment", "shipping"}:
        signals.add("delivery")
    if tokens & {"fulfillment", "fulfilment", "fulfilled", "fulfill"}:
        signals.add("fulfillment")
    if tokens & {"receiving", "receipt", "receipts", "received", "receive", "grn"}:
        signals.add("receiving")
    if tokens & {"invoice", "invoices", "invoiced", "uninvoiced"}:
        signals.add("invoice")
    if tokens & {"match", "matching", "matched", "unmatched", "reconcile", "reconciliation", "variance", "discrepancy", "discrepancies"}:
        signals.add("reconciliation")
    if tokens & {"payment", "payments", "paid", "payable", "check", "cash"}:
        signals.add("payment")
    if tokens & {"liability", "liabilities", "exposure", "accrual", "accrued", "unpaid"}:
        signals.add("liability")
    if tokens & {"lifecycle", "bottleneck", "bottlenecks", "delay", "delayed", "aging", "exception", "issue", "issues"}:
        signals.add("lifecycle")
    return sorted(signals)


def _infer_operational_domains(text: str, tokens: list[str]) -> list[str]:
    token_set = set(tokens)
    domains: set[str] = set()
    cash_receipt_context = "cash" in token_set and bool(token_set & {"receipt", "receipts"})
    if token_set & {"ap", "payable", "payables", "supplier", "vendor"}:
        domains.add("accounts_payable")
    if token_set & {"po", "purchase", "procurement", "receiving", "received"}:
        domains.add("procurement")
    if token_set & {"receipt", "receipts"} and not cash_receipt_context and not (token_set & {"ar", "receivable", "receivables", "customer"}):
        domains.add("procurement")
    if token_set & {"ar", "receivable", "receivables", "customer", "cash"}:
        domains.add("accounts_receivable")
    if token_set & {"sales", "billing", "customer", "shipment", "delivery"} and "po" not in token_set:
        domains.add("sales")
    if token_set & {"warehouse", "ewm", "shipment", "shipping", "delivery", "fulfillment", "fulfilment"}:
        domains.add("logistics")
    if token_set & {"gl", "ledger", "journal", "accounting", "account"}:
        domains.add("general_ledger")
    if "invoice" in token_set and not domains:
        domains.add("invoicing")
    return sorted(domains)


def _infer_temporal_columns(entry: dict) -> list[str]:
    temporal_columns: set[str] = set()
    roles = entry.get("column_semantic_roles") or {}
    for column_name, role_value in roles.items():
        parsed = _parse_role(str(role_value) if role_value else None)
        if parsed and parsed[0] == "date":
            temporal_columns.add(str(column_name))
    for column_name in entry.get("column_names") or []:
        name = str(column_name).lower()
        if any(term in name for term in ("date", "time", "period", "year")):
            temporal_columns.add(str(column_name))
    return sorted(temporal_columns)


def _infer_workflow_grain(text: str, tokens: list[str], role_labels: list[str]) -> str:
    token_set = set(tokens)
    if token_set & {"line", "lines", "item", "items"}:
        return "line"
    if token_set & {"distribution", "distributions"}:
        return "distribution"
    if token_set & {"schedule", "schedules", "location", "locations"}:
        return "schedule"
    if token_set & {"header", "headers"}:
        return "header"
    if token_set & {"payment", "payments"}:
        return "payment"
    if token_set & {"master", "supplier", "vendor", "customer"} and "invoice" not in token_set:
        return "master"
    if any("header" in label for label in role_labels):
        return "header"
    return "record"


def _is_workflow_like(query_tokens: list[str], intent_plan: Any) -> bool:
    token_set = set(query_tokens)
    if token_set & _OPERATIONAL_TERMS:
        return True
    behaviors = set(getattr(intent_plan, "behaviors", []) or []) if intent_plan else set()
    return bool(behaviors & {"open_items", "multi_step"})


def _has_analytical_intent(query_tokens: list[str], intent_plan: Any) -> bool:
    token_set = set(query_tokens)
    if token_set & _ANALYTICAL_TERMS and not (token_set & {"approval", "matching", "unmatched", "receiving", "receipt", "status"}):
        return True
    behaviors = set(getattr(intent_plan, "behaviors", []) or []) if intent_plan else set()
    return bool(behaviors & {"aggregation", "group_by", "top_n"})


def _prefer_objects(existing: list[str], preferred: list[str]) -> list[str]:
    merged = list(dict.fromkeys([*preferred, *existing]))
    return merged


def _prefer_domains(existing: list[str], preferred: list[str]) -> list[str]:
    merged = list(dict.fromkeys([*preferred, *existing]))
    return merged


def _compatibility(targets: list[str], observed: list[str], *, neutral: float) -> float:
    if not targets:
        return neutral
    if not observed:
        return 0.20
    target_set = set(targets)
    observed_set = set(observed)
    if target_set & observed_set:
        return 1.0
    if _related_business_objects(target_set, observed_set):
        return 0.62
    return 0.18


def _related_business_objects(targets: set[str], observed: set[str]) -> bool:
    related_groups = [
        {"purchase_order", "supplier", "goods_receipt", "invoice"},
        {"invoice", "payment", "liability"},
        {"delivery", "fulfillment", "warehouse_task", "goods_receipt"},
        {"customer", "sales_order", "delivery", "cash_receipt"},
        {"ledger_entry", "liability", "payment", "invoice"},
    ]
    return any(targets & group and observed & group for group in related_groups)


def _process_boundary_conflicts(
    *,
    query_business_objects: list[str],
    query_operational_domains: list[str],
    primitive: WorkflowSemanticPrimitive,
) -> list[str]:
    conflicts: list[str] = []
    target_objects = set(query_business_objects)
    target_domains = set(query_operational_domains)
    observed_objects = set(primitive.business_objects)
    observed_domains = set(primitive.operational_domains)

    if "goods_receipt" in target_objects and "cash_receipt" in observed_objects and "goods_receipt" not in observed_objects:
        conflicts.append("cash_receipt_not_goods_receipt")
    if target_domains & {"procurement", "accounts_payable"}:
        if "accounts_receivable" in observed_domains and not (observed_domains & {"procurement", "accounts_payable"}):
            conflicts.append("accounts_receivable_boundary_mismatch")
    if "purchase_order" in target_objects:
        if "sales_order" in observed_objects and "purchase_order" not in observed_objects:
            conflicts.append("sales_order_not_purchase_order")
    return conflicts


def _signal_overlap(targets: list[str], observed: list[str], *, neutral: float) -> float:
    if not targets:
        return neutral
    if not observed:
        return 0.15
    target_set = set(targets)
    observed_set = set(observed)
    overlap = len(target_set & observed_set)
    if overlap:
        return min(1.0, 0.45 + (overlap / max(1, len(target_set))) * 0.55)
    if _related_signals(target_set, observed_set):
        return 0.50
    return 0.18


def _related_signals(targets: set[str], observed: set[str]) -> bool:
    related = [
        {"invoice", "reconciliation", "payment", "liability"},
        {"delivery", "fulfillment", "receiving"},
        {"approval", "status", "lifecycle"},
    ]
    return any(targets & group and observed & group for group in related)


def _build_workflow_warnings(decisions: list[WorkflowCandidateDecision]) -> list[str]:
    warnings: list[str] = []
    if any(decision.temporal_eligibility.status == "outside_window" for decision in decisions):
        warnings.append("outside_window_candidates_penalized")
    if any("process_boundary_mismatch" in decision.rejection_reasons for decision in decisions):
        warnings.append("process_boundary_mismatches_detected")
    if any("transformed_object_penalized" in decision.rejection_reasons for decision in decisions):
        warnings.append("transformed_objects_penalized_for_operational_workflow")
    if not any(decision.selected for decision in decisions):
        warnings.append("no_candidate_passed_workflow_validation")
    return warnings


def _parse_catalog_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _date_str(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _empty_decision() -> WorkflowCandidateDecision:
    temporal = TemporalEligibility(status="unknown", score=0.0, allowed_primary=False)
    authority = TransactionalAuthority(
        source_type="unknown",
        score=0.0,
        transformation_level="unknown",
        workflow_grain="unknown",
    )
    return WorkflowCandidateDecision(
        file_id="",
        blob_path="",
        selected=False,
        score=0.0,
        best_task_id=None,
        score_components={},
        temporal_eligibility=temporal,
        transactional_authority=authority,
        workflow_fit=0.0,
        business_object_compatibility=0.0,
        process_continuity=0.0,
        lifecycle_relevance=0.0,
        transformed_object_penalty=0.0,
    )
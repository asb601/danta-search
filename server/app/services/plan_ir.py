"""Plan IR and validator for bounded analytics execution."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.services.brain_context import BrainContext, ExecutionEnvelope


@dataclass(frozen=True)
class PlanStage:
    id: str
    operation: str
    file_ids: list[str]
    depends_on: list[str] = field(default_factory=list)
    contracts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KPIContract:
    id: str
    metric: str
    aggregation: str | None
    grain: str | None
    source_memory_id: str | None = None


@dataclass(frozen=True)
class PlanIR:
    id: str
    stages: list[PlanStage]
    kpi_contracts: list[KPIContract] = field(default_factory=list)
    lifecycle_validations: list[str] = field(default_factory=list)

    def to_prompt_section(self, validation: "PlanValidationResult") -> str:
        if not self.stages and not validation.issues:
            return ""
        lines = ["--- PLAN VALIDATION ---"]
        lines.append(f"stages={len(self.stages)}; kpi_contracts={len(self.kpi_contracts)}")
        for stage in self.stages[:6]:
            lines.append(f"  - {stage.id}:{stage.operation}; files={len(stage.file_ids)}")
        if validation.issues:
            lines.append("validator findings:")
            for issue in validation.issues[:6]:
                lines.append(f"  - {issue.severity}:{issue.code}:{issue.message}")
        lines.append("---")
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanValidationIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class PlanValidationResult:
    ok: bool
    issues: list[PlanValidationIssue] = field(default_factory=list)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.__dict__ for issue in self.issues[:10]],
        }


def build_plan_ir_from_context(
    *,
    intent_plan: Any,
    catalog: list[dict],
    sql_ctx: Any,
    exec_strategy: Any,
    brain_context: BrainContext | None,
) -> PlanIR:
    settings = get_settings()
    max_stages = max(1, int(settings.PLAN_IR_MAX_STAGES))
    behaviors = set(getattr(intent_plan, "behaviors", []) or [])
    operation = "aggregate" if "aggregation" in behaviors else "detail"
    if "multi_step" in behaviors:
        operation = "decompose"

    stages: list[PlanStage] = []
    catalog_file_ids = {entry["file_id"] for entry in catalog if entry.get("file_id")}
    domain_stage_count = 0
    if brain_context:
        for domain in brain_context.domains:
            if domain.domain_type not in {"workflow_domain", "lifecycle_domain", "business_capability"}:
                continue
            domain_file_ids = [file_id for file_id in domain.contributor_file_ids if file_id in catalog_file_ids]
            if not domain_file_ids:
                continue
            domain_stage_count += 1
            stages.append(PlanStage(
                id=f"domain_stage_{domain_stage_count}",
                operation=domain.domain_type,
                file_ids=domain_file_ids[:8],
                contracts=[domain.id],
            ))
            if len(stages) >= max_stages:
                break

    clusters = list(getattr(exec_strategy, "clusters", []) or [])
    if stages:
        pass
    elif clusters:
        for idx, cluster in enumerate(clusters[:max_stages], 1):
            stages.append(PlanStage(
                id=f"stage_{idx}",
                operation="joined_sql" if getattr(cluster, "strategy", "") == "joined_sql" else operation,
                file_ids=list(getattr(cluster, "file_ids", []) or []),
            ))
    else:
        file_ids = [entry["file_id"] for entry in catalog if entry.get("file_id")][:max_stages]
        for idx, file_id in enumerate(file_ids, 1):
            stages.append(PlanStage(id=f"stage_{idx}", operation=operation, file_ids=[file_id]))

    contracts: list[KPIContract] = []
    if brain_context:
        metric_records = [record for record in brain_context.records if record.memory_type == "metric"]
        for record in metric_records[: max(1, int(settings.PLAN_IR_MAX_CONTRACTS))]:
            contracts.append(KPIContract(
                id=f"kpi_{uuid.uuid4().hex[:8]}",
                metric=record.title,
                aggregation="SUM" if "aggregation" in record.behaviors else None,
                grain=None,
                source_memory_id=record.id,
            ))
        remaining = max(0, int(settings.PLAN_IR_MAX_CONTRACTS) - len(contracts))
        for domain in [d for d in brain_context.domains if d.domain_type == "kpi_domain"][:remaining]:
            metric = (domain.kpi_terms or domain.terms or [domain.title])[0]
            contracts.append(KPIContract(
                id=f"kpi_{uuid.uuid4().hex[:8]}",
                metric=metric,
                aggregation="SUM" if "aggregation" in (domain.kpi_terms + domain.terms) else None,
                grain=None,
                source_memory_id=domain.id,
            ))

    lifecycle_validations = []
    if any(getattr(record, "memory_type", "") == "temporal" for record in (brain_context.records if brain_context else [])):
        lifecycle_validations.append("temporal_scope_available")
    if any(getattr(domain, "domain_type", "") == "lifecycle_domain" for domain in (brain_context.domains if brain_context else [])):
        lifecycle_validations.append("lifecycle_domain_available")
    if any(getattr(domain, "domain_type", "") == "workflow_domain" for domain in (brain_context.domains if brain_context else [])):
        lifecycle_validations.append("workflow_domain_available")
    if getattr(sql_ctx, "approved_joins", None):
        lifecycle_validations.append("approved_join_scope_available")

    return PlanIR(id=f"plan_{uuid.uuid4().hex[:12]}", stages=stages, kpi_contracts=contracts, lifecycle_validations=lifecycle_validations)


def validate_plan_ir(plan: PlanIR, envelope: ExecutionEnvelope, *, intent_plan: Any | None = None) -> PlanValidationResult:
    settings = get_settings()
    issues: list[PlanValidationIssue] = []
    allowed_files = set(envelope.shortlist_file_ids or envelope.anchor_file_ids)

    if len(plan.stages) > int(settings.PLAN_IR_MAX_STAGES):
        issues.append(PlanValidationIssue("error", "too_many_stages", "plan exceeds stage cap"))

    seen_stage_ids: set[str] = set()
    for stage in plan.stages:
        if stage.id in seen_stage_ids:
            issues.append(PlanValidationIssue("error", "duplicate_stage", f"duplicate stage id {stage.id}"))
        seen_stage_ids.add(stage.id)
        unknown = [file_id for file_id in stage.file_ids if allowed_files and file_id not in allowed_files]
        if unknown:
            issues.append(PlanValidationIssue("error", "file_outside_envelope", "stage references file outside execution envelope"))
        for dep in stage.depends_on:
            if dep not in seen_stage_ids:
                issues.append(PlanValidationIssue("warning", "dependency_order", f"stage {stage.id} depends on later/unknown stage {dep}"))

    behaviors = set(getattr(intent_plan, "behaviors", []) or []) if intent_plan else set()
    if "aggregation" in behaviors and not plan.kpi_contracts:
        issues.append(PlanValidationIssue("warning", "missing_kpi_contract", "aggregation intent has no governed metric memory contract"))

    for contract in plan.kpi_contracts:
        if not contract.metric:
            issues.append(PlanValidationIssue("error", "missing_metric", "KPI contract is missing metric"))
        if not contract.aggregation:
            issues.append(PlanValidationIssue("warning", "missing_aggregation", f"KPI contract {contract.id} has no default aggregation"))

    if envelope.ambiguity_flags:
        issues.append(PlanValidationIssue("warning", "semantic_ambiguity", ", ".join(envelope.ambiguity_flags[:3])))

    return PlanValidationResult(ok=not any(issue.severity == "error" for issue in issues), issues=issues)
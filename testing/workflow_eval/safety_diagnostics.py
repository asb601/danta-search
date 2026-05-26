"""
Business-Truth Safety Diagnostics

PURPOSE:
  Detect unsafe answers — where the planner might assert business facts
  (approvals, payments, delivery status, amounts) without the supporting
  domain evidence being in the shortlist.

CORE PRINCIPLE:
  An answer that claims "Invoice #123 was paid" is UNSAFE if the payment
  domain is not in the shortlist. The planner has no evidentiary basis for
  that claim. This is analogous to a hallucination at the domain level.

WHAT IT CHECKS:
  For each benchmark category, define which domains must be present for
  which types of business claims. If a required claim domain is absent,
  flag a SafetyViolation.

  Claims are not analyzed in the LLM output text — safety is assessed
  purely from what's in the shortlist. This is deterministic and fast.

SAFETY REQUIREMENTS TABLE:
  Each category maps to SafetyRequirements that define:
    - claim_type:       type of business claim (e.g. "payment_status")
    - required_domain:  short_name of domain that must be present
    - if_absent_risk:   what incorrect claim might the LLM make?

USAGE:
  report = run_safety_diagnostics(benchmark, score, world)
  print(report.is_safe_to_answer)    # True/False
  print(report.safety_score)         # 0.0 = unsafe, 1.0 = fully safe
  for v in report.safety_violations: print(v)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from testing.workflow_eval.benchmarks import WorkflowBenchmark
from testing.workflow_eval.completeness_scorer import WorkflowCompletenessScore
from testing.workflow_eval.world_state_inspector import PlannerWorldState


# ── Safety requirement definitions ────────────────────────────────────────────

@dataclass
class SafetyRequirement:
    """
    A single business-truth safety check for a workflow category.

    Attributes:
        claim_type:       human-readable claim type (e.g. "payment_status")
        domain_short_name: short_name of the domain that must be in shortlist
        if_absent_risk:   risk description if domain is absent
        severity:         "critical" | "major" | "minor"
    """
    claim_type: str
    domain_short_name: str
    if_absent_risk: str
    severity: Literal["critical", "major", "minor"] = "major"


# ── Per-category safety requirements ─────────────────────────────────────────
# These are generic — no ERP/SAP names. Expressed in terms of domain short_names
# that are defined in each benchmark's expected_domains list.

CATEGORY_SAFETY_REQUIREMENTS: dict[str, list[SafetyRequirement]] = {

    "invoice_matching": [
        SafetyRequirement(
            claim_type="invoice_status",
            domain_short_name="invoice_transactions",
            if_absent_risk="LLM may fabricate invoice match status without invoice data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="purchase_order_match",
            domain_short_name="purchase_order_transactions",
            if_absent_risk="LLM may claim PO match result without PO data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="vendor_identification",
            domain_short_name="vendor_master",
            if_absent_risk="LLM may misattribute invoices to wrong vendor",
            severity="major",
        ),
    ],

    "pending_approvals": [
        SafetyRequirement(
            claim_type="approval_status",
            domain_short_name="approval_status",
            if_absent_risk="LLM may claim approval status without approval records",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="temporal_filter",
            domain_short_name="date_dimension",
            if_absent_risk="LLM cannot correctly filter 'this month' without date data",
            severity="major",
        ),
    ],

    "delivery_delays": [
        SafetyRequirement(
            claim_type="delivery_status",
            domain_short_name="delivery_transactions",
            if_absent_risk="LLM may invent delivery delay facts without shipment data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="date_comparison",
            domain_short_name="delivery_date",
            if_absent_risk="LLM cannot compute delays without scheduled/actual dates",
            severity="critical",
        ),
    ],

    "partial_receipts": [
        SafetyRequirement(
            claim_type="quantity_status",
            domain_short_name="purchase_order",
            if_absent_risk="LLM may fabricate ordered quantity without PO data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="receipt_quantity",
            domain_short_name="goods_receipt",
            if_absent_risk="LLM may fabricate received quantity without GRN data",
            severity="critical",
        ),
    ],

    "overdue_invoices": [
        SafetyRequirement(
            claim_type="payment_status",
            domain_short_name="invoice_transactions",
            if_absent_risk="LLM may claim invoice is overdue without payment/due-date data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="outstanding_amount",
            domain_short_name="vendor_master",
            if_absent_risk="LLM cannot aggregate by vendor without vendor master",
            severity="major",
        ),
    ],

    "open_liabilities": [
        SafetyRequirement(
            claim_type="liability_amount",
            domain_short_name="liability_transactions",
            if_absent_risk="LLM may claim liability figures without GL/liability data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="cost_center_attribution",
            domain_short_name="cost_center",
            if_absent_risk="LLM may misattribute liabilities to wrong cost center",
            severity="major",
        ),
    ],

    "vendor_reconciliation": [
        SafetyRequirement(
            claim_type="vendor_balance",
            domain_short_name="vendor_master",
            if_absent_risk="LLM may compute incorrect vendor balance without master",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="invoice_total",
            domain_short_name="invoice_transactions",
            if_absent_risk="LLM may invent invoice totals",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="payment_total",
            domain_short_name="payment_transactions",
            if_absent_risk="LLM may invent payment totals — reconciliation will be wrong",
            severity="critical",
        ),
    ],

    "po_lifecycle": [
        SafetyRequirement(
            claim_type="lifecycle_stage",
            domain_short_name="purchase_order",
            if_absent_risk="LLM may describe PO lifecycle without PO records",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="payment_completion",
            domain_short_name="payment",
            if_absent_risk="LLM may claim payment was made without payment records",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="receipt_confirmation",
            domain_short_name="goods_receipt",
            if_absent_risk="LLM may claim goods received without GRN records",
            severity="critical",
        ),
    ],

    "grn_mismatches": [
        SafetyRequirement(
            claim_type="grn_quantity",
            domain_short_name="goods_receipt",
            if_absent_risk="LLM may invent GRN quantity figures",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="po_quantity",
            domain_short_name="purchase_order",
            if_absent_risk="LLM may invent PO quantity figures",
            severity="critical",
        ),
    ],

    "procurement_bottlenecks": [
        SafetyRequirement(
            claim_type="stage_timing",
            domain_short_name="requisition",
            if_absent_risk="LLM may fabricate stage duration without request records",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="bottleneck_identification",
            domain_short_name="stage_dimension",
            if_absent_risk="LLM may claim bottleneck location without stage data",
            severity="major",
        ),
    ],

    "cost_center_leakage": [
        SafetyRequirement(
            claim_type="budget_variance",
            domain_short_name="budget",
            if_absent_risk="LLM may compute wrong variance without budget data",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="actual_spend",
            domain_short_name="actual_spend",
            if_absent_risk="LLM may invent spend figures",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="cost_center_attribution",
            domain_short_name="cost_center",
            if_absent_risk="LLM may misattribute budget overruns",
            severity="major",
        ),
    ],

    "payment_aging": [
        SafetyRequirement(
            claim_type="days_to_pay",
            domain_short_name="invoice_transactions",
            if_absent_risk="LLM cannot compute aging without invoice issue dates",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="payment_confirmation",
            domain_short_name="payment_transactions",
            if_absent_risk="LLM may claim payment was made without payment records",
            severity="critical",
        ),
    ],

    "inventory_movement": [
        SafetyRequirement(
            claim_type="stock_quantity",
            domain_short_name="stock_movements",
            if_absent_risk="LLM may invent stock figures without movement records",
            severity="critical",
        ),
        SafetyRequirement(
            claim_type="material_identification",
            domain_short_name="material_master",
            if_absent_risk="LLM may misidentify items without material master",
            severity="major",
        ),
    ],
}


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SafetyViolation:
    """A detected business-truth safety violation."""
    claim_type: str
    missing_domain: str
    risk_description: str
    severity: str
    evidence: str


@dataclass
class BusinessTruthReport:
    """
    Business-truth safety assessment.

    Attributes:
        category:               workflow category checked
        requirements_checked:   how many safety requirements were evaluated
        safety_violations:      list of detected violations
        violation_count:        total violations
        critical_violation_count: critical-severity violations
        safety_score:           1.0 - (violations / requirements_checked)
        is_safe_to_answer:      True if no critical violations
        unsatisfied_claim_types: claim types that cannot be safely answered
        risk_summary:           human-readable risk description
    """
    category: str
    requirements_checked: int
    safety_violations: list[SafetyViolation]
    violation_count: int
    critical_violation_count: int
    safety_score: float
    is_safe_to_answer: bool
    unsatisfied_claim_types: list[str]
    risk_summary: str

    def summary_line(self) -> str:
        verdict = "SAFE" if self.is_safe_to_answer else "UNSAFE"
        return (
            f"safety={verdict} | "
            f"score={self.safety_score:.0%} | "
            f"violations={self.violation_count} "
            f"(critical={self.critical_violation_count})"
        )


# ── Main diagnostics function ─────────────────────────────────────────────────

def run_safety_diagnostics(
    benchmark: WorkflowBenchmark,
    score: WorkflowCompletenessScore,
    world: PlannerWorldState,
) -> BusinessTruthReport:
    """
    Run business-truth safety diagnostics for a benchmark evaluation.

    Checks whether the shortlist contains the domains required to make
    safe business claims for this workflow category.

    Args:
        benchmark: the benchmark being evaluated
        score:     completeness score (has domain coverage detail)
        world:     world state (has shortlist info)

    Returns:
        BusinessTruthReport with safety assessment
    """
    category = benchmark.workflow_category
    requirements = CATEGORY_SAFETY_REQUIREMENTS.get(category, [])

    if not requirements:
        # No safety requirements defined for this category
        return BusinessTruthReport(
            category=category,
            requirements_checked=0,
            safety_violations=[],
            violation_count=0,
            critical_violation_count=0,
            safety_score=1.0,
            is_safe_to_answer=True,
            unsatisfied_claim_types=[],
            risk_summary="No safety requirements defined for this category.",
        )

    # Build a set of satisfied domain short_names from the completeness score
    satisfied_domains: set[str] = {
        d.requirement.short_name
        for d in score.domain_coverage
        if d.satisfied
    }

    violations: list[SafetyViolation] = []
    unsatisfied_claims: list[str] = []

    for req in requirements:
        if req.domain_short_name not in satisfied_domains:
            violations.append(SafetyViolation(
                claim_type=req.claim_type,
                missing_domain=req.domain_short_name,
                risk_description=req.if_absent_risk,
                severity=req.severity,
                evidence=(
                    f"domain '{req.domain_short_name}' not in shortlist "
                    f"(shortlist_count={world.shortlist_file_count})"
                ),
            ))
            unsatisfied_claims.append(req.claim_type)

    critical_count = sum(1 for v in violations if v.severity == "critical")
    safety_score = 1.0 - (len(violations) / len(requirements)) if requirements else 1.0
    is_safe = critical_count == 0

    risk_summary = _build_risk_summary(violations, category, is_safe)

    return BusinessTruthReport(
        category=category,
        requirements_checked=len(requirements),
        safety_violations=violations,
        violation_count=len(violations),
        critical_violation_count=critical_count,
        safety_score=max(0.0, safety_score),
        is_safe_to_answer=is_safe,
        unsatisfied_claim_types=unsatisfied_claims,
        risk_summary=risk_summary,
    )


def _build_risk_summary(
    violations: list[SafetyViolation],
    category: str,
    is_safe: bool,
) -> str:
    if not violations:
        return f"All safety requirements for '{category}' are satisfied."

    critical = [v for v in violations if v.severity == "critical"]
    major = [v for v in violations if v.severity == "major"]

    parts: list[str] = []
    if critical:
        claims = ", ".join(v.claim_type for v in critical[:3])
        parts.append(f"CRITICAL: unsafe to claim {claims}")
    if major:
        claims = ", ".join(v.claim_type for v in major[:3])
        parts.append(f"MAJOR: weakly evidenced {claims}")

    verdict = "ANSWER UNSAFE" if not is_safe else "ANSWER MARGINAL"
    return f"{verdict} — {'; '.join(parts)}"

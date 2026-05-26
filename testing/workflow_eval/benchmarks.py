"""
Workflow Benchmark Definitions — enterprise workflow scenarios.

PURPOSE:
  Define the ground-truth expectations for 13 enterprise workflow scenarios.
  Each benchmark describes WHAT SEMANTIC DOMAINS are required to answer a
  business question — expressed as role-type patterns, not table names.

DESIGN PRINCIPLES:
  - NO hardcoded SAP/ERP/Oracle names. Benchmarks use semantic abstractions.
  - Domains are described by: role_type (entity/transaction/dimension)
    + label_hints (tokens that should appear in the role label).
  - A benchmark passes when the planner can see files satisfying each
    required DomainRequirement from the shortlist or topology hints.

WORKFLOW CATEGORIES:
  invoice_matching, pending_approvals, delivery_delays, partial_receipts,
  overdue_invoices, open_liabilities, vendor_reconciliation, po_lifecycle,
  grn_mismatches, procurement_bottlenecks, cost_center_leakage,
  payment_aging, inventory_movement
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Domain requirement ─────────────────────────────────────────────────────────

@dataclass
class DomainRequirement:
    """
    One semantic domain required to answer this workflow question.

    Satisfied when the planner can see at least one file where:
      column_semantic_roles contains a role with:
        kind   in _kind_set_for(role_type)
        label  containing at least one token from label_hints

    Fields:
        role_type:   "entity" | "transaction" | "dimension"
        label_hints: tokens that should appear in the role label
                     (any match = satisfied)
        is_required: True = FAIL if missing; False = WARN if missing
        rationale:   why this domain is needed
        short_name:  human-readable identifier for reporting
    """
    role_type: Literal["entity", "transaction", "dimension"]
    label_hints: list[str]
    is_required: bool
    rationale: str
    short_name: str

    def matches_role(self, role_kind: str, role_label: str) -> bool:
        """Check if a role (kind, label) satisfies this requirement."""
        kind_map = {
            "entity": {"entity_key"},
            "transaction": {"reference_key", "additive_measure", "non_additive_measure"},
            "dimension": {"date", "attribute"},
        }
        if role_kind not in kind_map.get(self.role_type, set()):
            return False
        label_lower = role_label.lower()
        return any(hint.lower() in label_lower for hint in self.label_hints)


@dataclass
class TopologyRequirement:
    """
    An expected join topology between two domain types.

    join_type:
        "entity-transaction"  — entity master joins to a transaction file
        "entity-entity"       — two entity masters join (e.g. vendor → contact)
        "transaction-dimension" — transaction joins to a dimension (e.g. date)

    source_short_name: short_name of the source domain requirement
    target_short_name: short_name of the target domain requirement
    """
    join_type: Literal["entity-transaction", "entity-entity", "transaction-dimension"]
    source_short_name: str
    target_short_name: str
    is_required: bool = True
    rationale: str = ""


@dataclass
class WorkflowBenchmark:
    """
    One complete benchmark scenario for a business workflow question.

    Evaluation passes when:
      1. All required DomainRequirements are satisfied by shortlist OR topology
      2. workflow_completeness >= min_workflow_completeness
      3. At least one required topology join path exists (if any specified)
      4. No business-truth safety violations flagged
    """
    id: str
    workflow_query: str
    workflow_category: str
    description: str
    expected_domains: list[DomainRequirement]
    topology_requirements: list[TopologyRequirement]
    min_workflow_completeness: float        # minimum to pass
    complexity: Literal["single_table", "two_table", "multi_table"]
    business_truth_requirements: list[str]  # semantic facts that must be accessible
    # Scoring weights: how critical is this benchmark to overall health
    weight: float = 1.0

    @property
    def required_domains(self) -> list[DomainRequirement]:
        return [d for d in self.expected_domains if d.is_required]

    @property
    def optional_domains(self) -> list[DomainRequirement]:
        return [d for d in self.expected_domains if not d.is_required]


# ── Benchmark definitions ──────────────────────────────────────────────────────

def _b(
    id: str,
    query: str,
    category: str,
    description: str,
    domains: list[DomainRequirement],
    topology: list[TopologyRequirement],
    min_completeness: float,
    complexity: str,
    truth_reqs: list[str],
    weight: float = 1.0,
) -> WorkflowBenchmark:
    return WorkflowBenchmark(
        id=id,
        workflow_query=query,
        workflow_category=category,
        description=description,
        expected_domains=domains,
        topology_requirements=topology,
        min_workflow_completeness=min_completeness,
        complexity=complexity,
        business_truth_requirements=truth_reqs,
        weight=weight,
    )


BENCHMARK_REGISTRY: list[WorkflowBenchmark] = [

    # ── 1. Invoice matching ────────────────────────────────────────────────────
    _b(
        id="WF-001",
        query="Show me all invoices that don't match their purchase orders",
        category="invoice_matching",
        description=(
            "Three-way match: invoice vs. PO vs. goods receipt. Requires "
            "invoice entity, purchase_order transaction reference, and "
            "goods receipt confirmation records."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["invoice", "bill", "payable", "ap"],
                is_required=True,
                rationale="Invoice header/line records are the primary subject",
                short_name="invoice_transactions",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po", "order"],
                is_required=True,
                rationale="PO records needed for matching",
                short_name="purchase_order_transactions",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier", "creditor"],
                is_required=True,
                rationale="Vendor master needed to resolve vendor dimensions",
                short_name="vendor_master",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["receipt", "grn", "goods_receipt", "delivery"],
                is_required=False,
                rationale="GRN data improves three-way match quality",
                short_name="goods_receipt",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="vendor_master",
                target_short_name="invoice_transactions",
                is_required=True,
                rationale="Vendor-invoice join is core to the match",
            ),
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="purchase_order_transactions",
                target_short_name="invoice_transactions",
                is_required=True,
                rationale="PO-invoice join enables matching",
            ),
        ],
        min_completeness=0.75,
        complexity="multi_table",
        truth_reqs=[
            "invoice_domain_present",
            "purchase_order_domain_present",
            "vendor_resolution_possible",
        ],
    ),

    # ── 2. Pending approvals ───────────────────────────────────────────────────
    _b(
        id="WF-002",
        query="What purchase orders are still pending approval this month?",
        category="pending_approvals",
        description=(
            "Approval workflow status. Requires transaction records with "
            "approval status dimension and temporal filtering capability."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po", "requisition", "order"],
                is_required=True,
                rationale="PO records with approval status",
                short_name="order_transactions",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["status", "approval", "workflow", "stage"],
                is_required=True,
                rationale="Status dimension to filter pending",
                short_name="approval_status",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "created", "submitted", "period"],
                is_required=True,
                rationale="Temporal dimension for 'this month' filter",
                short_name="date_dimension",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier", "requestor", "requester"],
                is_required=False,
                rationale="Requestor/vendor dimension improves context",
                short_name="requestor_entity",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="transaction-dimension",
                source_short_name="order_transactions",
                target_short_name="date_dimension",
                is_required=True,
                rationale="Temporal filter requires date join",
            ),
        ],
        min_completeness=0.70,
        complexity="two_table",
        truth_reqs=[
            "approval_status_dimension_present",
            "date_range_filterable",
        ],
    ),

    # ── 3. Delivery delays ────────────────────────────────────────────────────
    _b(
        id="WF-003",
        query="Which deliveries are delayed beyond their expected delivery date?",
        category="delivery_delays",
        description=(
            "Delivery performance analysis. Requires shipment/delivery records "
            "with scheduled vs. actual date comparison."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["delivery", "shipment", "dispatch", "logistics", "shipping"],
                is_required=True,
                rationale="Delivery/shipment records are the primary subject",
                short_name="delivery_transactions",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "scheduled", "planned", "expected", "actual"],
                is_required=True,
                rationale="Date dimensions for scheduled vs. actual comparison",
                short_name="delivery_date",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po", "order"],
                is_required=False,
                rationale="PO reference improves delay attribution",
                short_name="order_reference",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "carrier", "supplier", "transporter"],
                is_required=False,
                rationale="Carrier/vendor entity for delay attribution",
                short_name="carrier_entity",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="transaction-dimension",
                source_short_name="delivery_transactions",
                target_short_name="delivery_date",
                is_required=True,
                rationale="Date join required for delay computation",
            ),
        ],
        min_completeness=0.65,
        complexity="two_table",
        truth_reqs=[
            "delivery_domain_present",
            "date_comparison_possible",
        ],
    ),

    # ── 4. Partial receipts ───────────────────────────────────────────────────
    _b(
        id="WF-004",
        query="List all purchase orders with partial goods receipt — ordered but not fully delivered",
        category="partial_receipts",
        description=(
            "Partial delivery analysis requires ordered quantity vs. received "
            "quantity comparison across PO and GRN records."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po", "order"],
                is_required=True,
                rationale="PO with ordered quantity",
                short_name="purchase_order",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["receipt", "grn", "goods_receipt", "receiving"],
                is_required=True,
                rationale="GRN with received quantity",
                short_name="goods_receipt",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier"],
                is_required=False,
                rationale="Vendor context for follow-up",
                short_name="vendor_master",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="purchase_order",
                target_short_name="goods_receipt",
                is_required=True,
                rationale="PO-GRN join enables quantity comparison",
            ),
        ],
        min_completeness=0.80,
        complexity="two_table",
        truth_reqs=[
            "quantity_comparison_possible",
            "po_grn_join_available",
        ],
    ),

    # ── 5. Overdue invoices ───────────────────────────────────────────────────
    _b(
        id="WF-005",
        query="Show all overdue invoices and the total outstanding amount by vendor",
        category="overdue_invoices",
        description=(
            "Accounts payable aging. Requires invoice records with payment "
            "status and due date, plus vendor entity for grouping."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["invoice", "payable", "bill"],
                is_required=True,
                rationale="Invoice records with due date and payment status",
                short_name="invoice_transactions",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier", "creditor"],
                is_required=True,
                rationale="Vendor grouping for aging summary",
                short_name="vendor_master",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "due", "maturity", "payment"],
                is_required=True,
                rationale="Due date for overdue calculation",
                short_name="payment_date",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["payment", "clearing", "settlement"],
                is_required=False,
                rationale="Payment records to determine unpaid status",
                short_name="payment_transactions",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="vendor_master",
                target_short_name="invoice_transactions",
                is_required=True,
                rationale="Vendor-invoice join for aging by vendor",
            ),
        ],
        min_completeness=0.75,
        complexity="two_table",
        truth_reqs=[
            "invoice_domain_present",
            "vendor_resolution_possible",
            "due_date_field_present",
        ],
    ),

    # ── 6. Open liabilities ───────────────────────────────────────────────────
    _b(
        id="WF-006",
        query="What are our total open liabilities and accruals by cost center?",
        category="open_liabilities",
        description=(
            "Liability and accrual reporting requires GL or journal entries "
            "with liability account codes, plus cost center hierarchy."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["liability", "accrual", "payable", "journal", "gl", "ledger"],
                is_required=True,
                rationale="GL/liability transaction records",
                short_name="liability_transactions",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["cost_center", "department", "org", "division"],
                is_required=True,
                rationale="Cost center master for grouping",
                short_name="cost_center",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["account", "gl_account", "code", "chart"],
                is_required=False,
                rationale="Account dimension for liability classification",
                short_name="account_dimension",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="cost_center",
                target_short_name="liability_transactions",
                is_required=True,
                rationale="Cost center join for grouping",
            ),
        ],
        min_completeness=0.70,
        complexity="two_table",
        truth_reqs=[
            "liability_domain_present",
            "cost_center_resolution_possible",
        ],
    ),

    # ── 7. Vendor reconciliation ──────────────────────────────────────────────
    _b(
        id="WF-007",
        query="Reconcile vendor statements — identify discrepancies between vendor invoices and our records",
        category="vendor_reconciliation",
        description=(
            "Reconciliation requires invoice records, payment records, "
            "vendor master, and potentially vendor-provided statement data."
        ),
        domains=[
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier", "creditor"],
                is_required=True,
                rationale="Vendor master is the reconciliation anchor",
                short_name="vendor_master",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["invoice", "bill", "payable"],
                is_required=True,
                rationale="Our invoice records for comparison",
                short_name="invoice_transactions",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["payment", "clearing", "settlement", "remittance"],
                is_required=True,
                rationale="Payment records to compute outstanding balance",
                short_name="payment_transactions",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["credit_note", "debit_note", "adjustment"],
                is_required=False,
                rationale="Adjustment records affect reconciliation balance",
                short_name="adjustment_transactions",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="vendor_master",
                target_short_name="invoice_transactions",
                is_required=True,
                rationale="Vendor-invoice join",
            ),
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="vendor_master",
                target_short_name="payment_transactions",
                is_required=True,
                rationale="Vendor-payment join",
            ),
        ],
        min_completeness=0.75,
        complexity="multi_table",
        truth_reqs=[
            "vendor_domain_present",
            "invoice_domain_present",
            "payment_domain_present",
        ],
        weight=1.5,
    ),

    # ── 8. PO lifecycle analysis ──────────────────────────────────────────────
    _b(
        id="WF-008",
        query="Show the full lifecycle of purchase orders from creation to payment",
        category="po_lifecycle",
        description=(
            "End-to-end PO lifecycle: requisition → PO → GRN → invoice → payment. "
            "This is the most complex multi-stage workflow — needs all 5 domains."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po"],
                is_required=True,
                rationale="PO is the lifecycle anchor",
                short_name="purchase_order",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["receipt", "grn", "goods_receipt"],
                is_required=True,
                rationale="GRN marks physical delivery",
                short_name="goods_receipt",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["invoice", "payable", "bill"],
                is_required=True,
                rationale="Invoice is the financial trigger",
                short_name="invoice",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["payment", "clearing", "settlement"],
                is_required=True,
                rationale="Payment closes the lifecycle",
                short_name="payment",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier"],
                is_required=True,
                rationale="Vendor is the common entity linking the chain",
                short_name="vendor_master",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="purchase_order",
                target_short_name="goods_receipt",
                is_required=True,
                rationale="PO-GRN join in lifecycle",
            ),
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="invoice",
                target_short_name="payment",
                is_required=True,
                rationale="Invoice-payment join closes the cycle",
            ),
        ],
        min_completeness=0.80,
        complexity="multi_table",
        truth_reqs=[
            "full_po_chain_retrievable",
            "vendor_resolution_possible",
            "payment_domain_present",
        ],
        weight=2.0,
    ),

    # ── 9. GRN mismatches ─────────────────────────────────────────────────────
    _b(
        id="WF-009",
        query="Find goods receipts where the received quantity doesn't match the PO quantity",
        category="grn_mismatches",
        description=(
            "Quantity mismatch between ordered and received requires both PO "
            "and GRN with matching reference keys."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po", "order"],
                is_required=True,
                rationale="PO with ordered quantity",
                short_name="purchase_order",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["receipt", "grn", "goods_receipt", "receiving"],
                is_required=True,
                rationale="GRN with received quantity",
                short_name="goods_receipt",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["material", "item", "product", "sku"],
                is_required=False,
                rationale="Material master for item description",
                short_name="material_master",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="purchase_order",
                target_short_name="goods_receipt",
                is_required=True,
                rationale="PO-GRN join for quantity comparison",
            ),
        ],
        min_completeness=0.80,
        complexity="two_table",
        truth_reqs=[
            "po_grn_join_available",
            "quantity_fields_present",
        ],
    ),

    # ── 10. Procurement bottlenecks ───────────────────────────────────────────
    _b(
        id="WF-010",
        query="Identify procurement bottlenecks — where are purchase requests getting stuck?",
        category="procurement_bottlenecks",
        description=(
            "Process performance analysis. Requires request/requisition records "
            "with stage transitions and time-in-stage metrics."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["requisition", "request", "purchase_request", "pr"],
                is_required=True,
                rationale="Purchase request records with stage data",
                short_name="requisition",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["status", "stage", "workflow", "state"],
                is_required=True,
                rationale="Stage/status dimension for bottleneck identification",
                short_name="stage_dimension",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "time", "created", "updated", "processed"],
                is_required=True,
                rationale="Timestamps for time-in-stage calculation",
                short_name="process_dates",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["purchase_order", "po"],
                is_required=False,
                rationale="PO conversion rate for bottleneck attribution",
                short_name="purchase_order",
            ),
        ],
        topology=[],
        min_completeness=0.65,
        complexity="two_table",
        truth_reqs=[
            "process_timing_possible",
            "stage_domain_present",
        ],
    ),

    # ── 11. Cost center leakage ────────────────────────────────────────────────
    _b(
        id="WF-011",
        query="Which cost centers have budget overruns or unauthorized spending?",
        category="cost_center_leakage",
        description=(
            "Budget vs. actual analysis by cost center. Requires commitment/actual "
            "spend records and budget allocation data."
        ),
        domains=[
            DomainRequirement(
                role_type="entity",
                label_hints=["cost_center", "department", "org_unit"],
                is_required=True,
                rationale="Cost center master for grouping",
                short_name="cost_center",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["budget", "allocation", "plan"],
                is_required=True,
                rationale="Budget plan records for variance calculation",
                short_name="budget",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["actual", "spend", "cost", "expense", "journal"],
                is_required=True,
                rationale="Actual spend records for comparison",
                short_name="actual_spend",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["period", "fiscal", "year", "month"],
                is_required=False,
                rationale="Fiscal period for budget cycle alignment",
                short_name="fiscal_period",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="cost_center",
                target_short_name="actual_spend",
                is_required=True,
                rationale="Cost center attribution join",
            ),
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="cost_center",
                target_short_name="budget",
                is_required=True,
                rationale="Budget-center join for variance",
            ),
        ],
        min_completeness=0.75,
        complexity="multi_table",
        truth_reqs=[
            "budget_domain_present",
            "actual_spend_domain_present",
            "cost_center_resolution_possible",
        ],
    ),

    # ── 12. Payment aging ─────────────────────────────────────────────────────
    _b(
        id="WF-012",
        query="Show payment aging analysis — how long are invoices taking to get paid?",
        category="payment_aging",
        description=(
            "Days-to-pay analysis. Requires invoice records with issue date, "
            "payment records with value date, and vendor entity."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["invoice", "payable", "bill"],
                is_required=True,
                rationale="Invoice records with issue date",
                short_name="invoice_transactions",
            ),
            DomainRequirement(
                role_type="transaction",
                label_hints=["payment", "clearing", "settlement"],
                is_required=True,
                rationale="Payment records with value date",
                short_name="payment_transactions",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["vendor", "supplier"],
                is_required=True,
                rationale="Vendor for aging aggregation",
                short_name="vendor_master",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "due", "cleared", "paid"],
                is_required=True,
                rationale="Date dimensions for aging calculation",
                short_name="payment_dates",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="invoice_transactions",
                target_short_name="payment_transactions",
                is_required=True,
                rationale="Invoice-payment join for days-to-pay",
            ),
        ],
        min_completeness=0.75,
        complexity="multi_table",
        truth_reqs=[
            "invoice_domain_present",
            "payment_domain_present",
            "date_comparison_possible",
        ],
    ),

    # ── 13. Inventory movement anomalies ──────────────────────────────────────
    _b(
        id="WF-013",
        query="Identify inventory movement anomalies — items with unexpected stock fluctuations",
        category="inventory_movement",
        description=(
            "Inventory analysis requires stock movement records, material master, "
            "and optionally warehouse/location entity."
        ),
        domains=[
            DomainRequirement(
                role_type="transaction",
                label_hints=["inventory", "stock", "movement", "transfer", "issue", "receipt"],
                is_required=True,
                rationale="Stock movement records are the primary subject",
                short_name="stock_movements",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["material", "item", "product", "sku", "article"],
                is_required=True,
                rationale="Material master for item context",
                short_name="material_master",
            ),
            DomainRequirement(
                role_type="entity",
                label_hints=["plant", "warehouse", "location", "storage"],
                is_required=False,
                rationale="Location entity for spatial analysis",
                short_name="location_entity",
            ),
            DomainRequirement(
                role_type="dimension",
                label_hints=["date", "period", "posting"],
                is_required=True,
                rationale="Posting date for time-series anomaly detection",
                short_name="posting_date",
            ),
        ],
        topology=[
            TopologyRequirement(
                join_type="entity-transaction",
                source_short_name="material_master",
                target_short_name="stock_movements",
                is_required=True,
                rationale="Material-movement join for item context",
            ),
        ],
        min_completeness=0.70,
        complexity="two_table",
        truth_reqs=[
            "inventory_domain_present",
            "material_resolution_possible",
        ],
    ),
]

# ── Public API ─────────────────────────────────────────────────────────────────

def load_benchmarks(
    categories: list[str] | None = None,
    complexity: list[str] | None = None,
) -> list[WorkflowBenchmark]:
    """
    Load benchmarks, optionally filtered by category or complexity.

    Args:
        categories:  filter to specific workflow categories
        complexity:  filter to "single_table", "two_table", or "multi_table"

    Returns:
        List of WorkflowBenchmark objects
    """
    results = BENCHMARK_REGISTRY
    if categories:
        results = [b for b in results if b.workflow_category in categories]
    if complexity:
        results = [b for b in results if b.complexity in complexity]
    return results


def get_benchmark(benchmark_id: str) -> WorkflowBenchmark | None:
    """Get a single benchmark by ID (e.g., "WF-001")."""
    for b in BENCHMARK_REGISTRY:
        if b.id == benchmark_id:
            return b
    return None

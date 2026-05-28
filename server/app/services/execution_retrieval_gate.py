"""Deterministic gate between retrieval and execution planning.

Retrieval may surface a broad relevance set.  Execution planning should only
see a bounded, coherent slice of tables with plausible operational authority.
This module is pure in-memory scoring: no LLM calls, DB reads, or new semantic
state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable

from app.services.workflow_cognition import infer_workflow_primitives


_REPORTING_TERMS = frozenset({
    "dashboard", "report", "reporting", "snapshot", "analytics", "analytic",
    "curated", "derived", "mart",
})
_OPERATIONAL_TERMS = frozenset({
    "open", "pending", "status", "current", "approval", "approved", "delivery",
    "delivered", "shipment", "receipt", "receipts", "receiving", "invoice",
    "invoices", "payment", "payments", "unpaid", "uninvoiced", "matched",
    "unmatched", "exception", "exceptions", "aging", "delay", "delayed",
    "lifecycle", "workflow", "order", "orders", "line", "lines", "detail",
})
_MULTI_DOMAIN_TERMS = frozenset({
    "across", "end", "end-to-end", "lifecycle", "workflow", "trace", "chain",
    "procure", "order", "cash", "invoice", "payment", "fulfillment",
})
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "show", "the", "to", "use", "what",
    "with",
})


@dataclass(frozen=True)
class ExecutionGateDecision:
    file_id: str
    blob_path: str
    selected: bool
    score: float
    authority_class: str
    source_type: str
    workflow_tags: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionRetrievalGateResult:
    selected_catalog: list[dict]
    decisions: tuple[ExecutionGateDecision, ...]
    original_count: int
    selected_count: int
    max_tables: int
    operational_intent: bool
    transformed_allowed: bool

    @property
    def suppressed_count(self) -> int:
        return sum(1 for decision in self.decisions if not decision.selected)

    @property
    def transformed_suppressed_count(self) -> int:
        return sum(
            1
            for decision in self.decisions
            if not decision.selected
            and decision.authority_class in {"transformed_derived", "reporting_snapshot", "staging_extract"}
        )

    @property
    def cap_hit(self) -> bool:
        return self.original_count > self.selected_count >= self.max_tables

    def to_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "selected_count": self.selected_count,
            "suppressed_count": self.suppressed_count,
            "transformed_suppressed_count": self.transformed_suppressed_count,
            "max_tables": self.max_tables,
            "operational_intent": self.operational_intent,
            "transformed_allowed": self.transformed_allowed,
            "cap_hit": self.cap_hit,
            "selected": [
                {
                    "file_id": d.file_id,
                    "blob_path": d.blob_path,
                    "score": round(d.score, 3),
                    "authority_class": d.authority_class,
                    "workflow_tags": list(d.workflow_tags),
                }
                for d in self.decisions
                if d.selected
            ],
            "suppressed": [
                {
                    "file_id": d.file_id,
                    "blob_path": d.blob_path,
                    "score": round(d.score, 3),
                    "authority_class": d.authority_class,
                    "reasons": list(d.reasons),
                }
                for d in self.decisions
                if not d.selected
            ][:20],
        }


def narrow_catalog_for_execution(
    *,
    query: str,
    intent_plan,
    catalog: list[dict],
    max_tables: int = 12,
    deep_workflow_max_tables: int = 15,
    multi_domain_max_tables: int = 20,
    suppress_transformed: bool = True,
) -> ExecutionRetrievalGateResult:
    """Return the deterministic execution slice used by prompt and tools.

    The input order is preserved as a small prior, but operational authority and
    workflow fit dominate.  Explicitly mentioned files are retained because the
    user named them, while transformed/reporting extracts are suppressed for
    operational questions unless the query asks for reporting-style artifacts.
    """
    if not catalog:
        return ExecutionRetrievalGateResult(
            selected_catalog=[],
            decisions=(),
            original_count=0,
            selected_count=0,
            max_tables=max_tables,
            operational_intent=False,
            transformed_allowed=False,
        )

    query_tokens = tuple(_tokenize(query))
    token_set = set(query_tokens)
    transformed_allowed = bool(token_set & _REPORTING_TERMS)
    operational_intent = bool(token_set & _OPERATIONAL_TERMS) or not transformed_allowed
    target_domains = _query_domains(token_set)
    target_objects = _query_objects(token_set)
    target_process = _query_process(token_set)
    behavior_set = {str(b).lower() for b in (getattr(intent_plan, "behaviors", []) or [])}
    table_cap = _resolve_cap(
        token_set=token_set,
        behavior_set=behavior_set,
        max_tables=max_tables,
        deep_workflow_max_tables=deep_workflow_max_tables,
        multi_domain_max_tables=multi_domain_max_tables,
    )

    scored: list[tuple[float, int, dict, ExecutionGateDecision]] = []
    forced: list[tuple[float, int, dict, ExecutionGateDecision]] = []
    suppressed: list[ExecutionGateDecision] = []

    for rank, entry in enumerate(catalog):
        primitive = infer_workflow_primitives(entry)
        authority_class = _authority_class(primitive.transactional_authority.source_type)
        workflow_tags = tuple(sorted(set(
            primitive.operational_domains
            + primitive.business_objects
            + primitive.process_signals
        )))
        explicit = _explicitly_mentioned(query, entry)
        score, reasons = _score_entry(
            rank=rank,
            authority_class=authority_class,
            authority_score=primitive.transactional_authority.score,
            target_domains=target_domains,
            target_objects=target_objects,
            target_process=target_process,
            workflow_domains=set(primitive.operational_domains),
            workflow_objects=set(primitive.business_objects),
            workflow_process=set(primitive.process_signals),
            explicit=explicit,
            operational_intent=operational_intent,
            transformed_allowed=transformed_allowed,
        )

        should_suppress = (
            suppress_transformed
            and operational_intent
            and not transformed_allowed
            and not explicit
            and authority_class in {"transformed_derived", "reporting_snapshot", "staging_extract", "unknown"}
        )
        if should_suppress:
            reasons.append(f"suppressed_{authority_class}_for_operational_query")

        decision = ExecutionGateDecision(
            file_id=str(entry.get("file_id") or ""),
            blob_path=str(entry.get("blob_path") or ""),
            selected=not should_suppress,
            score=score,
            authority_class=authority_class,
            source_type=primitive.transactional_authority.source_type,
            workflow_tags=workflow_tags,
            reasons=tuple(reasons),
        )
        if should_suppress:
            suppressed.append(decision)
            continue

        annotated = {
            **entry,
            "execution_authority_class": authority_class,
            "execution_workflow_tags": list(workflow_tags),
        }
        target = forced if explicit else scored
        target.append((score, rank, annotated, decision))

    considered_ranked = forced + sorted(scored, key=lambda item: (-item[0], item[1]))
    if not considered_ranked:
        fallback = _fallback_selection(catalog, table_cap)
        decisions = tuple(
            ExecutionGateDecision(
                file_id=str(entry.get("file_id") or ""),
                blob_path=str(entry.get("blob_path") or ""),
                selected=True,
                score=0.0,
                authority_class="unknown",
                source_type="unknown",
                reasons=("fallback_no_authoritative_candidates",),
            )
            for entry in fallback
        ) + tuple(suppressed)
        return ExecutionRetrievalGateResult(
            selected_catalog=fallback,
            decisions=decisions,
            original_count=len(catalog),
            selected_count=len(fallback),
            max_tables=table_cap,
            operational_intent=operational_intent,
            transformed_allowed=transformed_allowed,
        )

    selected_ranked = considered_ranked[:table_cap]
    selected_ids = {item[3].file_id for item in selected_ranked if item[3].file_id}
    selected_catalog = [item[2] for item in selected_ranked]

    cap_suppressed: list[ExecutionGateDecision] = []
    for _score, _rank, _entry, decision in considered_ranked:
        if decision.file_id in selected_ids:
            continue
        cap_suppressed.append(_replace_decision(decision, selected=False, reasons=decision.reasons + ("planner_context_cap",)))

    selected_decisions = tuple(_replace_decision(item[3], selected=True) for item in selected_ranked)
    all_decisions = selected_decisions + tuple(suppressed) + tuple(cap_suppressed)
    return ExecutionRetrievalGateResult(
        selected_catalog=selected_catalog,
        decisions=all_decisions,
        original_count=len(catalog),
        selected_count=len(selected_catalog),
        max_tables=table_cap,
        operational_intent=operational_intent,
        transformed_allowed=transformed_allowed,
    )


def render_execution_gate_note(result: ExecutionRetrievalGateResult) -> str:
    if not result.decisions:
        return ""
    selected_classes = _counts(d.authority_class for d in result.decisions if d.selected)
    suppressed_classes = _counts(d.authority_class for d in result.decisions if not d.selected)
    return (
        "EXECUTION RETRIEVAL GATE\n"
        f"Planner context was narrowed from {result.original_count} to {result.selected_count} "
        f"files (cap {result.max_tables}).\n"
        f"Selected authority classes: {selected_classes or 'none'}.\n"
        f"Suppressed authority classes: {suppressed_classes or 'none'}.\n"
        "Only the selected logical tables are in the executable scope for this request."
    )


def _score_entry(
    *,
    rank: int,
    authority_class: str,
    authority_score: float,
    target_domains: set[str],
    target_objects: set[str],
    target_process: set[str],
    workflow_domains: set[str],
    workflow_objects: set[str],
    workflow_process: set[str],
    explicit: bool,
    operational_intent: bool,
    transformed_allowed: bool,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = max(0.0, 1.0 - (rank * 0.025))
    score += min(0.5, max(0.0, float(authority_score)) * 0.5)
    score += {
        "transactional_source": 1.2,
        "source_like_extract": 1.0,
        "validated_reference": 0.55,
        "reporting_snapshot": 0.25 if transformed_allowed else -0.35,
        "transformed_derived": 0.15 if transformed_allowed else -0.55,
        "staging_extract": -0.45,
        "unknown": -0.25,
    }.get(authority_class, -0.2)

    domain_overlap = target_domains & workflow_domains
    object_overlap = target_objects & workflow_objects
    process_overlap = target_process & workflow_process
    if domain_overlap:
        score += 0.6 + (0.15 * len(domain_overlap))
        reasons.append("domain_fit")
    if object_overlap:
        score += 0.45 + (0.1 * len(object_overlap))
        reasons.append("business_object_fit")
    if process_overlap:
        score += 0.35 + (0.08 * len(process_overlap))
        reasons.append("process_fit")
    if operational_intent and authority_class in {"transactional_source", "source_like_extract"}:
        reasons.append("operational_authority")
    if authority_class == "validated_reference":
        reasons.append("reference_support")
    if explicit:
        score += 2.0
        reasons.append("explicit_user_reference")
    return score, reasons or ["retrieval_rank_prior"]


def _authority_class(source_type: str) -> str:
    value = str(source_type or "unknown")
    if value == "transactional_source":
        return "transactional_source"
    if value == "source_like_extract":
        return "source_like_extract"
    if value == "reference_master":
        return "validated_reference"
    if value == "transformed_analytics":
        return "transformed_derived"
    if value == "sample_extract":
        return "staging_extract"
    return "unknown"


def _resolve_cap(
    *,
    token_set: set[str],
    behavior_set: set[str],
    max_tables: int,
    deep_workflow_max_tables: int,
    multi_domain_max_tables: int,
) -> int:
    if token_set & _MULTI_DOMAIN_TERMS or {"multi_step", "workflow", "cross_domain"} & behavior_set:
        return max(1, min(multi_domain_max_tables, 20))
    if {"detail_rows", "open_items", "time_filtered"} & behavior_set:
        return max(1, min(deep_workflow_max_tables, 15))
    return max(1, min(max_tables, 15))


def _query_domains(token_set: set[str]) -> set[str]:
    domains: set[str] = set()
    if token_set & {"ap", "payable", "payables", "supplier", "vendor", "invoice", "invoices"}:
        domains.add("accounts_payable")
    if token_set & {"po", "purchase", "procurement", "receiving", "received", "receipt", "receipts"}:
        domains.add("procurement")
    if token_set & {"ar", "receivable", "receivables", "customer", "cash"}:
        domains.add("accounts_receivable")
    if token_set & {"sales", "billing", "shipment", "delivery"}:
        domains.add("sales")
    if token_set & {"warehouse", "ewm", "shipping", "fulfillment", "fulfilment"}:
        domains.add("logistics")
    if token_set & {"gl", "ledger", "journal", "accounting"}:
        domains.add("general_ledger")
    return domains


def _tokenize(text: str) -> list[str]:
    expanded = str(text or "").lower().replace("_", " ").replace("-", " ")
    return [token for token in _TOKEN_RE.findall(expanded) if token and token not in _STOPWORDS]


def _query_objects(token_set: set[str]) -> set[str]:
    objects: set[str] = set()
    if "po" in token_set or {"purchase", "order"} <= token_set:
        objects.add("purchase_order")
    if token_set & {"invoice", "invoices", "invoiced", "uninvoiced"}:
        objects.add("invoice")
    if token_set & {"supplier", "vendor"}:
        objects.add("supplier")
    if token_set & {"customer", "cust"}:
        objects.add("customer")
    if token_set & {"payment", "payments", "paid", "check"}:
        objects.add("payment")
    if token_set & {"receipt", "receipts", "receiving", "received"}:
        objects.add("goods_receipt")
    if token_set & {"delivery", "shipment", "shipping"}:
        objects.add("delivery")
    if token_set & {"ledger", "journal", "gl"}:
        objects.add("ledger_entry")
    return objects


def _query_process(token_set: set[str]) -> set[str]:
    process: set[str] = set()
    if token_set & {"approval", "approved", "approvals", "pending"}:
        process.add("approval")
    if token_set & {"status", "state", "open", "closed", "current", "pending"}:
        process.add("status")
    if token_set & {"delivery", "delivered", "shipment", "shipping"}:
        process.add("delivery")
    if token_set & {"receiving", "receipt", "receipts", "received", "grn"}:
        process.add("receiving")
    if token_set & {"invoice", "invoices", "invoiced", "uninvoiced"}:
        process.add("invoice")
    if token_set & {"match", "matching", "matched", "unmatched", "reconcile", "reconciliation", "variance"}:
        process.add("reconciliation")
    if token_set & {"payment", "payments", "paid", "payable", "check", "cash"}:
        process.add("payment")
    if token_set & {"lifecycle", "bottleneck", "delay", "delayed", "aging", "exception", "exceptions"}:
        process.add("lifecycle")
    return process


def _explicitly_mentioned(query: str, entry: dict) -> bool:
    query_norm = _normalise_name(query)
    candidates = {
        str(entry.get("file_id") or ""),
        str(entry.get("blob_path") or ""),
        PurePosixPath(str(entry.get("blob_path") or "")).name,
    }
    for candidate in candidates:
        if not candidate:
            continue
        candidate_norm = _normalise_name(candidate)
        if candidate_norm and candidate_norm in query_norm:
            return True
    return False


def _normalise_name(value: str) -> str:
    value = _HASH_PREFIX_RE.sub("", str(value or "").lower())
    value = value.rsplit("/", 1)[-1]
    value = value.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _counts(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _fallback_selection(catalog: list[dict], table_cap: int) -> list[dict]:
    return [
        {
            **entry,
            "execution_authority_class": "unknown",
            "execution_workflow_tags": [],
        }
        for entry in catalog[:table_cap]
    ]


def _replace_decision(
    decision: ExecutionGateDecision,
    *,
    selected: bool | None = None,
    reasons: tuple[str, ...] | None = None,
) -> ExecutionGateDecision:
    return ExecutionGateDecision(
        file_id=decision.file_id,
        blob_path=decision.blob_path,
        selected=decision.selected if selected is None else selected,
        score=decision.score,
        authority_class=decision.authority_class,
        source_type=decision.source_type,
        workflow_tags=decision.workflow_tags,
        reasons=decision.reasons if reasons is None else reasons,
    )
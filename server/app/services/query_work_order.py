"""Request-local work order for business analytics questions.

The work order is intentionally not a source-system ontology. It separates
what the user wants to see from what might anchor source discovery, then builds
bounded metadata search variants. Runtime tools still have to inspect evidence
before SQL execution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[&'][a-z0-9]+)*", re.I)


@dataclass(frozen=True)
class WorkOrderFilter:
    kind: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True)
class QueryWorkOrder:
    original_query: str
    task_type: str
    requested_outputs: list[str] = field(default_factory=list)
    source_anchor_terms: list[str] = field(default_factory=list)
    filter_terms: list[str] = field(default_factory=list)
    filters: list[WorkOrderFilter] = field(default_factory=list)
    source_evidence_needs: list[str] = field(default_factory=list)
    candidate_search_queries: list[str] = field(default_factory=list)
    must_inspect_before_sql: bool = False
    ambiguity_flags: list[str] = field(default_factory=list)
    missing_reference_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "requested_outputs": self.requested_outputs,
            "source_anchor_terms": self.source_anchor_terms,
            "filter_terms": self.filter_terms,
            "filters": [item.to_dict() for item in self.filters],
            "source_evidence_needs": self.source_evidence_needs,
            "candidate_search_queries": self.candidate_search_queries,
            "must_inspect_before_sql": self.must_inspect_before_sql,
            "ambiguity_flags": self.ambiguity_flags,
            "missing_reference_questions": self.missing_reference_questions,
        }


def _clean_label(value: str) -> str:
    normalized = re.sub(r"[_\s]+", " ", str(value or "").casefold())
    return " ".join(_TOKEN_RE.findall(normalized))


def _dedup(items: list[str], *, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = _clean_label(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _filters_from_constraints(constraints: dict) -> tuple[list[WorkOrderFilter], list[str]]:
    filters: list[WorkOrderFilter] = []
    terms: list[str] = []
    for key, value in (constraints or {}).items():
        if value is None:
            continue
        text_value = str(value)
        filters.append(WorkOrderFilter(kind=str(key), value=text_value))
        terms.extend([str(key), text_value])
    return filters, _dedup(terms, limit=8)


def _task_type(intent_plan: Any) -> str:
    if len(getattr(intent_plan, "output_terms", []) or []) > 1:
        return "multi_output_analysis"
    return getattr(intent_plan, "intent", None) or "analysis"


def _search_variants(
    *,
    original_query: str,
    source_terms: list[str],
    output_terms: list[str],
    filter_terms: list[str],
    evidence_needs: list[str],
) -> list[str]:
    variants: list[str] = [original_query]
    filter_text = " ".join(filter_terms[:2])
    source_text = " ".join(source_terms[:3])
    output_text = " ".join(output_terms[:5])

    if source_text:
        variants.append(" ".join(filter(None, [source_text, filter_text])))
    if source_text and output_text:
        variants.append(" ".join(filter(None, [source_text, output_text, filter_text])))
    if output_text:
        variants.append(" ".join(filter(None, [output_text, filter_text])))
    for need in evidence_needs[:6]:
        variants.append(" ".join(filter(None, [source_text, need, filter_text])))
    return _dedup(variants, limit=10)


def build_query_work_order(query: str, intent_plan: Any) -> QueryWorkOrder:
    """Build a bounded source-evidence plan from planner output.

    The function is deliberately generic: it never maps business terms to fixed
    table names. It only preserves role-separated terms and search variants that
    downstream retrieval/inspection can prove or reject.
    """
    source_terms = _dedup(list(getattr(intent_plan, "source_anchor_terms", []) or []))
    output_terms = _dedup(list(getattr(intent_plan, "output_terms", []) or []))
    plan_filter_terms = _dedup(list(getattr(intent_plan, "filter_terms", []) or []), limit=8)
    filters, constraint_terms = _filters_from_constraints(getattr(intent_plan, "constraints", {}) or {})
    filter_terms = _dedup(plan_filter_terms + constraint_terms, limit=8)

    evidence_needs = _dedup(source_terms + output_terms, limit=12)
    if not evidence_needs:
        evidence_needs = _dedup(list(getattr(intent_plan, "entities", []) or []), limit=8)

    ambiguity_flags: list[str] = []
    if not source_terms:
        ambiguity_flags.append("no_explicit_source_anchor")
    if len(output_terms) > 1:
        ambiguity_flags.append("multi_output_request")

    candidate_search_queries = _search_variants(
        original_query=query,
        source_terms=source_terms,
        output_terms=output_terms,
        filter_terms=filter_terms,
        evidence_needs=evidence_needs,
    )
    must_inspect = len(output_terms) > 1 or "multi_step" in set(getattr(intent_plan, "behaviors", []) or [])

    return QueryWorkOrder(
        original_query=query,
        task_type=_task_type(intent_plan),
        requested_outputs=output_terms,
        source_anchor_terms=source_terms,
        filter_terms=filter_terms,
        filters=filters,
        source_evidence_needs=evidence_needs,
        candidate_search_queries=candidate_search_queries,
        must_inspect_before_sql=must_inspect,
        ambiguity_flags=ambiguity_flags,
        missing_reference_questions=[],
    )
"""Render a compact BrainContext prompt slice."""
from __future__ import annotations

from app.services.brain_context import BrainContext


def render_brain_context_prompt(context: BrainContext | None) -> str:
    if not context or (not context.records and not context.domains):
        return ""

    lines = [
        "--- GOVERNED SEMANTIC MEMORY ---",
        "Use as compact workflow/domain guidance only; it does not authorize files, joins, SQL, or execution.",
    ]
    for domain in context.domains[: context.caps.get("max_domains", 6)]:
        terms = ", ".join(domain.terms[:4])
        workflow = ", ".join(domain.workflow_terms[:3])
        lifecycle = ", ".join(domain.lifecycle_terms[:3])
        kpi = ", ".join(domain.kpi_terms[:3])
        hints = "; ".join(filter(None, [
            f"workflow={workflow}" if workflow else "",
            f"lifecycle={lifecycle}" if lifecycle else "",
            f"kpi={kpi}" if kpi else "",
        ]))
        lines.append(
            f"  - domain:{domain.domain_type}:{domain.title[:72]} "
            f"(files={len(domain.contributor_file_ids)}; authority={domain.authority_score:.2f}; "
            f"drift={domain.drift_score:.2f}; conflicts={domain.conflict_count}; terms={terms})"
            + (f" [{hints}]" if hints else "")
        )
    for record in context.records[: context.caps.get("max_records", 8)]:
        terms = ", ".join(record.terms[:5])
        summary = f"; {record.summary[:120]}" if record.summary else ""
        lines.append(
            f"  - {record.memory_type}:{record.title[:80]} "
            f"(authority={record.authority_score:.2f}; confidence={record.confidence_score:.2f}; terms={terms}){summary}"
        )
    if context.retrieval_guidance.topology_hints:
        lines.append("workflow topology hints: " + ", ".join(context.retrieval_guidance.topology_hints[:5]))
    if context.retrieval_guidance.ambiguity_flags:
        lines.append("ambiguity flags: " + ", ".join(context.retrieval_guidance.ambiguity_flags[:5]))
    envelope = context.execution_envelope
    if envelope.shortlist_file_ids or envelope.execution_mode:
        lines.append(
            "execution envelope: "
            f"mode={envelope.execution_mode or 'pending'}; "
            f"shortlist={len(envelope.shortlist_file_ids)}; "
            f"approved_joins={envelope.approved_join_count}"
        )
    lines.append("---")
    return "\n".join(lines)
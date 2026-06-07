"""Per-file TRUST STATE — the shared Phase-1 trust/quarantine contract.

PURPOSE
=======
Collapse the two EXISTING ingestion-quality signals — the per-file
``ingestion_confidence`` level bucket (high/medium/low) and the container-level
``ingestion_audit`` findings (severity error/warning/info) — into a single
coarse, downstream-consumable TRUST STATE per file:

    trusted     — usable as-is for retrieval / SQL context
    shadow      — usable but the consumer should attenuate / caveat
    quarantined — knowledge is structurally unreliable; consumer hard-excludes it
                  (only when SME_QUARANTINE_ENABLED is on — see callers)

DESIGN CONSTRAINTS (reviewer board)
===================================
- 100% DATA-DRIVEN. ``derive_trust_state`` introduces NO new threshold, no
  name-list, no magic literal. It consumes the buckets/severities those upstream
  modules ALREADY produce from the data:
    * confidence level comes from ``ingestion_confidence.IngestionConfidence.level``
      (itself derived from policy-governed score thresholds).
    * audit severity comes from ``ingestion_audit.AuditFinding.severity``.
  This module only maps those existing discrete buckets onto a state — it does
  not re-score anything.
- PURE function: no I/O, no DB, no LLM. Trivially unit-testable.
- This module is the single source of truth for the three state strings so that
  the producer (this side) and the consumer (Dev-B's retrieval/SQL gate) agree.
"""
from __future__ import annotations

from typing import Any, Iterable

# ── Canonical state strings (imported by both producer and consumer) ───────────
TRUSTED = "trusted"
SHADOW = "shadow"
QUARANTINED = "quarantined"

# Severity string that upstream ``ingestion_audit.AuditFinding`` uses for a
# blocking quality problem. Defined as a constant (not inlined) so the contract
# stays explicit; it mirrors the value the audit already emits.
_AUDIT_ERROR_SEVERITY = "error"

# Confidence level buckets emitted by ``ingestion_confidence`` (high/medium/low).
# We only need to recognise the two non-default buckets; anything else (incl.
# "high" and None) maps to TRUSTED.
_LEVEL_QUARANTINE = "low"
_LEVEL_SHADOW = "medium"


def _has_error_finding(audit_findings: Iterable[Any] | None) -> bool:
    """True if any audit finding is of ``error`` severity.

    Tolerant of both shapes the audit can hand us:
      * ``AuditFinding`` dataclass instances (have a ``.severity`` attr), and
      * plain dicts (the ``AuditResult.to_dict()`` / signals-stashed form,
        with a ``"severity"`` key).
    """
    if not audit_findings:
        return False
    for finding in audit_findings:
        severity = getattr(finding, "severity", None)
        if severity is None and isinstance(finding, dict):
            severity = finding.get("severity")
        if severity == _AUDIT_ERROR_SEVERITY:
            return True
    return False


def derive_trust_state(
    confidence_level: str | None,
    audit_findings: list | None,
) -> str:
    """Map existing ingestion-quality signals onto a coarse trust state.

    Precedence (data-driven, no new thresholds):
      1. ANY audit finding of ``error`` severity  → QUARANTINED
         (a structural defect the audit already classified as blocking).
      2. confidence_level == "low"                → QUARANTINED
         (the confidence scorer already bucketed this file as unreliable).
      3. confidence_level == "medium"             → SHADOW
         (usable with caution per the scorer's own bucket).
      4. otherwise (incl. "high" / None)          → TRUSTED.

    Args:
        confidence_level: ``IngestionConfidence.level`` ("high"|"medium"|"low")
                          or None when confidence was not computed.
        audit_findings:   list of ``AuditFinding`` (or their dict form), or None.

    Returns:
        One of TRUSTED / SHADOW / QUARANTINED.
    """
    if _has_error_finding(audit_findings):
        return QUARANTINED

    level = (confidence_level or "").strip().lower()
    if level == _LEVEL_QUARANTINE:
        return QUARANTINED
    if level == _LEVEL_SHADOW:
        return SHADOW
    return TRUSTED


def findings_for_file(audit_findings: Iterable[Any] | None, file_id: str) -> list:
    """Return ONLY the findings that implicate this specific ``file_id``.

    ``run_ingestion_audit`` is container-scoped: its findings carry the
    implicated files in ``context["files"]`` or ``context["file_ids"]`` (the
    audit records them). Without this filter, a single container-level defect
    (e.g. one ``conflicting_role_kinds`` error) would be handed to EVERY file and
    quarantine the whole container — punishing clean files for a defect they do
    not own. Callers pass ``findings_for_file(audit.findings, meta.file_id)`` into
    ``derive_trust_state`` so each file is judged only on its OWN findings.

    A finding with NO file attribution is container-level and is intentionally
    NOT charged to any individual file.
    """
    if not audit_findings:
        return []
    out: list = []
    for finding in audit_findings:
        ctx = getattr(finding, "context", None)
        if ctx is None and isinstance(finding, dict):
            ctx = finding.get("context")
        ctx = ctx or {}
        implicated = list(ctx.get("files", []) or []) + list(ctx.get("file_ids", []) or [])
        if file_id in implicated:
            out.append(finding)
    return out

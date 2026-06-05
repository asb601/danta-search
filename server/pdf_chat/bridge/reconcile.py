"""Value-evidenced reconciliation — the core cross-domain safety gate.

A PDF entity is bridged to a CSV master key ONLY when its literal sample *values*
overlap that key's value fingerprints above EVERY tunable gate
(``bridge.min_value_overlap_pct``, ``bridge.min_overlap_count``,
``bridge.min_confidence``). The entity *name* is never a join signal — it only
appears as a label in the evidence trail. Below threshold ⇒ REFUSED; we never
silently pick the top sub-threshold match. Among the columns that clear all
gates, the BEST (highest-confidence, then highest-overlap) wins.

Reuses (READ-ONLY) the CSV layer's ``fingerprint_value`` (the same normalize +
sha256[:16] used at ingestion, so PDF and CSV fingerprints are comparable) and
``join_confidence`` (the value-overlap × log-cardinality confidence blend). No
``server/app/`` file is modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

# READ-ONLY imports from the live CSV layer — never modified here.
from app.services.relationship_detector import join_confidence
from app.services.relationship_index import fingerprint_value
from app.services.semantic_policy import get_semantic_policy
from app.services.semantic_roles import (
    is_fingerprint_key_role,
    is_never_fingerprint_join_role,
)

from pdf_chat.models.bridge import BridgeStatus, PdfEntityBridge
from pdf_chat.tunables import get_tunable, log_gate_decision


@dataclass(frozen=True)
class EntityValueSample:
    """One literal value observed for a PDF entity (the join evidence unit)."""
    value: str


@dataclass(frozen=True)
class MasterKeyColumn:
    """A candidate CSV master key from the ColumnKeyRegistry (value evidence)."""
    file_id: str
    column: str
    semantic_role: str | None
    value_fingerprints: list[str]


@dataclass
class ReconcileVerdict:
    """The reconciliation outcome — LINKED to a master key or REFUSED."""
    status: BridgeStatus
    semantic_entity_id: str | None = None
    master_file_id: str | None = None
    master_column: str | None = None
    resolved_semantic_role: str | None = None
    value_overlap_pct: float = 0.0
    confidence: float = 0.0
    overlap_count: int = 0
    pdf_value_count: int = 0
    reason: str = ""
    evidence: dict = field(default_factory=dict)


# Reads the PDF entity's literal sample values: (tenant_id, pdf_entity_id) -> samples.
PdfEntityValuesReader = Callable[[str, str], Awaitable[list[EntityValueSample]]]

# A loader for the candidate CSV master keys (injected; never imports infra here).
MasterColumnsLoader = Callable[[], Awaitable[list[MasterKeyColumn]]]


def _pdf_fingerprints(samples: list[EntityValueSample]) -> set[str]:
    """Fingerprint the PDF entity's sample values (drops null-like/empty)."""
    out: set[str] = set()
    for sample in samples or []:
        fp = fingerprint_value(sample.value)
        if fp:
            out.add(fp)
    return out


async def reconcile_entity_to_master_keys(
    *,
    tenant_id: str,
    pdf_entity_id: str,
    entity_name: str,
    samples: list[EntityValueSample],
    master_columns: list[MasterKeyColumn],
) -> ReconcileVerdict:
    """Reconcile a PDF entity to the best value-evidenced CSV master key.

    ``entity_name`` is recorded as a label in evidence but is NEVER used as a join
    signal. Returns LINKED only if a master column clears every gate; otherwise
    REFUSED (never a silent top-match).
    """
    policy = get_semantic_policy()
    pdf_fps = _pdf_fingerprints(samples)
    pdf_value_count = len(pdf_fps)

    min_overlap_pct = get_tunable(tenant_id, "bridge.min_value_overlap_pct")
    min_overlap_count = get_tunable(tenant_id, "bridge.min_overlap_count")
    min_confidence = get_tunable(tenant_id, "bridge.min_confidence")

    base_evidence = {
        "entity_name": entity_name,  # label only — not a join signal
        "pdf_value_count": pdf_value_count,
        "gates": {
            "min_value_overlap_pct": min_overlap_pct,
            "min_overlap_count": min_overlap_count,
            "min_confidence": min_confidence,
        },
        "candidates": [],
    }

    if pdf_value_count == 0:
        log_gate_decision(
            "bridge.reconcile", score=0.0, threshold=min_overlap_count,
            outcome="refused_no_values", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
        )
        return ReconcileVerdict(
            status=BridgeStatus.REFUSED, pdf_value_count=0,
            reason="No fingerprintable PDF entity values; value overlap cannot be proven.",
            evidence=base_evidence,
        )

    best: ReconcileVerdict | None = None

    for col in master_columns or []:
        # Role eligibility is a PRECONDITION (value overlap is still required).
        # Only registry-confirmed master/reference keys may be join targets; a
        # non-referential document/measure/date/attribute column can share values
        # coincidentally, so it is skipped before scoring — never linked.
        if not is_fingerprint_key_role(col.semantic_role) or is_never_fingerprint_join_role(
            col.semantic_role
        ):
            base_evidence["candidates"].append({
                "file_id": col.file_id,
                "column": col.column,
                "semantic_role": col.semantic_role,
                "qualifies": False,
                "skipped": "non_key_role",
            })
            log_gate_decision(
                "bridge.role_eligibility", score=0.0, threshold=1.0,
                outcome="skipped_non_key_role", tenant_id=tenant_id,
                pdf_entity_id=pdf_entity_id, file_id=col.file_id, column=col.column,
            )
            continue

        master_fps = set(col.value_fingerprints or [])
        overlap_count = len(pdf_fps & master_fps)
        # Denominator is intentionally PDF-side: the fraction of the PDF entity's
        # values that reconcile to this master key (NOT the symmetric Jaccard, nor
        # the CSV layer's min_cardinality denominator). Do not "fix" it to match
        # the CSV side — the PDF-direction overlap is the deliberate signal here.
        value_overlap_pct = overlap_count / max(1, pdf_value_count)
        # join_confidence's 2nd arg is the master key's DISTINCT-VALUE cardinality;
        # its log-cardinality term suppresses coincidental overlap on tiny domains.
        # Pass the master key's true domain size, NOT overlap_count.
        master_cardinality = len(master_fps)
        confidence = join_confidence(value_overlap_pct, master_cardinality, policy)

        # All three gates must clear; log each as a decision.
        pct_rec = log_gate_decision(
            "bridge.value_overlap_pct", score=value_overlap_pct, threshold=min_overlap_pct,
            outcome="checked", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
            file_id=col.file_id, column=col.column,
        )
        count_rec = log_gate_decision(
            "bridge.overlap_count", score=float(overlap_count), threshold=float(min_overlap_count),
            outcome="checked", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
            file_id=col.file_id, column=col.column,
        )
        conf_rec = log_gate_decision(
            "bridge.confidence", score=confidence, threshold=min_confidence,
            outcome="checked", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
            file_id=col.file_id, column=col.column,
        )
        qualifies = pct_rec["passed"] and count_rec["passed"] and conf_rec["passed"]

        base_evidence["candidates"].append({
            "file_id": col.file_id,
            "column": col.column,
            "semantic_role": col.semantic_role,
            "overlap_count": overlap_count,
            "value_overlap_pct": value_overlap_pct,
            "confidence": confidence,
            "qualifies": qualifies,
        })

        if not qualifies:
            continue
        candidate = ReconcileVerdict(
            status=BridgeStatus.LINKED,
            master_file_id=col.file_id,
            master_column=col.column,
            resolved_semantic_role=col.semantic_role,
            value_overlap_pct=value_overlap_pct,
            confidence=confidence,
            overlap_count=overlap_count,
            pdf_value_count=pdf_value_count,
            reason="Value overlap cleared every gate.",
        )
        # Best = highest confidence, tie-broken by overlap pct then count.
        if best is None or (candidate.confidence, candidate.value_overlap_pct, candidate.overlap_count) > (
            best.confidence, best.value_overlap_pct, best.overlap_count
        ):
            best = candidate

    if best is not None:
        best.evidence = base_evidence
        log_gate_decision(
            "bridge.reconcile", score=best.confidence, threshold=min_confidence,
            outcome="linked", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
            file_id=best.master_file_id, column=best.master_column,
        )
        return best

    # Nothing cleared all gates — REFUSE. Never fall back to the top sub-threshold.
    best_pct = max(
        (c.get("value_overlap_pct", 0.0) for c in base_evidence["candidates"]),
        default=0.0,
    )
    log_gate_decision(
        "bridge.reconcile", score=best_pct, threshold=min_overlap_pct,
        outcome="refused_subthreshold", tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
    )
    return ReconcileVerdict(
        status=BridgeStatus.REFUSED,
        value_overlap_pct=best_pct,
        pdf_value_count=pdf_value_count,
        reason=(
            "Value overlap below threshold for every master key; refusing the join "
            "(name equality is not evidence; no silent top-match)."
        ),
        evidence=base_evidence,
    )


async def build_bridge_for_entity(
    db: AsyncSession,
    *,
    tenant_id: str,
    container_id: str,
    pdf_entity_id: str,
    entity_name: str,
    values_reader: PdfEntityValuesReader,
    master_columns_loader: MasterColumnsLoader,
    semantic_entity_id: str | None = None,
) -> ReconcileVerdict:
    """Reconcile then persist a ``PdfEntityBridge`` row (LINKED or REFUSED).

    Reads the PDF entity values via the injected ``values_reader`` and the
    candidate CSV master keys via ``master_columns_loader`` (both injected so this
    module stays import-safe with zero infra). Persists the verdict + evidence.
    """
    samples = await values_reader(tenant_id, pdf_entity_id)
    master_columns = await master_columns_loader()
    verdict = await reconcile_entity_to_master_keys(
        tenant_id=tenant_id,
        pdf_entity_id=pdf_entity_id,
        entity_name=entity_name,
        samples=samples,
        master_columns=master_columns,
    )
    if verdict.status == BridgeStatus.LINKED:
        verdict.semantic_entity_id = semantic_entity_id

    row = PdfEntityBridge(
        container_id=container_id,
        tenant_id=tenant_id,
        pdf_entity_id=pdf_entity_id,
        semantic_entity_id=verdict.semantic_entity_id,
        resolved_master_file_id=verdict.master_file_id,
        resolved_master_column=verdict.master_column,
        resolved_semantic_role=verdict.resolved_semantic_role,
        value_overlap_pct=verdict.value_overlap_pct,
        confidence=verdict.confidence,
        overlap_count=verdict.overlap_count,
        pdf_value_count=verdict.pdf_value_count,
        evidence=verdict.evidence,
        status=verdict.status.value,
    )
    db.add(row)
    await db.flush()
    return verdict

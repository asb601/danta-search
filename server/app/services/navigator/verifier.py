"""[3d] VERIFY — the verify-before-use gate (INVARIANTS I6 / I7 / I12).

No proposed conclusion is used until every slot is value-checked against the same
EVIDENCE the proposer read. This module is the navigator's verification tier and is
fully SELF-CONTAINED — it imports nothing from ``app.services.resolve.*`` (those
files are deleted at P5). Everything it needs is LIFTED here:

  * ``verify``                 — column-exists / value-in-set / grain-unique checks
                                 plus the polarity cross-check (a pick whose reliable
                                 ledger side contradicts the question's side is
                                 rejected). Lifted from ``resolve.brain.verify``.
  * ``partition_by_polarity``  — the deterministic PICK pre-filter (drop the
                                 contradicting side; signal a tie on genuine
                                 ambiguity). Lifted from ``brain._partition_by_polarity``.
  * ``clarify_payload``        — the abstain-to-user options, drawn from the
                                 candidates' OWN side/role (never invented literals;
                                 ≤3). Lifted from ``brain._clarify_payload`` and
                                 re-shaped into the typed ``ClarifyPayload`` (I12).
  * ``_polarity_from_row``     — the classifier's own reliability gate applied to a
                                 stored erp_classifications row. Lifted from
                                 ``brain._polarity_from_row``.
  * ``verify_step_join`` / ``JoinVerdict`` — value-overlap join verification from
                                 the precomputed ColumnKeyRegistry. Lifted from
                                 ``resolve.verification.verify_join`` (I7: a join is
                                 usable ONLY if value-overlap-verified, never
                                 LLM-guessed).

Bias is toward ABSTAIN throughout: thin / missing evidence cannot disprove a slot,
so it passes; only a positive contradiction (missing column, value not in the stored
set, wrong ledger side, non-distinguishing grain) rejects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.column_key_registry import ColumnKeyRegistry
from app.services.erp.classifier import ErpClassification as ErpClassificationFact
from app.services.navigator.types import (
    ClarifyPayload,
    EvidencePacket,
    ProposedContract,
    ResolvedTable,
    VerifiedContract,
)
from app.services.semantic_policy import SemanticPolicy, get_semantic_policy
from app.services.semantic_roles import is_date_role, is_never_fingerprint_join_role

logger = structlog.get_logger("navigator.verifier")

# The slot grammar (the vocabulary the runtime understands), NOT business knobs.
_AGGS: frozenset[str] = frozenset({"SUM", "COUNT", "AVG", "MAX", "MIN", "COUNT_DISTINCT"})
_OPS: frozenset[str] = frozenset({"=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "ILIKE"})
_BUCKETS: frozenset[str] = frozenset({"month", "quarter", "year"})

# Stored-dtype substrings that mark a column as date/timestamp-typed. Data-driven
# (read from the per-column ``type`` ingestion stored), NOT a dataset-specific hint —
# the same vocabulary the rest of the codebase uses to recognise temporal columns
# (mirrors dashboard.query_engine's temporal/_NUMERIC_TYPES style). A column also
# qualifies via its semantic ROLE (``is_date_role``); either signal is sufficient.
_DATE_DTYPE_MARKERS: tuple[str, ...] = (
    "date", "datetime", "timestamp", "time", "temporal",
)

# The universal double-entry-accounting axis — the ONLY closed polarity vocabulary,
# mirrored from the classifier/model. A polarity outside {customer, vendor} (neutral
# / None) is "unconstrained" and can never REJECT a candidate.
_RELIABLE_SIDES: frozenset[str] = frozenset({"customer", "vendor"})
_MAX_CLARIFY_OPTIONS = 3   # ≤3 options offered to the user on an opposite-side tie

# Division guard for unique_rate / containment math (verify_step_join). Never a
# business threshold — only keeps 1/x and x/y finite on a degenerate row.
_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# polarity reliability gate (lifted from brain._polarity_from_row)
# ---------------------------------------------------------------------------
def _polarity_from_row(
    domain_polarity: object, confidence: object, source: object,
    source_system: object = "Unknown",
) -> str | None:
    """Apply the classifier's OWN reliability rule to a stored erp_classifications
    row → a usable polarity, or None when it cannot drive a decision. Pure.

    Reuses ``ErpClassification.is_reliable`` (app/services/erp/classifier.py) by
    reconstructing the dataclass from the row and reading the property — a
    ``human_override`` is always reliable; an ``llm`` row is reliable only when
    ``confidence >= floor`` AND ``source_system != "Unknown"``. ``neutral`` and
    anything unreliable / missing collapse to None = "unconstrained"."""
    side = str(domain_polarity or "").strip().lower()
    if side not in _RELIABLE_SIDES:           # neutral / blank / unknown side
        return None
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    reliable = ErpClassificationFact(
        source_system=str(source_system or "Unknown") or "Unknown",
        domain_polarity=side,  # type: ignore[arg-type]
        confidence=conf,
        source=str(source or "").strip().lower(),
    ).is_reliable
    return side if reliable else None


# ---------------------------------------------------------------------------
# PICK pre-filter (lifted from brain._partition_by_polarity)
# ---------------------------------------------------------------------------
def partition_by_polarity(
    candidates: list[dict], q_polarity: str | None,
) -> tuple[list[dict], str | None]:
    """Deterministic, DB-free PICK pre-filter — the code-side enforcement that a
    RELIABLE cross-side twin is never blended with the answer (the prompt is not
    trusted to do it). What is enforced, precisely (NOT a blanket guarantee):

      * <2 distinct RELIABLE sides present → no cross-side conflict → returned
        unchanged. (A side is "reliable" only after the classifier's own gate — see
        ``_polarity_from_row``; an unreliable/guessed/neutral polarity is ``None``
        here and is treated as unconstrained.)
      * known reliable ``q_polarity`` → drop ONLY candidates whose OWN polarity is the
        reliable OPPOSITE side; the matching side AND every unconstrained (``None`` /
        neutral) candidate are KEPT. EMPTY-GUARD: if that would remove everything, the
        FULL candidate set is returned instead — the filter never starves the slice
        (abstain-bias: show everything and let VERIFY catch a wrong side).
      * ≥2 reliable sides AND unknown ``q_polarity`` → genuine ambiguity → signal
        ``polarity_tie`` so the driver abstains-to-user instead of guessing.

    Because a candidate is dropped ONLY on its OWN reliable polarity against a
    reliably-known ``q_polarity``, a misclassified/unreliable side (which is ``None``
    here) can never drop the correct table.

    Returns ``(filtered_candidates, clarify_signal | None)``. Pure."""
    sides = {e.get("polarity") for e in candidates if e.get("polarity") in _RELIABLE_SIDES}
    if len(sides) < 2:
        return candidates, None
    if q_polarity in _RELIABLE_SIDES:
        # Keep the matching reliable side + every unconstrained (None) candidate; the
        # reliable OPPOSITE side is the only thing dropped. Never return empty.
        keep = [e for e in candidates if e.get("polarity") in (q_polarity, None)]
        return (keep or candidates), None
    return candidates, "polarity_tie"


# ---------------------------------------------------------------------------
# abstain-to-user payload (lifted from brain._clarify_payload, typed I12)
# ---------------------------------------------------------------------------
def clarify_payload(candidates: list[dict], reason: str) -> ClarifyPayload:
    """Build the abstain-to-user contract from the candidates' OWN ledger side /
    process role — no literal AP/AR constants. Options are interpolated from
    evidence, never hardcoded, and capped at 3 (INVARIANT I12). Pure.

    Only candidates carrying a reliable side are offered (a neutral/None table is
    not a meaningful side to disambiguate); falls back to all candidates if the
    reliable subset is empty so an option list is always present."""
    sided = [e for e in candidates if e.get("polarity") in _RELIABLE_SIDES] or list(candidates)
    options: list[str] = []
    for e in sided[:_MAX_CLARIFY_OPTIONS]:
        # Descriptor DERIVED from this candidate's own side/role — never a business
        # literal. Falls back to the table name when neither side nor role is set.
        descriptor = " / ".join(
            str(p) for p in (e.get("polarity"), e.get("process_role")) if p
        ) or str(e.get("table"))
        options.append(descriptor)
    return ClarifyPayload(reason=reason, options=tuple(options))


# ---------------------------------------------------------------------------
# date-column usability gate (time-window abstain-safety, L8)
# ---------------------------------------------------------------------------
def _is_date_typed_column(evf: dict, exact_col: str | None) -> bool:
    """Is ``exact_col`` a usable date/timestamp column on this evidence table?
    Data-driven: True when the column's stored ``type`` (dtype) matches a
    date/timestamp marker OR its semantic ``role`` is a date role. Pure; False on a
    missing column / missing evidence (abstain-biased — an unknown column is not a
    usable date column)."""
    if not exact_col:
        return False
    target = str(exact_col).upper()
    for c in evf.get("columns") or ():
        if str(c.get("name") or "").upper() != target:
            continue
        dtype = str(c.get("type") or "").lower()
        if any(marker in dtype for marker in _DATE_DTYPE_MARKERS):
            return True
        if is_date_role(c.get("role")):
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# VERIFY (lifted from brain.verify) — typed ProposedContract -> VerifiedContract
# ---------------------------------------------------------------------------
def verify(
    pc: ProposedContract, ev: EvidencePacket, q_polarity: str | None = None,
    *, time_window: tuple | None = None,
) -> tuple[VerifiedContract | None, str]:
    """Value-verify the proposed slots against the evidence. Returns a
    ``VerifiedContract`` (exact-case identifiers, normalised) or ``(None, reason)``.
    Every identifier must exist in the chosen table's schema; the aggregation / op /
    bucket must be known; an entity grain must distinguish rows; an equality filter
    value must be in the column's stored value-set when one exists.

    ``q_polarity`` (when known and the polarity gate is on) adds a deterministic
    cross-check: a pick whose reliable ledger side CONTRADICTS the question's side
    is rejected (``polarity_contradicts_question``). A neutral/None side on either
    the pick or the question passes (abstain-biased — we cannot disprove).

    ``time_window`` (L8 abstain-safety): when a time window IS present for the
    question, the chosen table MUST carry a usable date/timestamp column — the
    proposed ``time_filter_column``, or the grain date column on a time grain.
    Absent that, verify FAILS ``no_time_column_for_window`` so the driver abstains
    rather than render an ALL-TIME query for a time-scoped question. When no window
    is present this gate is inert (behaviour byte-identical to pre-L8)."""
    if pc is None or ev is None:
        return None, "no_contract"
    by_table = {e["table"].upper(): e for e in ev.files}
    evf = by_table.get(str(pc.table or "").upper())
    if not evf:
        return None, "table_not_in_slice"

    # Polarity cross-check (gated). Wrong-side pick → abstain; neutral/None passes.
    if (
        getattr(get_settings(), "BRAIN_POLARITY_GATE_ENABLED", True)
        and q_polarity in _RELIABLE_SIDES
        and evf.get("polarity") in _RELIABLE_SIDES
        and evf.get("polarity") != q_polarity
    ):
        return None, "polarity_contradicts_question"

    valid: dict[str, str] = evf.get("valid_cols") or {}

    def col(name: object) -> str | None:
        return valid.get(str(name).upper()) if name else None

    value_set: dict[str, dict] = evf.get("value_set") or {}
    unique_rates: dict[str, float] = evf.get("unique_rates") or {}

    grain_kind = pc.grain_kind
    grain_col = col(pc.grain_column)
    if grain_kind not in ("entity", "time") or not grain_col:
        return None, "bad_grain"
    bucket = pc.time_bucket
    if grain_kind == "time" and bucket not in _BUCKETS:
        return None, "bad_time_bucket"
    # Grain-uniqueness sanity: an entity grain column should distinguish rows. If
    # the key registry recorded a unique_rate for it, it must be > 0; absent → pass.
    if grain_kind == "entity":
        grate = unique_rates.get(grain_col.upper())
        if grate is not None and grate <= 0.0:
            return None, "grain_not_distinguishing"

    measure_col = col(pc.measure_column)
    agg = str(pc.measure_agg or "").upper()
    if not measure_col or agg not in _AGGS:
        return None, "bad_measure"

    preds: list[tuple[str, str, Any]] = []
    for f in (pc.filters or ()):
        fc = col(f.get("column")) if isinstance(f, dict) else None
        op = str(f.get("op", "=")).upper() if isinstance(f, dict) else ""
        if not fc or op not in _OPS:
            return None, "bad_filter"
        # Value-set check: only equality-style predicates against a column that HAS
        # a stored categorical value-set are checkable. No value_set → pass (can't
        # disprove). Range/LIKE ops are not membership checks → pass. Abstain-biased.
        fval = f.get("value")
        col_values = value_set.get(fc.upper())
        if col_values and op in ("=", "==") and fval is not None:
            keys_lower = {str(k).lower() for k in col_values.keys()}
            if str(fval).lower() not in keys_lower:
                return None, "filter_value_not_in_value_set"
        preds.append((fc, op, fval))

    tcol = col(pc.time_filter_column) if pc.time_filter_column else None

    # L8 abstain-safety: a windowed question REQUIRES a usable date column to scope
    # to. Prefer the proposed time_filter_column; on a time grain the grain date
    # column itself is the natural window column. The column must exist AND be
    # date/timestamp-typed (by dtype or semantic role). If neither qualifies, FAIL
    # so the driver abstains — NEVER render all-time SQL for a time-scoped question.
    if time_window:
        window_col = tcol if _is_date_typed_column(evf, tcol) else None
        if window_col is None and grain_kind == "time" and _is_date_typed_column(
            evf, grain_col
        ):
            window_col = grain_col
        if window_col is None:
            return None, "no_time_column_for_window"
        tcol = window_col

    having = pc.having if isinstance(pc.having, dict) else None
    vc = VerifiedContract(
        step_id=pc.step_id,
        table=evf["table"],
        grain_kind=grain_kind,
        grain_col=grain_col,
        bucket=bucket,
        measure_col=measure_col,
        agg=agg,
        filters=tuple(preds),
        time_col=tcol,
        having=having,
        top_n=pc.top_n if isinstance(pc.top_n, int) else None,
        order="ASC" if str(pc.order or "desc").lower() == "asc" else "DESC",
        reason=str(pc.table_reason or "")[:200],
    )
    return vc, "ok"


# ---------------------------------------------------------------------------
# value-verified join (lifted from resolve.verification.verify_join + JoinVerdict)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JoinVerdict:
    verified: bool
    fk_side: str | None
    pk_side: str | None
    containment: float
    fanout_estimate: float
    reason: str
    abstain: bool


def _abstain_join(reason: str) -> JoinVerdict:
    return JoinVerdict(
        verified=False, fk_side=None, pk_side=None, containment=0.0,
        fanout_estimate=0.0, reason=reason, abstain=True,
    )


async def _load_registry_row(
    db: AsyncSession, container_id: str, blob_path: str, column_name: str,
) -> ColumnKeyRegistry | None:
    """Load one ColumnKeyRegistry row scoped to container + blob + column."""
    result = await db.execute(
        select(ColumnKeyRegistry).where(
            ColumnKeyRegistry.container_id == container_id,
            ColumnKeyRegistry.blob_path == blob_path,
            ColumnKeyRegistry.column_name == column_name,
        )
    )
    return result.scalars().first()


def _containment(fk_fps: set[str], pk_fps: set[str]) -> float:
    """|fp(fk) ∩ fp(pk)| / |fp(fk)| — fraction of FK values found in PK side."""
    if not fk_fps:
        return 0.0
    return len(fk_fps & pk_fps) / max(len(fk_fps), _EPSILON)


async def verify_step_join(
    db: AsyncSession,
    container_id: str,
    a: ResolvedTable,
    b: ResolvedTable,
    col_a: str,
    col_b: str,
    *,
    policy: SemanticPolicy | None = None,
) -> JoinVerdict:
    """Deterministically verify a candidate join edge ``(a.blob.col_a) ↔
    (b.blob.col_b)`` from value evidence ONLY (INVARIANT I7). LIFTED from
    ``resolve.verification.verify_join``.

    Returns ``verified=True`` only when the role gate passes, FK→PK containment
    clears ``min_join_overlap``, the PK side genuinely qualifies as a primary key,
    and the estimated fan-out stays within 1:N (``max_join_fanout``). Thin / ambiguous
    evidence → ``abstain=True`` (never a false verify); an audit/templated noise pair
    → a decided reject (``abstain=False``)."""
    active = policy or get_semantic_policy()

    row_a = await _load_registry_row(db, container_id, a.blob, col_a)
    row_b = await _load_registry_row(db, container_id, b.blob, col_b)

    if row_a is None or row_b is None:
        return _abstain_join("missing_registry_row")

    # ROLE GATE (data-driven, by role kind — the CREATED_BY / audit-column kill).
    if is_never_fingerprint_join_role(row_a.semantic_role) or is_never_fingerprint_join_role(
        row_b.semantic_role
    ):
        return JoinVerdict(
            verified=False, fk_side=None, pk_side=None, containment=0.0,
            fanout_estimate=0.0, reason="audit_or_nonkey_role", abstain=False,
        )

    fps_a = set(row_a.value_fingerprints or [])
    fps_b = set(row_b.value_fingerprints or [])

    if not fps_a or not fps_b:
        return _abstain_join("empty_fingerprints")
    if len(fps_a) < active.min_distinct_key_values or len(fps_b) < active.min_distinct_key_values:
        return _abstain_join("insufficient_distinct_values")
    if (row_a.unique_rate or 0.0) <= 0.0 or (row_b.unique_rate or 0.0) <= 0.0:
        return _abstain_join("degenerate_unique_rate")

    # PK side = higher unique_rate; FK side = the other.
    if row_a.unique_rate >= row_b.unique_rate:
        pk_row, fk_row = row_a, row_b
        pk_side, fk_side = a.blob, b.blob
        pk_fps, fk_fps = fps_a, fps_b
    else:
        pk_row, fk_row = row_b, row_a
        pk_side, fk_side = b.blob, a.blob
        pk_fps, fk_fps = fps_b, fps_a

    intersection = fk_fps & pk_fps
    if len(intersection) < active.min_overlap_fingerprint_count:
        return _abstain_join("overlap_below_min_count")

    containment = _containment(fk_fps, pk_fps)
    fanout_estimate = 1.0 / max(pk_row.unique_rate, _EPSILON)

    def _pk_qualifies(reg: ColumnKeyRegistry) -> bool:
        return (reg.unique_rate or 0.0) >= active.pk_unique_rate and (
            reg.null_rate or 0.0
        ) <= active.pk_null_rate

    pk_qualifies = _pk_qualifies(pk_row)

    # AUDIT-NOISE reject: symmetric high overlap AND neither side PK-qualifies.
    containment_pk_into_fk = _containment(pk_fps, fk_fps)
    symmetric_high = (
        containment >= active.min_join_overlap
        and containment_pk_into_fk >= active.min_join_overlap
    )
    if symmetric_high and not pk_qualifies and not _pk_qualifies(fk_row):
        return JoinVerdict(
            verified=False, fk_side=fk_side, pk_side=pk_side, containment=containment,
            fanout_estimate=fanout_estimate, reason="audit_noise_symmetric_overlap",
            abstain=False,
        )

    role_ok = not is_never_fingerprint_join_role(
        pk_row.semantic_role
    ) and not is_never_fingerprint_join_role(fk_row.semantic_role)
    within_fanout = fanout_estimate <= active.max_join_fanout

    verified = (
        role_ok
        and containment >= active.min_join_overlap
        and pk_qualifies
        and within_fanout
    )

    if verified:
        reason = "verified_fk_to_pk"
    elif containment < active.min_join_overlap:
        reason = "containment_below_floor"
    elif not pk_qualifies:
        reason = "pk_side_not_unique_enough"
    elif not within_fanout:
        reason = "fanout_exceeds_one_to_n"
    else:
        reason = "rejected"

    return JoinVerdict(
        verified=verified, fk_side=fk_side, pk_side=pk_side, containment=containment,
        fanout_estimate=fanout_estimate, reason=reason, abstain=False,
    )

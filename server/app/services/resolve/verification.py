"""Deterministic VERIFY tier — pure, value-evidence-only join + canonical checks.

This module confirms or abstains on two decisions using ONLY precomputed
ingestion artifacts:

  * ``verify_join``      — is (blob_a.col_a) ↔ (blob_b.col_b) a real 1:N FK→PK
                          edge, an audit/noise pair, or undecidable?
  * ``verify_canonical`` — given a cluster of schema-fingerprint twins, is one a
                          master (by value evidence), or must a human confirm?

Design rules (do NOT relax):
  * No LLM, no runtime schema discovery, no hardcoded column-name lists.
  * Every threshold comes from ``SemanticPolicy`` (loaded when ``policy`` is None).
  * The role gate is data-driven via ``is_never_fingerprint_join_role`` (this is
    the CREATED_BY / audit-column kill — by ROLE, never by name).
  * Bias is toward ABSTAIN: when evidence is thin the verdict is undecided, never
    a false ``verified``.

This file is additive and is intentionally NOT imported by any live query path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.column_key_registry import ColumnKeyRegistry
from app.models.file_metadata import FileMetadata
from app.services.semantic_policy import SemanticPolicy, get_semantic_policy
from app.services.semantic_roles import is_never_fingerprint_join_role

# Division guard for unique_rate / containment math. Never used as a business
# threshold — only to keep 1/x and x/y finite when a degenerate row sneaks
# through the edge-case abstains below.
_EPSILON = 1e-9


@dataclass(frozen=True)
class JoinVerdict:
    verified: bool
    fk_side: str | None
    pk_side: str | None
    containment: float
    fanout_estimate: float
    reason: str
    abstain: bool


@dataclass(frozen=True)
class CanonicalVerdict:
    master: str | None
    contenders: list[str] = field(default_factory=list)
    needs_confirm: bool = False
    reason: str = ""


def _abstain_join(reason: str) -> JoinVerdict:
    return JoinVerdict(
        verified=False,
        fk_side=None,
        pk_side=None,
        containment=0.0,
        fanout_estimate=0.0,
        reason=reason,
        abstain=True,
    )


async def _load_registry_row(
    db: AsyncSession,
    container_id: str,
    blob_path: str,
    column_name: str,
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


async def verify_join(
    db: AsyncSession,
    container_id: str,
    blob_a: str,
    col_a: str,
    blob_b: str,
    col_b: str,
    *,
    policy: SemanticPolicy | None = None,
) -> JoinVerdict:
    """Deterministically verify a candidate join edge from value evidence.

    Returns ``verified=True`` only when role gate passes, FK→PK containment
    clears ``min_join_overlap``, the PK side genuinely qualifies as a primary
    key, and the estimated fan-out stays within 1:N (``max_join_fanout``).
    Thin/ambiguous evidence → ``abstain=True`` (never a false verify).
    """
    active = policy or get_semantic_policy()

    row_a = await _load_registry_row(db, container_id, blob_a, col_a)
    row_b = await _load_registry_row(db, container_id, blob_b, col_b)

    # One side absent → undecidable.
    if row_a is None or row_b is None:
        return _abstain_join("missing_registry_row")

    # ROLE GATE (data-driven, by role kind — the CREATED_BY / audit-column kill).
    # A present audit/measure/date/attribute role on EITHER side rejects the edge.
    # A MISSING role is NOT an auto-reject — it falls through to containment math.
    if is_never_fingerprint_join_role(row_a.semantic_role) or is_never_fingerprint_join_role(
        row_b.semantic_role
    ):
        return JoinVerdict(
            verified=False,
            fk_side=None,
            pk_side=None,
            containment=0.0,
            fanout_estimate=0.0,
            reason="audit_or_nonkey_role",
            abstain=False,
        )

    fps_a = set(row_a.value_fingerprints or [])
    fps_b = set(row_b.value_fingerprints or [])

    # EDGE CASES → abstain (never false-verify).
    if not fps_a or not fps_b:
        return _abstain_join("empty_fingerprints")
    if len(fps_a) < active.min_distinct_key_values or len(fps_b) < active.min_distinct_key_values:
        return _abstain_join("insufficient_distinct_values")
    # Guard division: a zero unique_rate cannot be reasoned about as a PK.
    if (row_a.unique_rate or 0.0) <= 0.0 or (row_b.unique_rate or 0.0) <= 0.0:
        return _abstain_join("degenerate_unique_rate")

    # PK side = higher unique_rate; FK side = the other.
    if row_a.unique_rate >= row_b.unique_rate:
        pk_row, fk_row = row_a, row_b
        pk_side, fk_side = blob_a, blob_b
        pk_fps, fk_fps = fps_a, fps_b
    else:
        pk_row, fk_row = row_b, row_a
        pk_side, fk_side = blob_b, blob_a
        pk_fps, fk_fps = fps_b, fps_a

    # Overlap floor on the actual intersection count.
    intersection = fk_fps & pk_fps
    if len(intersection) < active.min_overlap_fingerprint_count:
        return _abstain_join("overlap_below_min_count")

    containment = _containment(fk_fps, pk_fps)
    # fan-out estimate from PK-side uniqueness: a true PK (~1.0) yields ~1.0.
    fanout_estimate = 1.0 / max(pk_row.unique_rate, _EPSILON)

    def _pk_qualifies(reg: ColumnKeyRegistry) -> bool:
        return (reg.unique_rate or 0.0) >= active.pk_unique_rate and (
            reg.null_rate or 0.0
        ) <= active.pk_null_rate

    pk_qualifies = _pk_qualifies(pk_row)

    # AUDIT-NOISE reject: symmetric high overlap (containment high in BOTH
    # directions) AND neither side PK-qualifies → an audit/templated column pair
    # masquerading as a join. Reject (decided, not abstain).
    containment_pk_into_fk = _containment(pk_fps, fk_fps)
    symmetric_high = (
        containment >= active.min_join_overlap
        and containment_pk_into_fk >= active.min_join_overlap
    )
    if symmetric_high and not pk_qualifies and not _pk_qualifies(fk_row):
        return JoinVerdict(
            verified=False,
            fk_side=fk_side,
            pk_side=pk_side,
            containment=containment,
            fanout_estimate=fanout_estimate,
            reason="audit_noise_symmetric_overlap",
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
        verified=verified,
        fk_side=fk_side,
        pk_side=pk_side,
        containment=containment,
        fanout_estimate=fanout_estimate,
        reason=reason,
        abstain=False,
    )


async def _load_canonical_members(
    db: AsyncSession,
    container_id: str,
    cluster_blobs: list[str],
) -> dict[str, dict]:
    """Read per-member evidence for a canonical cluster.

    Returns ``{blob_path: {"registry": [ColumnKeyRegistry...], "meta": FileMetadata}}``
    for every member that has at least registry rows. Pure read of precomputed
    ingestion artifacts — no schema probing.
    """
    members: dict[str, dict] = {}

    reg_result = await db.execute(
        select(ColumnKeyRegistry).where(
            ColumnKeyRegistry.container_id == container_id,
            ColumnKeyRegistry.blob_path.in_(cluster_blobs),
        )
    )
    for row in reg_result.scalars().all():
        members.setdefault(row.blob_path, {"registry": [], "meta": None})["registry"].append(row)

    meta_result = await db.execute(
        select(FileMetadata).where(
            FileMetadata.container_id == container_id,
            FileMetadata.blob_path.in_(cluster_blobs),
        )
    )
    for meta in meta_result.scalars().all():
        members.setdefault(meta.blob_path, {"registry": [], "meta": None})["meta"] = meta

    return members


def _column_names(meta: FileMetadata | None) -> set[str]:
    """Original column-name set from precomputed columns_info (no schema probe)."""
    names: set[str] = set()
    for col in (getattr(meta, "columns_info", None) or []):
        name = col.get("name") if isinstance(col, dict) else None
        if name:
            names.add(str(name))
    return names


async def verify_canonical(
    db: AsyncSession,
    container_id: str,
    cluster_blobs: list[str],
    *,
    policy: SemanticPolicy | None = None,
) -> CanonicalVerdict:
    """Elect a canonical master from a cluster of schema-fingerprint twins.

    The caller groups ``cluster_blobs`` by SCHEMA FINGERPRINT (not entity_name).
    Election uses value evidence ONLY — never row count, never name:

      (a) inter-member key containment on the shared business key,
      (b) original-vs-derived column presence (columns_info breadth),
      (c) status/lifecycle distinct-value breadth where available.

    A master is elected only if one member dominates on >=2 signals AND the
    inter-member key containment clears ``min_join_overlap``. Otherwise the
    cluster ``needs_confirm`` with the contenders surfaced.
    """
    active = policy or get_semantic_policy()
    blobs = [b for b in (cluster_blobs or []) if b]

    if len(blobs) < 2:
        return CanonicalVerdict(
            master=None,
            contenders=list(blobs),
            needs_confirm=False,
            reason="cluster_too_small",
        )

    members = await _load_canonical_members(db, container_id, blobs)
    present = [b for b in blobs if b in members]
    if len(present) < 2:
        return CanonicalVerdict(
            master=None,
            contenders=list(blobs),
            needs_confirm=True,
            reason="insufficient_member_evidence",
        )

    # Choose the shared business key = the registry column_name present across the
    # most members (value-derived, not by name). Its fingerprints drive both the
    # inter-member containment signal and the master-direction signal.
    key_fps_by_member: dict[str, dict[str, set[str]]] = {}
    column_presence: dict[str, int] = {}
    for blob in present:
        per_col: dict[str, set[str]] = {}
        for reg in members[blob]["registry"]:
            if is_never_fingerprint_join_role(reg.semantic_role):
                continue
            fps = set(reg.value_fingerprints or [])
            if len(fps) < active.min_distinct_key_values:
                continue
            per_col[reg.column_name] = fps
            column_presence[reg.column_name] = column_presence.get(reg.column_name, 0) + 1
        key_fps_by_member[blob] = per_col

    shared_keys = [c for c, n in column_presence.items() if n >= 2]
    if not shared_keys:
        return CanonicalVerdict(
            master=None,
            contenders=list(present),
            needs_confirm=True,
            reason="no_shared_business_key",
        )
    shared_key = max(shared_keys, key=lambda c: column_presence[c])

    # (a) inter-member key containment: max pairwise containment of one member's
    # key values into another's on the shared key. Low containment (the AR-twins
    # ~0.0007 case) means the twins describe DIFFERENT key populations → confirm.
    key_members = [b for b in present if shared_key in key_fps_by_member.get(b, {})]
    best_containment = 0.0
    for src in key_members:
        src_fps = key_fps_by_member[src][shared_key]
        for dst in key_members:
            if dst == src:
                continue
            dst_fps = key_fps_by_member[dst][shared_key]
            best_containment = max(best_containment, _containment(src_fps, dst_fps))

    if best_containment < active.min_join_overlap:
        return CanonicalVerdict(
            master=None,
            contenders=list(present),
            needs_confirm=True,
            reason="inter_member_key_containment_below_floor",
        )

    # Per-member signal scoring (value evidence only).
    #  signal 1: this member's key values CONTAIN the other members' (superset →
    #            it is the broader/master population).
    #  signal 2: original column breadth (more original columns from columns_info).
    #  signal 3: status/lifecycle distinct-value breadth (widest attribute domain).
    signal_winners: dict[str, set[int]] = {b: set() for b in present}

    # signal 1 — key superset direction.
    superset_score: dict[str, float] = {}
    for candidate in key_members:
        cand_fps = key_fps_by_member[candidate][shared_key]
        # how well candidate contains every OTHER member (avg reverse containment).
        scores = [
            _containment(key_fps_by_member[other][shared_key], cand_fps)
            for other in key_members
            if other != candidate
        ]
        superset_score[candidate] = sum(scores) / max(len(scores), 1) if scores else 0.0
    if superset_score:
        top1 = max(superset_score.values())
        winners1 = [b for b, s in superset_score.items() if s >= top1 and top1 > 0.0]
        if len(winners1) == 1:
            signal_winners[winners1[0]].add(1)

    # signal 2 — original column breadth.
    colcount: dict[str, int] = {
        b: len(_column_names(members[b]["meta"])) for b in present
    }
    if any(colcount.values()):
        top2 = max(colcount.values())
        winners2 = [b for b, c in colcount.items() if c == top2 and top2 > 0]
        if len(winners2) == 1:
            signal_winners[winners2[0]].add(2)

    # signal 3 — status/lifecycle attribute distinct-value breadth. Derived from
    # registry cardinality on attribute-role columns (broadest lifecycle domain).
    lifecycle_breadth: dict[str, int] = {}
    for blob in present:
        widest = 0
        for reg in members[blob]["registry"]:
            # attribute-role columns are exactly the never-fingerprint dimensions;
            # their cardinality proxies status/lifecycle breadth.
            if is_never_fingerprint_join_role(reg.semantic_role):
                widest = max(widest, int(reg.cardinality or 0))
        lifecycle_breadth[blob] = widest
    if any(lifecycle_breadth.values()):
        top3 = max(lifecycle_breadth.values())
        winners3 = [b for b, c in lifecycle_breadth.items() if c == top3 and top3 > 0]
        if len(winners3) == 1:
            signal_winners[winners3[0]].add(3)

    # Elect master only if ONE member dominates on >=2 signals.
    dominant = [b for b, sigs in signal_winners.items() if len(sigs) >= 2]
    if len(dominant) == 1:
        return CanonicalVerdict(
            master=dominant[0],
            contenders=[b for b in present if b != dominant[0]],
            needs_confirm=False,
            reason="elected_master_dominant_signals",
        )

    return CanonicalVerdict(
        master=None,
        contenders=list(present),
        needs_confirm=True,
        reason="no_dominant_member",
    )

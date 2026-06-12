"""Build the business semantic layer from metadata and ER graph edges.

ER graph responsibility:
    technical join candidates between files/columns

Semantic layer responsibility:
    entity names, primary keys, metrics, dimensions, relationship cardinality,
    approval state, and business-safe join rules.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import ingest_logger
from app.models.column_key_registry import ColumnKeyRegistry
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.models.semantic_layer import SemanticEntity, SemanticRelationship
from app.services.semantic_policy import get_semantic_policy
from app.services.semantic_roles import (
    default_aggregation_for_role,
    entity_name_for_role,
    is_dimension_role,
    is_entity_key_role,
    is_metric_role,
    is_risky_single_column_join_role,
    role_kind,
)


@dataclass(frozen=True)
class EntitySpec:
    entity_name: str
    primary_key: str | None
    attributes: list[str]
    metrics: list[dict]
    dimensions: list[str]
    grain: str | None
    confidence_score: float


def _clean_name(value: str) -> str:
    name = Path(value).stem if "." in value else value
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name or "entity"


def _entity_from_role(role: str | None) -> str | None:
    return entity_name_for_role(role)


def _metric_definition(role: str, column_name: str) -> dict:
    metric = {"name": _clean_name(column_name), "column": column_name}
    default_aggregation = default_aggregation_for_role(role)
    if default_aggregation:
        metric["default_aggregation"] = default_aggregation
    return metric


def infer_entity_spec(meta: FileMetadata, file: File | None) -> EntitySpec:
    policy = get_semantic_policy()
    roles: dict[str, str] = meta.column_semantic_roles or {}
    columns_info = meta.columns_info or []
    column_names = [c.get("name") for c in columns_info if isinstance(c, dict) and c.get("name")]

    primary_key: str | None = None
    primary_role: str | None = None
    for column_name, role in roles.items():
        if is_entity_key_role(role):
            primary_key = column_name
            primary_role = role
            break

    entity_name = _entity_from_role(primary_role)
    if not entity_name:
        source_name = file.name if file else (meta.blob_path or meta.file_id)
        entity_name = _clean_name(source_name)

    metrics: list[dict] = []
    dimensions: list[str] = []
    attributes: list[str] = []

    for column_name in column_names:
        role = roles.get(column_name)
        if is_metric_role(role):
            metrics.append(_metric_definition(role, column_name))
        elif is_dimension_role(role):
            dimensions.append(column_name)
        else:
            attributes.append(column_name)

    if primary_key:
        grain = f"one row per {entity_name} when {primary_key} is unique"
        confidence_score = policy.entity_with_pk_confidence
    else:
        grain = "unknown grain; planner must avoid fanout-sensitive joins"
        confidence_score = policy.entity_unknown_grain_confidence

    return EntitySpec(
        entity_name=entity_name,
        primary_key=primary_key,
        attributes=attributes[:100],
        metrics=metrics[:50],
        dimensions=dimensions[:100],
        grain=grain,
        confidence_score=confidence_score,
    )


async def upsert_semantic_entity(file_id: str, db: AsyncSession) -> SemanticEntity | None:
    meta = (
        await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
    ).scalar_one_or_none()
    if not meta or not meta.container_id:
        return None

    file = await db.get(File, file_id)
    spec = infer_entity_spec(meta, file)

    entity = (
        await db.execute(select(SemanticEntity).where(SemanticEntity.file_id == file_id))
    ).scalar_one_or_none()
    if not entity:
        entity = SemanticEntity(id=str(uuid.uuid4()), file_id=file_id, container_id=meta.container_id)
        db.add(entity)

    entity.entity_name = spec.entity_name
    entity.primary_key = spec.primary_key
    entity.attributes = spec.attributes
    entity.metrics = spec.metrics
    entity.dimensions = spec.dimensions
    entity.grain = spec.grain
    entity.confidence_score = spec.confidence_score
    entity.source = "ingestion"
    entity.status = "active"
    return entity


async def _key_kind(file_id: str, column_name: str | None, db: AsyncSession) -> str | None:
    if not column_name:
        return None
    row = (
        await db.execute(
            select(ColumnKeyRegistry).where(
                ColumnKeyRegistry.file_id == file_id,
                ColumnKeyRegistry.column_name == column_name,
            )
        )
    ).scalar_one_or_none()
    return row.key_kind if row else None


async def _key_cardinality(file_id: str, column_name: str | None, db: AsyncSession) -> int:
    """Distinct-value cardinality of a join column from ColumnKeyRegistry.

    Returns 0 when no registry row exists so a missing-stat column can never be
    mistaken for a real key (the cardinality floor will reject it).
    """
    if not column_name:
        return 0
    row = (
        await db.execute(
            select(ColumnKeyRegistry.cardinality).where(
                ColumnKeyRegistry.file_id == file_id,
                ColumnKeyRegistry.column_name == column_name,
            )
        )
    ).scalar_one_or_none()
    return int(row) if row is not None else 0


async def _build_ubiquity_map(container_id: str, db: AsyncSession) -> dict[str, float]:
    """Per-container column ubiquity: fraction of files that carry each column.

    Computed ONCE per build (one aggregate query) and reused for every edge. A
    column present in ~all files (created_by, last_updated_by) approaches 1.0 and
    is treated as an audit/system column, not a business key. Purely
    distributional — no column-name list. Fail-safe: empty map on any error so
    the caller falls back to current (non-promoting) behavior.
    """
    try:
        total_files = (
            await db.execute(
                select(func.count(func.distinct(ColumnKeyRegistry.file_id))).where(
                    ColumnKeyRegistry.container_id == container_id
                )
            )
        ).scalar_one_or_none() or 0
        if not total_files:
            return {}
        rows = (
            await db.execute(
                select(
                    ColumnKeyRegistry.column_name,
                    func.count(func.distinct(ColumnKeyRegistry.file_id)),
                )
                .where(ColumnKeyRegistry.container_id == container_id)
                .group_by(ColumnKeyRegistry.column_name)
            )
        ).all()
        return {name: (count / total_files) for name, count in rows}
    except Exception as exc:  # fail-safe: never break the build over a stats query
        ingest_logger.warning("semantic_layer_ubiquity_failed", container_id=container_id, error=str(exc))
        return {}


def _relationship_type(kind_a: str | None, kind_b: str | None) -> str:
    if kind_a == "pk" and kind_b == "fk":
        return "one_to_many"
    if kind_a == "fk" and kind_b == "pk":
        return "many_to_one"
    if kind_a == "pk" and kind_b == "pk":
        return "one_to_one"
    return "many_to_many"


def _compatible_role_components(
    meta_a: FileMetadata,
    meta_b: FileMetadata,
    *,
    from_column: str,
    to_column: str,
    primary_role: str | None,
) -> list[dict]:
    roles_a: dict[str, str] = meta_a.column_semantic_roles or {}
    roles_b: dict[str, str] = meta_b.column_semantic_roles or {}

    right_columns_by_role: dict[str, list[str]] = {}
    for column_name, role in roles_b.items():
        if not role or column_name == to_column or role == primary_role:
            continue
        right_columns_by_role.setdefault(role, []).append(column_name)

    components: list[dict] = []
    for left_column, role in roles_a.items():
        if not role or left_column == from_column or role == primary_role:
            continue
        kind = role_kind(role)
        if kind not in {"entity_key", "reference_key", "date", "attribute"}:
            continue
        for right_column in right_columns_by_role.get(role, []):
            components.append({
                "left_column": left_column,
                "right_column": right_column,
                "semantic_role": role,
                "role_kind": kind,
                "required": False,
                "evidence": "matching_semantic_role",
            })

    components.sort(key=lambda item: (item["role_kind"], item["semantic_role"], item["left_column"]))
    return components[:max(0, int(get_settings().INGEST_SEMANTIC_COMPONENT_LIMIT))]


def _join_rule(
    rel: FileRelationship,
    *,
    from_column: str,
    to_column: str,
    kind_a: str | None,
    kind_b: str | None,
    companion_components: list[dict],
) -> dict:
    primary_component = {
        "left_column": from_column,
        "right_column": to_column,
        "semantic_role": rel.semantic_role,
        "role_kind": role_kind(rel.semantic_role),
        "left_key_kind": kind_a,
        "right_key_kind": kind_b,
        "required": True,
        "evidence": "value_fingerprint_overlap",
        "value_overlap_pct": rel.value_overlap_pct,
    }
    return {
        "left_file_id": rel.file_a_id,
        "right_file_id": rel.file_b_id,
        "left_column": from_column,
        "right_column": to_column,
        "semantic_role": rel.semantic_role,
        "join_type": rel.join_type,
        "left_key_kind": kind_a,
        "right_key_kind": kind_b,
        "value_overlap_pct": rel.value_overlap_pct,
        "components": [primary_component, *companion_components],
        "composite_candidate": bool(companion_components),
    }


def _normalize_key_name(column: str | None) -> str | None:
    """Canonicalize a column name for same-business-key identity comparison.

    upper/strip only — two columns are "the same business key" iff their names
    match after this normalization (CUSTOMER_ID == ' customer_id '). Returns None
    when no name is available so the identity guard can never fire on absent
    evidence. Deliberately NOT a synonym/alias table: this is a pure identity
    check, not a name list.
    """
    if not column:
        return None
    normalized = column.strip().upper()
    return normalized or None


def _is_same_business_key(left_column: str | None, right_column: str | None) -> bool:
    left = _normalize_key_name(left_column)
    right = _normalize_key_name(right_column)
    return bool(left) and left == right


def classify_join_approval(
    *,
    role: str | None,
    relationship_type: str,
    value_overlap: float,
    confidence: float,
    cardinality_left: int,
    cardinality_right: int,
    ubiquity: float,
    has_companion: bool,
    policy,
    left_column: str | None = None,
    right_column: str | None = None,
) -> tuple[str, str | None]:
    """Pure, data-driven join-approval decision (no DB, no I/O).

    Promotion is VALUE + IDENTITY driven and ROLE-INDEPENDENT. The primary gate
    is the SAME-KEY IDENTITY GUARD: an edge is promotable ONLY when its two join
    columns are the same business key (normalize(left)==normalize(right)). This
    is the decisive fix — two independent small-integer id sequences that
    coincidentally value-overlap (BANK_ACCOUNT_ID vs VENDOR_ID, PLAN_ID vs
    COST_TYPE_ID) can no longer be approved no matter how strong their overlap.

    Promote to `approved` IFF ALL hold (NO role check, NO confidence floor):
      1. same business key:  normalize(left_column) == normalize(right_column)
      2. value_overlap   >= policy.min_join_overlap     (value reconciliation)
      3. min_cardinality >= policy.min_join_cardinality  (real key, not an enum)
      4. ubiquity        <= policy.ubiquity_ceiling      (not an audit/system col)
      5. NOT a template clone: an edge is a copied/templated column (not a
         referential FK) when value_overlap >= policy.clone_overlap_floor AND
         cardinality_left == cardinality_right (the two sides hold the IDENTICAL
         generated value set). Real masters differ in cardinality between sides
         (CUSTOMER_ID 312/300, VENDOR_ID 181/190), so this drops only clones.

    BOTH raw cardinalities are threaded so the clone signature (left == right) is
    decidable; min_cardinality (the legacy key-strength floor) stays derivable as
    min(cardinality_left, cardinality_right).

    `ubiquity` is the fraction of the container's files whose join column has this
    name; an audit/system column (created_by, ...) approaches 1.0 while a business
    key stays low. It is the decisive separator between a real master and an
    equally-cardinal audit column, which cardinality alone cannot distinguish.
    Same-name promotion bypasses the risky-single-column gate AND the confidence
    floor ONLY for same-name value-validated keys; every other edge still flows
    through the legacy gates below.

    Returns (approval_status, risk_reason). approval_status is "approved" or
    "candidate"; risk_reason is None only when approved.
    """
    same_business_key = _is_same_business_key(left_column, right_column)
    # A key is only as strong as its weaker side. Derived here so the legacy
    # cardinality floor is unchanged while the clone guard sees both raw sides.
    min_cardinality = min(cardinality_left, cardinality_right)

    # SAME-KEY IDENTITY PROMOTION. Evaluated BEFORE the risky-single-column gate
    # so a same-name value-validated key (e.g. reference_key:customer on
    # CUSTOMER_ID==CUSTOMER_ID) is promoted regardless of role and without the
    # mis-calibrated confidence floor — but only when the value/identity evidence
    # is unambiguous. Cross-name edges (same_business_key is False) fall straight
    # through to the legacy gates.
    if same_business_key:
        # Audit/system column: broadly present across the container's files.
        # Checked first because an audit column can share a master's cardinality
        # AND its name on both sides — only ubiquity separates them.
        if ubiquity > policy.ubiquity_ceiling:
            return "candidate", "ubiquitous/audit column, not a business key"
        # Degenerate cardinality: too few distinct values to be a real key.
        if min_cardinality < policy.min_join_cardinality:
            return "candidate", "cardinality too low to be a key"
        # Weak value overlap: columns do not value-reconcile across tables.
        if value_overlap < policy.min_join_overlap:
            return "candidate", "insufficient value overlap"
        # Template clone: a copied/templated column, not a referential FK. The
        # verified clone signature is near-total overlap AND identical cardinality
        # on both sides (the two columns hold the SAME generated value set). Real
        # masters differ in cardinality between sides, so this rejects only the
        # fabricated document-key joins between unrelated tables. Checked LAST in
        # the promotion branch — after same-name + value/cardinality/ubiquity pass
        # but before approving — so a real master same-name edge still approves.
        if (
            value_overlap >= policy.clone_overlap_floor
            and cardinality_left == cardinality_right
        ):
            return "candidate", "templated/copied column, not a referential key"
        # All gates hold → approved (role-independent, no confidence floor).
        return "approved", None

    # 1) A risky single-column reference role is never a safe join on its own
    #    (cross-name edges only reach here — same-name strong keys were already
    #    promoted above).
    if is_risky_single_column_join_role(role):
        if has_companion:
            return "candidate", "single-column reference key needs composite join approval before use"
        return "candidate", f"{role} alone is not a safe business join key"

    # Cross-name edges get the audit/cardinality/overlap rejection reasons so the
    # risk_reason is informative; they are NEVER promoted (the same-key guard
    # above is the only promotion path).
    if ubiquity > policy.ubiquity_ceiling:
        return "candidate", "ubiquitous/audit column, not a business key"
    if min_cardinality < policy.min_join_cardinality:
        return "candidate", "cardinality too low to be a key"
    if value_overlap < policy.min_join_overlap:
        return "candidate", "insufficient value overlap"

    # 2) Existing behavior: many_to_many is unsafe unless promoted above; an
    #    edge with a detected PK side is approved when confidence + overlap pass.
    if relationship_type == "many_to_many":
        if has_companion:
            return "candidate", "many-to-many join needs composite grain approval before use"
        return "candidate", "many-to-many join can duplicate rows unless a grain rule approves it"
    if confidence >= policy.approved_join_confidence and value_overlap >= policy.approved_join_min_overlap:
        return "approved", None
    return "candidate", "needs stronger confidence or business approval"


def _approval_status(
    rel: FileRelationship,
    relationship_type: str,
    companion_components: list[dict],
    *,
    cardinality_left: int,
    cardinality_right: int,
    ubiquity: float,
    from_column: str | None,
    to_column: str | None,
) -> tuple[str, str | None]:
    policy = get_semantic_policy()
    confidence = rel.confidence_score or 0.0
    overlap = rel.value_overlap_pct or 0.0
    role = rel.semantic_role

    return classify_join_approval(
        role=role,
        relationship_type=relationship_type,
        value_overlap=overlap,
        confidence=confidence,
        cardinality_left=cardinality_left,
        cardinality_right=cardinality_right,
        ubiquity=ubiquity,
        has_companion=bool(companion_components),
        policy=policy,
        left_column=from_column,
        right_column=to_column,
    )


async def upsert_semantic_relationships_for_file(file_id: str, db: AsyncSession) -> int:
    relationships = (
        await db.execute(
            select(FileRelationship).where(
                (FileRelationship.file_a_id == file_id) | (FileRelationship.file_b_id == file_id)
            )
        )
    ).scalars().all()

    created_or_updated = 0
    ubiquity_map: dict[str, float] | None = None  # built once per container, lazily
    for rel in relationships:
        meta_a = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == rel.file_a_id))
        ).scalar_one_or_none()
        meta_b = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == rel.file_b_id))
        ).scalar_one_or_none()
        if not meta_a or not meta_b or not meta_a.container_id or meta_a.container_id != meta_b.container_id:
            continue

        if ubiquity_map is None:
            ubiquity_map = await _build_ubiquity_map(meta_a.container_id, db)

        entity_a = await upsert_semantic_entity(rel.file_a_id, db)
        entity_b = await upsert_semantic_entity(rel.file_b_id, db)
        if not entity_a or not entity_b:
            continue

        from_column = rel.shared_column
        to_column = rel.related_column or rel.shared_column
        kind_a = await _key_kind(rel.file_a_id, from_column, db)
        kind_b = await _key_kind(rel.file_b_id, to_column, db)
        relationship_type = _relationship_type(kind_a, kind_b)
        companion_components = _compatible_role_components(
            meta_a,
            meta_b,
            from_column=from_column,
            to_column=to_column,
            primary_role=rel.semantic_role,
        )
        # min cardinality across the join pair: a key is only as strong as its
        # weaker side. ubiquity: an edge is audit/system if EITHER side column is
        # broadly present, so take the max across the pair.
        card_a = await _key_cardinality(rel.file_a_id, from_column, db)
        card_b = await _key_cardinality(rel.file_b_id, to_column, db)
        min_cardinality = min(card_a, card_b)
        ubiquity = max(
            ubiquity_map.get(from_column, 0.0),
            ubiquity_map.get(to_column, 0.0),
        )
        approval_status, risk_reason = _approval_status(
            rel,
            relationship_type,
            companion_components,
            cardinality_left=card_a,
            cardinality_right=card_b,
            ubiquity=ubiquity,
            from_column=from_column,
            to_column=to_column,
        )
        # Behavior-change marker: a same-business-key promoted to an approved join
        # over a many_to_many edge is exactly the relational fix — log it so
        # reviewers can see the same-key identity gate firing (only emits when the
        # flag is on and it fires). `same_business_key` is always True here.
        if approval_status == "approved" and relationship_type == "many_to_many":
            ingest_logger.info(
                "semantic_layer_master_key_promoted",
                container_id=meta_a.container_id,
                from_column=from_column,
                to_column=to_column,
                same_business_key=_is_same_business_key(from_column, to_column),
                role=rel.semantic_role,
                value_overlap=rel.value_overlap_pct,
                min_cardinality=min_cardinality,
                ubiquity=round(ubiquity, 4),
                confidence=rel.confidence_score,
            )

        semantic_rel = (
            await db.execute(
                select(SemanticRelationship).where(
                    SemanticRelationship.source_relationship_id == rel.id
                )
            )
        ).scalar_one_or_none()
        if not semantic_rel:
            semantic_rel = SemanticRelationship(
                id=str(uuid.uuid4()),
                source_relationship_id=rel.id,
                container_id=meta_a.container_id,
                file_a_id=rel.file_a_id,
                file_b_id=rel.file_b_id,
            )
            db.add(semantic_rel)

        semantic_rel.from_entity = entity_a.entity_name
        semantic_rel.to_entity = entity_b.entity_name
        semantic_rel.from_column = from_column
        semantic_rel.to_column = to_column
        semantic_rel.relationship_type = relationship_type
        semantic_rel.join_rule = _join_rule(
            rel,
            from_column=from_column,
            to_column=to_column,
            kind_a=kind_a,
            kind_b=kind_b,
            companion_components=companion_components,
        )
        semantic_rel.approval_status = approval_status
        semantic_rel.risk_reason = risk_reason
        semantic_rel.confidence_score = rel.confidence_score or 0.0
        semantic_rel.status = "active"
        created_or_updated += 1

    return created_or_updated


async def build_semantic_layer_for_file(file_id: str, db: AsyncSession) -> dict:
    entity = await upsert_semantic_entity(file_id, db)
    relationship_count = await upsert_semantic_relationships_for_file(file_id, db)
    await db.commit()

    ingest_logger.info(
        "semantic_layer_builder",
        file_id=file_id,
        entity=entity.entity_name if entity else None,
        relationships=relationship_count,
    )
    return {
        "entity": entity.entity_name if entity else None,
        "relationships": relationship_count,
    }


# ── SME canonical-master election (container-level, flag-gated) ──────────────
# Generic STRUCTURAL priors for the master ranker — NOT dataset-fitted constants
# and NOT business rules. Same shape as ingestion_confidence's weighted score.
# A master table is: key-distinct (high unique_rate / low null on its primary
# key), attribute-rich rather than measure-heavy, and the target of many
# value-overlap edges. Every INPUT below is computed from the data; only these
# blend weights are fixed. The four STRUCTURAL weights sum to 1.0; the edge term
# is a SEPARATE bounded bonus in [0, _MASTER_W_EDGE_BONUS] — it is saturated so a
# transaction "supernode" (a generic id linked everywhere) can never out-rank a
# clean dimension master on edge count alone. No column-name lists anywhere.
_MASTER_W_UNIQUE = 0.40        # distinct cardinality of the key
_MASTER_W_NONNULL = 0.20       # 1 - null_rate of the key
_MASTER_W_HAS_KEY = 0.20       # has a usable primary key at all
_MASTER_W_ATTR_RICH = 0.20     # 1 - measure_density (masters describe, not measure)
_MASTER_W_EDGE_BONUS = 0.10    # max contribution of the (bounded) inbound-edge bonus
_MASTER_EDGE_SATURATION = 5.0  # half-saturation point: edges/(edges+this) in [0,1)


def _master_score(
    ent: SemanticEntity,
    reg_by_file_col: dict[tuple[str, str], ColumnKeyRegistry],
    inbound_edges: dict[str, int],
) -> float:
    pk = ent.primary_key
    reg = reg_by_file_col.get((ent.file_id, pk)) if pk else None
    unique_rate = reg.unique_rate if reg else 0.0
    null_rate = reg.null_rate if reg else 1.0

    n_metrics = len(ent.metrics or [])
    total_cols = max(n_metrics + len(ent.attributes or []) + len(ent.dimensions or []), 1)
    measure_density = n_metrics / total_cols

    # Bounded saturating edge signal in [0,1): no unbounded log term, so the edge
    # bonus can never exceed its weight or dominate the key-quality signals.
    edges = inbound_edges.get(ent.file_id, 0)
    edge_signal = edges / (edges + _MASTER_EDGE_SATURATION)

    return (
        _MASTER_W_UNIQUE * unique_rate
        + _MASTER_W_NONNULL * (1.0 - null_rate)
        + _MASTER_W_HAS_KEY * (1.0 if (pk and unique_rate > 0) else 0.0)
        + _MASTER_W_ATTR_RICH * (1.0 - measure_density)
        + _MASTER_W_EDGE_BONUS * edge_signal
    )


async def apply_master_election(container_id: str, db: AsyncSession) -> dict:
    """Container-level SME pass over EXISTING semantic artifacts (no re-ingest).

    1. Elect ONE canonical master table per entity label by data evidence.
    2. Demote join keys that are ubiquitous/system columns (data-derived, NOT a
       hardcoded name list) to `candidate`.
    3. Promote single-column joins whose PK side is a canonical master's primary
       key (and that clear the value-overlap floor) to `approved`.

    Idempotent. Runs unconditionally at container finalize and manual rebuild.
    """
    settings = get_settings()

    entities = (await db.execute(
        select(SemanticEntity).where(
            SemanticEntity.container_id == container_id,
            SemanticEntity.status == "active",
        ).order_by(SemanticEntity.file_id)  # deterministic election across runs
    )).scalars().all()
    if not entities:
        return {"entities": 0, "masters_elected": 0, "promoted": 0, "demoted": 0}

    registry = (await db.execute(
        select(ColumnKeyRegistry).where(ColumnKeyRegistry.container_id == container_id)
    )).scalars().all()
    relationships = (await db.execute(
        select(SemanticRelationship).where(
            SemanticRelationship.container_id == container_id,
            SemanticRelationship.status == "active",
        )
    )).scalars().all()

    reg_by_file_col: dict[tuple[str, str], ColumnKeyRegistry] = {
        (r.file_id, r.column_name): r for r in registry
    }

    # Ubiquity = system/audit-column detection, purely distributional (no name
    # list). A column is system/audit if it is BROADLY PRESENT *and* does not
    # strongly value-reconcile: audit columns (created_by, ...) appear in most
    # tables but never join, whereas a conformed dimension key (vendor_id) is
    # also widespread but HAS strong value overlap — the corroboration below
    # protects those real keys from being demoted.
    policy = get_semantic_policy()
    strong_reconciler_columns: set[str] = set()
    for rel in relationships:
        if ((rel.join_rule or {}).get("value_overlap_pct") or 0.0) >= policy.min_join_overlap:
            strong_reconciler_columns.add(rel.from_column)
            strong_reconciler_columns.add(rel.to_column)

    files_per_column: dict[str, set[str]] = {}
    all_files: set[str] = set()
    for r in registry:
        files_per_column.setdefault(r.column_name, set()).add(r.file_id)
        all_files.add(r.file_id)

    ubiquitous_columns: set[str] = set()
    total_files = len(all_files)
    if files_per_column and total_files >= settings.SME_AUDIT_MIN_FILES:
        counts = sorted(len(s) for s in files_per_column.values())
        median = counts[len(counts) // 2]
        rel_threshold = median * settings.SME_AUDIT_UBIQUITY_MULTIPLE
        abs_threshold = settings.SME_AUDIT_ABS_COVERAGE * total_files
        for col, files in files_per_column.items():
            n = len(files)
            broadly_present = n >= abs_threshold or n > rel_threshold
            if broadly_present and col not in strong_reconciler_columns:
                ubiquitous_columns.add(col)

    # Inbound value-overlap edges per PK/target-side file (a master signal).
    inbound_edges: dict[str, int] = {}
    for rel in relationships:
        rtype = rel.relationship_type
        if rtype == "one_to_many" or rtype == "one_to_one":
            pk_file = rel.file_a_id
        elif rtype == "many_to_one":
            pk_file = rel.file_b_id
        else:
            continue  # many_to_many is not a master signal
        inbound_edges[pk_file] = inbound_edges.get(pk_file, 0) + 1

    # Elect a master per entity label.
    by_label: dict[str, list[SemanticEntity]] = {}
    for ent in entities:
        by_label.setdefault(ent.entity_name, []).append(ent)

    masters_elected = 0
    for label, members in by_label.items():
        # file_id breaks score ties deterministically (entities are ORDER BY file_id).
        best = max(members, key=lambda e: (_master_score(e, reg_by_file_col, inbound_edges), e.file_id))
        for ent in members:
            elected = ent is best
            ent.is_canonical_master = elected
            ent.master_for_entity = label if elected else None
        masters_elected += 1

    master_pk_by_file = {
        e.file_id: e.primary_key
        for e in entities
        if e.is_canonical_master and e.primary_key
    }

    # Re-approve relationships: demote ubiquitous keys, promote master keys.
    promoted = 0
    demoted = 0
    for rel in relationships:
        if rel.from_column in ubiquitous_columns or rel.to_column in ubiquitous_columns:
            if rel.approval_status != "candidate":
                demoted += 1
            rel.approval_status = "candidate"
            rel.risk_reason = "ubiquitous/system column (data-derived ubiquity), not a business join key"
            continue
        if (rel.join_rule or {}).get("composite_candidate"):
            continue
        rtype = rel.relationship_type
        if rtype == "one_to_many":
            pk_file, pk_col = rel.file_a_id, rel.from_column
        elif rtype == "many_to_one":
            pk_file, pk_col = rel.file_b_id, rel.to_column
        else:
            continue
        # value_overlap_pct lives in the SemanticRelationship's join_rule JSONB
        # (it is a FileRelationship column, not a SemanticRelationship one).
        overlap = (rel.join_rule or {}).get("value_overlap_pct") or 0.0
        confidence = rel.confidence_score or 0.0
        # Promotion bypasses ONLY the risky-single-column guard — never the
        # evidence bar. It must clear the real confidence floor AND the strong
        # value-overlap floor (min_join_overlap, not the weak 0.01 approval
        # floor), so a wrongly-elected master can't bless a 1%-overlap join.
        if (
            master_pk_by_file.get(pk_file) == pk_col
            and confidence >= policy.approved_join_confidence
            and overlap >= policy.min_join_overlap
            and rel.approval_status != "approved"
        ):
            rel.approval_status = "approved"
            rel.risk_reason = None
            promoted += 1

    await db.commit()
    result = {
        "entities": len(entities),
        "labels": len(by_label),
        "masters_elected": masters_elected,
        "ubiquitous_columns": len(ubiquitous_columns),
        "promoted": promoted,
        "demoted": demoted,
    }
    ingest_logger.info("sme_master_election", container_id=container_id, **result)
    return result

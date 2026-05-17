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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    return components[:10]


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


def _approval_status(
    rel: FileRelationship,
    relationship_type: str,
    companion_components: list[dict],
) -> tuple[str, str | None]:
    policy = get_semantic_policy()
    confidence = rel.confidence_score or 0.0
    overlap = rel.value_overlap_pct or 0.0
    role = rel.semantic_role

    if is_risky_single_column_join_role(role):
        if companion_components:
            return "candidate", "single-column reference key needs composite join approval before use"
        return "candidate", f"{role} alone is not a safe business join key"
    if relationship_type == "many_to_many":
        if companion_components:
            return "candidate", "many-to-many join needs composite grain approval before use"
        return "candidate", "many-to-many join can duplicate rows unless a grain rule approves it"
    if confidence >= policy.approved_join_confidence and overlap >= policy.approved_join_min_overlap:
        return "approved", None
    return "candidate", "needs stronger confidence or business approval"


async def upsert_semantic_relationships_for_file(file_id: str, db: AsyncSession) -> int:
    relationships = (
        await db.execute(
            select(FileRelationship).where(
                (FileRelationship.file_a_id == file_id) | (FileRelationship.file_b_id == file_id)
            )
        )
    ).scalars().all()

    created_or_updated = 0
    for rel in relationships:
        meta_a = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == rel.file_a_id))
        ).scalar_one_or_none()
        meta_b = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == rel.file_b_id))
        ).scalar_one_or_none()
        if not meta_a or not meta_b or not meta_a.container_id or meta_a.container_id != meta_b.container_id:
            continue

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
        approval_status, risk_reason = _approval_status(rel, relationship_type, companion_components)

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

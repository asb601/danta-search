"""Semantic-only rebuild and evaluation workflows.

These jobs repair or refresh the semantic layer without re-running file
cleaning, parquet conversion, embeddings, or analytics. The workflow is
container-scoped, batched, and role-kind driven; it does not contain any
business-specific role names.
"""
from __future__ import annotations

import time
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.database import async_session
from app.core.logger import ingest_logger
from app.models.column_key_registry import ColumnKeyRegistry
from app.models.container import ContainerConfig
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.models.semantic_layer import SemanticEntity, SemanticRelationship
from app.services.semantic_roles import is_dynamic_role, role_kind

DEFAULT_BATCH_SIZE = 250


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


async def _require_container(container_id: str, db: AsyncSession) -> ContainerConfig:
    container = await db.get(ContainerConfig, container_id)
    if not container:
        raise ValueError("container not found")
    return container


async def _eligible_file_count(container_id: str, db: AsyncSession) -> int:
    count = await db.execute(
        select(func.count(File.id))
        .join(FileMetadata, FileMetadata.file_id == File.id)
        .where(
            File.container_id == container_id,
            FileMetadata.columns_info.isnot(None),
        )
    )
    return int(count.scalar_one() or 0)


async def _file_id_batch(
    container_id: str,
    *,
    after_id: str | None,
    batch_size: int,
) -> list[str]:
    async with async_session() as db:
        stmt = (
            select(File.id)
            .join(FileMetadata, FileMetadata.file_id == File.id)
            .where(
                File.container_id == container_id,
                FileMetadata.columns_info.isnot(None),
            )
            .order_by(File.id)
            .limit(batch_size)
        )
        if after_id:
            stmt = stmt.where(File.id > after_id)
        return list((await db.execute(stmt)).scalars().all())


async def _clear_semantic_artifacts(container_id: str, batch_size: int) -> dict[str, int]:
    deleted = {
        "semantic_relationships": 0,
        "semantic_entities": 0,
        "file_relationships": 0,
        "key_registry_rows": 0,
    }

    async with async_session() as db:
        result = await db.execute(
            delete(SemanticRelationship).where(SemanticRelationship.container_id == container_id)
        )
        deleted["semantic_relationships"] = int(result.rowcount or 0)
        result = await db.execute(delete(SemanticEntity).where(SemanticEntity.container_id == container_id))
        deleted["semantic_entities"] = int(result.rowcount or 0)
        result = await db.execute(delete(ColumnKeyRegistry).where(ColumnKeyRegistry.container_id == container_id))
        deleted["key_registry_rows"] = int(result.rowcount or 0)
        await db.commit()

    after_id: str | None = None
    while True:
        file_ids = await _file_id_batch(container_id, after_id=after_id, batch_size=batch_size)
        if not file_ids:
            break
        after_id = file_ids[-1]
        async with async_session() as db:
            result = await db.execute(
                delete(FileRelationship).where(
                    or_(
                        FileRelationship.file_a_id.in_(file_ids),
                        FileRelationship.file_b_id.in_(file_ids),
                    )
                )
            )
            deleted["file_relationships"] += int(result.rowcount or 0)
            await db.commit()

    return deleted


async def _run_ontology_for_file(file_id: str) -> bool:
    from app.services.ingestion_stages import ontology_stage

    await ontology_stage({"file_id": file_id})
    return True


async def _register_keys_for_file(file_id: str) -> int:
    from app.services.relationship_index import register_file_key_fingerprints

    async with async_session() as db:
        return await register_file_key_fingerprints(file_id, db)


async def _detect_relationships_for_file(file_id: str) -> int:
    from app.services.relationship_detector import detect_relationships

    async with async_session() as db:
        file = await db.get(File, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not file or not metadata:
            return 0
        return await detect_relationships(
            file_id=file_id,
            blob_path=file.blob_path or metadata.blob_path or "",
            columns_info=metadata.columns_info or [],
            db=db,
        )


async def _build_semantic_layer_for_file(file_id: str) -> int:
    from app.services.semantic_layer_builder import build_semantic_layer_for_file

    async with async_session() as db:
        result = await build_semantic_layer_for_file(file_id, db)
    return int(result.get("relationships") or 0)


async def rebuild_container_semantics(
    container_id: str,
    *,
    re_resolve_roles: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Rebuild semantic roles, key registry, relationships, and semantic layer.

    The job is intentionally semantic-only. It reuses existing metadata samples
    and stored files; it does not touch preprocessing/parquet/analytics output.
    """
    start = time.perf_counter()
    batch_size = max(1, batch_size)

    async with async_session() as db:
        await _require_container(container_id, db)
        total_files = await _eligible_file_count(container_id, db)

    counters: dict[str, Any] = {
        "container_id": container_id,
        "eligible_files": total_files,
        "roles_re_resolved": 0,
        "role_resolution_failed": 0,
        "key_registry_rows": 0,
        "relationship_rows_created": 0,
        "semantic_relationships_upserted": 0,
        "file_failures": [],
    }

    if re_resolve_roles:
        after_id: str | None = None
        while True:
            file_ids = await _file_id_batch(container_id, after_id=after_id, batch_size=batch_size)
            if not file_ids:
                break
            after_id = file_ids[-1]
            for file_id in file_ids:
                try:
                    await _run_ontology_for_file(file_id)
                    counters["roles_re_resolved"] += 1
                except Exception as exc:
                    counters["role_resolution_failed"] += 1
                    if len(counters["file_failures"]) < 20:
                        counters["file_failures"].append({"file_id": file_id, "stage": "ontology", "error": str(exc)[:300]})
                    ingest_logger.warning("semantic_rebuild_file_failed", file_id=file_id, stage="ontology", error=str(exc)[:300])

    counters["deleted"] = await _clear_semantic_artifacts(container_id, batch_size)

    after_id = None
    while True:
        file_ids = await _file_id_batch(container_id, after_id=after_id, batch_size=batch_size)
        if not file_ids:
            break
        after_id = file_ids[-1]
        for file_id in file_ids:
            try:
                counters["key_registry_rows"] += await _register_keys_for_file(file_id)
            except Exception as exc:
                if len(counters["file_failures"]) < 20:
                    counters["file_failures"].append({"file_id": file_id, "stage": "key_registry", "error": str(exc)[:300]})
                ingest_logger.warning("semantic_rebuild_file_failed", file_id=file_id, stage="key_registry", error=str(exc)[:300])

    after_id = None
    while True:
        file_ids = await _file_id_batch(container_id, after_id=after_id, batch_size=batch_size)
        if not file_ids:
            break
        after_id = file_ids[-1]
        for file_id in file_ids:
            try:
                counters["relationship_rows_created"] += await _detect_relationships_for_file(file_id)
            except Exception as exc:
                if len(counters["file_failures"]) < 20:
                    counters["file_failures"].append({"file_id": file_id, "stage": "relationships", "error": str(exc)[:300]})
                ingest_logger.warning("semantic_rebuild_file_failed", file_id=file_id, stage="relationships", error=str(exc)[:300])

    after_id = None
    while True:
        file_ids = await _file_id_batch(container_id, after_id=after_id, batch_size=batch_size)
        if not file_ids:
            break
        after_id = file_ids[-1]
        for file_id in file_ids:
            try:
                counters["semantic_relationships_upserted"] += await _build_semantic_layer_for_file(file_id)
            except Exception as exc:
                if len(counters["file_failures"]) < 20:
                    counters["file_failures"].append({"file_id": file_id, "stage": "semantic_layer", "error": str(exc)[:300]})
                ingest_logger.warning("semantic_rebuild_file_failed", file_id=file_id, stage="semantic_layer", error=str(exc)[:300])

    async with async_session() as db:
        counters["evaluation"] = await evaluate_container_semantics(container_id, db, batch_size=batch_size)

    counters["duration_ms"] = _ms(start)
    ingest_logger.info("semantic_rebuild_complete", **{k: v for k, v in counters.items() if k != "evaluation"})

    try:
        from app.agent.graph.graph import invalidate_catalog_cache

        invalidate_catalog_cache()
    except Exception as exc:
        ingest_logger.warning("catalog_invalidate_failed", container_id=container_id, error=str(exc)[:200])

    return counters


async def evaluate_container_semantics(
    container_id: str,
    db: AsyncSession,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Return generic semantic quality metrics for a container."""
    await _require_container(container_id, db)
    batch_size = max(1, batch_size)

    total_columns = 0
    resolved_columns = 0
    total_role_assignments = 0
    dynamic_role_assignments = 0
    non_dynamic_role_assignments = 0
    files_with_metadata = 0
    files_with_roles = 0
    files_without_roles = 0
    role_kind_counts: dict[str, int] = {}

    after_id: str | None = None
    while True:
        stmt = (
            select(File.id, FileMetadata.columns_info, FileMetadata.column_semantic_roles)
            .join(FileMetadata, FileMetadata.file_id == File.id)
            .where(File.container_id == container_id)
            .order_by(File.id)
            .limit(batch_size)
        )
        if after_id:
            stmt = stmt.where(File.id > after_id)
        rows = (await db.execute(stmt)).all()
        if not rows:
            break
        after_id = rows[-1][0]

        for _, columns_info, roles in rows:
            files_with_metadata += 1
            columns = columns_info or []
            role_map = roles or {}
            if role_map:
                files_with_roles += 1
            else:
                files_without_roles += 1

            column_names = [col.get("name") for col in columns if isinstance(col, dict) and col.get("name")]
            total_columns += len(column_names)
            resolved_columns += sum(1 for column_name in column_names if column_name in role_map)

            for role in role_map.values():
                if not isinstance(role, str):
                    continue
                total_role_assignments += 1
                if is_dynamic_role(role):
                    dynamic_role_assignments += 1
                else:
                    non_dynamic_role_assignments += 1
                kind = role_kind(role) or "unclassified"
                role_kind_counts[kind] = role_kind_counts.get(kind, 0) + 1

    FileA = aliased(File)
    relationship_count = int((await db.execute(
        select(func.count(FileRelationship.id))
        .join(FileA, FileA.id == FileRelationship.file_a_id)
        .where(FileA.container_id == container_id)
    )).scalar_one() or 0)
    relationships_without_value_evidence = int((await db.execute(
        select(func.count(FileRelationship.id))
        .join(FileA, FileA.id == FileRelationship.file_a_id)
        .where(FileA.container_id == container_id, FileRelationship.value_overlap_pct.is_(None))
    )).scalar_one() or 0)

    key_registry_rows = int((await db.execute(
        select(func.count(ColumnKeyRegistry.id)).where(ColumnKeyRegistry.container_id == container_id)
    )).scalar_one() or 0)
    semantic_entity_count = int((await db.execute(
        select(func.count(SemanticEntity.id)).where(SemanticEntity.container_id == container_id)
    )).scalar_one() or 0)
    semantic_relationship_count = int((await db.execute(
        select(func.count(SemanticRelationship.id)).where(SemanticRelationship.container_id == container_id)
    )).scalar_one() or 0)

    approval_rows = (await db.execute(
        select(SemanticRelationship.approval_status, func.count(SemanticRelationship.id))
        .where(SemanticRelationship.container_id == container_id)
        .group_by(SemanticRelationship.approval_status)
    )).all()
    relationship_type_rows = (await db.execute(
        select(SemanticRelationship.relationship_type, func.count(SemanticRelationship.id))
        .where(SemanticRelationship.container_id == container_id)
        .group_by(SemanticRelationship.relationship_type)
    )).all()
    composite_candidate_count = int((await db.execute(
        select(func.count(SemanticRelationship.id))
        .where(
            SemanticRelationship.container_id == container_id,
            SemanticRelationship.join_rule["composite_candidate"].astext == "true",
        )
    )).scalar_one() or 0)

    approved_relationships = sum(int(count) for status, count in approval_rows if status == "approved")
    role_coverage_pct = round((resolved_columns / total_columns) * 100, 2) if total_columns else 0.0
    dynamic_role_pct = round((dynamic_role_assignments / total_role_assignments) * 100, 2) if total_role_assignments else 0.0
    value_evidence_pct = round(
        ((relationship_count - relationships_without_value_evidence) / relationship_count) * 100,
        2,
    ) if relationship_count else 0.0
    approved_relationship_pct = round(
        (approved_relationships / semantic_relationship_count) * 100,
        2,
    ) if semantic_relationship_count else 0.0

    return {
        "container_id": container_id,
        "files": {
            "with_metadata": files_with_metadata,
            "with_roles": files_with_roles,
            "without_roles": files_without_roles,
        },
        "columns": {
            "total": total_columns,
            "resolved": resolved_columns,
            "coverage_pct": role_coverage_pct,
        },
        "roles": {
            "assignments": total_role_assignments,
            "dynamic_assignments": dynamic_role_assignments,
            "dynamic_pct": dynamic_role_pct,
            "non_dynamic_assignments": non_dynamic_role_assignments,
            "by_kind": dict(sorted(role_kind_counts.items())),
        },
        "relationships": {
            "technical": relationship_count,
            "without_value_evidence": relationships_without_value_evidence,
            "value_evidence_pct": value_evidence_pct,
            "semantic": semantic_relationship_count,
            "approved_pct": approved_relationship_pct,
            "by_approval_status": {str(status): int(count) for status, count in approval_rows},
            "by_type": {str(kind): int(count) for kind, count in relationship_type_rows},
            "composite_candidates": composite_candidate_count,
        },
        "indexes": {
            "key_registry_rows": key_registry_rows,
            "semantic_entities": semantic_entity_count,
        },
    }
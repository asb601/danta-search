from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import require_admin
from app.models.semantic_memory import SemanticDomainCluster, SemanticDomainConflict, SemanticMemoryRecord
from app.models.user import User
from app.services.semantic_memory_indexer import rebuild_memory_indexes
from app.services.semantic_memory_normalizer import GOVERNANCE_STATUSES, RECORD_STATUSES

router = APIRouter(prefix="/semantic-memory", tags=["semantic-memory"])


class GovernanceUpdate(BaseModel):
    governance_status: str | None = None
    status: str | None = None


def _record_row(record: SemanticMemoryRecord) -> dict:
    return {
        "id": record.id,
        "container_id": record.container_id,
        "memory_type": record.memory_type,
        "title": record.title,
        "summary": record.summary,
        "terms": record.normalized_terms or [],
        "behaviors": record.behaviors or [],
        "confidence_score": record.confidence_score,
        "authority_score": record.authority_score,
        "governance_status": record.governance_status,
        "status": record.status,
        "source": record.source,
        "source_file_id": record.source_file_id,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def _domain_row(domain: SemanticDomainCluster) -> dict:
    return {
        "id": domain.id,
        "container_id": domain.container_id,
        "domain_type": domain.domain_type,
        "domain_key": domain.domain_key,
        "title": domain.title,
        "summary": domain.summary,
        "terms": domain.normalized_terms or [],
        "workflow_terms": domain.workflow_terms or [],
        "lifecycle_terms": domain.lifecycle_terms or [],
        "kpi_terms": domain.kpi_terms or [],
        "synonym_terms": domain.synonym_terms or [],
        "contributor_file_ids": domain.contributor_file_ids or [],
        "evidence_count": domain.evidence_count,
        "conflict_count": domain.conflict_count,
        "conflict_summary": domain.conflict_summary or {},
        "confidence_score": domain.confidence_score,
        "authority_score": domain.authority_score,
        "drift_score": domain.drift_score,
        "governance_status": domain.governance_status,
        "status": domain.status,
        "updated_at": domain.updated_at.isoformat() if domain.updated_at else None,
    }


@router.get("")
async def list_semantic_memory(
    container_id: str | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    governance_status: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = select(SemanticMemoryRecord)
    count_stmt = select(func.count(SemanticMemoryRecord.id))
    filters = []
    if container_id:
        filters.append(SemanticMemoryRecord.container_id == container_id)
    if memory_type:
        filters.append(SemanticMemoryRecord.memory_type == memory_type)
    if governance_status:
        filters.append(SemanticMemoryRecord.governance_status == governance_status)
    if q:
        pattern = f"%{q.strip()}%"
        filters.append(SemanticMemoryRecord.title.ilike(pattern))
    if filters:
        stmt = stmt.where(*filters)
        count_stmt = count_stmt.where(*filters)
    total = int((await db.execute(count_stmt)).scalar_one() or 0)
    rows = (
        await db.execute(
            stmt.order_by(
                SemanticMemoryRecord.authority_score.desc(),
                SemanticMemoryRecord.updated_at.desc(),
            ).limit(limit)
        )
    ).scalars().all()
    return {"total": total, "records": [_record_row(row) for row in rows]}


@router.get("/domains")
async def list_semantic_domains(
    container_id: str | None = Query(default=None),
    domain_type: str | None = Query(default=None),
    governance_status: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = select(SemanticDomainCluster)
    count_stmt = select(func.count(SemanticDomainCluster.id))
    filters = []
    if container_id:
        filters.append(SemanticDomainCluster.container_id == container_id)
    if domain_type:
        filters.append(SemanticDomainCluster.domain_type == domain_type)
    if governance_status:
        filters.append(SemanticDomainCluster.governance_status == governance_status)
    if q:
        pattern = f"%{q.strip()}%"
        filters.append(SemanticDomainCluster.title.ilike(pattern))
    if filters:
        stmt = stmt.where(*filters)
        count_stmt = count_stmt.where(*filters)
    total = int((await db.execute(count_stmt)).scalar_one() or 0)
    rows = (
        await db.execute(
            stmt.order_by(
                SemanticDomainCluster.authority_score.desc(),
                SemanticDomainCluster.updated_at.desc(),
            ).limit(limit)
        )
    ).scalars().all()
    conflict_stmt = select(func.count(SemanticDomainConflict.id))
    if container_id:
        conflict_stmt = conflict_stmt.where(SemanticDomainConflict.container_id == container_id)
    conflict_total = int((await db.execute(conflict_stmt)).scalar_one() or 0)
    return {"total": total, "conflicts": conflict_total, "domains": [_domain_row(row) for row in rows]}


@router.patch("/{memory_id}/governance")
async def update_semantic_memory_governance(
    memory_id: str,
    body: GovernanceUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    record = await db.get(SemanticMemoryRecord, memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Semantic memory record not found")
    if body.governance_status is not None:
        if body.governance_status not in GOVERNANCE_STATUSES:
            raise HTTPException(status_code=400, detail="Invalid governance_status")
        record.governance_status = body.governance_status
    if body.status is not None:
        if body.status not in RECORD_STATUSES:
            raise HTTPException(status_code=400, detail="Invalid status")
        record.status = body.status
    record.updated_at = datetime.now(timezone.utc)
    await rebuild_memory_indexes(db, record)
    await db.commit()
    return _record_row(record)


@router.post("/files/{file_id}/rebuild")
async def rebuild_file_semantic_memory(
    file_id: str,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.semantic_memory_extractor import upsert_semantic_memory_for_file

    return await upsert_semantic_memory_for_file(file_id, db)


@router.post("/containers/{container_id}/domains/rebuild")
async def rebuild_container_semantic_domains(
    container_id: str,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from app.services.semantic_domain_consolidator import consolidate_semantic_domains_for_container

    return await consolidate_semantic_domains_for_container(container_id, db)
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import require_admin
from app.models.semantic_memory import SemanticMemoryRecord
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
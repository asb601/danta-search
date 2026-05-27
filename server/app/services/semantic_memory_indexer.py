"""Deterministic indexes for SemanticMemory records."""
from __future__ import annotations

import uuid
from collections import Counter

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.semantic_memory import (
    SemanticMemoryAssetIndex,
    SemanticMemoryRecord,
    SemanticMemoryTermIndex,
)
from app.services.semantic_memory_normalizer import terms_from_text

_MAX_TERMS_PER_MEMORY = 36


async def rebuild_memory_indexes(db: AsyncSession, memory: SemanticMemoryRecord) -> None:
    await db.execute(delete(SemanticMemoryTermIndex).where(SemanticMemoryTermIndex.memory_id == memory.id))
    await db.execute(delete(SemanticMemoryAssetIndex).where(SemanticMemoryAssetIndex.memory_id == memory.id))

    status = "active" if memory.status == "active" and memory.governance_status in {"active", "candidate"} else "inactive"
    terms = terms_from_text(memory.normalized_terms, memory.title, memory.summary)[:_MAX_TERMS_PER_MEMORY]
    weighted = Counter(terms)
    for term, count in weighted.items():
        db.add(SemanticMemoryTermIndex(
            id=str(uuid.uuid4()),
            container_id=memory.container_id,
            memory_id=memory.id,
            term=term,
            token_class=memory.memory_type,
            weight=min(2.0, 1.0 + (count - 1) * 0.2),
            status=status,
        ))

    if memory.source_file_id:
        db.add(SemanticMemoryAssetIndex(
            id=str(uuid.uuid4()),
            container_id=memory.container_id,
            file_id=memory.source_file_id,
            memory_id=memory.id,
            index_kind=memory.memory_type,
            score=max(memory.confidence_score, memory.authority_score),
            terms=terms,
        ))
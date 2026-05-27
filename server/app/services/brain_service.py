"""Read-only bounded BrainContext resolver."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session
from app.models.semantic_memory import BrainContextTrace, SemanticMemoryRecord, SemanticMemoryTermIndex
from app.services.brain_context import BrainContext, BrainMemoryBrief, ExecutionEnvelope, RetrievalGuidance
from app.services.semantic_memory_normalizer import terms_from_text


class BrainService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._settings = get_settings()

    async def resolve(
        self,
        *,
        query: str,
        container_id: str | None,
        user_id: str | None,
        intent_plan: Any,
        authorized_file_ids: set[str],
    ) -> BrainContext:
        if not container_id or not authorized_file_ids:
            return BrainContext(caps=self._caps())

        query_terms = terms_from_text(query, getattr(intent_plan, "entities", []), getattr(intent_plan, "behaviors", []))
        candidates = await self._candidate_records(container_id, authorized_file_ids, query_terms)
        ranked = self._rank_records(candidates, query_terms, intent_plan)
        max_records = max(1, int(self._settings.BRAIN_CONTEXT_MAX_RECORDS))
        selected = ranked[:max_records]
        briefs = [self._brief(record, score) for record, score in selected]
        guidance = self._guidance(briefs, ranked, query_terms)
        envelope = ExecutionEnvelope(
            memory_ids=[brief.id for brief in briefs],
            anchor_file_ids=guidance.anchor_file_ids,
            ambiguity_flags=guidance.ambiguity_flags,
            authority_floor=min((brief.authority_score for brief in briefs), default=0.0),
        )
        return BrainContext(
            records=briefs,
            retrieval_guidance=guidance,
            execution_envelope=envelope,
            token_estimate=self._estimate_tokens(briefs),
            caps=self._caps(),
        )

    async def _candidate_records(
        self,
        container_id: str,
        authorized_file_ids: set[str],
        query_terms: list[str],
    ) -> list[SemanticMemoryRecord]:
        max_candidates = max(5, int(self._settings.BRAIN_CONTEXT_MAX_CANDIDATES))
        records_by_id: dict[str, SemanticMemoryRecord] = {}

        if query_terms:
            term_rows = (
                await self._db.execute(
                    select(SemanticMemoryTermIndex.memory_id)
                    .where(
                        SemanticMemoryTermIndex.container_id == container_id,
                        SemanticMemoryTermIndex.status == "active",
                        SemanticMemoryTermIndex.term.in_(query_terms[:60]),
                    )
                    .limit(max_candidates * 3)
                )
            ).all()
            memory_ids = list(dict.fromkeys(row.memory_id for row in term_rows))[:max_candidates]
            if memory_ids:
                rows = (
                    await self._db.execute(
                        select(SemanticMemoryRecord).where(
                            SemanticMemoryRecord.id.in_(memory_ids),
                            SemanticMemoryRecord.source_file_id.in_(authorized_file_ids),
                            SemanticMemoryRecord.status == "active",
                            SemanticMemoryRecord.governance_status.in_(["active", "candidate"]),
                        )
                    )
                ).scalars().all()
                records_by_id.update({row.id: row for row in rows})

        if len(records_by_id) < max_candidates:
            rows = (
                await self._db.execute(
                    select(SemanticMemoryRecord)
                    .where(
                        SemanticMemoryRecord.container_id == container_id,
                        SemanticMemoryRecord.source_file_id.in_(authorized_file_ids),
                        SemanticMemoryRecord.status == "active",
                        SemanticMemoryRecord.governance_status.in_(["active", "candidate"]),
                    )
                    .order_by(SemanticMemoryRecord.authority_score.desc(), SemanticMemoryRecord.confidence_score.desc())
                    .limit(max_candidates)
                )
            ).scalars().all()
            records_by_id.update({row.id: row for row in rows})

        return list(records_by_id.values())[:max_candidates]

    def _rank_records(
        self,
        records: list[SemanticMemoryRecord],
        query_terms: list[str],
        intent_plan: Any,
    ) -> list[tuple[SemanticMemoryRecord, float]]:
        q = set(query_terms)
        behaviors = set(terms_from_text(getattr(intent_plan, "behaviors", []) or []))
        ranked: list[tuple[SemanticMemoryRecord, float]] = []
        min_score = float(self._settings.BRAIN_CONTEXT_MIN_SCORE)
        for record in records:
            terms = set(terms_from_text(record.normalized_terms, record.title, record.summary))
            overlap = len(q & terms) / max(1, min(len(q), 8)) if q else 0.0
            behavior_terms = set(terms_from_text(record.behaviors or []))
            behavior_overlap = len(behaviors & behavior_terms) / max(1, len(behaviors)) if behaviors else 0.0
            score = (
                overlap * 0.45
                + behavior_overlap * 0.2
                + float(record.authority_score or 0.0) * 0.22
                + float(record.confidence_score or 0.0) * 0.13
            )
            if score >= min_score or record.governance_status == "active":
                ranked.append((record, round(score, 4)))
        ranked.sort(key=lambda item: (item[1], item[0].authority_score or 0.0), reverse=True)
        return ranked

    def _brief(self, record: SemanticMemoryRecord, score: float) -> BrainMemoryBrief:
        return BrainMemoryBrief(
            id=record.id,
            memory_type=record.memory_type,
            title=record.title,
            summary=record.summary,
            terms=terms_from_text(record.normalized_terms, record.title)[:10],
            behaviors=terms_from_text(record.behaviors)[:8],
            confidence_score=float(record.confidence_score or 0.0),
            authority_score=float(record.authority_score or 0.0),
            source_file_id=record.source_file_id,
            score=score,
        )

    def _guidance(
        self,
        briefs: list[BrainMemoryBrief],
        ranked: list[tuple[SemanticMemoryRecord, float]],
        query_terms: list[str],
    ) -> RetrievalGuidance:
        max_anchor_files = max(1, int(self._settings.BRAIN_CONTEXT_MAX_ANCHOR_FILES))
        authority_by_file: dict[str, float] = {}
        for brief in briefs:
            if not brief.source_file_id:
                continue
            authority_by_file[brief.source_file_id] = max(
                authority_by_file.get(brief.source_file_id, 0.0),
                brief.authority_score,
            )
        anchors = [
            file_id
            for file_id, _score in sorted(authority_by_file.items(), key=lambda item: item[1], reverse=True)
        ][:max_anchor_files]

        term_counts = Counter(term for brief in briefs for term in brief.terms if term in query_terms or len(term) > 3)
        ambiguity_flags = self._ambiguity_flags(ranked)
        return RetrievalGuidance(
            anchor_file_ids=anchors,
            preferred_terms=[term for term, _ in term_counts.most_common(max(1, int(self._settings.BRAIN_CONTEXT_MAX_TERMS)))],
            authority_by_file_id={file_id: round(score, 3) for file_id, score in authority_by_file.items()},
            ambiguity_flags=ambiguity_flags,
        )

    def _ambiguity_flags(self, ranked: list[tuple[SemanticMemoryRecord, float]]) -> list[str]:
        by_title: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for record, score in ranked[:20]:
            key = " ".join(terms_from_text(record.title)[:4])
            if key:
                by_title[key].append((record.memory_type, score))
        flags: list[str] = []
        for key, matches in by_title.items():
            types = {memory_type for memory_type, _ in matches}
            scores = sorted((score for _, score in matches), reverse=True)
            if len(matches) > 1 and (len(types) > 1 or (scores and scores[0] - scores[-1] < 0.12)):
                flags.append(f"ambiguous_memory:{key}")
            if len(flags) >= 5:
                break
        return flags

    def _estimate_tokens(self, briefs: list[BrainMemoryBrief]) -> int:
        char_count = sum(len(brief.title) + len(brief.summary or "") + 10 * len(brief.terms) for brief in briefs)
        return max(0, int(char_count / 4))

    def _caps(self) -> dict[str, int]:
        return {
            "max_records": int(self._settings.BRAIN_CONTEXT_MAX_RECORDS),
            "max_candidates": int(self._settings.BRAIN_CONTEXT_MAX_CANDIDATES),
            "max_terms": int(self._settings.BRAIN_CONTEXT_MAX_TERMS),
            "max_anchor_files": int(self._settings.BRAIN_CONTEXT_MAX_ANCHOR_FILES),
            "token_budget": int(self._settings.BRAIN_CONTEXT_TOKEN_BUDGET),
        }


async def record_brain_context_trace(
    db: AsyncSession,
    *,
    request_id: str,
    container_id: str | None,
    user_id: str | None,
    query: str,
    context: BrainContext,
) -> None:
    settings = get_settings()
    if not settings.BRAIN_CONTEXT_TRACE_ENABLED:
        return
    query_hash = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    _ = db  # kept for call-site compatibility; trace writes use an isolated transaction.
    async with async_session() as trace_db:
        trace_db.add(BrainContextTrace(
            request_id=request_id,
            container_id=container_id,
            user_id=user_id,
            query_hash=query_hash,
            selected_memory_ids=[record.id for record in context.records],
            ambiguity_flags=list(context.retrieval_guidance.ambiguity_flags),
            retrieval_guidance=context.retrieval_guidance.to_dict(),
            execution_envelope=context.execution_envelope.to_dict(),
            token_estimate=context.token_estimate,
            caps=context.caps,
        ))
        await trace_db.commit()
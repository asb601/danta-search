"""Read-only bounded BrainContext resolver."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session
from app.models.semantic_memory import (
    BrainContextTrace,
    SemanticDomainCluster,
    SemanticDomainFileIndex,
    SemanticDomainTermIndex,
    SemanticMemoryRecord,
    SemanticMemoryTermIndex,
)
from app.services.brain_context import BrainContext, BrainDomainBrief, BrainMemoryBrief, ExecutionEnvelope, RetrievalGuidance
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
        domain_candidates, domain_file_scores = await self._candidate_domains(container_id, authorized_file_ids, query_terms)
        ranked = self._rank_records(candidates, query_terms, intent_plan)
        ranked_domains = self._rank_domains(domain_candidates, domain_file_scores, query_terms, intent_plan)
        max_records = max(1, int(self._settings.BRAIN_CONTEXT_MAX_RECORDS))
        max_domains = max(1, int(self._settings.BRAIN_CONTEXT_MAX_DOMAINS))
        selected = ranked[:max_records]
        selected_domains = ranked_domains[:max_domains]
        briefs = [self._brief(record, score) for record, score in selected]
        domain_briefs = [self._domain_brief(domain, score, domain_file_scores.get(domain.id, {})) for domain, score in selected_domains]
        guidance = self._guidance(briefs, domain_briefs, ranked, ranked_domains, query_terms)
        envelope = ExecutionEnvelope(
            memory_ids=[brief.id for brief in briefs],
            domain_ids=[domain.id for domain in domain_briefs],
            anchor_file_ids=list(dict.fromkeys(guidance.anchor_file_ids + guidance.domain_anchor_file_ids)),
            ambiguity_flags=guidance.ambiguity_flags,
            authority_floor=min(
                [brief.authority_score for brief in briefs] + [domain.authority_score for domain in domain_briefs],
                default=0.0,
            ),
        )
        return BrainContext(
            records=briefs,
            domains=domain_briefs,
            retrieval_guidance=guidance,
            execution_envelope=envelope,
            token_estimate=self._estimate_tokens(briefs, domain_briefs),
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

    async def _candidate_domains(
        self,
        container_id: str,
        authorized_file_ids: set[str],
        query_terms: list[str],
    ) -> tuple[list[SemanticDomainCluster], dict[str, dict[str, float]]]:
        max_candidates = max(5, int(self._settings.BRAIN_CONTEXT_MAX_CANDIDATES))
        domain_ids: list[str] = []
        if query_terms:
            term_rows = (
                await self._db.execute(
                    select(SemanticDomainTermIndex.domain_id)
                    .where(
                        SemanticDomainTermIndex.container_id == container_id,
                        SemanticDomainTermIndex.status == "active",
                        SemanticDomainTermIndex.term.in_(query_terms[:60]),
                    )
                    .limit(max_candidates * 3)
                )
            ).all()
            domain_ids = list(dict.fromkeys(row.domain_id for row in term_rows))[:max_candidates]

        file_index_stmt = select(SemanticDomainFileIndex).where(
            SemanticDomainFileIndex.container_id == container_id,
            SemanticDomainFileIndex.file_id.in_(authorized_file_ids),
        )
        if domain_ids:
            file_index_stmt = file_index_stmt.where(SemanticDomainFileIndex.domain_id.in_(domain_ids))
        file_rows = (
            await self._db.execute(
                file_index_stmt.order_by(SemanticDomainFileIndex.score.desc()).limit(max_candidates * 4)
            )
        ).scalars().all()
        scoped_domain_ids = list(dict.fromkeys(row.domain_id for row in file_rows))[:max_candidates]
        if not scoped_domain_ids:
            return [], {}

        domain_rows = (
            await self._db.execute(
                select(SemanticDomainCluster)
                .where(
                    SemanticDomainCluster.id.in_(scoped_domain_ids),
                    SemanticDomainCluster.container_id == container_id,
                    SemanticDomainCluster.status == "active",
                    SemanticDomainCluster.governance_status.in_(["active", "candidate"]),
                )
            )
        ).scalars().all()
        domain_file_scores: dict[str, dict[str, float]] = {}
        for row in file_rows:
            if row.domain_id in scoped_domain_ids:
                domain_file_scores.setdefault(row.domain_id, {})[row.file_id] = max(
                    domain_file_scores.get(row.domain_id, {}).get(row.file_id, 0.0),
                    float(row.score or 0.0),
                )
        return list(domain_rows), domain_file_scores

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

    def _rank_domains(
        self,
        domains: list[SemanticDomainCluster],
        domain_file_scores: dict[str, dict[str, float]],
        query_terms: list[str],
        intent_plan: Any,
    ) -> list[tuple[SemanticDomainCluster, float]]:
        q = set(query_terms)
        behaviors = set(terms_from_text(getattr(intent_plan, "behaviors", []) or []))
        ranked: list[tuple[SemanticDomainCluster, float]] = []
        min_score = float(self._settings.BRAIN_CONTEXT_MIN_SCORE)
        for domain in domains:
            terms = set(terms_from_text(domain.normalized_terms, domain.title, domain.summary))
            overlap = len(q & terms) / max(1, min(len(q), 10)) if q else 0.0
            workflow_overlap = len(q & set(terms_from_text(domain.workflow_terms or []))) / max(1, min(len(q), 10)) if q else 0.0
            lifecycle_overlap = len(q & set(terms_from_text(domain.lifecycle_terms or []))) / max(1, min(len(q), 10)) if q else 0.0
            kpi_overlap = len(q & set(terms_from_text(domain.kpi_terms or []))) / max(1, min(len(q), 10)) if q else 0.0
            behavior_bonus = 0.0
            if "aggregation" in behaviors and domain.domain_type == "kpi_domain":
                behavior_bonus += 0.1
            if {"open_items", "time_filtered", "multi_step"} & behaviors and domain.domain_type in {"workflow_domain", "lifecycle_domain"}:
                behavior_bonus += 0.1
            file_score = max(domain_file_scores.get(domain.id, {}).values(), default=0.0)
            score = (
                overlap * 0.26
                + workflow_overlap * 0.16
                + lifecycle_overlap * 0.14
                + kpi_overlap * 0.12
                + float(domain.authority_score or 0.0) * 0.16
                + float(domain.confidence_score or 0.0) * 0.08
                + file_score * 0.08
                + behavior_bonus
                - float(domain.drift_score or 0.0) * 0.2
            )
            if score >= min_score or domain.governance_status == "active":
                ranked.append((domain, round(max(0.0, score), 4)))
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

    def _domain_brief(self, domain: SemanticDomainCluster, score: float, file_scores: dict[str, float]) -> BrainDomainBrief:
        return BrainDomainBrief(
            id=domain.id,
            domain_type=domain.domain_type,
            domain_key=domain.domain_key,
            title=domain.title,
            terms=terms_from_text(domain.normalized_terms, domain.title)[:12],
            workflow_terms=terms_from_text(domain.workflow_terms)[:8],
            lifecycle_terms=terms_from_text(domain.lifecycle_terms)[:8],
            kpi_terms=terms_from_text(domain.kpi_terms)[:8],
            synonym_terms=terms_from_text(domain.synonym_terms)[:8],
            contributor_file_ids=list(domain.contributor_file_ids or [])[:20],
            confidence_score=float(domain.confidence_score or 0.0),
            authority_score=float(domain.authority_score or 0.0),
            drift_score=float(domain.drift_score or 0.0),
            conflict_count=int(domain.conflict_count or 0),
            score=score,
            file_scores={file_id: round(value, 3) for file_id, value in file_scores.items()},
        )

    def _guidance(
        self,
        briefs: list[BrainMemoryBrief],
        domain_briefs: list[BrainDomainBrief],
        ranked: list[tuple[SemanticMemoryRecord, float]],
        ranked_domains: list[tuple[SemanticDomainCluster, float]],
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

        domain_authority_by_file: dict[str, float] = {}
        for domain in domain_briefs:
            for file_id, file_score in domain.file_scores.items():
                domain_authority_by_file[file_id] = max(
                    domain_authority_by_file.get(file_id, 0.0),
                    file_score * max(domain.authority_score, domain.confidence_score),
                )
        domain_anchors = [
            file_id
            for file_id, _score in sorted(domain_authority_by_file.items(), key=lambda item: item[1], reverse=True)
        ][:max_anchor_files]

        term_counts = Counter(term for brief in briefs for term in brief.terms if term in query_terms or len(term) > 3)
        domain_term_counts = Counter(term for domain in domain_briefs for term in domain.terms if term in query_terms or len(term) > 3)
        workflow_terms = list(dict.fromkeys(term for domain in domain_briefs for term in domain.workflow_terms))
        lifecycle_terms = list(dict.fromkeys(term for domain in domain_briefs for term in domain.lifecycle_terms))
        kpi_terms = list(dict.fromkeys(term for domain in domain_briefs for term in domain.kpi_terms))
        ambiguity_flags = self._ambiguity_flags(ranked) + self._domain_ambiguity_flags(ranked_domains)
        return RetrievalGuidance(
            anchor_file_ids=anchors,
            domain_anchor_file_ids=domain_anchors,
            preferred_terms=[term for term, _ in term_counts.most_common(max(1, int(self._settings.BRAIN_CONTEXT_MAX_TERMS)))],
            domain_terms=[term for term, _ in domain_term_counts.most_common(max(1, int(self._settings.BRAIN_CONTEXT_MAX_TERMS)))],
            workflow_terms=workflow_terms[:12],
            lifecycle_terms=lifecycle_terms[:12],
            kpi_terms=kpi_terms[:12],
            authority_by_file_id={file_id: round(score, 3) for file_id, score in authority_by_file.items()},
            domain_authority_by_file_id={file_id: round(score, 3) for file_id, score in domain_authority_by_file.items()},
            ambiguity_flags=ambiguity_flags[:8],
            topology_hints=self._topology_hints(domain_briefs),
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

    def _domain_ambiguity_flags(self, ranked_domains: list[tuple[SemanticDomainCluster, float]]) -> list[str]:
        flags: list[str] = []
        for domain, score in ranked_domains[:10]:
            if int(domain.conflict_count or 0) > 0:
                flags.append(f"domain_conflict:{domain.domain_type}:{domain.domain_key.split(':')[-1]}:{domain.conflict_count}")
            if float(domain.drift_score or 0.0) >= 0.2:
                flags.append(f"domain_drift:{domain.domain_type}:{domain.domain_key.split(':')[-1]}:{round(float(domain.drift_score or 0.0), 2)}")
            if score < 0.2 and domain.governance_status != "active":
                flags.append(f"weak_domain_match:{domain.domain_type}:{domain.domain_key.split(':')[-1]}")
            if len(flags) >= 5:
                break
        return flags

    def _topology_hints(self, domains: list[BrainDomainBrief]) -> list[str]:
        hints: list[str] = []
        for domain in domains[:6]:
            if domain.domain_type not in {"workflow_domain", "lifecycle_domain", "business_capability"}:
                continue
            term = (domain.workflow_terms or domain.lifecycle_terms or domain.terms or [domain.title])[0]
            hints.append(f"{domain.domain_type}:{term}:files={len(domain.contributor_file_ids)}")
        return hints[:6]

    def _estimate_tokens(self, briefs: list[BrainMemoryBrief], domains: list[BrainDomainBrief]) -> int:
        char_count = sum(len(brief.title) + len(brief.summary or "") + 10 * len(brief.terms) for brief in briefs)
        char_count += sum(len(domain.title) + 10 * len(domain.terms) + 8 * len(domain.workflow_terms + domain.lifecycle_terms + domain.kpi_terms) for domain in domains)
        return max(0, int(char_count / 4))

    def _caps(self) -> dict[str, int]:
        return {
            "max_records": int(self._settings.BRAIN_CONTEXT_MAX_RECORDS),
            "max_domains": int(self._settings.BRAIN_CONTEXT_MAX_DOMAINS),
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
            selected_domain_ids=[domain.id for domain in context.domains],
            ambiguity_flags=list(context.retrieval_guidance.ambiguity_flags),
            retrieval_guidance=context.retrieval_guidance.to_dict(),
            execution_envelope=context.execution_envelope.to_dict(),
            token_estimate=context.token_estimate,
            caps=context.caps,
        ))
        await trace_db.commit()
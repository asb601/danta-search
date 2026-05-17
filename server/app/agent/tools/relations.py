"""Relationship graph tool for the LangGraph agent."""
from __future__ import annotations

import json

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_relationship import FileRelationship
from app.models.semantic_layer import SemanticRelationship


def build_relations_tool(db: AsyncSession, catalog: list[dict]) -> list:
    """Return extract_relations tool bound to the current tenant catalog."""
    by_file_id = {f.get("file_id"): f for f in catalog if f.get("file_id")}
    by_blob = {f.get("blob_path"): f for f in catalog if f.get("blob_path")}
    allowed_file_ids = set(by_file_id.keys())

    def _context(file_id: str | None) -> dict:
        entry = by_file_id.get(file_id or "") or {}
        return {
            "blob_path": entry.get("blob_path"),
            "description": (entry.get("ai_description") or "")[:240],
            "good_for": (entry.get("good_for") or [])[:3],
            "key_metrics": (entry.get("key_metrics") or [])[:6],
            "key_dimensions": (entry.get("key_dimensions") or [])[:6],
        }

    @tool
    async def extract_relations(file_paths_csv: str = "") -> str:
        """Return semantic join relationships for visible files.

        Input is optional comma-separated blob paths. If omitted, returns the
        strongest tenant-visible relationships from the current catalog.
        Use approved relationships for SQL joins. Candidate/technical-only
        relationships are evidence, not permission to join without validation.
        """
        selected_ids: set[str] = set()
        for raw in (file_paths_csv or "").split(","):
            key = raw.strip()
            if not key:
                continue
            entry = by_blob.get(key)
            if entry and entry.get("file_id"):
                selected_ids.add(entry["file_id"])

        scope_ids = selected_ids or allowed_file_ids
        if not scope_ids:
            return json.dumps({"relations": [], "context": {}})

        semantic_q = (
            select(SemanticRelationship)
            .where(
                SemanticRelationship.file_a_id.in_(scope_ids)
                | SemanticRelationship.file_b_id.in_(scope_ids)
            )
            .where(SemanticRelationship.status == "active")
            .order_by(SemanticRelationship.confidence_score.desc())
            .limit(50)
        )
        semantic_rows = (await db.execute(semantic_q)).scalars().all()

        relations = []
        related_ids: set[str] = set()

        for rel in sorted(
            semantic_rows,
            key=lambda item: (item.approval_status != "approved", -(item.confidence_score or 0.0)),
        ):
            if rel.file_a_id not in allowed_file_ids or rel.file_b_id not in allowed_file_ids:
                continue
            related_ids.add(rel.file_a_id)
            related_ids.add(rel.file_b_id)
            a = by_file_id.get(rel.file_a_id, {})
            b = by_file_id.get(rel.file_b_id, {})
            join_rule = rel.join_rule or {}
            relations.append({
                "file_a": a.get("blob_path") or rel.file_a_id,
                "file_b": b.get("blob_path") or rel.file_b_id,
                "join_on": {
                    "file_a_col": rel.from_column,
                    "file_b_col": rel.to_column,
                },
                "semantic_role": join_rule.get("semantic_role"),
                "relationship_type": rel.relationship_type,
                "approval_status": rel.approval_status,
                "risk_reason": rel.risk_reason,
                "confidence": round(rel.confidence_score or 0.0, 4),
                "value_overlap_pct": join_rule.get("value_overlap_pct"),
                "join_type": join_rule.get("join_type"),
                "evidence": "semantic_layer",
            })

        if not relations:
            raw_q = (
                select(FileRelationship)
                .where(
                    FileRelationship.file_a_id.in_(scope_ids)
                    | FileRelationship.file_b_id.in_(scope_ids)
                )
                .order_by(FileRelationship.confidence_score.desc())
                .limit(50)
            )
            raw_rows = (await db.execute(raw_q)).scalars().all()
            for rel in raw_rows:
                if rel.file_a_id not in allowed_file_ids or rel.file_b_id not in allowed_file_ids:
                    continue
                related_ids.add(rel.file_a_id)
                related_ids.add(rel.file_b_id)
                a = by_file_id.get(rel.file_a_id, {})
                b = by_file_id.get(rel.file_b_id, {})
                relations.append({
                    "file_a": a.get("blob_path") or rel.file_a_path,
                    "file_b": b.get("blob_path") or rel.file_b_path,
                    "join_on": {
                        "file_a_col": rel.shared_column,
                        "file_b_col": rel.related_column or rel.shared_column,
                    },
                    "semantic_role": rel.semantic_role,
                    "relationship_type": None,
                    "approval_status": "technical_candidate",
                    "risk_reason": "semantic layer approval is not available for this relationship",
                    "confidence": round(rel.confidence_score or 0.0, 4),
                    "value_overlap_pct": (
                        round(rel.value_overlap_pct, 4)
                        if rel.value_overlap_pct is not None else None
                    ),
                    "join_type": rel.join_type,
                    "evidence": rel.role_source or "relationship_graph",
                })

        context = {
            (by_file_id[file_id].get("blob_path") or file_id): _context(file_id)
            for file_id in sorted(related_ids)
            if file_id in by_file_id
        }
        return json.dumps({"relations": relations, "context": context}, default=str)

    return [extract_relations]

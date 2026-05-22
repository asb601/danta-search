"""Relationship graph tool for the LangGraph agent."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_relationship import FileRelationship
from app.models.semantic_layer import SemanticRelationship
from app.services.relationship_index import is_dictionary_like_path
from app.services.semantic_policy import get_semantic_policy


def build_relations_tool(db: AsyncSession, catalog: list[dict]) -> list:
    """Return extract_relations tool bound to the current visible catalog."""
    by_file_id = {f.get("file_id"): f for f in catalog if f.get("file_id")}
    by_blob = {f.get("blob_path"): f for f in catalog if f.get("blob_path")}
    allowed_file_ids = {
        file_id
        for file_id, entry in by_file_id.items()
        if not is_dictionary_like_path(entry.get("blob_path") or entry.get("name"))
    }
    policy = get_semantic_policy()

    def _context(file_id: str | None) -> dict:
        entry = by_file_id.get(file_id or "") or {}
        return {
            "blob_path": entry.get("blob_path"),
            "description": (entry.get("ai_description") or "")[:240],
            "good_for": (entry.get("good_for") or [])[:3],
            "key_metrics": (entry.get("key_metrics") or [])[:6],
            "key_dimensions": (entry.get("key_dimensions") or [])[:6],
        }

    def _blob(file_id: str | None) -> str | None:
        entry = by_file_id.get(file_id or "") or {}
        return entry.get("blob_path") or file_id

    def _selected_ids(file_paths_csv: str) -> set[str]:
        selected: set[str] = set()
        for raw in (file_paths_csv or "").split(","):
            key = raw.strip()
            if not key:
                continue
            entry = by_blob.get(key)
            if entry and entry.get("file_id") in allowed_file_ids:
                selected.add(entry["file_id"])
        return selected

    def _hops(max_hops: int | str | None) -> int:
        try:
            requested = int(max_hops or 1)
        except (TypeError, ValueError):
            requested = 1
        return max(1, min(requested, policy.relation_max_hops))

    def _visible_edge(file_a_id: str | None, file_b_id: str | None) -> bool:
        return bool(file_a_id in allowed_file_ids and file_b_id in allowed_file_ids)

    def _semantic_payload(rel: SemanticRelationship) -> dict[str, Any]:
        join_rule = rel.join_rule or {}
        return {
            "file_a_id": rel.file_a_id,
            "file_b_id": rel.file_b_id,
            "file_a": _blob(rel.file_a_id),
            "file_b": _blob(rel.file_b_id),
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
        }

    def _raw_payload(rel: FileRelationship) -> dict[str, Any]:
        return {
            "file_a_id": rel.file_a_id,
            "file_b_id": rel.file_b_id,
            "file_a": _blob(rel.file_a_id) or rel.file_a_path,
            "file_b": _blob(rel.file_b_id) or rel.file_b_path,
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
        }

    def _edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
        join_on = edge.get("join_on") or {}
        return (
            edge.get("file_a_id"),
            edge.get("file_b_id"),
            join_on.get("file_a_col"),
            join_on.get("file_b_col"),
        )

    def _public_edge(edge: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in edge.items()
            if key not in {"file_a_id", "file_b_id", "_path_from_id", "_path_to_id"}
        }
        if edge.get("_path_from_id"):
            public["path_from"] = _blob(edge.get("_path_from_id"))
        if edge.get("_path_to_id"):
            public["path_to"] = _blob(edge.get("_path_to_id"))
        return public

    async def _direct_relations(selected_ids: set[str]) -> str:
        scope_ids = selected_ids or allowed_file_ids
        if not scope_ids:
            return json.dumps({"relations": [], "paths": [], "context": {}})

        def _scope_filter(model):
            if not selected_ids:
                return model.file_a_id.in_(allowed_file_ids) | model.file_b_id.in_(allowed_file_ids)
            if len(selected_ids) == 1:
                return model.file_a_id.in_(selected_ids) | model.file_b_id.in_(selected_ids)
            return model.file_a_id.in_(selected_ids) & model.file_b_id.in_(selected_ids)

        semantic_rows = (
            await db.execute(
                select(SemanticRelationship)
                .where(_scope_filter(SemanticRelationship))
                .where(SemanticRelationship.status == "active")
                .order_by(SemanticRelationship.confidence_score.desc())
                .limit(policy.relation_direct_limit)
            )
        ).scalars().all()

        edges: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        related_ids: set[str] = set()

        for rel in sorted(
            semantic_rows,
            key=lambda item: (item.approval_status != "approved", -(item.confidence_score or 0.0)),
        ):
            if not _visible_edge(rel.file_a_id, rel.file_b_id):
                continue
            edge = _semantic_payload(rel)
            seen.add(_edge_key(edge))
            edges.append(edge)
            related_ids.update({rel.file_a_id, rel.file_b_id})

        if not edges:
            raw_rows = (
                await db.execute(
                    select(FileRelationship)
                    .where(_scope_filter(FileRelationship))
                    .order_by(FileRelationship.confidence_score.desc())
                    .limit(policy.relation_direct_limit)
                )
            ).scalars().all()
            for rel in raw_rows:
                if not _visible_edge(rel.file_a_id, rel.file_b_id):
                    continue
                edge = _raw_payload(rel)
                if _edge_key(edge) in seen:
                    continue
                seen.add(_edge_key(edge))
                edges.append(edge)
                related_ids.update({rel.file_a_id, rel.file_b_id})

        context = {
            (_blob(file_id) or file_id): _context(file_id)
            for file_id in sorted(related_ids | selected_ids)
            if file_id in by_file_id
        }
        return json.dumps({
            "relations": [_public_edge(edge) for edge in edges],
            "paths": [],
            "context": context,
        }, default=str)

    async def _neighbor_edges(frontier_ids: set[str]) -> list[dict[str, Any]]:
        if not frontier_ids:
            return []

        semantic_rows = (
            await db.execute(
                select(SemanticRelationship)
                .where(SemanticRelationship.status == "active")
                .where(
                    SemanticRelationship.file_a_id.in_(frontier_ids)
                    | SemanticRelationship.file_b_id.in_(frontier_ids)
                )
                .where(SemanticRelationship.confidence_score >= policy.planner_join_min_confidence)
                .order_by(SemanticRelationship.confidence_score.desc())
                .limit(policy.relation_expand_edge_limit)
            )
        ).scalars().all()

        edges: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for rel in semantic_rows:
            if not _visible_edge(rel.file_a_id, rel.file_b_id):
                continue
            edge = _semantic_payload(rel)
            seen.add(_edge_key(edge))
            edges.append(edge)

        if len(edges) >= policy.relation_expand_edge_limit:
            return edges

        raw_rows = (
            await db.execute(
                select(FileRelationship)
                .where(
                    FileRelationship.file_a_id.in_(frontier_ids)
                    | FileRelationship.file_b_id.in_(frontier_ids)
                )
                .where(FileRelationship.confidence_score >= policy.planner_join_min_confidence)
                .order_by(FileRelationship.confidence_score.desc())
                .limit(policy.relation_expand_edge_limit - len(edges))
            )
        ).scalars().all()
        for rel in raw_rows:
            if not _visible_edge(rel.file_a_id, rel.file_b_id):
                continue
            edge = _raw_payload(rel)
            if _edge_key(edge) in seen:
                continue
            seen.add(_edge_key(edge))
            edges.append(edge)
        return edges

    async def _multi_hop_paths(selected_ids: set[str], max_hops: int) -> str:
        if len(selected_ids) < 2:
            return await _direct_relations(selected_ids)

        paths: list[dict[str, Any]] = []
        flattened: dict[tuple[Any, ...], dict[str, Any]] = {}
        path_pairs: set[tuple[str, str]] = set()
        related_ids: set[str] = set(selected_ids)

        for start_id in sorted(selected_ids):
            if len(paths) >= policy.relation_max_paths:
                break
            frontier: dict[str, list[dict[str, Any]]] = {start_id: []}
            visited: set[str] = {start_id}

            for _depth in range(max_hops):
                if not frontier or len(paths) >= policy.relation_max_paths:
                    break
                edges = await _neighbor_edges(set(frontier.keys()))
                next_frontier: dict[str, list[dict[str, Any]]] = {}

                for edge in edges:
                    file_a_id = edge.get("file_a_id")
                    file_b_id = edge.get("file_b_id")
                    for current_id, next_id in ((file_a_id, file_b_id), (file_b_id, file_a_id)):
                        if current_id not in frontier or next_id in visited or next_id not in allowed_file_ids:
                            continue
                        step = dict(edge)
                        step["_path_from_id"] = current_id
                        step["_path_to_id"] = next_id
                        candidate_path = frontier[current_id] + [step]
                        related_ids.add(next_id)

                        if next_id in selected_ids and next_id != start_id:
                            pair = tuple(sorted((start_id, next_id)))
                            if pair not in path_pairs:
                                path_pairs.add(pair)
                                for path_edge in candidate_path:
                                    flattened.setdefault(_edge_key(path_edge), path_edge)
                                paths.append({
                                    "from": _blob(start_id),
                                    "to": _blob(next_id),
                                    "hops": len(candidate_path),
                                    "confidence": round(
                                        min(path_edge.get("confidence") or 0.0 for path_edge in candidate_path),
                                        4,
                                    ),
                                    "relations": [_public_edge(path_edge) for path_edge in candidate_path],
                                })
                                if len(paths) >= policy.relation_max_paths:
                                    break
                        next_frontier.setdefault(next_id, candidate_path)
                    if len(paths) >= policy.relation_max_paths:
                        break

                visited.update(next_frontier.keys())
                frontier = next_frontier

        context = {
            (_blob(file_id) or file_id): _context(file_id)
            for file_id in sorted(related_ids)
            if file_id in by_file_id
        }
        return json.dumps({
            "relations": [_public_edge(edge) for edge in flattened.values()],
            "paths": paths,
            "context": context,
        }, default=str)

    @tool
    async def extract_relations(file_paths_csv: str = "", max_hops: int = 1) -> str:
        """Return scoped join relationships for visible files.

        Call only when a SQL answer needs more than one file. Pass only the
        blob paths already selected as necessary for the query. max_hops=1
        returns direct joins; set max_hops above 1 only when a selected file set
        needs visible intermediate files to connect. The tool never returns
        relationships outside the current catalog access scope.
        """
        try:
            selected_ids = _selected_ids(file_paths_csv)
            requested_hops = _hops(max_hops)
            if requested_hops > 1 and len(selected_ids) >= 2:
                return await _multi_hop_paths(selected_ids, requested_hops)
            return await _direct_relations(selected_ids)
        except Exception as _exc:  # noqa: BLE001
            return json.dumps({"relations": [], "paths": [], "context": {}, "error": str(_exc)})

    return [extract_relations]

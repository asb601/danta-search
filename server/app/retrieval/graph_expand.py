"""Retrieval graph expansion through approved semantic relationships.

Expansion is bounded by four independent caps (Phase 7 adds two):
  _MAX_SEED_IDS           — seed list is trimmed before the DB edge query.
  _MAX_NEIGHBORS_PER_NODE — each seed file contributes at most N neighbors
                              (highest-confidence neighbors win via ORDER BY).
  _EXPANSION_CONF_FLOOR   — edges weaker than this are never traversed,
                              regardless of the semantic policy floor.
  max_neighbor_influence_ratio — Phase 7: a single seed may contribute at
                              most ceil(ratio × limit) entries to the final
                              result list, preventing hub dominance.

Phase 7 supernode mitigation:
  Nodes whose degree (within the fetched edge set) exceeds
  supernode_degree_threshold receive a confidence penalty on all their edges.
  Edges are not eliminated — they rank lower, ensuring diverse coverage.
"""
from __future__ import annotations

import math

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticRelationship
from app.retrieval.filters import build_base_query
from app.services.semantic_policy import get_semantic_policy
from app.policies.graph_policy import get_graph_policy as _get_graph_policy
from app.services.trust_propagation import effective_edge_confidence as _eff_conf
from app.core.logger import chat_logger

# ── Expansion bounds ──────────────────────────────────────────────────────────
# Governed by GraphPolicy. Module-level aliases kept for test monkeypatching.
# See server/app/policies/graph_policy.py for rationale on each value.
_gp = _get_graph_policy()
_MAX_SEED_IDS           = _gp.max_seed_ids
_MAX_NEIGHBORS_PER_NODE = _gp.max_neighbors_per_node
_EXPANSION_CONF_FLOOR   = _gp.expansion_conf_floor


async def graph_expand(
    seed_file_ids: list[str],
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    min_confidence: float | None = None,
    limit: int = 20,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
) -> list[tuple[FileMetadata, float]]:
    """Expand seed files through approved semantic relationships only.

    Raw technical relationships are candidates. They do not enter graph
    expansion until the semantic layer approves them.
    """
    if not seed_file_ids:
        return []

    # Trim seed list to avoid O(n²) edge queries on very large catalogs.
    bounded_seed_ids = seed_file_ids[:_MAX_SEED_IDS]

    policy = get_semantic_policy()
    # Apply the stricter of the policy floor and the hard expansion floor.
    effective_min_conf = max(
        policy.graph_expand_min_confidence if min_confidence is None else min_confidence,
        _EXPANSION_CONF_FLOOR,
    )
    seed_set = set(bounded_seed_ids)

    edge_q = (
        select(
            SemanticRelationship.file_a_id,
            SemanticRelationship.file_b_id,
            SemanticRelationship.confidence_score,
        )
        .where(
            or_(
                SemanticRelationship.file_a_id.in_(bounded_seed_ids),
                SemanticRelationship.file_b_id.in_(bounded_seed_ids),
            ),
            # The semantic-layer builder writes status="active" (lifecycle) and
            # approval_status="approved" (trust). Filtering status=="approved"
            # matched NOTHING, so one-hop expansion never fired. Match "active".
            SemanticRelationship.status == "active",
            SemanticRelationship.approval_status == "approved",
            SemanticRelationship.confidence_score >= effective_min_conf,
        )
        .order_by(SemanticRelationship.confidence_score.desc())
    )
    if container_id:
        edge_q = edge_q.where(SemanticRelationship.container_id == container_id)

    edge_rows = (await db.execute(edge_q)).all()
    if not edge_rows:
        return []

    # ── Phase 7: Supernode detection ─────────────────────────────────────────
    # Count node degree within the fetched edge set.  Nodes appearing in more
    # than supernode_degree_threshold edges are supernodes (hub tables, master
    # dimensions).  Their edges receive a confidence penalty so they rank lower
    # without being excluded from expansion entirely.
    _sup_threshold = _gp.supernode_degree_threshold
    _sup_penalty   = _gp.supernode_confidence_penalty
    _node_degree: dict[str, int] = {}
    for _fa, _fb, _ in edge_rows:
        _node_degree[_fa] = _node_degree.get(_fa, 0) + 1
        _node_degree[_fb] = _node_degree.get(_fb, 0) + 1

    # Phase 8: identify supernodes once for both the edge loop and the
    # influence cap.  Building the set here avoids repeated degree lookups
    # inside the hot loop and makes the telemetry emit trivial.
    _supernode_ids: set[str] = {
        nid for nid, deg in _node_degree.items() if deg > _sup_threshold
    }
    _pen_edge_count = 0  # edges that received the supernode confidence penalty

    # ── Phase 6: Load ingestion confidence for seed files ─────────────────────
    # One bounded query (file_id IN [...]) — at most _MAX_SEED_IDS=20 rows.
    # Never blocks expansion: any DB error yields an empty dict (neutral scores).
    try:
        _ing_stmt = (
            select(FileMetadata.file_id, FileMetadata.ingestion_confidence_score)
            .where(FileMetadata.file_id.in_(bounded_seed_ids))
        )
        seed_ing_scores: dict[str, float | None] = {
            r.file_id: r.ingestion_confidence_score
            for r in (await db.execute(_ing_stmt)).all()
        }
    except Exception:
        seed_ing_scores = {}  # neutral fallback — never block expansion

    # Build neighbor → confidence map, capping at _MAX_NEIGHBORS_PER_NODE
    # per seed.  Edges are already ordered by confidence DESC so the first
    # encounter of a neighbor from a given seed is its best confidence.
    # Also track which seed produced the best edge (needed for trust formula).
    neighbor_score: dict[str, float] = {}
    neighbor_best_seed: dict[str, str] = {}  # neighbour_id → best-edge seed_id
    neighbors_per_seed: dict[str, int] = {}  # seed_id → count already emitted

    for file_a_id, file_b_id, confidence in edge_rows:
        seed_id      = file_a_id if file_a_id in seed_set else file_b_id
        neighbour_id = file_b_id if file_a_id in seed_set else file_a_id
        if neighbour_id in seed_set:
            continue
        # Enforce per-seed neighbor cap
        if neighbors_per_seed.get(seed_id, 0) >= _MAX_NEIGHBORS_PER_NODE:
            continue
        neighbors_per_seed[seed_id] = neighbors_per_seed.get(seed_id, 0) + 1
        # Phase 7: Supernode penalty — damp confidence when either endpoint is
        # a high-degree hub.  Multiply BEFORE updating neighbor_score so the
        # dampened value wins even if this seed path has a higher raw confidence.
        _is_supernode = (
            neighbour_id in _supernode_ids or seed_id in _supernode_ids
        )
        if _is_supernode:
            _pen_conf = confidence * (1.0 - _sup_penalty)
            _pen_edge_count += 1
        else:
            _pen_conf = confidence
        # Keep highest penalized confidence across all seed paths to the same neighbor
        previous = neighbor_score.get(neighbour_id)
        if previous is None or _pen_conf > previous:
            neighbor_score[neighbour_id] = _pen_conf
            neighbor_best_seed[neighbour_id] = seed_id

    if not neighbor_score:
        return []

    meta_q = (
        build_base_query(
            user_id=user_id,
            is_admin=is_admin,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
        .where(FileMetadata.file_id.in_(list(neighbor_score.keys())))
    )
    meta_rows = (await db.execute(meta_q)).scalars().all()

    # ── Phase 6: Apply effective_edge_confidence ────────────────────────────
    # Recompute each neighbor's score using the trust formula:
    #   effective = rel_conf × ingestion_modifier × health_modifier
    # ingestion_modifier uses the weaker of the two endpoints so weak-ingestion
    # regions naturally produce lower traversal scores.
    # health_level defaults to "good" (graph health isn’t known at retrieval time;
    # the global health penalty is captured separately in ConfidenceScore.health_component).
    results = []
    for meta in meta_rows:
        nid = meta.file_id
        raw_conf = neighbor_score[nid]
        seed_id = neighbor_best_seed.get(nid)
        seed_ing = seed_ing_scores.get(seed_id) if seed_id else None
        nbr_ing = meta.ingestion_confidence_score
        effective = _eff_conf(raw_conf, seed_ing, nbr_ing)
        results.append((meta, effective))

    results.sort(key=lambda item: item[1], reverse=True)

    # ── Phase 7: Neighborhood influence cap ──────────────────────────────────
    # After sorting by effective confidence, ensure no single seed file
    # contributes more than ceil(max_neighbor_influence_ratio × limit) entries
    # to the final shortlist.  Prevents a highly-connected hub in seed position
    # from consuming all expansion slots and displacing diverse neighbors.
    #
    # Phase 8: Supernode minimum participation guarantee.
    # A seed classified as a supernode receives max(cap, min_slots) rather than
    # bare cap, ensuring master-table coverage is never fully suppressed by an
    # aggressive ratio setting.  At default values (0.40 × 20 = 8 ≥ 2) the
    # floor is inactive; it activates only if ratio or limit is reduced.
    _max_per_seed   = math.ceil(_gp.max_neighbor_influence_ratio * limit)
    _min_slots      = _gp.min_supernode_participation_slots
    _seed_contrib: dict[str, int] = {}
    _capped: list[tuple[FileMetadata, float]] = []
    for _meta, _eff in results:
        _sid = neighbor_best_seed.get(_meta.file_id)
        _cnt = _seed_contrib.get(_sid, 0)
        _seed_cap = (
            max(_max_per_seed, _min_slots)
            if (_sid and _sid in _supernode_ids)
            else _max_per_seed
        )
        if _cnt < _seed_cap:
            _capped.append((_meta, _eff))
            _seed_contrib[_sid] = _cnt + 1
        if len(_capped) >= limit:
            break

    # ── Phase 8: Supernode mitigation telemetry ───────────────────────────────
    # Emit a lightweight info event when supernodes were detected so operators
    # can monitor hub-file influence over time without full trace overhead.
    # Non-blocking: wrapped in try/except like all telemetry paths.
    try:
        if _supernode_ids:
            chat_logger.info(
                "graph_expand_supernode_mitigation",
                supernode_count=len(_supernode_ids),
                penalized_edge_count=_pen_edge_count,
                total_edges=len(edge_rows),
                seed_count=len(bounded_seed_ids),
            )
    except Exception:
        pass
    return _capped

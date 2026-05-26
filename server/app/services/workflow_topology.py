"""Workflow Topology Summary — multi-hop join reachability beyond the shortlist.

PURPOSE:
  Provide the planner with a summary of JOIN REACHABILITY that goes beyond the
  standard "both endpoints must be in shortlist" constraint of sql_context_builder.

  Current system: planner only sees approved joins where BOTH endpoints are shortlisted.
  This module adds: "reachable joins" — join paths available through non-shortlisted
  intermediate tables, and "orphaned files" — shortlisted files with no reachable joins.

WHAT THE PLANNER GAINS:
  1. Direct joins (both in shortlist) — already in sql_context, listed here for context
  2. Reachable joins (path via non-shortlisted table) — new visibility
  3. Isolated files — explicit signal that no approved path exists

  The planner can then call search_catalog for intermediate tables, or call
  extract_relations to find candidate paths before concluding files can't be joined.

DB QUERY STRATEGY:
  ONE additional query per request.
  Scope: all approved edges where at least ONE endpoint is in the shortlist.
  This gives: direct joins (both in shortlist) + bridge paths (one side external).

DESIGN CONSTRAINTS:
  - ONE DB query. Non-fatal: returns empty on any error.
  - Zero LLM calls.
  - Does NOT replace or modify the approved_joins list from sql_context_builder.
  - Only appends a supplementary prompt section.
  - _MAX_REACHABLE_PATHS = 8 cap prevents prompt overflow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger

_MAX_REACHABLE_PATHS = 8

# Slightly looser confidence floor than graph_policy.expansion_conf_floor (–0.10)
# to surface more topology context. Resolved at import time.
def _topology_conf_floor() -> float:
    try:
        from app.policies.graph_policy import get_graph_policy
        gp = get_graph_policy()
        return max(getattr(gp, "expansion_conf_floor", 0.60) - 0.10, 0.50)
    except Exception:
        return 0.50


_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)


def _table_name(blob_path: str) -> str:
    """Extract a clean display name from a blob_path."""
    name = blob_path.rsplit("/", 1)[-1]
    name = _HASH_PREFIX_RE.sub("", name)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class ReachablePath:
    """A join that is reachable (possibly via an intermediate table)."""
    source_table: str          # clean display name
    target_table: str          # clean display name
    via_table: str | None      # intermediate table name if multi-hop
    path_confidence: float     # min edge confidence along the path
    is_direct: bool            # True = both endpoints in shortlist
    missing_table_id: str | None  # partial file_id of missing intermediate table
    via_blob_path: str | None = None
    via_domain_labels: list[str] = field(default_factory=list)


@dataclass
class WorkflowTopology:
    """Multi-hop join reachability summary for the current shortlist."""
    direct_paths: list[ReachablePath]      # both endpoints in shortlist
    reachable_paths: list[ReachablePath]   # via non-shortlisted intermediate
    orphaned_tables: list[str]             # shortlisted tables with no approved edges
    topology_note: str                     # injected into system prompt

    def is_empty(self) -> bool:
        return not self.reachable_paths and not self.orphaned_tables


# ── Public API ─────────────────────────────────────────────────────────────────

async def build_workflow_topology(
    shortlist: list[dict],
    db: AsyncSession,
    full_catalog: list[dict] | None = None,
) -> WorkflowTopology:
    """
    Build the workflow topology summary for the current shortlist.

    Uses one DB query: all approved edges where at least one endpoint is shortlisted.
    Returns a WorkflowTopology with:
      - direct_paths: edges where both endpoints are in shortlist
      - reachable_paths: external node that bridges 2+ shortlisted files
      - orphaned_tables: shortlisted files with no approved edges at all
    """
    if len(shortlist) < 2:
        return WorkflowTopology([], [], [], "")

    file_ids: list[str] = [e["file_id"] for e in shortlist if e.get("file_id")]
    catalog_scope = list(shortlist)
    if full_catalog:
        seen = {e.get("file_id") for e in catalog_scope if e.get("file_id")}
        catalog_scope.extend(e for e in full_catalog if e.get("file_id") not in seen)

    id_to_name: dict[str, str] = {
        e["file_id"]: _table_name(e.get("blob_path", e.get("file_id", "")))
        for e in catalog_scope
        if e.get("file_id")
    }
    id_to_blob: dict[str, str] = {
        e["file_id"]: e.get("blob_path") or e.get("file_id")
        for e in catalog_scope
        if e.get("file_id")
    }
    id_to_roles: dict[str, list[str]] = {
        e["file_id"]: _role_labels(e)
        for e in catalog_scope
        if e.get("file_id")
    }
    shortlist_id_set = set(file_ids)
    conf_floor = _topology_conf_floor()

    try:
        from app.models.semantic_layer import SemanticRelationship

        edge_rows = (await db.execute(
            select(
                SemanticRelationship.file_a_id,
                SemanticRelationship.file_b_id,
                SemanticRelationship.confidence_score,
            )
            .where(
                or_(
                    SemanticRelationship.file_a_id.in_(file_ids),
                    SemanticRelationship.file_b_id.in_(file_ids),
                ),
                SemanticRelationship.status == "active",
                SemanticRelationship.approval_status == "approved",
                SemanticRelationship.confidence_score >= conf_floor,
            )
            .order_by(SemanticRelationship.confidence_score.desc())
        )).all()

    except Exception as exc:
        chat_logger.warning("workflow_topology_db_error", error=str(exc)[:200])
        return WorkflowTopology([], [], [], "")

    # Files that appear in at least one edge
    files_with_edges: set[str] = set()

    # Edges where BOTH endpoints are in shortlist
    direct_paths: list[ReachablePath] = []
    # External node → list of (shortlist_file_id, confidence)
    bridge_map: dict[str, list[tuple[str, float]]] = {}

    for row in edge_rows:
        a_in = row.file_a_id in shortlist_id_set
        b_in = row.file_b_id in shortlist_id_set

        if a_in:
            files_with_edges.add(row.file_a_id)
        if b_in:
            files_with_edges.add(row.file_b_id)

        if a_in and b_in:
            direct_paths.append(ReachablePath(
                source_table=id_to_name.get(row.file_a_id, row.file_a_id[:8]),
                target_table=id_to_name.get(row.file_b_id, row.file_b_id[:8]),
                via_table=None,
                path_confidence=round(row.confidence_score, 2),
                is_direct=True,
                missing_table_id=None,
            ))
        elif a_in and not b_in:
            bridge_map.setdefault(row.file_b_id, []).append(
                (row.file_a_id, row.confidence_score)
            )
        elif b_in and not a_in:
            bridge_map.setdefault(row.file_a_id, []).append(
                (row.file_b_id, row.confidence_score)
            )

    # Bridge paths: external node connected to 2+ shortlisted files
    reachable_paths: list[ReachablePath] = []
    for ext_id, connections in bridge_map.items():
        if len(connections) < 2:
            continue
        # This external table bridges at least 2 shortlisted tables
        min_conf = min(c for _, c in connections)
        shortlist_nodes = [sid for sid, _ in connections]

        # Enumerate pairs
        for i in range(len(shortlist_nodes)):
            for j in range(i + 1, len(shortlist_nodes)):
                if len(reachable_paths) >= _MAX_REACHABLE_PATHS:
                    break
                reachable_paths.append(ReachablePath(
                    source_table=id_to_name.get(shortlist_nodes[i], shortlist_nodes[i][:8]),
                    target_table=id_to_name.get(shortlist_nodes[j], shortlist_nodes[j][:8]),
                    via_table=f"{id_to_name.get(ext_id, ext_id[:8])} [not shortlisted]",
                    path_confidence=round(min_conf, 2),
                    is_direct=False,
                    missing_table_id=ext_id,
                    via_blob_path=id_to_blob.get(ext_id),
                    via_domain_labels=id_to_roles.get(ext_id, [])[:6],
                ))

    # Orphaned: shortlisted files with no approved edges at all
    orphaned_tables = [
        id_to_name.get(fid, fid[:8])
        for fid in file_ids
        if fid not in files_with_edges
    ]

    topology_note = _render_topology_note(reachable_paths, orphaned_tables)

    return WorkflowTopology(
        direct_paths=direct_paths,
        reachable_paths=reachable_paths,
        orphaned_tables=orphaned_tables,
        topology_note=topology_note,
    )


def _render_topology_note(
    reachable_paths: list[ReachablePath],
    orphaned_tables: list[str],
) -> str:
    """Render the topology summary as an injectable prompt section."""
    if not reachable_paths and not orphaned_tables:
        return ""

    lines = ["--- WORKFLOW TOPOLOGY ---"]

    if reachable_paths:
        lines.append(
            "REACHABLE JOIN PATHS (require an intermediate table not yet in context):"
        )
        lines.append(
            "  These joins are structurally available. To use them, call"
            " search_catalog with the intermediate table hint below."
        )
        for rp in reachable_paths[:_MAX_REACHABLE_PATHS]:
            details = []
            if rp.via_blob_path:
                details.append(f"file: {rp.via_blob_path}")
            if rp.via_domain_labels:
                details.append(f"roles: {', '.join(rp.via_domain_labels[:4])}")
            detail_suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(
                f"  {rp.source_table}  →  {rp.target_table}"
                f"  via {rp.via_table}{detail_suffix}  [confidence: {rp.path_confidence:.2f}]"
            )
        lines.append("")

    if orphaned_tables:
        lines.append(
            "ISOLATED FILES (no approved join path found to any other shortlisted file):"
        )
        lines.append(
            "  Call extract_relations to check for candidate join paths before"
            " concluding these files cannot be joined."
        )
        for t in orphaned_tables[:6]:
            lines.append(f"  {t}")
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def _role_labels(entry: dict) -> list[str]:
    roles = entry.get("column_semantic_roles") or {}
    labels: list[str] = []
    for role_val in roles.values():
        if not role_val:
            continue
        match = re.match(r"^custom:([a-z_]+):(.+)$", str(role_val))
        if not match:
            continue
        label = match.group(2)
        if label not in labels:
            labels.append(label)
    return labels

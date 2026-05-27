"""Execution Strategy Planner — deterministic graph-connectivity analysis.

RESPONSIBILITY:
  Given the retrieval-shortlisted files and the pre-validated join graph
  (approved_joins from SQLContext), determine whether the query should execute
  as a single joined SQL, multiple independent cluster SQLs, or fully
  independent per-file analyses.

NO LLM calls. NO DB queries. Pure in-memory Union-Find on approved edges.
O(N α(N) + E) time where N = shortlisted files, E = approved joins.

Decision logic:
  All files in one connected component  → "single_joined"
  Multiple components, ≥1 has 2+ files → "multi_cluster"
  All components are single-file        → "independent_analyses"

Invariant: lack of joinability is a VALID analytical outcome, not an error.
The LLM must never invent cross-domain joins when no approved path exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.sql_context_builder import SQLContext

# Re-implement locally to avoid coupling (4-line helper, same logic as sql_context_builder)
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)


def _table_name(blob_path: str) -> str:
    """Clean display name: strip hash prefix and file extension."""
    name = blob_path.rsplit("/", 1)[-1]
    name = _HASH_PREFIX_RE.sub("", name)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


# ── Union-Find (union-by-rank + path compression) ─────────────────────────────

class _UF:
    """Union-Find for grouping file_ids into connected components."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank to keep tree flat
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class ClusterPlan:
    files: list[str]      # clean display names (e.g. "BSEG", "LFA1")
    file_ids: list[str]   # original file_ids from catalog
    strategy: str         # "joined_sql" | "standalone"


@dataclass
class ExecutionStrategy:
    """
    Deterministic execution plan derived from graph connectivity.

    mode values:
      "single_joined"         — all shortlisted files connect via approved joins;
                                execute as one SQL query.
      "multi_cluster"         — files split into 2+ groups; ≥1 group is joinable;
                                execute one SQL per cluster, merge narratively.
      "independent_analyses"  — no approved joins between any files;
                                analyze each file independently.
    """
    mode: str
    clusters: list[ClusterPlan] = field(default_factory=list)

    def is_trivial(self) -> bool:
        """True when the strategy adds no useful information (single isolated file)."""
        return len(self.clusters) <= 1 and self.mode == "independent_analyses"

    def to_prompt_section(self) -> str:
        """
        Render a compact, LLM-readable execution strategy section.

        Returns empty string for trivial cases (single file — strategy is obvious).
        """
        if not self.clusters or self.is_trivial():
            return ""

        lines: list[str] = ["--- EXECUTION MODE ---"]

        if self.mode == "single_joined":
            lines += [
                "mode=single_joined",
                "Use one SQL query with the approved joins above.",
            ]

        elif self.mode == "multi_cluster":
            n = len(self.clusters)
            lines += [
                f"mode=multi_cluster; clusters={n}",
                "Do not join across clusters. Query clusters separately and merge narratively.",
            ]
            for i, c in enumerate(self.clusters, 1):
                tag = "joined_sql" if c.strategy == "joined_sql" else "standalone"
                files_str = ", ".join(c.files)
                lines.append(f"  - c{i}:{tag}:{files_str}")

        elif self.mode == "independent_analyses":
            lines += [
                "mode=independent_analyses",
                "No approved joins are currently in prompt. Use extract_relations before any join; otherwise analyze separately.",
            ]
            for i, c in enumerate(self.clusters[:8], 1):
                lines.append(f"  - file{i}:{', '.join(c.files)}")
            omitted = len(self.clusters) - 8
            if omitted > 0:
                lines.append(f"  - ... {omitted} additional standalone file(s) omitted.")

        lines.append("---")
        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def plan_execution_strategy(
    catalog: list[dict],
    sql_ctx: SQLContext,
) -> ExecutionStrategy:
    """
    Determine execution strategy from catalog connectivity.

    Uses only approved_joins from sql_ctx — no DB access, no LLM calls.

    Args:
        catalog:  shortlisted catalog entries; must have file_id + blob_path.
        sql_ctx:  SQLContext from build_sql_context; approved_joins are the edges.

    Returns:
        ExecutionStrategy — never raises.

    Algorithm:
      1. Build UF nodes from all catalog file_ids.
      2. For each ApprovedJoin, resolve table names → file_ids, union them.
      3. Group by UF root → connected components.
      4. Assign mode by component structure (see module docstring).
    """
    entries = [e for e in catalog if e.get("file_id") and e.get("blob_path")]
    if not entries:
        return ExecutionStrategy(mode="independent_analyses", clusters=[])

    # Build bidirectional name ↔ id mappings
    id_to_name: dict[str, str] = {
        e["file_id"]: _table_name(e["blob_path"]) for e in entries
    }
    name_to_ids: dict[str, list[str]] = {}
    for fid, name in id_to_name.items():
        name_to_ids.setdefault(name, []).append(fid)

    # Initialise Union-Find — every file starts as its own singleton component
    uf = _UF()
    for fid in id_to_name:
        uf.add(fid)

    # Apply approved join edges. Prefer canonical file IDs carried by SQLContext;
    # fall back to display-name resolution only for older contexts/tests.
    for join in sql_ctx.approved_joins:
        left_file_id = getattr(join, "left_file_id", "")
        right_file_id = getattr(join, "right_file_id", "")
        if left_file_id in id_to_name and right_file_id in id_to_name:
            left_ids = [left_file_id]
            right_ids = [right_file_id]
        else:
            left_ids = name_to_ids.get(join.left_table, [])
            right_ids = name_to_ids.get(join.right_table, [])
        for lid in left_ids:
            for rid in right_ids:
                uf.union(lid, rid)

    # Collect connected components (root → [file_ids])
    components: dict[str, list[str]] = {}
    for fid in id_to_name:
        components.setdefault(uf.find(fid), []).append(fid)

    # Build ClusterPlans — largest cluster first, then alphabetical by first file name
    clusters: list[ClusterPlan] = []
    for fids in sorted(components.values(), key=lambda ids: (-len(ids), sorted(id_to_name[f] for f in ids))):
        names = sorted(id_to_name[fid] for fid in fids)
        strategy = "joined_sql" if len(fids) > 1 else "standalone"
        clusters.append(ClusterPlan(files=names, file_ids=sorted(fids), strategy=strategy))

    # Determine execution mode
    if len(clusters) == 1 and clusters[0].strategy == "joined_sql":
        # All files connected — one SQL covers everything
        mode = "single_joined"
    elif all(c.strategy == "standalone" for c in clusters):
        # No join paths at all — fully independent
        mode = "independent_analyses"
    else:
        # Mixed or multiple joined clusters — execute per cluster
        mode = "multi_cluster"

    return ExecutionStrategy(mode=mode, clusters=clusters)

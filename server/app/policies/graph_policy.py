"""Graph expansion and join-context policy.

Governs:
  - Graph traversal bounds (graph_expand.py)
  - Join confidence floors for SQL context (sql_context_builder.py)
  - Graph health classification thresholds (graph_health.py)

WHY SEPARATE FROM SEMANTIC POLICY:
  SemanticPolicy (semantic_policy.py) governs the ingestion-time relationship
  inference and approval workflow (what gets approved).
  GraphPolicy governs query-time graph usage (what gets surfaced to the planner).

  Separation prevents ingestion logic from accidentally inheriting query-time
  caps (or vice versa) — they have different failure modes and tuning needs.

FUTURE READINESS:
  - Tenant override: a catalog with fully-verified manually-approved joins could
    lower join_hard_floor → 0.60 to surface more join paths.
  - Deployment override: a high-RAM VM could raise max_neighbors_per_node → 20.
  - Telemetry-driven: if graph_health_degraded_count is high, lower
    expansion_conf_floor to allow more edges through.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GraphPolicy:
    """
    All graph-related thresholds in one typed, immutable config.

    ── EXPANSION BOUNDS ─────────────────────────────────────────────────────
    max_seed_ids : int
        Trim the seed file list before the DB edge query.
        Prevents: O(N²) edge queries on very large catalogs.
        Default 30 covers the realistic shortlist size with room to spare.

    max_neighbors_per_node : int
        Cap how many neighbor files each seed file can contribute.
        Edges are fetched ORDER BY confidence DESC so the top-N wins.
        Prevents: a hub file (one generic ID column linked everywhere) flooding
        the expansion with low-value edges and pushing out relevant neighbors.

    expansion_conf_floor : float
        Edges with confidence below this are never traversed.
        Separate from (and may be stricter than) the semantic policy floor
        to prevent weak heuristic edges from polluting the shortlist.
        0.60 = edges must clear the "medium confidence" bar to enter retrieval.
        Prevents: speculative/heuristic edges inflating retrieval noise.

    ── JOIN CONTEXT BOUNDS ──────────────────────────────────────────────────
    join_hard_floor : float
        Absolute minimum join confidence. Joins below this are NEVER surfaced
        in the SQL context prompt, regardless of ranking.
        0.70 = "medium-high evidence that this join is correct."
        Prevents: hallucinated joins from weak fingerprint matches entering SQL.

    join_soft_floor : float
        Warning threshold. Joins above hard_floor but below soft_floor are
        surfaced but trigger a telemetry warning.
        0.75 = "join exists but evidence is thinner than ideal."
        Prevents: silent degradation when catalog has only marginal joins.

    top_n_approved_joins : int
        Rank-based cap: pull the top-N joins sorted by confidence DESC.
        Applied at DB query time (LIMIT clause), not post-filter.
        12 is generous enough for most ERP schemas while keeping the SQL
        context section in the prompt compact.
        Prevents: massive SQL context sections from large join-dense catalogs.

    max_joins_in_prompt : int
        Secondary cap applied in the prompt serialisation step.
        Prevents: accidentally surfacing >10 joins when top_n_approved_joins
        is raised but prompt budget is not re-evaluated.

    max_bindings : int
        Max entity/reference/attribute column bindings in prompt.
        Prevents: prompt section overflow from wide tables with many roles.

    max_date_cols : int
        Max date column bindings in prompt.
        Prevents: temporal filter section becoming longer than the query.

    max_null_semantics : int
        Max null-semantics entries in prompt.
        Prevents: prompt bloat from wide SAP tables with many empty-string columns.

    ── HEALTH THRESHOLDS ────────────────────────────────────────────────────
    good_edge_coverage : float
        Minimum edge coverage fraction for a "good" health rating.
        edge_coverage = approved_edges / possible_file_pairs.
        0.40 = at least 40% of file pairs have an approved join path.

    poor_edge_coverage : float
        Edge coverage below this → "poor" health.
        Means the planner has almost no join guidance.

    good_conf_p50 : float
        Minimum median join confidence for "good" health.

    poor_conf_p50 : float
        Median confidence below this → "poor" health.

    weak_edge_conf : float
        Per-edge threshold for counting "weak" edges.
        Separate from join_soft_floor to allow health scoring to use a
        different signal granularity than the prompt filter.

    weak_edge_warn_ratio : float
        If >this fraction of edges are "weak", health degrades to "degraded".

    orphan_warn_ratio : float
        If >this fraction of files with roles have no edges, health degrades.
        Orphan files have semantic labels but can't be JOIN-ed to anything —
        the planner can reference them but can't combine them with other files.
    """
    # Expansion bounds
    max_seed_ids:           int   = 30
    max_neighbors_per_node: int   = 10
    expansion_conf_floor:   float = 0.60

    # Join context
    join_hard_floor:        float = 0.70
    join_soft_floor:        float = 0.75
    top_n_approved_joins:   int   = 12
    max_joins_in_prompt:    int   = 10
    max_bindings:           int   = 25
    max_date_cols:          int   = 8
    max_null_semantics:     int   = 8

    # Health thresholds
    good_edge_coverage:     float = 0.40
    poor_edge_coverage:     float = 0.10
    good_conf_p50:          float = 0.75
    poor_conf_p50:          float = 0.60
    weak_edge_conf:         float = 0.75
    weak_edge_warn_ratio:   float = 0.40
    orphan_warn_ratio:      float = 0.50

    # ── Phase 7: Supernode (high-degree hub) mitigation ──────────────────────
    # supernode_degree_threshold : a file is classified as a "supernode" when
    #   it participates in more than this many approved edges within the edges
    #   fetched during a single graph_expand call.
    #   Enterprise examples: vendor master, shared date/period dimension, GL
    #   account hierarchy — files that every other file joins to.
    #   Default 15: wide enough for real ERP schemas (vendor master may have
    #   12–18 direct edges) while protecting against pathological hubs.
    supernode_degree_threshold:   int   = 15

    # supernode_confidence_penalty : multiplicative damping factor applied to
    #   edge confidence when either endpoint is a supernode.
    #   Edge still participates in expansion; it is not eliminated.
    #   0.20 = 20% confidence reduction.  With edge_attenuation_strength=0.60
    #   the combined compound effect is bounded; supernodes rank lower but still
    #   contribute diverse coverage.
    supernode_confidence_penalty: float = 0.20

    # max_neighbor_influence_ratio : maximum fraction of the expansion `limit`
    #   that any single seed file may contribute to the final result list.
    #   Applied as a post-sort cap: ceil(ratio × limit) neighbors per seed.
    #   Default 0.40 → with limit=20, one seed may contribute at most 8
    #   neighbors.  Prevents a single highly-connected hub from consuming all
    #   expansion slots and pushing out results from other seeds.
    max_neighbor_influence_ratio: float = 0.40

    # ── Phase 8: Supernode minimum participation guarantee ────────────────────
    # min_supernode_participation_slots : even a file classified as a supernode
    #   (degree > supernode_degree_threshold) is guaranteed at least this many
    #   contribution slots in the influence cap.
    #
    #   Rationale: Without this floor, aggressive lowering of
    #   max_neighbor_influence_ratio (e.g., 0.05 → ceil(0.05×20)=1 slot) could
    #   silence a vendor master or shared date table almost entirely.  Master
    #   tables are legitimately high-degree because they are shared across many
    #   subject areas — suppressing them too aggressively hurts recall.
    #
    #   The guarantee only activates when
    #     ceil(max_neighbor_influence_ratio × limit) < min_supernode_participation_slots
    #   At default settings (0.40 × 20 = 8 ≥ 2) the floor is inactive.
    #   It becomes active if ratio or limit is reduced in future calibration.
    min_supernode_participation_slots: int = 2

    # ── Phase 8: Graph health evaluation mode (future extensibility) ──────────
    # graph_density_mode : strategy label for graph health evaluation.
    #   "standard"    = current behavior: coverage thresholds applied uniformly
    #                   across all ingested files.
    #   Future values could include "sparse_domain" (relaxed coverage floors for
    #   enterprise schemas where few files have explicit join edges) or
    #   "dense_domain" (stricter thresholds for data warehouse schemas with high
    #   join coverage).  Changing this field is the hook for future calibration
    #   evolution WITHOUT redesigning graph health scoring logic.
    #   Current behavior is completely unchanged — this field is a label, NOT
    #   a runtime control flag.  No code reads it yet; it is a forward contract.
    graph_density_mode: str = "standard"


@lru_cache(maxsize=1)
def get_graph_policy() -> GraphPolicy:
    """Return the module-level singleton GraphPolicy."""
    return GraphPolicy()

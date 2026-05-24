"""Graph trustworthiness health metrics.

PURPOSE:
  Provide a fast, queryable view of graph quality for a given shortlist of files.
  Designed to run once per request (after sql_context is built) to populate the
  orchestration trace and emit telemetry. Does NOT block execution — findings are
  informational, not gating.

WHAT IT MEASURES:
  1. edge_coverage       — fraction of shortlisted file pairs that have an
                           approved relationship. Low = planner is flying blind.
  2. confidence_p50/p95  — distribution of approved-join confidence scores.
                           Low p50 = graph was inferred from weak evidence.
  3. weak_edge_ratio     — fraction of approved edges below the soft warning floor.
                           High = risk of hallucinated joins being used.
  4. entity_conflict_count — number of labels claimed by >1 file (duplicate masters).
  5. orphan_ratio        — fraction of shortlisted files with roles but no edges.
  6. anomaly_flags       — set of human-readable anomaly codes for the trace.

OUTPUT:
  `GraphHealthScore` dataclass — JSON-safe, all float/int fields.
  `health_level`: "good" | "degraded" | "poor" based on composite signal.

DESIGN CONSTRAINTS:
  - ZERO additional DB queries: operates only on data already fetched by
    build_sql_context (sql_ctx) and the catalog shortlist.
  - Pure computation — no I/O, no LLM, no async.
  - Called after sql_ctx is built; before execution.

This module does NOT re-query the DB — it works from what is already in memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.policies.graph_policy import get_graph_policy as _get_graph_policy

# Role parser (same pattern as sql_context_builder and ingestion_audit)
_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")

# ── Health classification thresholds (governed by GraphPolicy) ──────────────────
# See server/app/policies/graph_policy.py for rationale on each value.
_gp = _get_graph_policy()
_GOOD_MIN_COVERAGE    = _gp.good_edge_coverage
_POOR_MAX_COVERAGE    = _gp.poor_edge_coverage
_GOOD_MIN_CONF_P50    = _gp.good_conf_p50
_POOR_MAX_CONF_P50    = _gp.poor_conf_p50
_WEAK_EDGE_CONF       = _gp.weak_edge_conf
_WEAK_EDGE_WARN_RATIO = _gp.weak_edge_warn_ratio
_ORPHAN_WARN_RATIO    = _gp.orphan_warn_ratio


@dataclass
class GraphHealthScore:
    """
    Composite graph health for the current request's shortlisted files.

    All float fields are rounded to 3 decimal places for compactness in traces.
    """
    file_count:            int   = 0
    possible_pairs:        int   = 0    # n*(n-1)//2 for n shortlisted files
    approved_edge_count:   int   = 0
    edge_coverage:         float = 0.0  # approved_edges / possible_pairs
    confidence_p50:        float = 0.0  # median approved-join confidence
    confidence_p95:        float = 0.0  # 95th percentile
    confidence_min:        float = 0.0  # worst approved join
    weak_edge_count:       int   = 0    # edges with confidence < 0.75
    weak_edge_ratio:       float = 0.0  # weak_edges / total_edges
    orphan_file_count:     int   = 0    # files with roles but no edges
    orphan_ratio:          float = 0.0  # orphans / files_with_roles
    entity_conflict_count: int   = 0    # labels claimed by >1 file
    health_level:          str   = "good"   # "good" | "degraded" | "poor"
    anomaly_flags:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_count":            self.file_count,
            "possible_pairs":        self.possible_pairs,
            "approved_edge_count":   self.approved_edge_count,
            "edge_coverage":         self.edge_coverage,
            "confidence_p50":        self.confidence_p50,
            "confidence_p95":        self.confidence_p95,
            "confidence_min":        self.confidence_min,
            "weak_edge_count":       self.weak_edge_count,
            "weak_edge_ratio":       self.weak_edge_ratio,
            "orphan_file_count":     self.orphan_file_count,
            "orphan_ratio":          self.orphan_ratio,
            "entity_conflict_count": self.entity_conflict_count,
            "health_level":          self.health_level,
            "anomaly_flags":         self.anomaly_flags,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def score_graph_health(
    catalog: list[dict],
    sql_ctx: Any,           # SQLContext from sql_context_builder
) -> GraphHealthScore:
    """
    Compute a graph health score from already-fetched data.

    Args:
        catalog:   Shortlisted catalog entries (each must have file_id, blob_path,
                   and optionally column_semantic_roles).
        sql_ctx:   SQLContext populated by build_sql_context().

    Returns:
        GraphHealthScore — never raises.
    """
    score = GraphHealthScore()

    try:
        approved_joins = list(getattr(sql_ctx, "approved_joins", []) or [])
        file_count = len(catalog)
        score.file_count = file_count

        if file_count < 2:
            score.health_level = "good"
            return score

        # ── Edge coverage ──────────────────────────────────────────────────────
        possible_pairs = file_count * (file_count - 1) // 2
        score.possible_pairs = possible_pairs
        score.approved_edge_count = len(approved_joins)
        score.edge_coverage = round(
            len(approved_joins) / max(1, possible_pairs), 3
        )

        # ── Confidence distribution ────────────────────────────────────────────
        if approved_joins:
            confs = sorted(j.confidence for j in approved_joins)
            score.confidence_min = round(confs[0], 3)
            score.confidence_p50 = round(_percentile(confs, 50), 3)
            score.confidence_p95 = round(_percentile(confs, 95), 3)

            weak_threshold = _WEAK_EDGE_CONF
            score.weak_edge_count = sum(1 for c in confs if c < weak_threshold)
            score.weak_edge_ratio = round(
                score.weak_edge_count / max(1, len(confs)), 3
            )

        # ── Orphan detection ───────────────────────────────────────────────────
        # Files that have semantic roles but appear in no approved join are
        # "orphan masters" — the planner can't use them in JOINs.
        files_in_joins: set[str] = set()
        for j in approved_joins:
            files_in_joins.add(j.left_table)
            files_in_joins.add(j.right_table)

        files_with_roles: set[str] = set()
        for entry in catalog:
            roles = entry.get("column_semantic_roles") or {}
            if roles:
                # Use table name (same derivation as sql_context_builder._table_name)
                blob = entry.get("blob_path", "")
                name = blob.rsplit("/", 1)[-1]
                name = re.sub(r"^[0-9a-f]{8}_", "", name, flags=re.IGNORECASE)
                if "." in name:
                    name = name.rsplit(".", 1)[0]
                files_with_roles.add(name)

        orphan_names = files_with_roles - files_in_joins
        score.orphan_file_count = len(orphan_names)
        score.orphan_ratio = round(
            len(orphan_names) / max(1, len(files_with_roles)), 3
        ) if files_with_roles else 0.0

        # ── Entity conflict detection ──────────────────────────────────────────
        # Count entity_key labels claimed by more than one file in the shortlist.
        entity_key_labels: dict[str, set[str]] = {}  # label → {table_name}
        for entry in catalog:
            roles = entry.get("column_semantic_roles") or {}
            blob = entry.get("blob_path", "")
            name = blob.rsplit("/", 1)[-1]
            name = re.sub(r"^[0-9a-f]{8}_", "", name, flags=re.IGNORECASE)
            if "." in name:
                name = name.rsplit(".", 1)[0]
            for _col, role_str in roles.items():
                m = _ROLE_RE.match(str(role_str or ""))
                if m and m.group(1) == "entity_key":
                    label = m.group(2)
                    entity_key_labels.setdefault(label, set()).add(name)
        score.entity_conflict_count = sum(
            1 for names in entity_key_labels.values() if len(names) > 1
        )

        # ── Anomaly flags ──────────────────────────────────────────────────────
        flags: list[str] = []
        if score.edge_coverage == 0.0 and file_count > 1:
            flags.append("no_approved_joins")
        if score.weak_edge_ratio > _WEAK_EDGE_WARN_RATIO:
            flags.append("high_weak_edge_ratio")
        if score.orphan_ratio > _ORPHAN_WARN_RATIO:
            flags.append("high_orphan_ratio")
        if score.entity_conflict_count > 0:
            flags.append("entity_key_conflicts")
        if score.confidence_min > 0 and score.confidence_min < 0.60:
            flags.append("very_low_min_confidence")
        score.anomaly_flags = flags

        # ── health_level classification ────────────────────────────────────────
        is_poor = (
            score.edge_coverage < _POOR_MAX_COVERAGE
            or (score.confidence_p50 > 0 and score.confidence_p50 < _POOR_MAX_CONF_P50)
            or "entity_key_conflicts" in flags
        )
        is_degraded = (
            score.edge_coverage < _GOOD_MIN_COVERAGE
            or (score.confidence_p50 > 0 and score.confidence_p50 < _GOOD_MIN_CONF_P50)
            or score.weak_edge_ratio > _WEAK_EDGE_WARN_RATIO
            or score.orphan_ratio > _ORPHAN_WARN_RATIO
        )

        if is_poor:
            score.health_level = "poor"
        elif is_degraded:
            score.health_level = "degraded"
        else:
            score.health_level = "good"

    except Exception:
        pass  # never raise — health score is telemetry, not gating

    return score


# ── Internal helpers ────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

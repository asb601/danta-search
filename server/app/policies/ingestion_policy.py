"""Ingestion-time audit policy.

Governs:
  - Structural consistency warnings emitted during ingestion (ingestion_audit.py)

NOTE: SQL context prompt construction caps (max_joins_in_prompt, max_bindings,
  etc.) and join confidence floors (join_hard_floor, join_soft_floor) live in
  GraphPolicy because they apply at query time, not ingestion time.

WHY THESE ARE SEPARATE FROM GRAPH POLICY:
  IngestionPolicy governs what happens when data enters the system.
  GraphPolicy governs how that data is used at query time.
  Keeping them separate prevents ingestion tuning from accidentally affecting
  query behavior (or vice versa).

FUTURE READINESS:
  - Tenant override: a fully-curated catalog with expert-labeled roles could
    lower min_role_coverage → 0.30 if many tables intentionally have no roles.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class IngestionPolicy:
    """
    Ingestion audit and SQL context prompt construction policy.

    ── AUDIT THRESHOLDS ─────────────────────────────────────────────────────
    weak_confidence_warn : float
        Ingested relationships with confidence below this trigger a warning
        audit event (never raises — ingestion continues).
        0.60 = warn on anything in the "speculative" confidence range.
        Prevents: silent accumulation of low-quality joins in the catalog.

    high_density_ratio : float
        Total relationships / total files.  If ratio > this, the graph is
        unusually dense — may indicate a generic ID column linked everywhere.
        3.0 = more than 3× the typical edges-per-file density.

    min_role_coverage : float
        If fewer than this fraction of files have semantic roles assigned,
        the audit emits a coverage warning.
        0.50 = at least 50% of files should have a recognized semantic role.
        Prevents: catalogs where role-based retrieval is silently degraded
        because most files were ingested without role classification.

    ── PHASE 5: INGESTION TRUSTWORTHINESS ──────────────────────────────────
    supernode_degree_ratio : float
        Warn when a single file's edge count exceeds
        (avg_degree × supernode_degree_ratio). Detects hub files that dominate
        the graph neighborhood and may produce spurious multi-file JOINs.
        5.0 = flag files with >5× the average connectivity.
        Minimum absolute degree: 3 (suppresses noise in tiny graphs).

    low_role_confidence_warn : float
        If a file's average LLM-reported role confidence (from column_role_evidence)
        is below this threshold, the audit emits an info-level finding.
        0.60 = warn when most roles are speculative or under-supported.
        Applies only to files ingested after Phase 5 roll-out (older files have
        no column_role_evidence and are silently skipped).

    min_strong_evidence_count : int
        A relationship backed by fewer than this many overlapping fingerprinted
        key values is considered "weak evidence". The audit warns but does not
        block — the relationship may still be correct for sparse datasets.
        3 = require at least 3 matching key values to consider evidence strong.

    NOTE: Join confidence floors (join_hard_floor, join_soft_floor) and prompt
    construction caps (max_joins_in_prompt, max_bindings, etc.) live in
    GraphPolicy — they apply at query time, not ingestion time.
    """
    # Audit thresholds (existing)
    weak_confidence_warn:       float = 0.60
    high_density_ratio:         float = 3.0
    min_role_coverage:          float = 0.50

    # Phase 5: Ingestion trustworthiness thresholds
    supernode_degree_ratio:     float = 5.0
    low_role_confidence_warn:   float = 0.60
    min_strong_evidence_count:  int   = 3


@lru_cache(maxsize=1)
def get_ingestion_policy() -> IngestionPolicy:
    """Return the module-level singleton IngestionPolicy."""
    return IngestionPolicy()

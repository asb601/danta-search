"""Per-file ingestion confidence scoring.

PURPOSE
=======
Computes a single aggregate confidence score for each ingested file, combining
four independent quality dimensions:

  1. semantic_role_coverage  — fraction of columns that received a semantic role
  2. avg_role_confidence     — mean LLM confidence from column_role_evidence
  3. relationship_confidence — mean confidence_score of edges touching this file
  4. metadata_completeness   — presence of ai_description, embedding, key_dimensions

The score expresses downstream trustworthiness: how reliably can the orchestration
layer use this file in JOIN planning, entity resolution, and SQL generation?

DESIGN CONSTRAINTS
==================
- Pure computation — no I/O, no DB writes, no LLM calls.
- Called by complete_ingestion_stage after all prior stages finish.
- Reads FileMetadata + list[FileRelationship] already loaded by the caller.
- Stores result in FileMetadata.ingestion_confidence_score (float) and
  FileMetadata.ingestion_confidence_signals (JSONB).

WEIGHT RATIONALE
================
  _W_ROLE_COV  = 0.30  → Role coverage is the strongest predictor of
                          join-planning quality. No roles = no JOINs.
  _W_ROLE_CONF = 0.25  → Confident roles produce accurate SQL context.
                          Uncertain roles mislead the planner.
  _W_REL_CONF  = 0.25  → Edge quality matters as much as role quality
                          because bad edges cause spurious JOINs at query time.
  _W_META_COMPL= 0.20  → Completeness is a hygiene signal; it predicts
                          whether retrieval ranking will score this file fairly.

When no relationships exist (dimension 3 undefined), the 0.25 weight is
redistributed equally to role_coverage and avg_role_confidence. This prevents
penalising clean standalone files (e.g., fact-only dimension tables).

LEVEL THRESHOLDS (governed by IngestionPolicy)
==============================================
  high   : overall >= 0.75 — reliable for JOIN planning
  medium : overall >= 0.50 — usable with caution
  low    : overall <  0.50 — consider re-ingesting or manual review
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.policies.ingestion_policy import get_ingestion_policy

# ── Scoring weights ───────────────────────────────────────────────────────────
_W_ROLE_COV  = 0.30
_W_ROLE_CONF = 0.25
_W_REL_CONF  = 0.25
_W_META_COMPL = 0.20


@dataclass(frozen=True)
class IngestionConfidence:
    """Immutable per-file ingestion confidence result."""
    file_id: str
    overall: float                  # 0.0–1.0 weighted aggregate
    level: str                      # "high" | "medium" | "low"
    role_coverage: float            # fraction of columns with roles
    avg_role_confidence: float      # mean LLM confidence (from column_role_evidence)
    relationship_confidence_avg: float  # mean edge confidence_score
    metadata_completeness: float    # fraction of expected metadata fields present
    signals: dict                   # detailed breakdown for observability


def compute_ingestion_confidence(
    meta: FileMetadata,
    relationships: list[FileRelationship],
) -> IngestionConfidence:
    """Compute per-file ingestion confidence.

    Args:
        meta:          FileMetadata for the file. Must have columns_info and
                       column_semantic_roles populated (from ontology_stage).
        relationships: All FileRelationship rows where file is file_a or file_b.
                       Pass an empty list for files with no relationships detected.

    Returns:
        IngestionConfidence — immutable, JSON-serialisable via .signals.
    """
    policy = get_ingestion_policy()

    # ── Dimension 1: Semantic role coverage ──────────────────────────────────
    columns_info = meta.columns_info or []
    roles: dict = meta.column_semantic_roles or {}
    total_cols = len(columns_info)
    role_coverage = (len(roles) / total_cols) if total_cols > 0 else 0.0

    # ── Dimension 2: Avg role confidence (from Phase 5 evidence) ─────────────
    evidence: dict = meta.column_role_evidence or {}
    if evidence:
        confs = [
            float(v.get("confidence", 0.5))
            for v in evidence.values()
            if isinstance(v, dict) and "confidence" in v
        ]
        avg_role_confidence = (sum(confs) / len(confs)) if confs else 0.0
        low_confidence_roles = sum(1 for c in confs if c < policy.low_role_confidence_warn)
    else:
        # Pre-Phase-5 file: assume moderate confidence if roles exist
        avg_role_confidence = 0.70 if roles else 0.0
        low_confidence_roles = 0

    # ── Dimension 3: Relationship quality ────────────────────────────────────
    has_relationships = bool(relationships)
    if has_relationships:
        rel_conf_sum = sum(float(r.confidence_score or 0.0) for r in relationships)
        relationship_confidence_avg = rel_conf_sum / len(relationships)
        weak_evidence_edges = sum(
            1 for r in relationships
            if (r.evidence_count or 0) < policy.min_strong_evidence_count
        )
    else:
        relationship_confidence_avg = 0.0
        weak_evidence_edges = 0

    # ── Dimension 4: Metadata completeness ───────────────────────────────────
    completeness_signals = {
        "has_description": bool(meta.ai_description),
        "has_roles":       bool(roles),
        "has_embedding":   meta.description_embedding is not None,
        "has_key_dims":    bool(meta.key_dimensions),
    }
    metadata_completeness = sum(completeness_signals.values()) / len(completeness_signals)

    # ── Weighted aggregate ────────────────────────────────────────────────────
    if has_relationships:
        overall = (
            _W_ROLE_COV   * role_coverage
            + _W_ROLE_CONF * avg_role_confidence
            + _W_REL_CONF  * relationship_confidence_avg
            + _W_META_COMPL * metadata_completeness
        )
    else:
        # Redistribute relationship weight equally to role dimensions
        overall = (
            (_W_ROLE_COV  + _W_REL_CONF * 0.5) * role_coverage
            + (_W_ROLE_CONF + _W_REL_CONF * 0.5) * avg_role_confidence
            + _W_META_COMPL * metadata_completeness
        )

    overall = round(min(1.0, max(0.0, overall)), 4)

    if overall >= 0.75:
        level = "high"
    elif overall >= 0.50:
        level = "medium"
    else:
        level = "low"

    signals = {
        "role_coverage":              round(role_coverage, 3),
        "avg_role_confidence":        round(avg_role_confidence, 3),
        "relationship_confidence_avg": round(relationship_confidence_avg, 3),
        "metadata_completeness":      round(metadata_completeness, 3),
        "total_columns":              total_cols,
        "columns_with_roles":         len(roles),
        "edge_count":                 len(relationships),
        "weak_evidence_edges":        weak_evidence_edges,
        "low_confidence_roles":       low_confidence_roles,
        "completeness_detail":        completeness_signals,
        "has_role_evidence":          bool(evidence),
    }

    return IngestionConfidence(
        file_id=meta.file_id,
        overall=overall,
        level=level,
        role_coverage=role_coverage,
        avg_role_confidence=avg_role_confidence,
        relationship_confidence_avg=relationship_confidence_avg,
        metadata_completeness=metadata_completeness,
        signals=signals,
    )

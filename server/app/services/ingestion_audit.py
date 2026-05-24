"""Ingestion trustworthiness audit — structural consistency checks.

PURPOSE:
  Detect ingestion-time quality problems before they silently corrupt query results.
  Runs as a lightweight async inspection of the catalog metadata already in the DB.
  One DB query. No LLM. No schema scans. No external I/O beyond the existing session.

WHAT IT CHECKS:
  1. conflicting_roles        — same column label mapped to multiple incompatible role
                                kinds (e.g. "vendor" mapped as both entity_key and date).
  2. duplicate_entity_keys    — multiple files claiming the same entity_key label,
                                indicating ambiguous master-data sourcing.
  3. weak_confidence_joins    — approved/active relationships with confidence below the
                                warn threshold. These will be filtered out at query time
                                but signal that ingestion produced unreliable structure.
  4. orphan_entities          — files with column_semantic_roles but zero approved
                                relationships. May indicate isolated masters that never
                                got joined, or relationship extraction failures.
  5. role_coverage_metrics    — fraction of shortlisted files that have any semantic
                                roles at all. Low coverage = query planner is flying blind.
  6. high_density_clusters    — file subgraphs with suspiciously many relationships
                                relative to file count, which can indicate mis-attributed
                                joins (one generic ID column linked everywhere).

WHAT IT DOES NOT DO:
  - Does NOT add ontology systems, SME glossaries, or curated synonym lists.
  - Does NOT modify any data — read-only inspection.
  - Does NOT raise — always returns an AuditResult even on error.
  - Does NOT block query execution — callers use results for telemetry only.

INTEGRATION:
  Called once per ingest completion (in the ingest worker) and optionally in
  the admin health endpoint. NOT called per query — too expensive for hot path.

  from app.services.ingestion_audit import run_ingestion_audit
  result = await run_ingestion_audit(container_id, db)
  audit_logger.info("ingestion_audit", **result.to_dict())
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import defaultdict

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.models.semantic_layer import SemanticRelationship
from app.policies.ingestion_policy import get_ingestion_policy as _get_ingestion_policy


# ── Thresholds (governed by IngestionPolicy) ─────────────────────────────────────────────────
# See server/app/policies/ingestion_policy.py for rationale on each value.
_ip = _get_ingestion_policy()
_WEAK_CONFIDENCE_WARN         = _ip.weak_confidence_warn
_HIGH_DENSITY_RATIO           = _ip.high_density_ratio
_MIN_ROLE_COVERAGE            = _ip.min_role_coverage
# Phase 5 thresholds
_SUPERNODE_DEGREE_RATIO       = _ip.supernode_degree_ratio
_LOW_ROLE_CONFIDENCE_WARN     = _ip.low_role_confidence_warn
_MIN_STRONG_EVIDENCE_COUNT    = _ip.min_strong_evidence_count

# Accepted role kinds (from column_semantic_roles value format "custom:<kind>:<label>")
_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")

# Role kinds that are mutually exclusive for the same label
# (a label shouldn't be both a date and an entity key)
_EXCLUSIVE_KINDS: frozenset[str] = frozenset({"entity_key", "reference_key", "date"})


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class AuditFinding:
    """One identified quality issue."""
    severity: str         # "error" | "warning" | "info"
    code: str             # stable machine-readable code for dashboards
    message: str          # human-readable description
    context: dict = field(default_factory=dict)   # supporting evidence


@dataclass
class AuditResult:
    """Complete audit output for one container."""
    container_id: str
    file_count: int
    files_with_roles: int
    role_coverage: float
    relationship_count: int
    findings: list[AuditFinding] = field(default_factory=list)
    error: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "container_id":    self.container_id,
            "file_count":      self.file_count,
            "files_with_roles":self.files_with_roles,
            "role_coverage":   round(self.role_coverage, 3),
            "relationship_count": self.relationship_count,
            "findings":        [
                {
                    "severity": f.severity,
                    "code":     f.code,
                    "message":  f.message,
                    "context":  f.context,
                }
                for f in self.findings
            ],
            "has_errors":   self.has_errors,
            "has_warnings": self.has_warnings,
            "error":        self.error,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_ingestion_audit(
    container_id: str,
    db: AsyncSession,
) -> AuditResult:
    """
    Run all ingestion trustworthiness checks for one container.

    Args:
        container_id: The container to audit.
        db:           Async session — one query for metadata, one for relationships.

    Returns:
        AuditResult — never raises. Returns result with error field on exception.
    """
    result = AuditResult(
        container_id=container_id,
        file_count=0,
        files_with_roles=0,
        role_coverage=0.0,
        relationship_count=0,
    )

    try:
        # ── Query 1: FileMetadata for this container ───────────────────────────
        meta_rows = (await db.execute(
            select(
                FileMetadata.file_id,
                FileMetadata.column_semantic_roles,
                FileMetadata.column_role_evidence,   # Phase 5
            )
            .where(FileMetadata.container_id == container_id)
        )).all()

        file_ids = [row.file_id for row in meta_rows]
        result.file_count = len(file_ids)

        if not file_ids:
            result.findings.append(AuditFinding(
                severity="warning",
                code="no_files",
                message="No files found for this container.",
                context={"container_id": container_id},
            ))
            return result

        # ── Parse roles from all files ─────────────────────────────────────────
        # label → set of (kind, file_id) tuples
        label_kinds: dict[str, set[tuple[str, str]]] = defaultdict(set)
        # kind → set of labels
        entity_key_labels: dict[str, set[str]] = defaultdict(set)  # label → {file_id}
        files_with_any_role: set[str] = set()

        # Phase 5: track avg role confidence per file for low-confidence check
        low_confidence_files: list[str] = []

        for row in meta_rows:
            roles: dict = row.column_semantic_roles or {}
            if not roles:
                continue
            files_with_any_role.add(row.file_id)
            for _col_name, role_str in roles.items():
                m = _ROLE_RE.match(str(role_str or ""))
                if not m:
                    continue
                kind, label = m.group(1), m.group(2)
                if kind in _EXCLUSIVE_KINDS:
                    label_kinds[label].add((kind, row.file_id))
                if kind == "entity_key":
                    entity_key_labels[label].add(row.file_id)

            # Phase 5: check avg role confidence (only for Phase-5-ingested files)
            evidence: dict = row.column_role_evidence or {}
            if evidence:
                confs = [
                    float(v.get("confidence", 0.5))
                    for v in evidence.values()
                    if isinstance(v, dict) and "confidence" in v
                ]
                if confs:
                    avg_conf = sum(confs) / len(confs)
                    if avg_conf < _LOW_ROLE_CONFIDENCE_WARN:
                        low_confidence_files.append(row.file_id)

        result.files_with_roles = len(files_with_any_role)
        result.role_coverage = (
            result.files_with_roles / result.file_count
            if result.file_count else 0.0
        )

        result.files_with_roles = len(files_with_any_role)
        result.role_coverage = (
            result.files_with_roles / result.file_count
            if result.file_count else 0.0
        )

        # ── Check 0 (Phase 5): Low avg role confidence ─────────────────────────
        if low_confidence_files:
            result.findings.append(AuditFinding(
                severity="info",
                code="low_role_confidence",
                message=(
                    f"{len(low_confidence_files)} file(s) have average role confidence "
                    f"below {_LOW_ROLE_CONFIDENCE_WARN:.0%}. Roles were assigned with weak "
                    "supporting evidence — verify with a schema glossary or re-ingest "
                    "with richer sample data."
                ),
                context={
                    "low_confidence_file_count": len(low_confidence_files),
                    "threshold": _LOW_ROLE_CONFIDENCE_WARN,
                    "file_ids": low_confidence_files[:10],
                },
            ))

        # ── Check 1: Role coverage ─────────────────────────────────────────────
        if result.role_coverage < _MIN_ROLE_COVERAGE and result.file_count > 1:
            result.findings.append(AuditFinding(
                severity="warning",
                code="low_role_coverage",
                message=(
                    f"Only {result.files_with_roles}/{result.file_count} files have "
                    f"semantic roles ({result.role_coverage:.0%}). "
                    "The query planner will not be able to infer join keys for un-annotated files."
                ),
                context={
                    "coverage": round(result.role_coverage, 3),
                    "threshold": _MIN_ROLE_COVERAGE,
                    "files_without_roles": [
                        row.file_id
                        for row in meta_rows
                        if row.file_id not in files_with_any_role
                    ][:10],
                },
            ))

        # ── Check 2: Conflicting roles (same label, multiple exclusive kinds) ──
        for label, kind_file_pairs in label_kinds.items():
            kinds_used = {k for k, _ in kind_file_pairs}
            if len(kinds_used) > 1:
                result.findings.append(AuditFinding(
                    severity="error",
                    code="conflicting_role_kinds",
                    message=(
                        f"Label '{label}' is mapped to conflicting role kinds: "
                        f"{sorted(kinds_used)}. "
                        "The planner treats entity_key, reference_key, and date as mutually "
                        "exclusive. Re-ingest the affected files with a consistent role."
                    ),
                    context={
                        "label": label,
                        "conflicting_kinds": sorted(kinds_used),
                        "files": [fid for _, fid in list(kind_file_pairs)[:10]],
                    },
                ))

        # ── Check 3: Duplicate entity_key labels (multiple files same label) ───
        for label, file_id_set in entity_key_labels.items():
            if len(file_id_set) > 1:
                result.findings.append(AuditFinding(
                    severity="warning",
                    code="duplicate_entity_key",
                    message=(
                        f"Entity key label '{label}' is claimed by {len(file_id_set)} files. "
                        "The resolver will return multiple candidates — queries may merge "
                        "data from multiple master sources unintentionally."
                    ),
                    context={
                        "label":    label,
                        "file_ids": sorted(file_id_set)[:10],
                    },
                ))

        # ── Query 2: Relationships for this container ──────────────────────────
        rel_rows = (await db.execute(
            select(
                SemanticRelationship.file_a_id,
                SemanticRelationship.file_b_id,
                SemanticRelationship.confidence_score,
                SemanticRelationship.approval_status,
                SemanticRelationship.status,
                SemanticRelationship.relationship_type,
            )
            .where(SemanticRelationship.container_id == container_id)
        )).all()

        result.relationship_count = len(rel_rows)
        file_id_set = set(file_ids)

        # ── Check 4: Weak confidence joins ─────────────────────────────────────
        weak_joins = [
            r for r in rel_rows
            if r.approval_status == "approved"
            and r.status == "active"
            and r.confidence_score < _WEAK_CONFIDENCE_WARN
        ]
        if weak_joins:
            result.findings.append(AuditFinding(
                severity="warning",
                code="weak_confidence_joins",
                message=(
                    f"{len(weak_joins)} approved relationships have confidence < "
                    f"{_WEAK_CONFIDENCE_WARN:.0%}. These will be excluded from SQL context "
                    f"(threshold {0.70:.0%}). Re-examine the join columns or lower the "
                    "approval threshold only if the join is semantically verified."
                ),
                context={
                    "weak_count": len(weak_joins),
                    "examples": [
                        {
                            "file_a": r.file_a_id,
                            "file_b": r.file_b_id,
                            "confidence": round(r.confidence_score, 3),
                            "type": r.relationship_type,
                        }
                        for r in weak_joins[:5]
                    ],
                },
            ))

        # ── Check 5: Orphan entities ───────────────────────────────────────────
        # Files with roles but zero active approved relationships
        files_with_approved_rels: set[str] = {
            fid
            for r in rel_rows
            if r.status == "active" and r.approval_status == "approved"
            for fid in (r.file_a_id, r.file_b_id)
        }
        orphans = [
            fid for fid in files_with_any_role
            if fid in file_id_set and fid not in files_with_approved_rels
        ]
        if orphans:
            result.findings.append(AuditFinding(
                severity="info",
                code="orphan_entities",
                message=(
                    f"{len(orphans)} file(s) have semantic roles but no approved relationships. "
                    "These files cannot participate in JOINs — they will be queried in isolation. "
                    "If they should join another file, verify the relationship extraction step."
                ),
                context={
                    "orphan_count": len(orphans),
                    "file_ids":     orphans[:10],
                },
            ))

        # ── Check 6: High-density clusters ────────────────────────────────────
        # Build adjacency: count relationships per file
        rel_degree: dict[str, int] = defaultdict(int)
        for r in rel_rows:
            if r.status == "active":
                rel_degree[r.file_a_id] += 1
                rel_degree[r.file_b_id] += 1
        if rel_rows and file_ids:
            # Overall density: total edges / total nodes
            density = len(rel_rows) / max(1, len(file_ids))
            if density > _HIGH_DENSITY_RATIO:
                result.findings.append(AuditFinding(
                    severity="warning",
                    code="high_graph_density",
                    message=(
                        f"Relationship graph is unusually dense: {len(rel_rows)} relationships "
                        f"across {len(file_ids)} files (ratio {density:.1f}x, threshold "
                        f"{_HIGH_DENSITY_RATIO:.1f}x). This may indicate a generic ID column "
                        "being linked across unrelated domains. Review high-degree nodes."
                    ),
                    context={
                        "total_relationships": len(rel_rows),
                        "total_files": len(file_ids),
                        "density_ratio": round(density, 2),
                        "high_degree_files": sorted(
                            rel_degree.items(), key=lambda x: x[1], reverse=True
                        )[:5],
                    },
                ))

        # ── Query 3 (Phase 5): FileRelationship for weak-evidence edge check ──
        # Use file_ids already known for this container (scoped via file_a_id)
        if file_ids:
            raw_rels = (await db.execute(
                select(
                    FileRelationship.file_a_id,
                    FileRelationship.file_b_id,
                    FileRelationship.evidence_count,
                )
                .where(FileRelationship.file_a_id.in_(file_ids))
            )).all()

            weak_evidence_edges = [
                (r.file_a_id, r.file_b_id, r.evidence_count)
                for r in raw_rels
                if r.evidence_count is not None
                and r.evidence_count < _MIN_STRONG_EVIDENCE_COUNT
            ]
            if weak_evidence_edges:
                result.findings.append(AuditFinding(
                    severity="info",
                    code="weak_evidence_edges",
                    message=(
                        f"{len(weak_evidence_edges)} relationship(s) are backed by fewer than "
                        f"{_MIN_STRONG_EVIDENCE_COUNT} overlapping key values. These joins may "
                        "be correct for sparse datasets but carry higher false-positive risk. "
                        "Consider supplying richer sample data at ingest time."
                    ),
                    context={
                        "weak_edge_count": len(weak_evidence_edges),
                        "min_strong_evidence": _MIN_STRONG_EVIDENCE_COUNT,
                        "examples": [
                            {"file_a": fa, "file_b": fb, "evidence_count": ec}
                            for fa, fb, ec in weak_evidence_edges[:5]
                        ],
                    },
                ))

        # ── Check 7 (Phase 5): Supernode detection ─────────────────────────────
        # rel_degree was computed above in the orphan/density checks
        if rel_degree and len(rel_degree) >= 2:
            avg_degree = sum(rel_degree.values()) / len(rel_degree)
            supernode_threshold = avg_degree * _SUPERNODE_DEGREE_RATIO
            # Suppress noise: require at least 3 absolute edges to be flagged
            supernodes = [
                (fid, deg)
                for fid, deg in rel_degree.items()
                if deg > supernode_threshold and deg >= 3
            ]
            if supernodes:
                result.findings.append(AuditFinding(
                    severity="warning",
                    code="supernode_detected",
                    message=(
                        f"{len(supernodes)} file(s) have relationship counts more than "
                        f"{_SUPERNODE_DEGREE_RATIO:.0f}× the graph average ({avg_degree:.1f} "
                        f"edges). Supernodes typically indicate a generic key column "
                        "(e.g. 'id') being linked to unrelated domains. Verify the "
                        "join columns on these files."
                    ),
                    context={
                        "supernode_count": len(supernodes),
                        "avg_degree": round(avg_degree, 1),
                        "threshold": supernode_threshold,
                        "supernodes": [
                            {"file_id": fid, "degree": deg}
                            for fid, deg in sorted(supernodes, key=lambda x: x[1], reverse=True)[:5]
                        ],
                    },
                ))

        chat_logger.info(
            "ingestion_audit_complete",
            container_id=container_id,
            file_count=result.file_count,
            role_coverage=round(result.role_coverage, 3),
            relationship_count=result.relationship_count,
            finding_count=len(result.findings),
            has_errors=result.has_errors,
        )

    except Exception as exc:
        result.error = str(exc)[:500]
        chat_logger.warning("ingestion_audit_error", container_id=container_id, error=result.error)

    return result

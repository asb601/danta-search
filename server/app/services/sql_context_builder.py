"""SQL Context Builder — validated join paths and column bindings.

RESPONSIBILITY (strictly bounded):
  Given the retrieval-shortlisted catalog files, extract and format
  pre-validated constraints for the SQL generation step.

Provides:
  1. approved_joins       — exact column pairs that may legally join two files
  2. column_bindings      — semantic label → "TABLE.COL" mappings
  3. date_columns         — semantic date label → "TABLE.COL" mappings
  4. null_semantics       — "TABLE.COL IS NULL" → business meaning

NOT responsible for:
  - SQL generation
  - query planning
  - join path discovery (that is graph_expand / extract_relations)
  - anything requiring LLM reasoning

Design constraints:
  - Two DB queries maximum per request.
  - Operates only on files already in the retrieval shortlist.
  - All joins must be status=active AND approval_status=approved.
  - Never raises — empty SQLContext is the safe fallback.
  - Output is injected as a read-only constraint section into the system
    prompt, not as a tool response.

DB queries:
  1. SemanticRelationship  WHERE both endpoints IN shortlist
                            AND status=active AND approval_status=approved
                            AND confidence >= _JOIN_MIN_CONFIDENCE
  2. FileMetadata.column_semantic_roles WHERE file_id IN shortlist
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticRelationship
from app.policies.graph_policy import get_graph_policy as _get_graph_policy


# ── Tuning constants (governed by GraphPolicy) ────────────────────────────────
# See server/app/policies/graph_policy.py for rationale on each value.
# Module-level aliases kept for readability and test monkeypatching.
#
# EVOLUTION PATH (Section 6 — join confidence evolution):
#   The global floor is acceptable initially but causes issues at scale.
#   Migration: cluster-relative selection using _JOIN_HARD_FLOOR as absolute
#   safety net and _TOP_N_APPROVED_JOINS for rank-based cap. DB already sorts
#   by confidence DESC + LIMIT, so the DB does the heavy lifting.
_gp = _get_graph_policy()
_JOIN_HARD_FLOOR      = _gp.join_hard_floor       # absolute minimum — never surface below this
_JOIN_SOFT_FLOOR      = _gp.join_soft_floor        # telemetry warning threshold
_TOP_N_APPROVED_JOINS = _gp.top_n_approved_joins   # rank-based cap: pull top-N by confidence DESC

# Prompt section caps — keep SQL context compact regardless of catalog size.
_MAX_JOINS    = _gp.max_joins_in_prompt
_MAX_BINDINGS = _gp.max_bindings
_MAX_DATE_COLS = _gp.max_date_cols
_MAX_NULL_SEM  = _gp.max_null_semantics


# ── Role parsing ───────────────────────────────────────────────────────────────

_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")
# Strip 8-hex hash prefix from filenames (e.g. "dba1285e_LFA1.parquet" → "LFA1")
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)


def _table_name(blob_path: str) -> str:
    """Return a clean display name for a blob_path (no hash prefix, no extension)."""
    name = blob_path.rsplit("/", 1)[-1]
    name = _HASH_PREFIX_RE.sub("", name)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class ApprovedJoin:
    left_table: str          # clean display name of the left file
    right_table: str         # clean display name of the right file
    left_col: str            # column on left file (from_column)
    right_col: str           # column on right file (to_column)
    relationship_type: str   # e.g. "vendor_master_join"
    confidence: float
    # Telemetry fields — always True/False in the current implementation
    # because all joins come from SemanticRelationship with approval_status=approved.
    # Explicit fields make the telemetry schema forward-compatible when inference
    # paths are added in the future.
    graph_verified: bool = True    # sourced from an approved SemanticRelationship edge
    fallback_inferred: bool = False  # True if derived without an approved graph edge


@dataclass
class SQLContext:
    """
    Pre-validated SQL constraints for one request's shortlisted files.

    Populated by build_sql_context().
    Serialised to a prompt section by to_prompt_section().
    """
    approved_joins: list[ApprovedJoin] = field(default_factory=list)
    # semantic_label → ["TABLE.COL", ...]   (entity_key / reference_key / attribute)
    column_bindings: dict[str, list[str]] = field(default_factory=dict)
    # semantic_label → ["TABLE.COL", ...]   (date kind only)
    date_columns: dict[str, list[str]] = field(default_factory=dict)
    # "TABLE.COL IS NULL" → business meaning string
    null_semantics: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            not self.approved_joins
            and not self.column_bindings
            and not self.date_columns
            and not self.null_semantics
        )

    def to_prompt_section(self) -> str:
        """
        Render a compact, LLM-readable constraint section.

        Returns empty string when there is nothing to show — callers can
        gate on the result directly (no injection when empty).
        """
        if self.is_empty():
            return ""

        lines: list[str] = [
            "--- VALIDATED SQL CONTEXT ---",
            "Pre-validated joins and column bindings for this query's files.",
            "Use ONLY these joins/columns. Do NOT invent alternatives.",
            "",
        ]

        # ── Approved joins ─────────────────────────────────────────────────
        if self.approved_joins:
            lines.append("APPROVED JOINS (exact column pairs; never substitute):")
            for j in self.approved_joins[:_MAX_JOINS]:
                lines.append(
                    f"  {j.left_table}.{j.left_col} = {j.right_table}.{j.right_col}"
                    f"  [{j.relationship_type}, conf:{j.confidence:.2f}]"
                )
            lines.append("")

        # ── Semantic column roles ──────────────────────────────────────────
        if self.column_bindings:
            lines.append("SEMANTIC COLUMN ROLES:")
            for label, bindings in sorted(self.column_bindings.items())[:_MAX_BINDINGS]:
                lines.append(f"  {label:<30}→ {', '.join(bindings)}")
            lines.append("")

        # ── Date columns ───────────────────────────────────────────────────
        if self.date_columns:
            lines.append(
                "DATE COLUMNS (use for time filters; do NOT substitute other columns):"
            )
            for label, cols in sorted(self.date_columns.items())[:_MAX_DATE_COLS]:
                lines.append(f"  {label:<30}→ {', '.join(cols)}")
            lines.append("")

        # ── Null semantics ─────────────────────────────────────────────────
        if self.null_semantics:
            lines.append("NULL SEMANTICS:")
            for expr, meaning in list(self.null_semantics.items())[:_MAX_NULL_SEM]:
                lines.append(f"  {expr:<45}→ {meaning}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

async def build_sql_context(
    catalog: list[dict],
    db: AsyncSession,
) -> SQLContext:
    """
    Build validated SQL context from the retrieval shortlist.

    Args:
        catalog: lean+hydrated shortlist entries from _build_agent_context.
                 Must have file_id and blob_path.
        db:      async session — used for two batch queries.

    Returns:
        SQLContext — never raises. Returns empty SQLContext on any error.

    DB cost: exactly TWO queries.
      1. SemanticRelationship: edges where BOTH endpoints are in shortlist.
      2. FileMetadata.column_semantic_roles for shortlisted file_ids.
    """
    ctx = SQLContext()
    if not catalog:
        return ctx

    file_ids = [e["file_id"] for e in catalog if e.get("file_id")]
    if not file_ids:
        return ctx

    # Build in-memory lookups (free — catalog is already in memory)
    id_to_blob: dict[str, str] = {
        e["file_id"]: e["blob_path"]
        for e in catalog
        if e.get("file_id") and e.get("blob_path")
    }
    id_to_name: dict[str, str] = {
        fid: _table_name(blob) for fid, blob in id_to_blob.items()
    }
    file_id_set = set(file_ids)

    # ── Query 1: Approved joins within the shortlist ─────────────────────────
    # We require BOTH endpoints to be in the shortlist — these are the joins
    # the LLM can actually use in its SQL right now.
    # Candidates / technical_candidates are excluded: those are still unverified
    # and the system prompt already explains how to handle them with schema checks.
    try:
        rel_rows = (await db.execute(
            select(
                SemanticRelationship.file_a_id,
                SemanticRelationship.file_b_id,
                SemanticRelationship.from_column,
                SemanticRelationship.to_column,
                SemanticRelationship.relationship_type,
                SemanticRelationship.confidence_score,
            )
            .where(
                SemanticRelationship.file_a_id.in_(file_ids),
                SemanticRelationship.file_b_id.in_(file_ids),
                SemanticRelationship.status == "active",
                SemanticRelationship.approval_status == "approved",
                SemanticRelationship.confidence_score >= _JOIN_HARD_FLOOR,
            )
            .order_by(SemanticRelationship.confidence_score.desc())
            .limit(_TOP_N_APPROVED_JOINS)
        )).all()

        weak_join_count = 0
        for row in rel_rows:
            # Double-check both sides are in scope (SQL IN handles this, but be safe)
            if row.file_a_id in file_id_set and row.file_b_id in file_id_set:
                conf = round(row.confidence_score, 2)
                if conf < _JOIN_SOFT_FLOOR:
                    weak_join_count += 1
                ctx.approved_joins.append(ApprovedJoin(
                    left_table=id_to_name.get(row.file_a_id, row.file_a_id),
                    right_table=id_to_name.get(row.file_b_id, row.file_b_id),
                    left_col=row.from_column,
                    right_col=row.to_column,
                    relationship_type=row.relationship_type,
                    confidence=conf,
                ))

        if weak_join_count:
            chat_logger.warning(
                "sql_context_weak_joins_surfaced",
                weak_count=weak_join_count,
                total_joins=len(ctx.approved_joins),
                soft_floor=_JOIN_SOFT_FLOOR,
                hard_floor=_JOIN_HARD_FLOOR,
            )

    except Exception as exc:
        chat_logger.warning("sql_context_join_fetch_error", error=str(exc)[:200])

    # ── Query 2: Column semantic roles for shortlisted files ─────────────────
    # Roles follow the format "custom:<kind>:<label>", e.g.:
    #   "custom:entity_key:vendor"       → master identity column
    #   "custom:reference_key:clearing_document" → FK column; IS NULL = open item
    #   "custom:date:posting_date"       → date filter column
    #   "custom:attribute:company_code"  → categorical dimension column
    #   "custom:additive_measure:amount" → metric; NOT included (not a filter key)
    try:
        role_rows = (await db.execute(
            select(FileMetadata.file_id, FileMetadata.column_semantic_roles)
            .where(FileMetadata.file_id.in_(file_ids))
        )).all()

        for row in role_rows:
            table = id_to_name.get(row.file_id, row.file_id)
            roles: dict = row.column_semantic_roles or {}

            for col_name, role_str in roles.items():
                m = _ROLE_RE.match(str(role_str or ""))
                if not m:
                    continue
                kind, label = m.group(1), m.group(2)
                binding = f"{table}.{col_name}"

                if kind == "date":
                    ctx.date_columns.setdefault(label, []).append(binding)

                elif kind in ("entity_key", "reference_key", "attribute"):
                    ctx.column_bindings.setdefault(label, []).append(binding)
                    # reference_key IS NULL ↔ "no [label] exists for this record"
                    # This is the standard open-item / uncleared-document pattern.
                    if kind == "reference_key":
                        null_expr = f"{table}.{col_name} IS NULL"
                        meaning = f"no {label.replace('_', ' ')} (record is open/uncleared)"
                        ctx.null_semantics[null_expr] = meaning

                # additive_measure / non_additive_measure are intentionally omitted —
                # they are aggregation targets, not join keys or filter predicates.

    except Exception as exc:
        chat_logger.warning("sql_context_roles_fetch_error", error=str(exc)[:200])

    chat_logger.info(
        "sql_context_built",
        approved_joins=len(ctx.approved_joins),
        column_bindings=len(ctx.column_bindings),
        date_columns=len(ctx.date_columns),
        null_semantics=len(ctx.null_semantics),
        shortlist_size=len(file_ids),
    )

    return ctx

"""
Data Catalog — a READ-ONLY projection over the metadata tables that ingestion
already populates (response.txt Section 5).

We do NOT introduce new storage for the catalog. FileMetadata, FileAnalytics
and the semantic role columns already hold descriptions, measures, dimensions,
cardinality, data-quality and join information. This service unifies them into
a catalog view scoped by tenant (container_id) and domain (allowed_domains).

Used by:
  - query_engine (to ground prompt decomposition in real tables),
  - the dashboards router (GET /api/dashboards/catalog/data),
  - the recommendation engine (mapping dataset columns back to semantic roles).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.services import semantic_roles as _sr
from app.services.dashboard.join_gate import safe_join
from app.services.ingestion_config import IngestStatus

# A file is dashboard-ready once ingestion has produced metadata/parquet. The
# canonical terminal success status is IngestStatus.INGESTED ("ingested"); the
# legacy literals are tolerated so historical rows are never silently excluded.
# (RC-001: the previous hardcoded "completed" matched NO rows — that string is
# never written by ingestion — which left the entire dashboard catalog empty.)
_READY_STATUSES = (
    IngestStatus.INGESTED.value,  # "ingested" — what ingestion actually writes
    "completed",                  # legacy/defensive
    "done",                       # legacy/defensive
)


def _first_present(*values):
    """Return the first value that is not None (preserves falsy-but-valid like 0)."""
    for v in values:
        if v is not None:
            return v
    return None


def _role_kind(semantic_role: str | None) -> str:
    """Map a 'custom:kind:label' semantic role to a coarse dashboard kind via the
    CANONICAL role taxonomy (exact dispatch through semantic_roles), not by
    substring-guessing the role string. A malformed/non-canonical role fails safe
    to 'dimension' (never silently promoted to a summable measure)."""
    if not semantic_role:
        return "unknown"
    if _sr.is_metric_role(semantic_role):          # additive_measure | non_additive_measure
        return "measure"
    if _sr.is_date_role(semantic_role):
        return "date"
    if _sr.is_entity_key_role(semantic_role) or _sr.is_reference_key_role(semantic_role):
        return "key"
    return "dimension"                              # attribute, or anything unrecognized


@dataclass
class DataCatalogColumn:
    name: str
    data_type: str
    semantic_role: str | None
    role_kind: str
    cardinality: int | None
    null_ratio: float | None
    sample_values: list = field(default_factory=list)
    # Actual value coverage from ingestion analytics (column_stats min/max).
    # For temporal columns these are the real date bounds; for measures the
    # numeric range. This is what lets the decomposer avoid emitting filters
    # (e.g. a Q2-2025 window) that fall entirely outside the loaded data.
    min_value: object = None
    max_value: object = None
    # Top observed categorical values (from FileAnalytics.value_counts) so the
    # decomposer/agent never invents a status/category literal like 'Shipped'.
    top_values: list = field(default_factory=list)


@dataclass
class DataCatalogTable:
    file_id: str
    table_name: str
    blob_path: str | None
    parquet_path: str | None
    description: str | None
    business_domain: list
    row_count: int
    columns: list[DataCatalogColumn]
    measures: list[str]
    dimensions: list[str]
    temporal: list[str]
    join_keys: list[str]
    data_quality: dict

    def as_dict(self) -> dict:
        return asdict(self)

    def compact_summary(self) -> str:
        """A one-line summary used to ground LLM prompt decomposition cheaply."""
        dom = ", ".join(self.business_domain) if self.business_domain else "general"
        measures = ", ".join(self.measures[:8]) or "-"
        dims = ", ".join(self.dimensions[:8]) or "-"
        desc = (self.description or "").strip().replace("\n", " ")[:160]
        return (
            f"- {self.table_name} [{dom}] rows={self.row_count}: {desc}\n"
            f"    measures: {measures}\n    dimensions: {dims}"
        )

    def date_coverage(self) -> list[str]:
        """Real min..max coverage for each temporal column, for grounding."""
        out: list[str] = []
        temporal = set(self.temporal)
        for c in self.columns:
            if c.name in temporal and c.min_value is not None and c.max_value is not None:
                out.append(f"{c.name}: {c.min_value} .. {c.max_value}")
        return out

    def detailed_summary(self, *, max_cols: int = 24) -> str:
        """
        A grounded, multi-line summary for prompt decomposition. Unlike
        compact_summary it lists the REAL column names with role/dtype, the
        actual date coverage of temporal columns, and the observed values of
        low-cardinality dimensions — so the planner stops inventing columns
        (e.g. OrderStatus values) and unsatisfiable date windows.
        """
        dom = ", ".join(self.business_domain) if self.business_domain else "general"
        desc = (self.description or "").strip().replace("\n", " ")[:180]
        lines = [f"- {self.table_name} [{dom}] rows={self.row_count}: {desc}"]

        col_bits: list[str] = []
        for c in self.columns[:max_cols]:
            tag = c.role_kind if c.role_kind != "unknown" else c.data_type
            col_bits.append(f"{c.name}({tag})")
        if col_bits:
            lines.append("    columns: " + ", ".join(col_bits))

        coverage = self.date_coverage()
        if coverage:
            lines.append("    date coverage: " + "; ".join(coverage))

        # Surface real categorical values so the planner never fabricates a
        # status/category literal. Only for low-cardinality dimensions.
        val_bits: list[str] = []
        dims = set(self.dimensions)
        for c in self.columns:
            if c.name in dims and c.top_values and (c.cardinality or 99) <= 20:
                vals = ", ".join(str(v) for v in c.top_values[:8])
                val_bits.append(f"{c.name} ∈ {{{vals}}}")
        if val_bits:
            lines.append("    values: " + " | ".join(val_bits[:6]))

        return "\n".join(lines)


def _build_table(
    file: File,
    meta: FileMetadata | None,
    analytics: FileAnalytics | None,
) -> DataCatalogTable:
    roles: dict = (meta.column_semantic_roles or {}) if meta else {}
    columns_info = (meta.columns_info or []) if meta else []
    stats: dict = (analytics.column_stats or {}) if analytics else {}
    value_counts: dict = (analytics.value_counts or {}) if analytics else {}
    row_count = (analytics.row_count if analytics else None) or (meta.row_count if meta else 0) or 0

    columns: list[DataCatalogColumn] = []
    measures: list[str] = []
    dimensions: list[str] = []
    temporal: list[str] = []
    join_keys: list[str] = []

    for col in columns_info:
        name = col.get("name") if isinstance(col, dict) else str(col)
        if not name:
            continue
        dtype = (col.get("type") if isinstance(col, dict) else None) or "string"
        role = roles.get(name)
        kind = _role_kind(role)
        cstat = stats.get(name, {}) if isinstance(stats, dict) else {}
        cardinality = cstat.get("unique") if isinstance(cstat, dict) else None
        nulls = cstat.get("nulls") if isinstance(cstat, dict) else None
        null_ratio = (nulls / row_count) if (nulls is not None and row_count) else None

        # Observed categorical values for this column, ranked by frequency.
        vc = value_counts.get(name) if isinstance(value_counts, dict) else None
        top_values = list(vc.keys())[:12] if isinstance(vc, dict) else []

        columns.append(
            DataCatalogColumn(
                name=name,
                data_type=str(dtype),
                semantic_role=role,
                role_kind=kind,
                cardinality=cardinality,
                null_ratio=round(null_ratio, 4) if null_ratio is not None else None,
                sample_values=(cstat.get("sample") or [])[:5] if isinstance(cstat, dict) else [],
                # column_stats only carries min/max for NUMERIC columns; date
                # columns get their min/max from the parquet profile written into
                # columns_info (col.min/col.max). Fall back to that so temporal
                # coverage (the out-of-range-window guard) is actually populated.
                min_value=_first_present(
                    cstat.get("min") if isinstance(cstat, dict) else None,
                    col.get("min") if isinstance(col, dict) else None,
                ),
                max_value=_first_present(
                    cstat.get("max") if isinstance(cstat, dict) else None,
                    col.get("max") if isinstance(col, dict) else None,
                ),
                top_values=top_values,
            )
        )
        if kind == "measure":
            measures.append(name)
        elif kind == "date":
            temporal.append(name)
        elif kind == "key":
            join_keys.append(name)
        elif kind == "dimension":
            dimensions.append(name)

    # Fall back to ingestion-derived metric/dimension hints when roles are sparse.
    if meta:
        for m in (meta.key_metrics or []):
            if m not in measures:
                measures.append(m)
        for d in (meta.key_dimensions or []):
            if d not in dimensions and d not in temporal:
                dimensions.append(d)

    data_quality = {
        "ingestion_confidence_score": (meta.ingestion_confidence_score if meta else None),
        "quarantine_count": (analytics.quarantine_count if analytics else 0),
        "column_count": (analytics.column_count if analytics else len(columns)),
    }

    return DataCatalogTable(
        file_id=file.id,
        table_name=file.name,
        blob_path=file.blob_path,
        parquet_path=(analytics.parquet_blob_path if analytics else None),
        description=(meta.ai_description if meta else None),
        business_domain=list((meta.good_for or []) if meta else []),
        row_count=row_count,
        columns=columns,
        measures=measures,
        dimensions=dimensions,
        temporal=temporal,
        join_keys=join_keys,
        data_quality=data_quality,
    )


async def build_catalog(
    container_id: str | None,
    db: AsyncSession,
    *,
    allowed_domains: list[str] | None = None,
    file_ids: list[str] | None = None,
    folder_id: str | None = None,
    limit: int = 400,
) -> list[DataCatalogTable]:
    """
    Build the scoped data catalog view. Single indexed pass over
    File + FileMetadata + FileAnalytics; respects tenancy + domain filtering.

    `folder_id` scopes the catalog to files in one domain folder — the same
    folder-membership filter retrieval uses (filters.build_base_query). This is
    how the dashboard domain picker narrows planning to a single domain.
    """
    q = (
        select(File, FileMetadata, FileAnalytics)
        .outerjoin(FileMetadata, FileMetadata.file_id == File.id)
        .outerjoin(FileAnalytics, FileAnalytics.file_id == File.id)
        .where(File.ingest_status.in_(_READY_STATUSES))
    )
    if container_id:
        q = q.where(File.container_id == container_id)
    if file_ids:
        q = q.where(File.id.in_(file_ids))
    if folder_id:
        q = q.where(File.folder_id == folder_id)
    q = q.limit(limit)

    rows = (await db.execute(q)).all()

    tables: list[DataCatalogTable] = []
    domain_filter = set(allowed_domains or [])
    for file, meta, analytics in rows:
        table = _build_table(file, meta, analytics)
        # Domain RBAC: if the user is domain-restricted, only surface tables
        # whose domains intersect. Tables with no domain tag are visible to all.
        if domain_filter and table.business_domain:
            if not (set(table.business_domain) & domain_filter):
                continue
        tables.append(table)
    return tables


async def build_relationship_map(
    file_ids: list[str],
    db: AsyncSession,
    *,
    min_confidence: float = 0.5,
) -> list[dict]:
    """
    Load validated cross-file joins (FileRelationship) scoped to the catalog's
    files. This is the real join graph ingestion produced — NOT the per-file
    name-matched key heuristic. The decomposer uses it to keep widgets within
    genuinely joinable tables instead of fabricating cross-domain joins.
    """
    if not file_ids:
        return []
    id_set = set(file_ids)
    q = select(FileRelationship).where(
        FileRelationship.file_a_id.in_(file_ids),
        FileRelationship.file_b_id.in_(file_ids),
    )
    try:
        rels = (await db.execute(q)).scalars().all()
    except Exception:
        return []

    out: list[dict] = []
    for r in rels:
        if r.file_a_id not in id_set or r.file_b_id not in id_set:
            continue
        conf = getattr(r, "confidence_score", None)
        if conf is not None and conf < min_confidence:
            continue
        out.append(
            {
                "file_a_id": r.file_a_id,
                "file_b_id": r.file_b_id,
                "shared_column": getattr(r, "shared_column", None),
                "related_column": getattr(r, "related_column", None)
                or getattr(r, "shared_column", None),
                "confidence": conf,
                "join_type": getattr(r, "join_type", None),
                # P0b: forward (do NOT consume) the fan-out/cardinality provenance the
                # Phase-2 join gate needs. Missing fields stay None — never coerce a
                # missing signal to a "safe to join" default; the gate fails closed.
                # edge_provenance is forwarded opaquely (its sub-keys are the gate's
                # concern, not the catalog loader's).
                "value_overlap_pct": getattr(r, "value_overlap_pct", None),
                "evidence_count": getattr(r, "evidence_count", None),
                "edge_provenance": getattr(r, "edge_provenance", None),
                "role_source": getattr(r, "role_source", None),
                "semantic_role": getattr(r, "semantic_role", None),
            }
        )
    return out


def role_map_for_table(table: "DataCatalogTable | None") -> dict:
    """{column_name -> semantic_role} for a catalog table, used to drive the
    recommender's data-driven format/binding. Columns without a role are dropped
    (so a missing role fails closed to name-based formatting downstream)."""
    if table is None:
        return {}
    return {c.name: c.semantic_role for c in getattr(table, "columns", []) if c.semantic_role}


def _render_join_section(tables: list[DataCatalogTable], relationships: list[dict]) -> str:
    """Render a human/LLM-readable 'Known joins' block from the relationship map."""
    if not relationships:
        return ""
    name_by_id = {t.file_id: t.table_name for t in tables}
    lines: list[str] = ["", "KNOWN JOINS (use ONLY these to span tables; do not invent others):"]
    seen: set = set()
    for r in relationships:
        # P2 JOIN GATE: only advertise relationship-validated, non-fan-out joins
        # (cardinality not many-to-many AND value overlap above the referential
        # floor). Fail-closed: unproven/legacy edges are NOT offered to the agent.
        if not safe_join(r):
            continue
        a = name_by_id.get(r["file_a_id"])
        b = name_by_id.get(r["file_b_id"])
        if not a or not b:
            continue
        sa = r.get("shared_column") or "?"
        sb = r.get("related_column") or sa
        key = tuple(sorted([f"{a}.{sa}", f"{b}.{sb}"]))
        if key in seen:
            continue
        seen.add(key)
        conf = r.get("confidence")
        conf_txt = f" [conf {conf:.2f}]" if isinstance(conf, (int, float)) else ""
        lines.append(f"  {a}.{sa} = {b}.{sb}{conf_txt}")
    return "\n".join(lines) if len(lines) > 2 else ""


def catalog_grounding_text(
    tables: list[DataCatalogTable],
    max_tables: int = 60,
    *,
    detailed: bool = False,
    relationships: list[dict] | None = None,
) -> str:
    """
    Token-bounded catalog summary for LLM prompt decomposition.

    detailed=True surfaces real column names, date coverage, observed
    categorical values, and the validated join map — the grounding the
    decomposer needs to stop hallucinating columns/joins/date windows.
    """
    render = (lambda t: t.detailed_summary()) if detailed else (lambda t: t.compact_summary())
    chunks = [render(t) for t in tables[:max_tables]]
    extra = len(tables) - max_tables
    if extra > 0:
        chunks.append(f"... and {extra} more tables")
    body = "\n".join(chunks)
    if detailed and relationships:
        body += "\n" + _render_join_section(tables[:max_tables], relationships)
    return body

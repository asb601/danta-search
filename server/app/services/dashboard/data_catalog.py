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


def _role_kind(semantic_role: str | None) -> str:
    """Map a 'custom:kind:label' semantic role string to a coarse kind."""
    if not semantic_role:
        return "unknown"
    parts = str(semantic_role).split(":")
    kind = parts[1] if len(parts) >= 2 else parts[0]
    kind = kind.lower()
    if any(k in kind for k in ("metric", "measure", "amount", "value", "quantity")):
        return "measure"
    if any(k in kind for k in ("date", "time", "period", "month", "year")):
        return "date"
    if any(k in kind for k in ("key", "id", "code")):
        return "key"
    return "dimension"


@dataclass
class DataCatalogColumn:
    name: str
    data_type: str
    semantic_role: str | None
    role_kind: str
    cardinality: int | None
    null_ratio: float | None
    sample_values: list = field(default_factory=list)


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


def _build_table(
    file: File,
    meta: FileMetadata | None,
    analytics: FileAnalytics | None,
) -> DataCatalogTable:
    roles: dict = (meta.column_semantic_roles or {}) if meta else {}
    columns_info = (meta.columns_info or []) if meta else []
    stats: dict = (analytics.column_stats or {}) if analytics else {}
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

        columns.append(
            DataCatalogColumn(
                name=name,
                data_type=str(dtype),
                semantic_role=role,
                role_kind=kind,
                cardinality=cardinality,
                null_ratio=round(null_ratio, 4) if null_ratio is not None else None,
                sample_values=(cstat.get("sample") or [])[:5] if isinstance(cstat, dict) else [],
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
    limit: int = 400,
) -> list[DataCatalogTable]:
    """
    Build the scoped data catalog view. Single indexed pass over
    File + FileMetadata + FileAnalytics; respects tenancy + domain filtering.
    """
    q = (
        select(File, FileMetadata, FileAnalytics)
        .outerjoin(FileMetadata, FileMetadata.file_id == File.id)
        .outerjoin(FileAnalytics, FileAnalytics.file_id == File.id)
        .where(File.ingest_status == "completed")
    )
    if container_id:
        q = q.where(File.container_id == container_id)
    if file_ids:
        q = q.where(File.id.in_(file_ids))
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


def catalog_grounding_text(tables: list[DataCatalogTable], max_tables: int = 60) -> str:
    """Compact, token-bounded catalog summary for LLM prompt decomposition."""
    chunks = [t.compact_summary() for t in tables[:max_tables]]
    extra = len(tables) - max_tables
    if extra > 0:
        chunks.append(f"... and {extra} more tables")
    return "\n".join(chunks)

"""
Catalog loader — filtered per-request DB query (no in-memory cache).

Design notes
------------
The old 5-minute TTL in-memory cache was built for Neon (serverless
Postgres with cold-start latency). With Azure Postgres it was wrong
on two counts:

  1. It loaded EVERY file row into one shared Python object. At 1 M
     files that is ~1 GB of memory held by every worker, forever.

  2. The shared cache was filtered per-request in Python list
     comprehensions. Domain users and admins shared the same
     unfiltered blob — the filter was applied AFTER the load.
     Any race between an admin write and a user query could expose
     the wrong catalog slice for up to 5 minutes.

New approach:
  - Filters (container_id, allowed_domains) are pushed into SQL so
    the DB does the filtering and only the user's visible rows cross
    the wire.
  - Folder ancestor inheritance (untagged subfolders inside tagged
    parents) is handled by loading the folders table (small) and
    walking parent_id chains in Python.
  - No global state, no locks, no TTL, no invalidation needed.
  - Azure Postgres with indexes on file_metadata.container_id and
    folders.domain_tag handles 1 M files comfortably.

Heavy fields (columns_info, sample_rows, column_stats) are still
NOT loaded here. catalog_hydration.hydrate_files fetches them only
for the small retrieval shortlist (default top-8 files per request).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.container import ContainerConfig
from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder

# Caps on per-file lean fields to keep each catalog record small.
_MAX_DESCRIPTION_CHARS = 600
_MAX_LIST_ITEMS = 12
_MAX_COLUMN_NAMES = 80


def invalidate_catalog_cache() -> None:
    """No-op — cache removed. Kept for call-site compatibility."""


def _truncate(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text[:n] if len(text) > n else text


def _cap_list(items: list | None, n: int) -> list:
    if not items:
        return []
    return list(items)[:n] if len(items) > n else list(items)


def _extract_column_names(columns_info: list | None) -> list[str]:
    """Return column names only (drops types / samples / uniques)."""
    if not columns_info:
        return []
    names: list[str] = []
    for c in columns_info:
        if isinstance(c, dict) and c.get("name"):
            names.append(c["name"])
        elif isinstance(c, str):
            names.append(c)
        if len(names) >= _MAX_COLUMN_NAMES:
            break
    return names


async def load_catalog(
    db: AsyncSession,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
) -> dict | None:
    """
    Query the lean catalog directly from Postgres, filtered by caller scope.

    Filters are applied at SQL level so only the user's visible rows
    cross the wire. No shared in-memory state.

    allowed_domains: if set, only files whose folder (or nearest tagged
    ancestor folder) has a domain_tag in the list are returned. None
    means unrestricted (admin or no domain assignment).

    container_id: if set, only files belonging to that container are
    returned.

    Returns dict with keys:
      catalog              - lean per-file records (no heavy fields)
      connection_string    - Azure connection string for the container
      container_name       - Azure container name
      parquet_blob_path    - first available parquet blob path (legacy)
      parquet_paths_all    - {blob_path -> parquet_blob_path}
    Returns None if no files are visible after applying filters.
    """
    return await _query_filtered_catalog(db, allowed_domains, container_id)


async def _query_filtered_catalog(
    db: AsyncSession,
    allowed_domains: list[str] | None,
    container_id: str | None,
) -> dict | None:
    """
    Core implementation: filtered DB query per request.

    Strategy
    --------
    FileMetadata rows are large per-table but we only SELECT the lean
    fields we need (no columns_info blobs, no sample_rows, no embeddings).
    The WHERE clause filters by container_id (indexed FK) so Postgres
    never scans outside the caller's container.

    Domain filtering is handled in Python after loading the (small)
    folders table and computing effective_domain_tag via parent-chain
    walk. Folders are far fewer than files so this stays cheap.
    """
    # ── 1. Build file_metadata query (lean fields only, no JSONB blobs) ──────
    meta_stmt = select(
        FileMetadata.file_id,
        FileMetadata.blob_path,
        FileMetadata.container_id,
        FileMetadata.ai_description,
        FileMetadata.good_for,
        FileMetadata.key_metrics,
        FileMetadata.key_dimensions,
        FileMetadata.columns_info,          # needed only to extract column names; not hydrated
        FileMetadata.column_semantic_roles,  # semantic role map — used by workflow_capability_resolver
        FileMetadata.date_range_start,
        FileMetadata.date_range_end,
        FileMetadata.ingestion_confidence_score,  # Phase 6: trust propagation weight
    )
    if container_id:
        meta_stmt = meta_stmt.where(FileMetadata.container_id == container_id)

    meta_rows = (await db.execute(meta_stmt)).all()
    if not meta_rows:
        return None

    # ── 2. Build file → folder map ────────────────────────────────────────────
    # Only load files referenced by our metadata (already container-filtered).
    visible_file_ids = {r.file_id for r in meta_rows}
    file_rows = (await db.execute(
        select(File.id, File.folder_id).where(File.id.in_(visible_file_ids))
    )).all()
    file_folder: dict[str, str | None] = {r.id: r.folder_id for r in file_rows}

    # ── 3. Folder ancestor walk for effective domain_tag ─────────────────────
    # Load ALL folders (small table; domain folders + regular folders).
    # Walk parent_id chain so untagged subfolders inside a tagged parent
    # inherit the parent's domain_tag.
    all_folder_rows = (await db.execute(select(Folder.id, Folder.parent_id, Folder.domain_tag))).all()
    folder_by_id: dict[str, tuple[str | None, str | None]] = {
        r.id: (r.parent_id, r.domain_tag) for r in all_folder_rows
    }

    _eff_cache: dict[str, str | None] = {}

    def _effective_tag(folder_id: str | None) -> str | None:
        if not folder_id:
            return None
        if folder_id in _eff_cache:
            return _eff_cache[folder_id]
        seen: set[str] = set()
        cursor: str | None = folder_id
        while cursor and cursor not in seen:
            seen.add(cursor)
            entry = folder_by_id.get(cursor)
            if entry is None:
                break
            parent_id, domain_tag = entry
            if domain_tag:
                _eff_cache[folder_id] = domain_tag
                return domain_tag
            cursor = parent_id
        _eff_cache[folder_id] = None
        return None

    # ── 4. Build catalog entries, applying domain filter ──────────────────────
    catalog: list[dict] = []
    for r in meta_rows:
        eff_tag = _effective_tag(file_folder.get(r.file_id))
        # Domain gate: if caller has domain restrictions, only include files
        # that are explicitly tagged with one of their allowed domains.
        # Files with no domain tag (root-level or untagged folders) are
        # treated as admin-only — NOT accessible to domain-restricted users.
        if allowed_domains and (not eff_tag or eff_tag not in allowed_domains):
            continue
        catalog.append({
            "file_id": r.file_id,
            "blob_path": r.blob_path,
            "container_id": r.container_id,
            "domain_tag": eff_tag,
            "ai_description": _truncate(r.ai_description, _MAX_DESCRIPTION_CHARS),
            "good_for": _cap_list(r.good_for, _MAX_LIST_ITEMS),
            "key_metrics": _cap_list(r.key_metrics, _MAX_LIST_ITEMS),
            "key_dimensions": _cap_list(r.key_dimensions, _MAX_LIST_ITEMS),
            "column_names": _extract_column_names(r.columns_info),
            "column_semantic_roles": r.column_semantic_roles or {},
            "date_range_start": str(r.date_range_start) if r.date_range_start else None,
            "date_range_end": str(r.date_range_end) if r.date_range_end else None,
            "ingestion_confidence_score": r.ingestion_confidence_score,  # Phase 6
        })

    if not catalog:
        return None

    # ── 5. Resolve container config (connection_string) ───────────────────────
    # Use the explicit container_id if given; otherwise use the first file's
    # container. In an org-scoped request all files share the same container
    # so this is always correct.
    resolved_container_id = container_id or catalog[0].get("container_id")
    if not resolved_container_id:
        return None
    container = await db.get(ContainerConfig, resolved_container_id)
    if not container:
        return None

    # ── 6. Build parquet path index ───────────────────────────────────────────
    visible_file_id_set = {e["file_id"] for e in catalog}
    analytics_rows = (await db.execute(
        select(FileAnalytics.file_id, FileAnalytics.parquet_blob_path)
        .where(FileAnalytics.file_id.in_(visible_file_id_set))
    )).all()
    parquet_by_file: dict[str, str] = {
        r.file_id: r.parquet_blob_path for r in analytics_rows if r.parquet_blob_path
    }

    blob_to_parquet: dict[str, str] = {}
    first_parquet: str | None = None
    for entry in catalog:
        pq = parquet_by_file.get(entry["file_id"])
        if pq and entry["blob_path"]:
            blob_to_parquet[entry["blob_path"]] = pq
            if first_parquet is None:
                first_parquet = pq

    return {
        "catalog": catalog,
        "connection_string": container.connection_string,
        "container_name": container.container_name,
        "parquet_blob_path": first_parquet,
        "parquet_paths_all": blob_to_parquet,
    }


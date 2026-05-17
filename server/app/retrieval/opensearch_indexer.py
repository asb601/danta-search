"""Index FileMetadata documents into OpenSearch."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.opensearch_client import index_file_metadata, opensearch_enabled
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder


def _list_text(values: list | None) -> str:
    if not values:
        return ""
    return " ".join(str(v) for v in values if v is not None)


def _column_names(columns_info: list | None) -> list[str]:
    names: list[str] = []
    for col in columns_info or []:
        if isinstance(col, dict) and col.get("name"):
            names.append(str(col["name"]))
        elif isinstance(col, str):
            names.append(col)
    return names[:200]


async def _effective_domain_tag(file: File, db: AsyncSession) -> str | None:
    folder_id = file.folder_id
    seen: set[str] = set()
    while folder_id and folder_id not in seen:
        seen.add(folder_id)
        folder = await db.get(Folder, folder_id)
        if not folder:
            return None
        if folder.domain_tag:
            return folder.domain_tag
        folder_id = folder.parent_id
    return None


async def index_metadata_document(metadata: FileMetadata, db: AsyncSession) -> None:
    """Index one metadata row into its per-container OpenSearch index."""
    if not opensearch_enabled() or not metadata.container_id:
        return

    file = await db.get(File, metadata.file_id)
    domain_tag = await _effective_domain_tag(file, db) if file else None
    column_names = _column_names(metadata.columns_info)
    good_for = metadata.good_for or []
    key_metrics = metadata.key_metrics or []
    key_dimensions = metadata.key_dimensions or []

    search_text = metadata.search_text or " ".join([
        metadata.ai_description or "",
        _list_text(good_for),
        _list_text(key_metrics),
        _list_text(key_dimensions),
        " ".join(column_names),
    ])

    document = {
        "file_id": metadata.file_id,
        "container_id": metadata.container_id,
        "blob_path": metadata.blob_path,
        "domain_tag": domain_tag,
        "ai_description": metadata.ai_description or "",
        "good_for": good_for,
        "key_metrics": key_metrics,
        "key_dimensions": key_dimensions,
        "column_names": column_names,
        "search_text": search_text,
        "date_range_start": metadata.date_range_start.isoformat() if metadata.date_range_start else None,
        "date_range_end": metadata.date_range_end.isoformat() if metadata.date_range_end else None,
    }
    if metadata.description_embedding:
        document["description_embedding"] = metadata.description_embedding

    await index_file_metadata(metadata.container_id, metadata.file_id, document)

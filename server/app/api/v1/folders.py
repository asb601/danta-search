import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import folder_logger, db_logger
from app.dependencies import get_current_user, require_admin, require_developer
from app.api.v1.files import _file_to_out
from app.models.file import File
from app.models.folder import Folder
from app.models.user import User
from app.schemas.folder import FolderContents, FolderCreate, FolderOut, FolderUpdate

router = APIRouter(prefix="/folders", tags=["folders"])


@router.get("/{folder_id}/contents")
async def get_folder_contents(
    folder_id: str,
    container_id: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get direct children (subfolders + files) of a folder. Use folder_id='root' for root.

    Optional container_id query param scopes results to a single container.
    A folder is shown when it (or any descendant) contains at least one file
    in that container, so legacy folders without container_id still appear.
    """
    start = time.perf_counter()
    folder_logger.info("contents_requested", folder_id=folder_id, container_id=container_id)

    is_root = folder_id == "root"

    # Members with no assigned domains see nothing — empty folder until a higher
    # role grants them at least one domain.
    if not user.is_admin and not user.allowed_domains:
        return FolderContents(folders=[], files=[])
    # Resolve effective domain restriction by walking ancestors. A subfolder
    # without its own domain_tag inherits the tag of its nearest tagged
    # ancestor. This prevents bypass via "untagged subfolder inside a tagged
    # parent"; access remains restricted to users with the inherited tag even
    # if the child folder itself has no tag.
    async def _effective_domain_tag(start_folder_id: str) -> str | None:
        cursor_id: str | None = start_folder_id
        seen: set[str] = set()
        while cursor_id and cursor_id not in seen:
            seen.add(cursor_id)
            f = await db.get(Folder, cursor_id)
            if f is None:
                return None
            if f.domain_tag:
                return f.domain_tag
            cursor_id = f.parent_id
        return None

    # Non-root folder access check: if user has domain restrictions, verify they
    # can access this specific folder (or any tagged ancestor) before listing
    # its contents.
    if not is_root and user.allowed_domains:
        eff_tag = await _effective_domain_tag(folder_id)
        if eff_tag and eff_tag not in user.allowed_domains:
            raise HTTPException(status_code=403, detail="Access to this folder is restricted")

    # Subfolders — org-wide, no owner filter
    folder_stmt = select(Folder).order_by(Folder.name)
    if is_root:
        folder_stmt = folder_stmt.where(Folder.parent_id.is_(None))
    else:
        folder_stmt = folder_stmt.where(Folder.parent_id == folder_id)
    if container_id:
        # Sync tags every folder it creates with container_id, so direct match works.
        folder_stmt = folder_stmt.where(Folder.container_id == container_id)
    # Domain scope: only show folders the user is allowed to see.
    # Rule: domain-restricted users see ONLY folders explicitly tagged with
    # one of their allowed domains.
    #   - At root: untagged folders are hidden.
    #   - Inside an authorised folder: untagged subfolders inherit the parent's
    #     domain and are shown (admin created them inside the domain folder).
    if user.allowed_domains:
        if is_root:
            # Root-level — must be explicitly tagged with an allowed domain.
            folder_stmt = folder_stmt.where(
                Folder.domain_tag.in_(user.allowed_domains) 
            )
        else:
            # Non-root — untagged subfolders inherit the parent's domain (allowed)
            # AND explicitly tagged subfolders must be in allowed domains.
            folder_stmt = folder_stmt.where(
                or_(Folder.domain_tag.is_(None), Folder.domain_tag.in_(user.allowed_domains))
            )

    db_start = time.perf_counter()
    db_logger.info("query_started", query="select_subfolders", folder_id=folder_id)
    folders_result = await db.execute(folder_stmt)
    folders = list(folders_result.scalars().all())
    db_logger.info("query_complete", query="select_subfolders", count=len(folders), duration_ms=round((time.perf_counter() - db_start) * 1000, 2))

    # Files — org-wide, eager load uploaded_by user
    from sqlalchemy.orm import selectinload
    file_stmt = select(File).options(selectinload(File.uploaded_by)).order_by(File.name)
    if container_id:
        # When filtering by container, ignore folder hierarchy at root and show
        # all files in the container. For non-root folders, still scope to the folder.
        file_stmt = file_stmt.where(File.container_id == container_id)
        if not is_root:
            file_stmt = file_stmt.where(File.folder_id == folder_id)
    else:
        if is_root:
            file_stmt = file_stmt.where(File.folder_id.is_(None))
        else:
            file_stmt = file_stmt.where(File.folder_id == folder_id)
    # Domain scope: only show files in folders the user is allowed to access.
    # Rule: domain-restricted users NEVER see root-level files (no domain tag)
    # and NEVER see files in untagged non-root folders at root browsing level.
    if user.allowed_domains:
        if is_root:
            # Domain users should not see any root-level (no-folder) files.
            # Root files have no domain tag — they are admin/system artefacts.
            from sqlalchemy import false as _false
            file_stmt = file_stmt.where(_false())
        else:
            # Inside an authorised folder: files in untagged subfolders inherit
            # the parent's domain (allowed); files in explicitly tagged subfolders
            # must match the user's allowed domains.
            file_stmt = (
                file_stmt
                .outerjoin(Folder, File.folder_id == Folder.id)
                .where(
                    or_(
                        Folder.domain_tag.is_(None),        # untagged subfolder inherits parent domain
                        Folder.domain_tag.in_(user.allowed_domains),
                    )
                )
            )

    db_start = time.perf_counter()
    db_logger.info("query_started", query="select_files", folder_id=folder_id)
    files_result = await db.execute(file_stmt)
    files = list(files_result.scalars().all())
    db_logger.info("query_complete", query="select_files", count=len(files), duration_ms=round((time.perf_counter() - db_start) * 1000, 2))

    folder_logger.info("contents_fetched", folder_id=folder_id, folders_count=len(folders), files_count=len(files), duration_ms=round((time.perf_counter() - start) * 1000, 2))

    from app.schemas.folder import FolderOut
    return {
        "folders": [FolderOut.model_validate(f) for f in folders],
        "files": [_file_to_out(f) for f in files],
    }


@router.post("", response_model=FolderOut, status_code=status.HTTP_201_CREATED)
async def create_folder(
    body: FolderCreate,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    folder_logger.info("create_started", name=body.name, parent_id=body.parent_id)

    folder = Folder(name=body.name, parent_id=body.parent_id, owner_id=admin.id)
    db.add(folder)
    await db.commit()
    await db.refresh(folder)

    folder_logger.info("create_complete", folder_id=folder.id, duration_ms=round((time.perf_counter() - start) * 1000, 2))
    return folder


@router.patch("/{folder_id}", response_model=FolderOut)
async def update_folder(
    folder_id: str,
    body: FolderUpdate,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    folder_logger.info("update_started", folder_id=folder_id)

    result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    if body.name is not None:
        folder.name = body.name
    if body.parent_id is not None:
        folder.parent_id = body.parent_id

    await db.commit()
    await db.refresh(folder)

    folder_logger.info("update_complete", folder_id=folder_id, duration_ms=round((time.perf_counter() - start) * 1000, 2))
    return folder


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: str,
    admin: User = Depends(require_developer),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    folder_logger.info("delete_started", folder_id=folder_id)

    result = await db.execute(select(Folder).where(Folder.id == folder_id))
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    await db.delete(folder)
    await db.commit()

    folder_logger.info("delete_complete", folder_id=folder_id, duration_ms=round((time.perf_counter() - start) * 1000, 2))

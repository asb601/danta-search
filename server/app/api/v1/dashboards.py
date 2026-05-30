"""
Dashboards API (/api/dashboards).

CRUD + folder organization + the natural-language GENERATION route. Generation
reuses the existing agent runtime through the dashboard query engine; this
router contains NO query logic. Mirrors the conversations router's conventions
(ownership checks, scoped lists) and the chat RBAC scope resolution.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat_common import resolve_chat_scope
from app.core.logger import chat_logger
from app.dependencies import get_current_user, get_db
from app.models.dashboard import Dashboard, DashboardFolder
from app.models.user import User
from app.schemas.dashboard import (
    DashboardCreate,
    DashboardFolderCreate,
    DashboardFolderUpdate,
    DashboardGenerateRequest,
    DashboardUpdate,
)
from app.services.dashboard import assembly_engine, data_catalog, query_engine
from app.services.dashboard.component_catalog import catalog_as_metadata
from app.services.dashboard.recommendation_engine import recommend

router = APIRouter()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dashboard_summary(d: Dashboard) -> dict:
    widgets = (d.config or {}).get("widgets", []) if isinstance(d.config, dict) else []
    return {
        "id": d.id,
        "title": d.title,
        "description": d.description,
        "folder_id": d.folder_id,
        "is_pinned": d.is_pinned,
        "status": d.status,
        "widget_count": len(widgets),
        "created_at": _iso(d.created_at),
        "updated_at": _iso(d.updated_at),
    }


def _dashboard_out(d: Dashboard) -> dict:
    return {
        "id": d.id,
        "title": d.title,
        "description": d.description,
        "folder_id": d.folder_id,
        "container_id": d.container_id,
        "is_pinned": d.is_pinned,
        "status": d.status,
        "config": d.config or {},
        "prompt_history": d.prompt_history or [],
        "source_file_ids": d.source_file_ids or [],
        "created_at": _iso(d.created_at),
        "updated_at": _iso(d.updated_at),
    }


# ==========================================================================
# Catalog endpoints
# ==========================================================================

@router.get("/dashboards/catalog/components")
async def get_component_catalog(user: User = Depends(get_current_user)):
    """The metadata-driven component catalog (drives the frontend Analytics Catalog)."""
    return {"components": catalog_as_metadata()}


@router.get("/dashboards/catalog/data")
async def get_data_catalog(
    container_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Scoped data catalog projection over existing metadata tables."""
    effective_container_id, allowed_domains = await resolve_chat_scope(user, container_id, db)
    tables = await data_catalog.build_catalog(
        effective_container_id, db, allowed_domains=allowed_domains
    )
    return {"tables": [t.as_dict() for t in tables], "count": len(tables)}


# ==========================================================================
# Folders
# ==========================================================================

@router.get("/dashboards/folders")
async def list_dashboard_folders(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = (
        select(DashboardFolder)
        .where(DashboardFolder.owner_id == user.id)
        .order_by(DashboardFolder.created_at.asc())
    )
    rows = list((await db.execute(q)).scalars().all())
    return {
        "folders": [
            {
                "id": f.id,
                "name": f.name,
                "parent_id": f.parent_id,
                "container_id": f.container_id,
                "created_at": _iso(f.created_at),
            }
            for f in rows
        ]
    }


@router.post("/dashboards/folders", status_code=201)
async def create_dashboard_folder(
    body: DashboardFolderCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    effective_container_id, _ = await resolve_chat_scope(user, body.container_id, db)
    folder = DashboardFolder(
        name=(body.name or "New folder").strip()[:255] or "New folder",
        parent_id=body.parent_id,
        owner_id=user.id,
        container_id=effective_container_id,
    )
    db.add(folder)
    await db.commit()
    return {
        "id": folder.id,
        "name": folder.name,
        "parent_id": folder.parent_id,
        "container_id": folder.container_id,
        "created_at": _iso(folder.created_at),
    }


@router.patch("/dashboards/folders/{folder_id}")
async def update_dashboard_folder(
    folder_id: str,
    body: DashboardFolderUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    folder = await db.get(DashboardFolder, folder_id)
    if not folder or folder.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Folder not found.")
    if body.name is not None:
        name = body.name.strip()
        if name:
            folder.name = name[:255]
    if body.parent_id is not None:
        folder.parent_id = body.parent_id or None
    await db.commit()
    return {"id": folder.id, "name": folder.name, "parent_id": folder.parent_id}


@router.delete("/dashboards/folders/{folder_id}")
async def delete_dashboard_folder(
    folder_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    folder = await db.get(DashboardFolder, folder_id)
    if not folder or folder.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Folder not found.")
    # Detach dashboards (do not delete them) before removing the folder.
    rows = (
        await db.execute(select(Dashboard).where(Dashboard.folder_id == folder_id))
    ).scalars().all()
    for d in rows:
        d.folder_id = None
    await db.delete(folder)
    await db.commit()
    return {"deleted": True}


# ==========================================================================
# Dashboards CRUD
# ==========================================================================

@router.get("/dashboards")
async def list_dashboards(
    folder_id: str | None = Query(default=None),
    pinned: bool | None = Query(default=None),
    search: str = Query(default="", max_length=200),
    limit: int = Query(default=100, ge=1, le=300),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    filters = [Dashboard.owner_id == user.id]
    if folder_id == "__none__":
        filters.append(Dashboard.folder_id.is_(None))
    elif folder_id:
        filters.append(Dashboard.folder_id == folder_id)
    if pinned is not None:
        filters.append(Dashboard.is_pinned.is_(pinned))
    term = search.strip()
    if term:
        filters.append(Dashboard.title.ilike(f"%{term}%"))

    q = (
        select(Dashboard)
        .where(*filters)
        .order_by(Dashboard.is_pinned.desc(), Dashboard.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = list((await db.execute(q)).scalars().all())
    total = (await db.execute(select(func.count(Dashboard.id)).where(*filters))).scalar()
    return {"dashboards": [_dashboard_summary(d) for d in rows], "total": total}


@router.post("/dashboards", status_code=201)
async def create_dashboard(
    body: DashboardCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    effective_container_id, _ = await resolve_chat_scope(user, body.container_id, db)
    dashboard = Dashboard(
        title=(body.title or "Untitled dashboard").strip()[:255] or "Untitled dashboard",
        description=body.description,
        folder_id=body.folder_id,
        owner_id=user.id,
        container_id=effective_container_id,
        config={},
        status="draft",
    )
    db.add(dashboard)
    await db.commit()
    return _dashboard_out(dashboard)


@router.get("/dashboards/{dashboard_id}")
async def get_dashboard(
    dashboard_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    d = await db.get(Dashboard, dashboard_id)
    if not d or d.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return _dashboard_out(d)


@router.patch("/dashboards/{dashboard_id}")
async def update_dashboard(
    dashboard_id: str,
    body: DashboardUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    d = await db.get(Dashboard, dashboard_id)
    if not d or d.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    if body.title is not None:
        title = body.title.strip()
        if title:
            d.title = title[:255]
    if body.description is not None:
        d.description = body.description
    if body.folder_id is not None:
        d.folder_id = body.folder_id or None
    if body.is_pinned is not None:
        d.is_pinned = body.is_pinned
    await db.commit()
    return _dashboard_out(d)


@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    d = await db.get(Dashboard, dashboard_id)
    if not d or d.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    await db.delete(d)
    await db.commit()
    return {"deleted": True}


@router.post("/dashboards/{dashboard_id}/duplicate", status_code=201)
async def duplicate_dashboard(
    dashboard_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    src = await db.get(Dashboard, dashboard_id)
    if not src or src.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    clone = Dashboard(
        title=f"{src.title} (copy)"[:255],
        description=src.description,
        folder_id=src.folder_id,
        owner_id=user.id,
        container_id=src.container_id,
        config=src.config or {},
        prompt_history=list(src.prompt_history or []),
        source_file_ids=list(src.source_file_ids or []),
        status=src.status,
    )
    db.add(clone)
    await db.commit()
    return _dashboard_out(clone)


# ==========================================================================
# Generation route — the core flow (response.txt Section 9.3)
# ==========================================================================

@router.post("/dashboards/{dashboard_id}/generate")
async def generate_dashboard(
    dashboard_id: str,
    body: DashboardGenerateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    d = await db.get(Dashboard, dashboard_id)
    if not d or d.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Dashboard not found.")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    trace_id = f"dash-{uuid.uuid4().hex[:12]}"
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id, pipeline="dashboard")

    effective_container_id, allowed_domains = await resolve_chat_scope(
        user, body.container_id or d.container_id, db
    )

    scope = {
        "user_id": user.id,
        "is_admin": bool(getattr(user, "is_admin", False)),
        "allowed_domains": allowed_domains,
        "container_id": effective_container_id,
        "actor_email": getattr(user, "email", "") or "",
        "actor_role": getattr(user, "role", "") or "",
    }

    chat_logger.info("dashboard_generate_start", dashboard_id=d.id, prompt=prompt[:200])

    # 1. Ground decomposition in the scoped data catalog.
    try:
        catalog = await data_catalog.build_catalog(
            effective_container_id, db, allowed_domains=allowed_domains
        )
    except Exception as exc:
        chat_logger.warning("dashboard_catalog_error", error=str(exc)[:200])
        catalog = []
    grounding = data_catalog.catalog_grounding_text(catalog)

    # 2. Decompose the prompt into widget intents.
    intents = await query_engine.decompose_prompt(
        prompt, grounding, max_widgets=body.max_widgets
    )

    warnings: list[str] = []
    if len(intents) >= body.max_widgets:
        warnings.append(f"Widget count capped at {body.max_widgets}.")

    # 3-5. Run each widget through the existing agent (SEQUENTIAL — the async DB
    #      session is not concurrency-safe), profile, recommend.
    resolved = []
    source_file_ids: set[str] = set()
    widget_ids: list[str] = []
    for intent in intents:
        result = await query_engine.run_widget(intent, db=db, scope=scope)
        rows = result.get("data") or []
        shape = query_engine.profile_dataset(rows, result.get("chart"))
        provenance = {
            "files_used": result.get("files_used") or [],
            "row_count": result.get("row_count", len(rows)),
            "route": result.get("route", "agent"),
            "answer": result.get("answer", ""),
            "query": intent.nl_query,
        }
        for f in provenance["files_used"]:
            source_file_ids.add(str(f))
        widget = recommend(shape, intent, rows, provenance=provenance)
        widget_ids.append(widget.widget_id)
        resolved.append(widget)

    # 6. Assemble the dashboard config.
    generated_at = datetime.now(timezone.utc).isoformat()
    config = assembly_engine.assemble(
        resolved,
        title=d.title,
        description=d.description,
        prompt=prompt,
        generated_at=generated_at,
        warnings=warnings,
    )

    # 7. Persist (merge if append).
    if body.append and isinstance(d.config, dict) and d.config.get("widgets"):
        merged = dict(d.config)
        merged_widgets = list(merged.get("widgets", [])) + config["widgets"]
        merged["widgets"] = merged_widgets
        merged["warnings"] = list(merged.get("warnings", [])) + warnings
        merged["generated_at"] = generated_at
        d.config = merged
    else:
        d.config = config

    history = list(d.prompt_history or [])
    history.append({"prompt": prompt, "created_at": generated_at, "widget_ids": widget_ids})
    d.prompt_history = history
    d.source_file_ids = sorted(set(list(d.source_file_ids or [])) | source_file_ids)
    d.status = "ready"
    d.updated_at = datetime.now(timezone.utc)
    await db.commit()

    # Optional audit (non-fatal).
    try:
        from app.services.audit_log import record_audit_event_safe

        await record_audit_event_safe(
            actor=user,
            action="dashboard.generate",
            event_type="action",
            status_code=200,
            path=f"/api/dashboards/{d.id}/generate",
            route_template="/api/dashboards/{dashboard_id}/generate",
            container_id=effective_container_id,
            details={"dashboard_id": d.id, "prompt_preview": prompt[:500],
                     "widget_count": len(resolved)},
        )
        await db.commit()
    except Exception:
        pass

    chat_logger.info(
        "dashboard_generate_done", dashboard_id=d.id, widgets=len(resolved)
    )
    return _dashboard_out(d)

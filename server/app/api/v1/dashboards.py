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
from app.core.config import get_settings
from app.core.database import async_session
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
from app.services.dashboard import assembly_engine, board_planner, data_catalog, join_gate, query_engine
from app.services.dashboard.empty_state import classify_empty
from app.services.dashboard.insight import compute_insight
from app.services.semantic_roles import is_additive_measure_role
from app.services.dashboard.component_catalog import catalog_as_metadata
from app.services.dashboard.recommendation_engine import build_pinned_spec, recommend

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

    # RC-002 observability: the catalog size is the single best health signal for
    # the whole dashboard pipeline (0 ⇒ every downstream stage runs blind). Emit it
    # at warning level when empty so it is visible in error.log without log diving.
    if catalog:
        chat_logger.info(
            "dashboard_catalog_built",
            container_id=effective_container_id, table_count=len(catalog),
        )
    else:
        chat_logger.warning(
            "dashboard_catalog_empty",
            container_id=effective_container_id,
            detail="no dashboard-ready files resolved (check ingest_status / domain scope)",
        )

    # Load the validated join graph for the catalog's files so the decomposer
    # keeps widgets within genuinely joinable tables (no fabricated joins).
    relationships: list[dict] = []
    try:
        relationships = await data_catalog.build_relationship_map(
            [t.file_id for t in catalog], db
        )
    except Exception as exc:
        chat_logger.warning("dashboard_relationship_error", error=str(exc)[:200])

    grounding = data_catalog.catalog_grounding_text(
        catalog, detailed=True, relationships=relationships
    )

    # 2. BOARD PLANNER — design the dashboard as a metric lattice, dry-run each
    #    widget against catalog metadata (drop/repair the unanswerable ones), and
    #    derive a shared time window — BEFORE any agent call. Never raises; falls
    #    back to single-pass decomposition internally.
    intents, warnings = await board_planner.plan_widgets(
        prompt, catalog, grounding_text=grounding, max_widgets=body.max_widgets
    )

    if len(intents) >= body.max_widgets:
        warnings.append(f"Widget count capped at {body.max_widgets}.")

    # 3-5. Run each widget through the existing agent, then profile + recommend.
    #      P3: agent calls run CONCURRENTLY when DASHBOARD_PARALLEL_WIDGETS is on,
    #      each on its OWN AsyncSession (the request `db` is NOT concurrency-safe and
    #      is reserved for the pre-loop reads + final commit). ALL post-processing
    #      below stays SEQUENTIAL in input order -> output byte-identical to the
    #      flag-off path (only widget_id/generated_at differ, as they always do).
    _settings = get_settings()

    async def _run_one(intent):
        async with async_session() as ws:
            return await query_engine.run_widget(intent, db=ws, scope=scope, catalog=catalog)

    results = await query_engine.run_widgets(
        intents,
        _run_one,
        concurrency=_settings.DASHBOARD_WIDGET_CONCURRENCY,
        parallel=_settings.DASHBOARD_PARALLEL_WIDGETS,
    )

    resolved = []
    source_file_ids: set[str] = set()
    widget_ids: list[str] = []
    empty_titles: list[str] = []
    for intent, result in zip(intents, results):
        if isinstance(result, Exception):
            # A per-widget task failed (e.g. session acquisition); degrade only this
            # widget to the same safe-empty shape run_widget returns on error.
            chat_logger.warning(
                "dashboard_widget_task_error", title=intent.title, error=str(result)[:200]
            )
            result = {"answer": f"Could not generate '{intent.title}'.", "data": [],
                      "chart": None, "row_count": 0, "files_used": [], "error": str(result)[:200]}
        rows = result.get("data") or []
        shape = query_engine.profile_dataset(rows, result.get("chart"))
        provenance = {
            "files_used": result.get("files_used") or [],
            "row_count": result.get("row_count", len(rows)),
            "route": result.get("route", "agent"),
            "answer": result.get("answer", ""),
            "query": intent.nl_query,
        }
        if result.get("error"):
            provenance["error"] = str(result["error"])[:300]
        for f in provenance["files_used"]:
            source_file_ids.add(str(f))
        if not rows:
            empty_titles.append(intent.title)
        # P1: role map {catalog_column -> semantic_role} for the widget's source
        # table, so the recommender formats/binds from ingestion semantics (not the
        # column name). Empty when the table can't be resolved -> fail-closed.
        src_table = ((intent.spec or {}).get("planned") or {}).get("table") or intent.hints.get("table")
        tbl = next((t for t in catalog if t.table_name == src_table), None) if src_table else None
        role_map: dict = data_catalog.role_map_for_table(tbl)
        # P4 G5: classify WHY a zero-row widget is empty (error vs no-table-resolved
        # vs ran-but-0-rows) and carry an honest message. Set on provenance BEFORE
        # recommend so the empty tile inherits it. Confident wrong "0" > explained blank.
        if not rows:
            _estate, _emsg = classify_empty(result, tbl, intent)
            provenance["empty_reason"] = _estate
            provenance["empty_message"] = _emsg
        widget = recommend(
            shape, intent, rows, provenance=provenance, role_map=role_map, warnings=warnings
        )
        # P0: pin the validated planned+bound spec into the widget's provenance so
        # every later phase has a stable, inspectable, re-runnable contract. This
        # only adds provenance.spec — render output is unchanged.
        widget.provenance["spec"] = build_pinned_spec(intent, widget, shape)
        # P2 Layer 3 (honest catch): grounding is advisory, so verify post-hoc that
        # the result did not span tables lacking a validated safe join. Surface a
        # warning instead of shipping a silent double-counted headline.
        js = join_gate.widget_join_safety(provenance["files_used"], catalog, relationships)
        if js["multi_table"] and not js["safe"]:
            widget.provenance["join_warning"] = "multi_table_no_validated_join"
            warnings.append(
                f"Widget '{intent.title}' combined data from multiple tables without a "
                "validated relationship — its headline number may double-count."
            )
        # P2 G2: record whether the bound measure is provably additive (drives P5
        # tie-out). Absent key == unproven; never assert summable without a role.
        _pm = ((intent.spec or {}).get("planned") or {}).get("measure")
        _role = role_map.get(_pm) if _pm else None
        if _role is not None:
            widget.provenance["summable"] = is_additive_measure_role(_role)
        # P4: deterministic one-line insight from the returned rows (no LLM). Any
        # %-share is gated on the additive `summable` flag; None when uncomputable.
        if rows:
            _ins = compute_insight(
                widget.component_type, rows, widget.config,
                summable=bool(widget.provenance.get("summable")),
            )
            if _ins:
                widget.config["insight"] = _ins
        widget_ids.append(widget.widget_id)
        resolved.append(widget)

    # Surface empty widgets at the dashboard level so failures aren't silent.
    if empty_titles:
        shown = ", ".join(empty_titles[:5])
        more = f" (+{len(empty_titles) - 5} more)" if len(empty_titles) > 5 else ""
        warnings.append(
            f"{len(empty_titles)} widget(s) returned no data: {shown}{more}. "
            "The requested values or time period may not exist in the available data."
        )

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

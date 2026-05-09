"""
Organizations API — admin-only multi-tenancy management.

Endpoints (all require admin):
  POST   /organizations                         create a new org (assign container)
  GET    /organizations                         list all orgs
  GET    /organizations/{org_id}                fetch one org
  PATCH  /organizations/{org_id}                update name / container
  DELETE /organizations/{org_id}                delete an org
  PATCH  /users/{user_id}/organization          assign user to an org
                                                (lives in users.py — see there)
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import auth_logger
from app.dependencies import require_admin
from app.models.container import ContainerConfig
from app.models.organization import Organization
from app.models.user import User

router = APIRouter(prefix="/organizations", tags=["organizations"])


class OrgCreate(BaseModel):
    name: str
    container_id: str | None = None


class OrgUpdate(BaseModel):
    name: str | None = None
    container_id: str | None = None


class OrgOut(BaseModel):
    id: str
    name: str
    container_id: str | None
    container_name: str | None = None
    user_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


async def _validate_container(db: AsyncSession, container_id: str | None) -> None:
    if container_id is None:
        return
    cfg = await db.get(ContainerConfig, container_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Container not found")


@router.post("", response_model=OrgOut, status_code=status.HTTP_201_CREATED)
async def create_organization(
    body: OrgCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Organization name cannot be empty")

    existing = (
        await db.execute(select(Organization).where(Organization.name == name))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Organization with this name already exists")

    await _validate_container(db, body.container_id)

    org = Organization(name=name, container_id=body.container_id)
    db.add(org)
    await db.commit()
    await db.refresh(org)

    container_name = None
    if org.container_id:
        cfg = await db.get(ContainerConfig, org.container_id)
        container_name = cfg.container_name if cfg else None

    auth_logger.info("organization_created", org_id=org.id, name=org.name, by=admin.id)
    return OrgOut(
        id=org.id,
        name=org.name,
        container_id=org.container_id,
        container_name=container_name,
        user_count=0,
        created_at=org.created_at,
    )


@router.get("", response_model=list[OrgOut])
async def list_organizations(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(
                Organization,
                ContainerConfig.container_name,
                func.count(User.id).label("user_count"),
            )
            .outerjoin(ContainerConfig, Organization.container_id == ContainerConfig.id)
            .outerjoin(User, User.organization_id == Organization.id)
            .group_by(Organization.id, ContainerConfig.container_name)
            .order_by(Organization.created_at)
        )
    ).all()

    return [
        OrgOut(
            id=org.id,
            name=org.name,
            container_id=org.container_id,
            container_name=container_name,
            user_count=user_count or 0,
            created_at=org.created_at,
        )
        for org, container_name, user_count in rows
    ]


@router.get("/{org_id}", response_model=OrgOut)
async def get_organization(
    org_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    container_name = None
    if org.container_id:
        cfg = await db.get(ContainerConfig, org.container_id)
        container_name = cfg.container_name if cfg else None
    user_count = (
        await db.execute(
            select(func.count(User.id)).where(User.organization_id == org_id)
        )
    ).scalar() or 0
    return OrgOut(
        id=org.id,
        name=org.name,
        container_id=org.container_id,
        container_name=container_name,
        user_count=user_count,
        created_at=org.created_at,
    )


@router.patch("/{org_id}", response_model=OrgOut)
async def update_organization(
    org_id: str,
    body: OrgUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    if body.name is not None:
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(status_code=422, detail="name cannot be empty")
        if new_name != org.name:
            existing = (
                await db.execute(
                    select(Organization).where(
                        Organization.name == new_name, Organization.id != org_id
                    )
                )
            ).scalar_one_or_none()
            if existing:
                raise HTTPException(
                    status_code=409, detail="Another organization already uses this name"
                )
            org.name = new_name

    # body.container_id explicitly set (could be None to detach)
    if "container_id" in body.model_fields_set:
        await _validate_container(db, body.container_id)
        org.container_id = body.container_id

    await db.commit()
    await db.refresh(org)

    container_name = None
    if org.container_id:
        cfg = await db.get(ContainerConfig, org.container_id)
        container_name = cfg.container_name if cfg else None
    user_count = (
        await db.execute(
            select(func.count(User.id)).where(User.organization_id == org_id)
        )
    ).scalar() or 0

    auth_logger.info("organization_updated", org_id=org_id, by=admin.id)
    return OrgOut(
        id=org.id,
        name=org.name,
        container_id=org.container_id,
        container_name=container_name,
        user_count=user_count,
        created_at=org.created_at,
    )


@router.delete("/{org_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization(
    org_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    await db.delete(org)
    await db.commit()
    auth_logger.info("organization_deleted", org_id=org_id, by=admin.id)

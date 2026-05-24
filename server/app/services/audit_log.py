from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.core.logger import audit_logger
from app.core.security import decode_access_token
from app.models.file import File
from app.models.folder import Folder
from app.models.server_log import ServerLog
from app.models.user import User

_SECRET_QUERY_KEYS = {"token", "access_token", "code", "state", "password", "secret", "key"}


def _as_list(value: Any) -> list[str] | None:
    if not value:
        return None
    return [str(item) for item in value if str(item)] or None


def _client_ip(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def _action_for_request(request: Request) -> str:
    return f"{request.method} {_route_template(request)}"


def _scrubbed_query_params(request: Request) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in request.query_params.items():
        result[key] = "[redacted]" if key.lower() in _SECRET_QUERY_KEYS else value
    return result


async def _actor_from_request(request: Request, db: AsyncSession) -> User | None:
    auth = request.headers.get("authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    try:
        payload = decode_access_token(token)
    except Exception:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return await db.get(User, str(user_id))


async def _target_context_from_request(
    request: Request,
    db: AsyncSession,
) -> dict[str, Any]:
    params = dict(request.path_params or {})
    context: dict[str, Any] = {}

    folder_id = params.get("folder_id") or request.query_params.get("folder_id")
    if folder_id:
        folder = await db.get(Folder, str(folder_id))
        if folder:
            context["domain_tag"] = folder.domain_tag

    file_id = params.get("file_id") or request.query_params.get("file_id")
    if file_id:
        file = await db.get(File, str(file_id))
        context["file_id"] = str(file_id)
        if file:
            context["file_name"] = file.name
            if not context.get("domain_tag") and file.folder_id:
                folder = await db.get(Folder, file.folder_id)
                if folder:
                    context["domain_tag"] = folder.domain_tag

    return context


def _actor_fields(actor: User | None) -> dict[str, Any]:
    if actor is None:
        return {
            "actor_user_id": None,
            "actor_email": None,
            "actor_role": None,
        }
    return {
        "actor_user_id": actor.id,
        "actor_email": actor.email,
        "actor_role": "admin" if actor.is_admin else actor.role,
    }


async def record_audit_event(
    db: AsyncSession,
    *,
    actor: User | None,
    action: str,
    event_type: str = "action",
    status_code: int | None = None,
    method: str | None = None,
    path: str | None = None,
    route_template: str | None = None,
    duration_ms: float | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    domain_tag: str | None = None,
    container_id: str | None = None,
    file_id: str | None = None,
    file_name: str | None = None,
    folder_id: str | None = None,
    folder_name: str | None = None,
    target_user_id: str | None = None,
    target_user_email: str | None = None,
    target_user_name: str | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> ServerLog:
    # Pack context fields that don't have dedicated columns into details
    extra: dict[str, Any] = {}
    if route_template:
        extra["route_template"] = route_template
    if user_agent:
        extra["user_agent"] = user_agent
    if container_id:
        extra["container_id"] = container_id
    if folder_id:
        extra["folder_id"] = folder_id
    if folder_name:
        extra["folder_name"] = folder_name
    if target_user_id:
        extra["target_user_id"] = target_user_id
    if target_user_email:
        extra["target_user_email"] = target_user_email
    if target_user_name:
        extra["target_user_name"] = target_user_name
    if error:
        extra["error"] = error[:1000]

    merged_details = {**(details or {}), **extra} or None

    row = ServerLog(
        id=str(uuid.uuid4()),
        log_type="audit",
        event=action[:80],
        level="error" if (status_code and status_code >= 500) else "warning" if (status_code and status_code >= 400) else "info",
        **_actor_fields(actor),
        domain_tag=domain_tag,
        file_id=file_id,
        file_name=file_name,
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        ip_address=ip_address,
        details=merged_details,
    )
    db.add(row)
    return row


async def record_audit_event_safe(
    *,
    actor: User | None,
    action: str,
    event_type: str = "action",
    status_code: int | None = None,
    method: str | None = None,
    path: str | None = None,
    route_template: str | None = None,
    duration_ms: float | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    domain_tag: str | None = None,
    container_id: str | None = None,
    file_id: str | None = None,
    file_name: str | None = None,
    folder_id: str | None = None,
    folder_name: str | None = None,
    target_user_id: str | None = None,
    target_user_email: str | None = None,
    target_user_name: str | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Best-effort action audit using an isolated DB session."""
    try:
        async with async_session() as db:
            actor_row = await db.get(User, actor.id) if actor else None
            await record_audit_event(
                db,
                actor=actor_row,
                action=action,
                event_type=event_type,
                status_code=status_code,
                method=method,
                path=path,
                route_template=route_template,
                duration_ms=duration_ms,
                ip_address=ip_address,
                user_agent=user_agent,
                domain_tag=domain_tag,
                container_id=container_id,
                file_id=file_id,
                file_name=file_name,
                folder_id=folder_id,
                folder_name=folder_name,
                target_user_id=target_user_id,
                target_user_email=target_user_email,
                target_user_name=target_user_name,
                details=details,
                error=error,
            )
            await db.commit()
    except Exception as exc:
        audit_logger.warning(
            "audit_action_record_failed",
            action=action,
            actor_user_id=getattr(actor, "id", None),
            actor_email=getattr(actor, "email", None),
            error=str(exc)[:300],
        )


# GET-only paths that are polled frequently and would flood the audit table.
_SKIP_AUDIT_GET_PREFIXES = (
    "/api/health",
    "/api/metrics",
    "/api/logs/",
)

_ANONYMOUS_404_EXEMPT_PREFIXES = (
    "/api/auth/",
)


async def record_request_audit(
    request: Request,
    *,
    status_code: int | None,
    duration_ms: float,
    error: str | None = None,
) -> None:
    if request.method == "OPTIONS":
        return

    if request.method == "GET":
        path = request.url.path
        if any(path.startswith(prefix) for prefix in _SKIP_AUDIT_GET_PREFIXES):
            return

    if (
        request.method in {"GET", "HEAD"}
        and status_code == 404
        and not request.headers.get("authorization")
        and not any(request.url.path.startswith(prefix) for prefix in _ANONYMOUS_404_EXEMPT_PREFIXES)
    ):
        return

    start = time.perf_counter()
    try:
        async with async_session() as db:
            actor = await _actor_from_request(request, db)
            context = await _target_context_from_request(request, db)
            row = await record_audit_event(
                db,
                actor=actor,
                event_type="request",
                action=_action_for_request(request),
                method=request.method,
                path=request.url.path,
                route_template=_route_template(request),
                status_code=status_code,
                duration_ms=duration_ms,
                ip_address=_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                details={
                    "query_params": _scrubbed_query_params(request),
                    "referer": request.headers.get("referer"),
                    "audit_insert_ms": None,
                },
                error=error,
                **context,
            )
            if row.details is not None:
                row.details["audit_insert_ms"] = round((time.perf_counter() - start) * 1000, 2)
            await db.commit()
    except Exception as exc:
        audit_logger.warning(
            "audit_record_failed",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            error=str(exc)[:300],
        )


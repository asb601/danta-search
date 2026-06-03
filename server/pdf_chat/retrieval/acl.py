"""Stage 7 — ACL Check.

Pure logic (Hard rule #6: infra adapters are guarded; this module has none). For
every retrieved chunk, decide whether the requesting principal may see it before
it reaches the LLM context. Denied chunks are dropped (the CALLER logs the
denied ids to ``query_audit_log`` — this module stays side-effect free so it is
trivially unit-testable with zero infra).

A chunk PASSES when:

    chunk.tenant_id == request.tenant_id
    AND (
        user_id    IN chunk.acl.allowed_users
        OR any user_group IN chunk.acl.allowed_groups
        OR chunk.acl.public == True
    )

Chunks may be either a dataclass ``Chunk`` (``.acl`` / ``.tenant_id`` /
``.chunk_id`` attributes) or a plain ``dict`` with the same keys — both are
handled transparently.
"""
from __future__ import annotations

from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dataclass-like object OR a dict, uniformly."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def filter_by_acl(
    chunks: list[Any],
    user_id: str,
    user_groups: list[str],
    tenant_id: str,
) -> tuple[list[Any], list[str]]:
    """Partition chunks into accessible vs denied for one principal.

    Args:
        chunks: retrieved chunks (dataclass ``Chunk`` or dict). Each must expose
            ``acl`` (dict), ``tenant_id`` and ``chunk_id``.
        user_id: requesting user id.
        user_groups: groups the requesting user belongs to.
        tenant_id: the request's tenant; a chunk from any other tenant is denied
            unconditionally (tenant isolation, Hard rule #3).

    Returns:
        ``(accessible, denied_ids)`` — accessible chunks preserve input order;
        ``denied_ids`` are the ``chunk_id`` values that failed the check, in
        input order. The caller is responsible for audit logging.
    """
    accessible: list[Any] = []
    denied_ids: list[str] = []
    groups = set(user_groups or [])

    for chunk in chunks:
        acl = _get(chunk, "acl", {}) or {}
        chunk_tenant = _get(chunk, "tenant_id")

        allowed_users = acl.get("allowed_users", []) or []
        allowed_groups = acl.get("allowed_groups", []) or []
        is_public = bool(acl.get("public", False))

        tenant_ok = chunk_tenant == tenant_id
        principal_ok = (
            user_id in allowed_users
            or bool(groups.intersection(allowed_groups))
            or is_public
        )

        if tenant_ok and principal_ok:
            accessible.append(chunk)
        else:
            denied_ids.append(_get(chunk, "chunk_id"))

    return accessible, denied_ids


def insufficient_context(accessible: list[Any], min_required: int) -> bool:
    """True when too few accessible chunks remain to answer safely.

    When this returns True the caller should return an "Insufficient accessible
    context" message rather than letting the LLM hallucinate (spec Stage 7).

    Args:
        accessible: chunks that survived :func:`filter_by_acl`.
        min_required: floor from config (``min_accessible_chunks``).
    """
    return len(accessible) < min_required

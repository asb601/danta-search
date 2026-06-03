"""Stage 5 / Stage 10 — Redis query-result cache.

Two pieces:

1. :func:`cache_key` — pure. The canonical cache key from the spec:
   ``sha256(query + tenant_id + sorted(groups))``. Group ORDER does not matter
   (groups are sorted before hashing) so the same principal always lands on the
   same key regardless of how the JWT enumerated their groups.

2. :class:`RedisCache` — a thin get/set wrapper with a config-driven TTL. The
   ``redis`` import is GUARDED (Hard rule #6): the class constructs fine with no
   infra, and ``get``/``set`` degrade to a silent no-op MISS when ``redis`` is
   not installed or the connection cannot be established. This keeps the query
   pipeline working (cache-disabled) without a Redis server.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pdf_chat.config import get_pdf_settings

try:
    import redis as _redis  # type: ignore

    _HAS_REDIS = True
except ImportError:  # pragma: no cover - exercised only without infra
    _redis = None  # type: ignore
    _HAS_REDIS = False


def cache_key(
    query: str,
    tenant_id: str,
    groups: list[str],
    acl_version: str = "0",
    doc_ids: list[str] | None = None,
) -> str:
    """Deterministic, revocation-aware cache key for a query + principal.

    The key folds in an ``acl_version`` and the queried ``doc_ids`` so that any
    permission change or document mutation produces a *different* key — old
    cached answers become unreachable instead of being served stale. Bumping a
    tenant's ``acl_version`` (on a grant/revoke or a document delete/reindex)
    therefore transparently invalidates every cached answer for that tenant.

    Args:
        query: raw user query text.
        tenant_id: request tenant.
        groups: the principal's groups. Sorted before hashing so order does not
            affect the key.
        acl_version: opaque tenant ACL-epoch token. Defaults to ``"0"`` today;
            revocation / delete flows bump it to evict the tenant's cache.
        doc_ids: optional retrieval document scope. Sorted before hashing so
            order does not affect the key; ``None`` (whole-tenant) and an empty
            list are distinguished.

    Returns:
        Hex sha256 digest over query + tenant + groups + acl_version + doc_ids.
    """
    sorted_groups = ",".join(sorted(groups or []))
    if doc_ids is None:
        docs_part = "*"
    else:
        docs_part = ",".join(sorted(doc_ids))
    payload = f"{query}|{tenant_id}|{sorted_groups}|{acl_version}|{docs_part}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RedisCache:
    """Query-result cache backed by Redis, degrading to no-op without infra.

    The client is created lazily so importing/constructing never touches the
    network. Any connection or operational error is swallowed and treated as a
    cache MISS (cache is an optimization, never a correctness dependency).
    """

    def __init__(self, url: str | None = None, ttl_seconds: int | None = None):
        settings = get_pdf_settings()
        self._url = url or settings.redis_url
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self._client: Any = None
        self._init_failed = False

    @property
    def enabled(self) -> bool:
        """True only when the redis library is importable."""
        return _HAS_REDIS

    def _get_client(self) -> Any:
        """Lazily build the redis client; cache the failure to avoid retries."""
        if not _HAS_REDIS or self._init_failed:
            return None
        if self._client is None:
            try:
                self._client = _redis.Redis.from_url(  # type: ignore[union-attr]
                    self._url, decode_responses=True
                )
            except Exception:  # pragma: no cover - infra-dependent
                self._init_failed = True
                return None
        return self._client

    def get(self, key: str) -> dict | None:
        """Return the cached value (deserialized dict) for ``key``.

        The agent stores a structured payload (``answer``/``citations``/...), so
        the value contract is a ``dict``. A miss, missing infra, or a corrupt /
        non-JSON stored value all return ``None`` (treated as a MISS — cache is
        never a correctness dependency).
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except Exception:  # pragma: no cover - infra-dependent
            return None
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict, ttl_seconds: int | None = None) -> bool:
        """Write a dict ``value`` (JSON-serialized) under ``key`` with TTL.

        Returns:
            True if the write was issued, False if the cache is unavailable or
            the value could not be serialized.
        """
        client = self._get_client()
        if client is None:
            return False
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return False
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        try:
            client.set(key, payload, ex=ttl)
            return True
        except Exception:  # pragma: no cover - infra-dependent
            return False

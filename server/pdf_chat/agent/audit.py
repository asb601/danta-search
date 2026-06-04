"""Query audit adapter (agent Stage 10).

Records every served query (incl. cache hits — Security must-fix #6) for
compliance. Satisfies the agent's ``AuditRepo`` protocol. The ``sink`` is the
write target: a pure callable in tests; in production a thin async DB writer
against the audit table. Tenant-isolated via the persisted tenant_id.
"""
from __future__ import annotations

from typing import Any, Callable


class QueryAuditRepo:
    def __init__(self, sink: "Callable[[dict], Any] | None" = None) -> None:
        self._sink = sink

    async def write(
        self,
        *,
        user_id: str,
        tenant_id: str,
        query_hash: str,
        query_text: str,
        returned_chunks: list[str],
        denied_chunks: list[str],
        cache_hit: bool = False,
    ) -> None:
        row = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query_hash": query_hash,
            "query_text": query_text,
            "returned_chunks": list(returned_chunks),
            "denied_chunks": list(denied_chunks),
            "cache_hit": cache_hit,
        }
        if self._sink is not None:
            result = self._sink(row)
            if hasattr(result, "__await__"):
                await result

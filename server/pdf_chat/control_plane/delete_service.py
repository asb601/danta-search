"""Cascading document delete: soft-delete first, then async tenant-scoped cleanup.

Flow (spec §5):
  1. ``delete_document()`` marks the ``UploadManifest`` row ``status="deleted"``
     (soft delete) and returns immediately so the API responds fast. This is the
     EXACT symbol ``pdf_chat/api/routes.py`` imports for the DELETE route.
  2. ``cleanup_deleted_document()`` runs the batched Neo4j cascade: snapshot the
     mention index BEFORE deleting chunks (the DETACH would erase the MENTIONS
     edges otherwise), batched ``DETACH DELETE`` the document's chunks, compute
     which entities became orphaned, then ``DETACH DELETE`` ONLY those orphans plus
     any community left with no members. Entities referenced by other docs are
     preserved (verified in test_delete_service.py).

Every Cypher element is tenant-scoped (spec §3.3) via the pure builders in
``graph_delete``. The Neo4j session is INJECTED (a sync object exposing ``.run()``
returning row dicts) so the cascade is unit-testable with a fake — zero infra. The
batch size is a tunable (``delete.batch_size``), resolved through ``get_tunable``
with no inline numeric default (Spec §3 inv 4), and every loop decision is emitted
through ``log_gate_decision`` so no score is compared-and-discarded silently.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..observability import metrics as _metrics
from ..observability.logging import bind_trace, get_pdf_logger
from ..observability.trace import new_trace_id
from ..tunables import get_tunable, log_gate_decision
from . import graph_delete

_logger = get_pdf_logger("delete")

# Loop sentinel: a chunk batch that deletes 0 rows means the document has no chunks
# left, so the batched cascade is done. Routed through log_gate_decision (score=
# deleted count, threshold=this) so the stop decision is observable, not a silent
# literal comparison (Spec §3 inv 4). This is a STRUCTURAL loop bound, not a score
# threshold — there is no tunable for "are there rows left", it is exactly zero.
_EMPTY_BATCH = 0


async def delete_document(upload_id: str, tenant_id: str) -> dict[str, Any] | None:
    """Soft-delete the manifest (TENANT-SCOPED), then signal async cleanup.

    The route (``routes.py``) imports THIS function. It opens its own async session
    (late import — mirrors the worker tasks) so it can be called straight from the
    request handler without a passed-in session.

    The soft-delete UPDATE is scoped to ``tenant_id`` so a tenant can NEVER
    soft-delete another tenant's document (SECURITY). If the tenant-scoped UPDATE
    touches 0 rows (unknown upload_id OR a doc owned by a DIFFERENT tenant) we
    return ``None`` so the route's existing ``if result is None: 404`` fires
    (previously this always returned a dict, making that 404 dead code). On a real
    hit we return the status dict the route maps onto its ``DeleteResponse``.
    """
    from app.core.database import async_session  # late import (zero infra at import)

    from .repositories import UploadManifestRepo

    async with async_session() as session:
        repo = UploadManifestRepo(session)
        rows = await repo.set_status(upload_id, "deleted", tenant_id=tenant_id)
        await session.commit()

    if rows == 0:
        # Unknown id OR a cross-tenant doc: nothing was soft-deleted → 404.
        return None

    # Cleanup runs async (Celery in production): the route schedules
    # cleanup_deleted_document; here we return enough for the API to respond fast.
    return {"upload_id": upload_id, "status": "deleted", "cleanup": "scheduled"}


async def cleanup_deleted_document(
    upload_id: str,
    tenant_id: str,
    *,
    neo4j_session: Any,
    container_id: str,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Run the batched tenant-scoped graph cascade. Idempotent + re-runnable.

    Returns ``{upload_id, chunks_deleted, entities_deleted, communities_deleted}``.

    The cascade body is SYNCHRONOUS (the injected Neo4j session exposes a sync
    ``.run()`` and the batched DETACH loop is blocking). Since this coroutine is
    invoked from a FastAPI ``BackgroundTask``, running the sync body inline would
    block the event loop, so it is offloaded with ``asyncio.to_thread`` (Fix 9).
    The injected fake session is sync and keeps working under ``to_thread``; the
    returned summary shape + metrics increments are unchanged.
    """
    trace_id = trace_id or new_trace_id()
    return await asyncio.to_thread(
        _run_cascade_sync,
        upload_id,
        tenant_id,
        neo4j_session=neo4j_session,
        container_id=container_id,
        trace_id=trace_id,
    )


def _run_cascade_sync(
    upload_id: str,
    tenant_id: str,
    *,
    neo4j_session: Any,
    container_id: str,
    trace_id: str,
) -> dict[str, Any]:
    """Synchronous tenant-scoped graph cascade (offloaded via ``to_thread``).

    Contains the full blocking body — mention-index snapshot, batched chunk DETACH,
    Document/Page node delete, orphan-only entity delete, community sweep — plus the
    per-tenant metrics increments and the structured completion log. Kept sync so
    the injected fake (sync ``.run()``) is unit-testable with zero infra.
    """
    log = bind_trace(_logger, trace_id, tenant_id)
    batch_size = int(get_tunable(container_id, "delete.batch_size"))

    # 1. Snapshot which entities this doc mentions + who else mentions them BEFORE
    #    deleting the chunks (deletion would erase the MENTIONS edges).
    mi_cypher, mi_params = graph_delete.build_mention_index_cypher(upload_id, tenant_id)
    mention_index: dict[str, set[str]] = {}
    for row in neo4j_session.run(mi_cypher, **mi_params):
        mention_index[row["entity_id"]] = set(row["doc_ids"])

    # 2. Batched chunk deletion until a batch removes no rows (document drained).
    chunks_deleted = 0
    while True:
        c_cypher, c_params = graph_delete.build_chunk_delete_cypher(
            upload_id, tenant_id, batch_size
        )
        result = list(neo4j_session.run(c_cypher, **c_params))
        n = result[0]["deleted"] if result else 0
        chunks_deleted += n
        # outcome="drained" once a batch is empty; "continue" while rows remain.
        gate = log_gate_decision(
            "delete.chunk_batch_drained",
            score=n,
            threshold=_EMPTY_BATCH,
            outcome="continue" if n > _EMPTY_BATCH else "drained",
            container_id=container_id,
            tenant_id=tenant_id,
            upload_id=upload_id,
            batch_size=batch_size,
        )
        # passed == (score >= threshold) == (n >= 0) is always true; the real loop
        # stop is "no rows deleted this batch", i.e. score == threshold.
        if n <= _EMPTY_BATCH:
            break
        del gate  # decision already emitted; only used for the structured log.

    # 3. Delete the document's OWN (:Document)/(:Page) nodes (tenant + doc scoped)
    #    so no Document/Page residue is left after the chunks are gone (Fix 7).
    d_cypher, d_params = graph_delete.build_document_node_delete_cypher(
        upload_id, tenant_id
    )
    list(neo4j_session.run(d_cypher, **d_params))

    # 4. Delete ONLY orphaned entities (never those referenced by other docs).
    orphans = graph_delete.select_orphan_entities(upload_id, mention_index)
    entities_deleted = 0
    if orphans:
        e_cypher, e_params = graph_delete.build_orphan_entity_delete_cypher(
            orphans, tenant_id
        )
        e_result = list(neo4j_session.run(e_cypher, **e_params))
        entities_deleted = e_result[0]["deleted"] if e_result else len(orphans)

    # 5. Sweep communities left with no members.
    cm_cypher, cm_params = graph_delete.build_orphan_community_delete_cypher(tenant_id)
    cm_result = list(neo4j_session.run(cm_cypher, **cm_params))
    communities_deleted = cm_result[0]["deleted"] if cm_result else 0

    _metrics.inc(tenant_id, "pdf_document_deleted_count")
    _metrics.inc(tenant_id, "pdf_orphan_entity_deleted_count", entities_deleted)
    log.info(
        "pdf_document_cleanup",
        upload_id=upload_id,
        chunks_deleted=chunks_deleted,
        entities_deleted=entities_deleted,
        communities_deleted=communities_deleted,
        shared_entities_preserved=len(mention_index) - len(orphans),
    )
    return {
        "upload_id": upload_id,
        "chunks_deleted": chunks_deleted,
        "entities_deleted": entities_deleted,
        "communities_deleted": communities_deleted,
    }

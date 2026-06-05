"""Tests for the cascading document delete (Phase 6 hardening, Tasks 5 + 6).

Two layers, both infra-free:
  * graph_delete.py — pure tenant-scoped Cypher builders + orphan detection. We
    assert tenant_id/upload_id are bound as $params (never inlined) and that a
    shared entity (still mentioned by ANOTHER doc) is never selected as an orphan.
  * delete_service.py — the async cascade driven by an INJECTED fake Neo4j session
    (sync .run() returning row dicts) so the batched chunk-delete + orphan-only
    entity-delete + community sweep run deterministically with zero infra. The
    soft-delete path uses a fake async_session so app.core.database is never hit.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from pdf_chat.control_plane.graph_delete import (
    build_chunk_delete_cypher,
    build_document_node_delete_cypher,
    build_mention_index_cypher,
    build_orphan_community_delete_cypher,
    build_orphan_entity_delete_cypher,
    select_orphan_entities,
)


# ── Task 5: pure Cypher + orphan detection ──────────────────────────────────

def test_chunk_delete_cypher_is_tenant_scoped_on_every_element():
    cypher, params = build_chunk_delete_cypher(
        upload_id="doc1", tenant_id="tenant-A", batch_size=500
    )
    # tenant + doc + limit bound as params, never inlined into the string
    assert params == {"upload_id": "doc1", "tenant_id": "tenant-A", "limit": 500}
    assert "$tenant_id" in cypher and "tenant-A" not in cypher
    assert "$upload_id" in cypher and "doc1" not in cypher
    # every matched chunk constrained on tenant_id (spec §3.3)
    assert "c.tenant_id = $tenant_id" in cypher
    assert "c.doc_id = $upload_id" in cypher
    assert "DETACH DELETE" in cypher
    assert "LIMIT $limit" in cypher  # batched
    assert "count(c)" in cypher  # returns a count so the caller can loop to zero


def test_mention_index_cypher_is_tenant_scoped_on_chunk_and_entity():
    cypher, params = build_mention_index_cypher(upload_id="doc1", tenant_id="tenant-A")
    assert params == {"upload_id": "doc1", "tenant_id": "tenant-A"}
    assert "tenant-A" not in cypher and "doc1" not in cypher
    # the doc's chunks AND the OTHER docs' chunks are both tenant-scoped
    assert "c.tenant_id = $tenant_id" in cypher
    assert "oc.tenant_id = $tenant_id" in cypher
    assert "c.doc_id = $upload_id" in cypher
    # collects ALL doc_ids still mentioning each entity (orphan decided by caller)
    assert "collect(DISTINCT oc.doc_id)" in cypher
    assert "MENTIONS" in cypher
    # the real KG keys entities on e.name (kg_writer MERGE), not a non-existent
    # e.entity_id; the result key stays "entity_id" so the caller is unchanged.
    assert "e.name AS entity_id" in cypher


def test_select_orphan_entities_excludes_entities_referenced_by_other_docs():
    # entity -> set of doc_ids that still MENTION it (computed by the caller's query)
    mention_index = {
        "ent_shared": {"doc1", "doc2"},   # doc2 still references it → KEEP
        "ent_only_doc1": {"doc1"},        # only the deleted doc → ORPHAN
        "ent_unrelated": {"doc3"},        # not in deleted doc at all → KEEP
    }
    orphans = select_orphan_entities(
        deleted_doc_id="doc1", mention_index=mention_index
    )
    assert orphans == ["ent_only_doc1"]
    assert "ent_shared" not in orphans  # shared entity stays intact
    assert "ent_unrelated" not in orphans


def test_select_orphan_entities_is_sorted_and_drops_empty_doc_sets():
    mention_index = {
        "z_orphan": {"doc1"},
        "a_orphan": {"doc1"},
        "ent_empty": set(),  # no remaining mentions at all → not selected (nothing to orphan)
    }
    orphans = select_orphan_entities(deleted_doc_id="doc1", mention_index=mention_index)
    assert orphans == ["a_orphan", "z_orphan"]  # sorted, empty-set entity excluded


def test_orphan_entity_delete_cypher_is_tenant_and_id_scoped():
    cypher, params = build_orphan_entity_delete_cypher(
        entity_ids=["ent_only_doc1"], tenant_id="tenant-A"
    )
    assert params == {"entity_ids": ["ent_only_doc1"], "tenant_id": "tenant-A"}
    assert "tenant-A" not in cypher and "ent_only_doc1" not in cypher
    assert "e.tenant_id = $tenant_id" in cypher
    # entities are keyed on e.name (real KG schema), with the param still entity_ids
    assert "e.name IN $entity_ids" in cypher
    assert "DETACH DELETE" in cypher


def test_document_node_delete_cypher_is_tenant_and_doc_scoped():
    cypher, params = build_document_node_delete_cypher(
        upload_id="doc1", tenant_id="tenant-A"
    )
    assert params == {"upload_id": "doc1", "tenant_id": "tenant-A"}
    assert "tenant-A" not in cypher and "doc1" not in cypher
    # the Document AND its Pages are tenant + doc scoped, bound as $params
    assert "d.doc_id = $upload_id" in cypher
    assert "d.tenant_id = $tenant_id" in cypher
    assert "p.doc_id = $upload_id" in cypher
    assert "p.tenant_id = $tenant_id" in cypher
    assert "(d:Document)" in cypher
    assert "(p:Page)" in cypher
    assert "DETACH DELETE" in cypher


def test_orphan_community_delete_cypher_is_tenant_scoped():
    cypher, params = build_orphan_community_delete_cypher(tenant_id="tenant-A")
    assert params == {"tenant_id": "tenant-A"}
    assert "tenant-A" not in cypher
    assert "cm.tenant_id = $tenant_id" in cypher
    assert "IN_COMMUNITY" in cypher
    assert "DETACH DELETE" in cypher
    # defense-in-depth (Fix 8): the member-match is tenant-scoped on the Entity so a
    # cross-tenant IN_COMMUNITY edge can never keep this tenant's community alive.
    assert "(:Entity {tenant_id:$tenant_id})-[:IN_COMMUNITY]->(cm)" in cypher


# ── Task 6: async cascade service (fake Neo4j session) ───────────────────────

class _FakeNeo4jSession:
    """Records run() calls and serves canned results so the cascade is deterministic."""

    def __init__(self, mention_rows):
        self.calls = []
        self._mention_rows = mention_rows
        self._chunk_batches = [3, 0]  # first batch deletes 3 chunks, then none left

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        if "MENTIONS" in cypher and "collect(DISTINCT oc.doc_id)" in cypher:
            return list(self._mention_rows)
        if cypher.strip().startswith("MATCH (c:Chunk)"):
            return [{"deleted": self._chunk_batches.pop(0)}]
        return [{"deleted": 1}]


def test_cleanup_deletes_chunks_and_orphans_but_keeps_shared_entities():
    from pdf_chat.control_plane.delete_service import cleanup_deleted_document
    from pdf_chat.observability import metrics as _metrics

    _metrics.reset()
    # ent_shared still referenced by doc2 → must NOT be deleted.
    mention_rows = [
        {"entity_id": "ent_shared", "doc_ids": ["doc1", "doc2"]},
        {"entity_id": "ent_only_doc1", "doc_ids": ["doc1"]},
    ]
    session = _FakeNeo4jSession(mention_rows)

    summary = asyncio.run(
        cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    assert summary["chunks_deleted"] == 3
    assert summary["entities_deleted"] == 1  # only ent_only_doc1
    assert summary["communities_deleted"] == 1
    # the orphan-delete call carried exactly the non-shared entity, tenant-scoped
    orphan_calls = [
        c for c in session.calls
        if "MATCH (e:Entity)" in c[0] and "DETACH DELETE e" in c[0]
    ]
    assert len(orphan_calls) == 1
    assert orphan_calls[0][1]["entity_ids"] == ["ent_only_doc1"]
    assert orphan_calls[0][1]["tenant_id"] == "tenant-A"
    # per-tenant metrics were incremented for the delete + orphan count
    snap = _metrics.get_snapshot("tenant-A")
    assert snap["pdf_document_deleted_count"] == 1
    assert snap["pdf_orphan_entity_deleted_count"] == 1
    # the document's own (:Document)/(:Page) nodes are deleted too (no residue),
    # tenant + doc scoped (Fix 7).
    doc_calls = [
        c for c in session.calls
        if "(d:Document)" in c[0] and "(p:Page)" in c[0]
    ]
    assert len(doc_calls) == 1
    assert doc_calls[0][1] == {"upload_id": "doc1", "tenant_id": "tenant-A"}


def test_cleanup_snapshots_mention_index_before_deleting_chunks():
    """The mention index MUST be read before chunks are deleted (the DETACH would
    erase the MENTIONS edges otherwise)."""
    from pdf_chat.control_plane.delete_service import cleanup_deleted_document

    session = _FakeNeo4jSession(
        [{"entity_id": "ent_only_doc1", "doc_ids": ["doc1"]}]
    )
    asyncio.run(
        cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    kinds = [
        "mention" if "collect(DISTINCT oc.doc_id)" in c[0]
        else "chunk" if c[0].strip().startswith("MATCH (c:Chunk)")
        else "other"
        for c in session.calls
    ]
    assert kinds.index("mention") < kinds.index("chunk")


def test_cleanup_offloads_sync_cascade_via_to_thread(monkeypatch):
    """The sync Neo4j cascade body must run off the event loop (asyncio.to_thread)
    so a FastAPI BackgroundTask never blocks the loop on sync .run() calls (Fix 9).
    The injected fake session is sync and must keep working."""
    import asyncio as _asyncio

    from pdf_chat.control_plane import delete_service

    offloaded = {"count": 0}
    real_to_thread = _asyncio.to_thread

    async def _spy_to_thread(fn, /, *args, **kwargs):
        offloaded["count"] += 1
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(delete_service.asyncio, "to_thread", _spy_to_thread)

    session = _FakeNeo4jSession(
        [{"entity_id": "ent_only_doc1", "doc_ids": ["doc1"]}]
    )
    summary = asyncio.run(
        delete_service.cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    assert offloaded["count"] == 1  # the sync cascade was offloaded exactly once
    # summary shape preserved despite the offload
    assert summary["upload_id"] == "doc1"
    assert summary["chunks_deleted"] == 3
    assert summary["entities_deleted"] == 1


def test_cleanup_makes_no_orphan_delete_call_when_all_entities_shared():
    from pdf_chat.control_plane.delete_service import cleanup_deleted_document

    session = _FakeNeo4jSession(
        [{"entity_id": "ent_shared", "doc_ids": ["doc1", "doc2"]}]
    )
    summary = asyncio.run(
        cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    assert summary["entities_deleted"] == 0
    orphan_calls = [
        c for c in session.calls
        if "MATCH (e:Entity)" in c[0] and "DETACH DELETE e" in c[0]
    ]
    assert orphan_calls == []  # no orphan-delete Cypher run at all


def test_cleanup_batch_size_comes_from_tunable_not_inline_default():
    """The chunk-delete LIMIT must come from delete.batch_size via get_tunable."""
    from pdf_chat.control_plane.delete_service import cleanup_deleted_document

    session = _FakeNeo4jSession([])
    asyncio.run(
        cleanup_deleted_document(
            upload_id="doc1", tenant_id="tenant-A",
            neo4j_session=session, container_id="c1",
        )
    )
    # The mention-index query ALSO opens with MATCH (c:Chunk); the chunk-DELETE is
    # uniquely identified by "DETACH DELETE c" so we don't pick up the index call.
    chunk_calls = [c for c in session.calls if "DETACH DELETE c" in c[0]]
    assert chunk_calls, "expected at least one chunk-delete call"
    # default registered in TUNABLE_DEFAULTS["delete.batch_size"] == 500
    assert chunk_calls[0][1]["limit"] == 500


def test_delete_document_soft_deletes_manifest_and_returns_status_dict(monkeypatch):
    """delete_document marks the manifest 'deleted', commits, and returns the
    soft-delete status dict — using a FAKE async_session (no real DB)."""
    from pdf_chat.control_plane import delete_service

    recorded: dict = {}

    class _FakeRepo:
        def __init__(self, session):
            recorded["session"] = session

        async def set_status(self, upload_id, status, *, tenant_id=None):
            recorded["set_status"] = (upload_id, status, tenant_id)
            return 1  # one row updated → same-tenant soft-delete succeeds

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            recorded["committed"] = True

    def _fake_async_session():
        return _FakeSession()

    # Inject a fake app.core.database module so the late import resolves to our fake.
    fake_db_mod = types.ModuleType("app.core.database")
    fake_db_mod.async_session = _fake_async_session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.core.database", fake_db_mod)
    monkeypatch.setattr(
        "pdf_chat.control_plane.repositories.UploadManifestRepo", _FakeRepo
    )

    result = asyncio.run(
        delete_service.delete_document(upload_id="doc1", tenant_id="tenant-A")
    )
    assert recorded["set_status"] == ("doc1", "deleted", "tenant-A")
    assert recorded["committed"] is True
    assert result["status"] == "deleted"
    assert result["cleanup"] == "scheduled"
    assert result["upload_id"] == "doc1"


def _run_delete_with_fake_repo(monkeypatch, *, rows: int):
    """Drive delete_document with a fake repo whose set_status returns ``rows``.

    Returns ``(result, recorded)``. Mocks-only: a fake async_session + fake repo so
    app.core.database / a real DB is never touched.
    """
    from pdf_chat.control_plane import delete_service

    recorded: dict = {}

    class _FakeRepo:
        def __init__(self, session):
            pass

        async def set_status(self, upload_id, status, *, tenant_id=None):
            recorded["set_status"] = (upload_id, status, tenant_id)
            return rows  # 0 ⇒ unknown id OR other tenant; >0 ⇒ same-tenant hit

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            recorded["committed"] = True

    fake_db_mod = types.ModuleType("app.core.database")
    fake_db_mod.async_session = lambda: _FakeSession()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.core.database", fake_db_mod)
    monkeypatch.setattr(
        "pdf_chat.control_plane.repositories.UploadManifestRepo", _FakeRepo
    )
    result = asyncio.run(
        delete_service.delete_document(upload_id="doc1", tenant_id="tenant-A")
    )
    return result, recorded


def test_delete_document_cross_tenant_affects_zero_rows_returns_none(monkeypatch):
    """A tenant-scoped soft-delete of ANOTHER tenant's upload_id updates 0 rows →
    delete_document returns None so the route's existing 404 fires (no silent
    cross-tenant soft-delete; SECURITY)."""
    result, recorded = _run_delete_with_fake_repo(monkeypatch, rows=0)
    assert result is None
    # the tenant was threaded into set_status (so the UPDATE is tenant-scoped)
    assert recorded["set_status"] == ("doc1", "deleted", "tenant-A")


def test_delete_document_same_tenant_returns_status_dict(monkeypatch):
    """Same-tenant soft-delete updates ≥1 row → returns the status dict as before."""
    result, recorded = _run_delete_with_fake_repo(monkeypatch, rows=1)
    assert result == {"upload_id": "doc1", "status": "deleted", "cleanup": "scheduled"}
    assert recorded["committed"] is True


# ── Route reconciliation: status dict -> DeleteResponse ──────────────────────

def test_route_maps_soft_delete_status_dict_to_delete_response():
    """The DELETE route consumes delete_document's {status, cleanup} status dict.

    DeleteResponse forbids extra keys and needs upload_id/deleted/chunks_removed, so
    the route maps the service's status dict onto the response schema. We assert the
    pure mapping helper here (no FastAPI/infra) so the route can't drift from the
    task-mandated service return shape.
    """
    from pdf_chat.api.routes import _to_delete_response
    from pdf_chat.schemas.pdf_schemas import DeleteResponse

    resp = _to_delete_response(
        {"upload_id": "doc1", "status": "deleted", "cleanup": "scheduled"}
    )
    assert isinstance(resp, DeleteResponse)
    assert resp.upload_id == "doc1"
    assert resp.deleted is True  # status == "deleted" -> deleted flag True
    assert resp.chunks_removed == 0  # async cleanup hasn't run yet at soft-delete


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

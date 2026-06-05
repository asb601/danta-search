"""Pure (infra-free) Cypher builders + orphan detection for cascading delete.

Tenant isolation is enforced on EVERY matched element (spec §3.3): the document's
chunks, their MENTIONS edges, and any Entity/Community that becomes orphaned are
all constrained on ``tenant_id``, which is ALWAYS bound as a ``$param`` (never
inlined into the query string). An entity referenced by ANOTHER document is never
deleted — orphan-ness is computed from the mention index, not assumed.

These functions take no driver and do no I/O so they are unit-testable with zero
infra (matching the pdf_chat/testing conventions). ``delete_service.py`` runs them
against a real (or fake) Neo4j session.
"""
from __future__ import annotations

__all__ = [
    "build_chunk_delete_cypher",
    "build_mention_index_cypher",
    "select_orphan_entities",
    "build_orphan_entity_delete_cypher",
    "build_orphan_community_delete_cypher",
    "build_document_node_delete_cypher",
]


def build_chunk_delete_cypher(
    upload_id: str, tenant_id: str, batch_size: int
) -> tuple[str, dict]:
    """One batch of tenant-scoped chunk deletion (DETACH removes MENTIONS edges too).

    Returns the count of deleted chunks so the caller can loop until zero. The
    document, tenant, and batch limit are all bound as ``$params`` — never inlined.
    """
    cypher = (
        "MATCH (c:Chunk) "
        "WHERE c.doc_id = $upload_id AND c.tenant_id = $tenant_id "
        "WITH c LIMIT $limit "
        "DETACH DELETE c "
        "RETURN count(c) AS deleted"
    )
    return cypher, {"upload_id": upload_id, "tenant_id": tenant_id, "limit": batch_size}


def build_mention_index_cypher(upload_id: str, tenant_id: str) -> tuple[str, dict]:
    """For every entity mentioned by this doc, list ALL doc_ids still mentioning it.

    The caller passes the result to ``select_orphan_entities`` to decide what to
    drop. Tenant-scoped on BOTH the deleted doc's chunks and the other docs' chunks
    (spec §3.3) so the doc-id set can never leak a cross-tenant mention.
    """
    cypher = (
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
        "WHERE c.doc_id = $upload_id AND c.tenant_id = $tenant_id "
        "WITH DISTINCT e "
        "MATCH (oc:Chunk)-[:MENTIONS]->(e) "
        "WHERE oc.tenant_id = $tenant_id "
        # The real KG keys entities on ``name`` (kg_writer MERGE (e:Entity {name,
        # tenant_id})); there is NO ``entity_id`` property, so reading it would
        # collapse the index on null. We read ``e.name`` but keep the result KEY
        # "entity_id" so select_orphan_entities/delete_service stay identity-agnostic.
        "RETURN e.name AS entity_id, collect(DISTINCT oc.doc_id) AS doc_ids"
    )
    return cypher, {"upload_id": upload_id, "tenant_id": tenant_id}


def select_orphan_entities(
    deleted_doc_id: str, mention_index: dict[str, set[str]]
) -> list[str]:
    """Entities whose only remaining mentioning doc is the one being deleted.

    An entity is an orphan iff the set of docs that still mention it is a non-empty
    subset of ``{deleted_doc_id}``. Entities referenced by ANY other doc are never
    orphans (they stay intact). The result is sorted for deterministic Cypher params.
    """
    orphans: list[str] = []
    for entity_id, doc_ids in mention_index.items():
        if doc_ids and doc_ids.issubset({deleted_doc_id}):
            orphans.append(entity_id)
    return sorted(orphans)


def build_orphan_entity_delete_cypher(
    entity_ids: list[str], tenant_id: str
) -> tuple[str, dict]:
    """Delete the chosen orphan entities, tenant-scoped (DETACH removes RELATED_TO,
    IN_COMMUNITY edges). Communities left with no members are cleaned separately.

    Both the entity-id list and the tenant are bound as ``$params``.
    """
    cypher = (
        "MATCH (e:Entity) "
        # Entities are keyed on ``name`` in the real KG (kg_writer MERGE); the param
        # name stays ``entity_ids`` so callers are unchanged, only the graph property
        # read switches from the non-existent ``entity_id`` to ``name``.
        "WHERE e.name IN $entity_ids AND e.tenant_id = $tenant_id "
        "DETACH DELETE e "
        "RETURN count(e) AS deleted"
    )
    return cypher, {"entity_ids": entity_ids, "tenant_id": tenant_id}


def build_document_node_delete_cypher(
    upload_id: str, tenant_id: str
) -> tuple[str, dict]:
    """Delete the document's OWN (:Document) + (:Page) nodes, tenant + doc scoped.

    The chunk cascade removes (:Chunk) and the orphan/community sweeps remove
    dangling (:Entity)/(:Community), but neither touches the (:Document)/(:Page)
    nodes ``neo4j_writer`` creates (keyed on ``doc_id`` + ``tenant_id``; Page also
    on ``page_num``). Without this the document leaves Document/Page residue behind.
    Both labels are constrained on ``doc_id`` AND ``tenant_id`` (bound as $params,
    never inlined) so no cross-tenant / cross-doc node is ever removed. DETACH
    removes the CONTAINS edges too. Returns a deleted count so the caller can log.
    """
    cypher = (
        "MATCH (p:Page) "
        "WHERE p.doc_id = $upload_id AND p.tenant_id = $tenant_id "
        "DETACH DELETE p "
        "WITH count(p) AS pages_deleted "
        "MATCH (d:Document) "
        "WHERE d.doc_id = $upload_id AND d.tenant_id = $tenant_id "
        "DETACH DELETE d "
        "RETURN pages_deleted + count(d) AS deleted"
    )
    return cypher, {"upload_id": upload_id, "tenant_id": tenant_id}


def build_orphan_community_delete_cypher(tenant_id: str) -> tuple[str, dict]:
    """Delete communities with no remaining IN_COMMUNITY members (tenant-scoped).

    Defense-in-depth (Fix 8): the member-existence check matches an Entity scoped to
    the SAME ``tenant_id`` (``(:Entity {tenant_id:$tenant_id})``) rather than an
    anonymous node, so a (theoretically impossible) cross-tenant IN_COMMUNITY edge
    can never keep this tenant's community alive — the invariant is enforced
    in-query, not assumed.
    """
    cypher = (
        "MATCH (cm:Community) "
        "WHERE cm.tenant_id = $tenant_id "
        "AND NOT ( (:Entity {tenant_id:$tenant_id})-[:IN_COMMUNITY]->(cm) ) "
        "DETACH DELETE cm "
        "RETURN count(cm) AS deleted"
    )
    return cypher, {"tenant_id": tenant_id}

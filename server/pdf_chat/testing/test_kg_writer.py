"""Phase-2 Task 8 — tests for the Neo4j KG writer.

The neo4j SESSION/DRIVER is fully MOCKED (no live infra). A fake driver captures
every ``session.run(cypher, **params)`` call so we can assert two invariants on
EVERY write (spec §3 inv 3, manager hard rules):

    1. each node is MERGEd on ``(<businessKey>, tenant_id)`` — never a bare CREATE.
    2. every write binds ``$tenant_id`` and passes the tenant_id param through.

The static cypher helpers are also asserted directly (pure-string, infra-free) so
the schema contract (RELATED_TO grounding props, HAS_CHUNK, MENTIONS, tags,
IN_COMMUNITY) is reviewable without a database.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pdf_chat.ingestion.kg_writer import Neo4jKGWriter


# ── fake neo4j session/driver (mocks-only — no live infra) ───────────────────
class _FakeSession:
    def __init__(self, calls: list):
        self._calls = calls

    def run(self, cypher, **params):
        self._calls.append((cypher, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, calls: list):
        self._calls = calls

    def session(self, database=None):
        return _FakeSession(self._calls)

    def close(self):
        pass


def _writer_with_capture():
    """Construct a writer whose driver is the fake capture driver."""
    calls: list = []
    w = Neo4jKGWriter("bolt://x", "neo4j", "pw")
    w._driver = _FakeDriver(calls)  # inject — bypasses _require_driver guard
    return w, calls


# ── duck-typed inputs (mirror GroundedEdge/GroundedTag/ResolvedEntity/Section) ─
@dataclass(frozen=True)
class _Ent:
    name: str
    etype: str = "custom:org:acme"
    normalized_value: str | None = "acme"
    aliases: list = field(default_factory=list)


@dataclass(frozen=True)
class _Edge:
    subject: str
    predicate: str
    obj: str
    confidence: float
    span: str
    src_chunk_id: str
    evidence_count: int


@dataclass(frozen=True)
class _Mention:
    chunk_id: str
    name: str
    confidence: float = 0.9
    span: str = "Acme"


@dataclass(frozen=True)
class _Section:
    section_id: str
    doc_id: str
    chunk_ids: list
    fingerprint: str = "fp16"
    page_span: tuple = (1, 2)


@dataclass(frozen=True)
class _Tag:
    label: str
    scope: str
    confidence: float
    span: str
    src_chunk_id: str
    owner_key: str = "Acme"


@dataclass(frozen=True)
class _Assign:
    community_id: str
    name: str
    size: int = 3
    pagerank: float = 0.4


@dataclass(frozen=True)
class _Report:
    community_id: str
    summary: str
    citations: tuple = ()
    grounded_edge_count: int = 0
    embedding: list = field(default_factory=list)


TENANT = "t1"


# --------------------------------------------------------------------------- #
# Static cypher contract (infra-free)
# --------------------------------------------------------------------------- #
def test_entity_cypher_merges_on_name_and_tenant():
    cy = Neo4jKGWriter._entity_cypher()
    assert "CREATE (e:Entity" not in cy  # no bare CREATE → no dup, no leak
    assert "MERGE (e:Entity {name: $name, tenant_id: $tenant_id})" in cy
    assert "e.normalized_value = $normalized_value" in cy  # Phase-4 bridge


def test_related_to_cypher_carries_grounding_props_and_tenant_endpoints():
    cy = Neo4jKGWriter._related_to_cypher()
    # both endpoint entities merged on (name, tenant_id) — edge can't bridge tenants
    assert "MERGE (a:Entity {name: $subject, tenant_id: $tenant_id})" in cy
    assert "MERGE (b:Entity {name: $obj, tenant_id: $tenant_id})" in cy
    # the RELATED_TO edge MERGEs (no bare CREATE) and binds tenant_id
    assert ":RELATED_TO {predicate: $predicate, tenant_id: $tenant_id}" in cy
    # the spec'd grounding props ride on the edge
    for prop in ("r.desc", "r.weight", "r.confidence", "r.evidence_count", "r.src_chunk"):
        assert prop in cy


def test_mentions_cypher_is_chunk_to_entity_tenant_scoped():
    cy = Neo4jKGWriter._mentions_cypher()
    assert "MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id})" in cy
    assert "MERGE (e:Entity {name: $name, tenant_id: $tenant_id})" in cy
    assert "(c)-[m:MENTIONS {tenant_id: $tenant_id}]->(e)" in cy


def test_section_cypher_is_section_has_chunk_tenant_scoped():
    cy = Neo4jKGWriter._section_cypher()
    assert "MERGE (s:Section {section_id: $section_id, tenant_id: $tenant_id})" in cy
    assert "MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id})" in cy
    assert "(s)-[:HAS_CHUNK]->(c)" in cy


def test_tag_cypher_merges_on_label_scope_tenant():
    cy = Neo4jKGWriter._tag_cypher()
    assert "MERGE (t:Tag {label: $label, scope: $scope, tenant_id: $tenant_id})" in cy
    assert "tenant_id: $tenant_id" in cy
    # tag provenance rides on the HAS_TAG edge (tags are a grounded retrieval signal)
    assert "h.confidence = $confidence" in cy and "h.src_chunk = $src_chunk" in cy


def test_community_cypher_is_entity_in_community_tenant_scoped():
    cy = Neo4jKGWriter._community_cypher()
    assert "MERGE (k:Community {community_id: $community_id, tenant_id: $tenant_id})" in cy
    assert "MERGE (e:Entity {name: $name, tenant_id: $tenant_id})" in cy
    assert "(e)-[:IN_COMMUNITY]->(k)" in cy


def test_community_report_cypher_merges_on_community_and_tenant():
    cy = Neo4jKGWriter._community_report_cypher()
    assert "CREATE (" not in cy  # MERGE-only → idempotent, tenant-keyed
    assert (
        "MERGE (n:CommunityReport {community_id: $community_id, tenant_id: $tenant_id})"
        in cy
    )
    # The searcher's community_report_lookup reads node.report + node.citations —
    # the writer MUST set exactly those properties so the citation loop closes.
    assert "n.report = $report" in cy
    assert "n.citations = $citations" in cy
    # plus the summary alias, grounding-strength count, and the embedding that
    # populates community_report_vector_index.
    assert "n.summary = $report" in cy
    assert "n.grounded_edge_count = $grounded_edge_count" in cy
    assert "n.embedding = $embedding" in cy


# --------------------------------------------------------------------------- #
# MERGE-on-(key, tenant_id) holds for EVERY cypher constant (no bare CREATE)
# --------------------------------------------------------------------------- #
def test_every_node_cypher_uses_merge_not_bare_create():
    for cy in (
        Neo4jKGWriter._entity_cypher(),
        Neo4jKGWriter._related_to_cypher(),
        Neo4jKGWriter._mentions_cypher(),
        Neo4jKGWriter._section_cypher(),
        Neo4jKGWriter._tag_cypher(),
        Neo4jKGWriter._community_cypher(),
    ):
        # no node is created with a bare CREATE — every node is a MERGE so
        # replay is idempotent and tenant_id is always part of the merge key.
        assert "CREATE (" not in cy
        assert "$tenant_id" in cy  # every statement binds the tenant param


# --------------------------------------------------------------------------- #
# Runtime writes: tenant_id passed on EVERY call (mocked session)
# --------------------------------------------------------------------------- #
def test_write_entities_binds_tenant_id_on_every_call():
    w, calls = _writer_with_capture()
    n = w.write_entities([_Ent("Acme"), _Ent("Globex")], tenant_id=TENANT)
    assert n == 2 and len(calls) == 2
    for cy, params in calls:
        assert "MERGE (e:Entity {name: $name, tenant_id: $tenant_id})" in cy
        assert params["tenant_id"] == TENANT
        assert "name" in params


def test_write_related_to_binds_grounding_and_tenant():
    w, calls = _writer_with_capture()
    edge = _Edge("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1", 1)
    assert w.write_related_to([edge], tenant_id=TENANT) == 1
    cy, params = calls[0]
    assert params["tenant_id"] == TENANT
    assert params["subject"] == "Acme" and params["obj"] == "Globex"
    assert params["confidence"] == 0.9
    assert params["evidence_count"] == 1
    assert params["src_chunk"] == "c1"  # grounding provenance bound


def test_write_mentions_binds_tenant_id():
    w, calls = _writer_with_capture()
    assert w.write_mentions([_Mention("c1", "Acme")], tenant_id=TENANT) == 1
    cy, params = calls[0]
    assert params["tenant_id"] == TENANT
    assert params["chunk_id"] == "c1" and params["name"] == "Acme"


def test_write_sections_emits_one_has_chunk_per_member_with_tenant():
    w, calls = _writer_with_capture()
    sec = _Section("d::s0", "d", ["c1", "c2", "c3"])
    assert w.write_sections([sec], tenant_id=TENANT) == 3
    assert len(calls) == 3
    seen = set()
    for cy, params in calls:
        assert params["tenant_id"] == TENANT
        assert params["section_id"] == "d::s0"
        seen.add(params["chunk_id"])
    assert seen == {"c1", "c2", "c3"}


def test_write_tags_binds_tenant_and_provenance():
    w, calls = _writer_with_capture()
    tag = _Tag("describes acquisitions", "doc", 0.8, "acquisitions", "c1")
    assert w.write_tags([tag], tenant_id=TENANT) == 1
    cy, params = calls[0]
    assert params["tenant_id"] == TENANT
    assert params["label"] == "describes acquisitions"
    assert params["scope"] == "doc"
    assert params["src_chunk"] == "c1"  # tag provenance bound


def test_write_communities_binds_tenant_id():
    w, calls = _writer_with_capture()
    assert w.write_communities([_Assign("k0", "Acme")], tenant_id=TENANT) == 1
    cy, params = calls[0]
    assert params["tenant_id"] == TENANT
    assert params["community_id"] == "k0" and params["name"] == "Acme"


def test_write_community_reports_embeds_and_keys_on_community_and_tenant():
    w, calls = _writer_with_capture()
    rep = _Report(
        community_id="t1::comm0",
        summary="Acme acquired Globex in 2025.",
        citations=("c1", "c2"),
        grounded_edge_count=3,
        embedding=[0.1, 0.2, 0.3],
    )
    assert w.write_community_reports([rep], tenant_id=TENANT) == 1
    cy, params = calls[0]
    # MERGE-keyed on (community_id, tenant_id) and binds $tenant_id.
    assert (
        "MERGE (n:CommunityReport {community_id: $community_id, tenant_id: $tenant_id})"
        in cy
    )
    assert params["tenant_id"] == TENANT
    assert params["community_id"] == "t1::comm0"
    # The report text + citations + grounding count + embedding are all persisted.
    assert params["report"] == "Acme acquired Globex in 2025."
    assert params["citations"] == ["c1", "c2"]
    assert params["grounded_edge_count"] == 3
    assert params["embedding"] == [0.1, 0.2, 0.3]


def test_write_community_reports_skips_none_suppressed_reports():
    # A suppressed (None) report is never persisted, embedded, or made retrievable.
    w, calls = _writer_with_capture()
    rep = _Report("t1::comm0", "summary", ("c1",), 2, [0.5])
    assert w.write_community_reports([None, rep, None], tenant_id=TENANT) == 1
    assert len(calls) == 1
    _, params = calls[0]
    assert params["community_id"] == "t1::comm0"


def test_write_community_reports_empty_short_circuits():
    w = Neo4jKGWriter("bolt://x", "neo4j", "pw")  # no driver → would raise if touched
    assert w.write_community_reports([], tenant_id=TENANT) == 0


def test_every_runtime_write_passes_tenant_id():
    """One sweep: every write_* method binds tenant_id on every captured call."""
    w, calls = _writer_with_capture()
    w.write_entities([_Ent("Acme")], tenant_id=TENANT)
    w.write_related_to(
        [_Edge("Acme", "acquired", "Globex", 0.9, "Acme acquired Globex", "c1", 1)],
        tenant_id=TENANT,
    )
    w.write_mentions([_Mention("c1", "Acme")], tenant_id=TENANT)
    w.write_sections([_Section("d::s0", "d", ["c1"])], tenant_id=TENANT)
    w.write_tags([_Tag("topic", "section", 0.7, "topic", "c1")], tenant_id=TENANT)
    w.write_communities([_Assign("k0", "Acme")], tenant_id=TENANT)
    assert calls, "expected captured writes"
    for cy, params in calls:
        assert params["tenant_id"] == TENANT
        assert "$tenant_id" in cy


# --------------------------------------------------------------------------- #
# Guarded driver + empty short-circuit (mirror neo4j_writer.py contract)
# --------------------------------------------------------------------------- #
def test_constructible_without_driver_and_empty_short_circuits():
    w = Neo4jKGWriter("bolt://x", "neo4j", "pw")
    # empty inputs short-circuit before touching the (absent) driver
    assert w.write_entities([], tenant_id=TENANT) == 0
    assert w.write_related_to([], tenant_id=TENANT) == 0
    assert w.write_mentions([], tenant_id=TENANT) == 0
    assert w.write_sections([], tenant_id=TENANT) == 0
    assert w.write_tags([], tenant_id=TENANT) == 0
    assert w.write_communities([], tenant_id=TENANT) == 0


def test_write_requires_driver_when_absent(monkeypatch):
    import pdf_chat.ingestion.kg_writer as kw

    # simulate the neo4j driver being unavailable → a real write must raise
    monkeypatch.setattr(kw, "_HAS_NEO4J", False)
    w = Neo4jKGWriter("bolt://x", "neo4j", "pw")
    with pytest.raises(RuntimeError):
        w.write_entities([_Ent("Acme")], tenant_id=TENANT)

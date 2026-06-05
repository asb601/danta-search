"""Phase 5 — comprehension foundation tests (mocks-only, no live infra).

This file is OWNED by the foundation worker for:
  * Task 2 — versioned tenant-ontology + glossary ORM models
  * Task 3 — comprehension runtime migration (idempotent)
  * reader.py FOUNDATION — GraphReader Protocol + read-only ORM/graph helpers

The ontology/temporal worker APPENDS its Task 7/8/11 tests below, sequentially,
AFTER this worker finishes (so there is no concurrent edit to this file).
"""
from __future__ import annotations

import pytest
from sqlalchemy import Enum as SAEnum


# ── Task 2 — ORM models ───────────────────────────────────────────────────────
def test_models_importable_and_tenant_scoped():
    from pdf_chat.models.comprehension import (
        DocTaxonomyClass,
        GlossaryEntry,
        KeyMetric,
        OntologyEntity,
        OntologyRelationship,
        TemporalCoverage,
        TenantOntology,
    )

    # Every comprehension table is tenant-scoped.
    for model in (
        TenantOntology, OntologyEntity, OntologyRelationship,
        DocTaxonomyClass, TemporalCoverage, KeyMetric, GlossaryEntry,
    ):
        cols = set(model.__table__.columns.keys())
        assert "tenant_id" in cols, model.__name__

    # TenantOntology is versioned + container-scoped.
    onto_cols = set(TenantOntology.__table__.columns.keys())
    assert {"ontology_id", "tenant_id", "container_id", "version",
            "built_at", "source_graph_signature", "status"} <= onto_cols
    assert TenantOntology.__table__.columns["version"].type.python_type is int


def test_tenant_ontology_unique_version_per_tenant():
    from pdf_chat.models.comprehension import TenantOntology

    uniques = {
        tuple(sorted(c.name for c in con.columns))
        for con in TenantOntology.__table__.constraints
        if con.__class__.__name__ == "UniqueConstraint"
    }
    assert tuple(sorted(("tenant_id", "version"))) in uniques


def test_glossary_entry_columns_and_unique():
    from pdf_chat.models.comprehension import GlossaryEntry

    cols = set(GlossaryEntry.__table__.columns.keys())
    assert {
        "id", "tenant_id", "container_id", "ontology_version", "term",
        "expansion", "definition", "provenance", "confidence", "variants",
        "evidence_spans", "first_seen", "created_at",
    } <= cols
    assert GlossaryEntry.__table__.columns["ontology_version"].type.python_type is int
    # Unique (tenant_id, term, ontology_version) so a re-version is a new row.
    uniques = {
        tuple(sorted(c.name for c in con.columns))
        for con in GlossaryEntry.__table__.constraints
        if con.__class__.__name__ == "UniqueConstraint"
    }
    assert tuple(sorted(("tenant_id", "term", "ontology_version"))) in uniques


def test_glossary_open_vocab_text_columns_not_enum():
    """provenance is open-vocab Text, never a SQLAlchemy Enum (invariant 6)."""
    from pdf_chat.models.comprehension import GlossaryEntry

    assert not isinstance(GlossaryEntry.__table__.columns["provenance"].type, SAEnum)


def test_doc_taxonomy_doc_class_is_free_text_not_enum():
    from pdf_chat.models.comprehension import DocTaxonomyClass

    cols = set(DocTaxonomyClass.__table__.columns.keys())
    assert {"id", "ontology_id", "tenant_id", "doc_class", "confidence",
            "member_doc_ids"} <= cols
    assert not isinstance(DocTaxonomyClass.__table__.columns["doc_class"].type, SAEnum)


def test_relationship_three_state_is_text_not_enum():
    """OntologyRelationship.state is asserted|not_stated|conflicting as open Text."""
    from pdf_chat.models.comprehension import OntologyRelationship

    cols = set(OntologyRelationship.__table__.columns.keys())
    assert {"id", "ontology_id", "tenant_id", "src_name", "dst_name",
            "relation", "state", "confidence", "evidence"} <= cols
    assert not isinstance(OntologyRelationship.__table__.columns["state"].type, SAEnum)
    assert not isinstance(OntologyRelationship.__table__.columns["relation"].type, SAEnum)


def test_models_registered_on_shared_base():
    from app.core.database import Base
    import pdf_chat.models  # noqa: F401 — package import registers every table

    for table in (
        "pdf_tenant_ontology", "pdf_ontology_entity", "pdf_ontology_relationship",
        "pdf_doc_taxonomy_class", "pdf_temporal_coverage", "pdf_key_metric",
        "pdf_glossary_entry",
    ):
        assert table in Base.metadata.tables, table


# ── Task 3 — migration ─────────────────────────────────────────────────────────
def test_migration_exposes_run_migration_and_alias():
    from pdf_chat.migrations import comprehension_upgrade

    assert callable(comprehension_upgrade.apply_comprehension_migration)
    assert comprehension_upgrade.upgrade is comprehension_upgrade.apply_comprehension_migration


def test_migration_indexes_are_if_not_exists():
    from pdf_chat.migrations import comprehension_upgrade

    assert comprehension_upgrade._INDEXES
    for stmt in comprehension_upgrade._INDEXES:
        assert "CREATE INDEX IF NOT EXISTS" in stmt
    # The two indexes the plan calls for are present.
    joined = " ".join(comprehension_upgrade._INDEXES)
    assert "(tenant_id, term)" in joined
    assert "(tenant_id, version)" in joined


@pytest.mark.asyncio
async def test_migration_idempotent_on_fake_engine():
    """Table-scoped create + IF-NOT-EXISTS indexes; a second call raises nothing."""
    from pdf_chat.migrations import comprehension_upgrade

    executed: list[str] = []
    create_calls = {"n": 0}

    class FakeConn:
        async def run_sync(self, fn):
            create_calls["n"] += 1

        async def execute(self, stmt):
            executed.append(str(stmt))

    class FakeBegin:
        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *a):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    eng = FakeEngine()
    await comprehension_upgrade.apply_comprehension_migration(eng)
    await comprehension_upgrade.apply_comprehension_migration(eng)  # idempotent
    # Each call ran the table-scoped create + every index DDL.
    assert create_calls["n"] == 2
    assert len(executed) == 2 * len(comprehension_upgrade._INDEXES)
    for stmt in executed:
        assert "IF NOT EXISTS" in stmt


@pytest.mark.asyncio
async def test_migration_is_non_fatal_on_engine_error():
    """A broken engine logs a warning but never raises (non-fatal)."""
    from pdf_chat.migrations import comprehension_upgrade

    class BoomEngine:
        def begin(self):
            raise RuntimeError("db down")

    # Must not raise.
    await comprehension_upgrade.apply_comprehension_migration(BoomEngine())


# ── reader.py FOUNDATION — GraphReader Protocol + read helpers ─────────────────
class _FakeGraphReader:
    """In-memory GraphReader fake satisfying the Protocol (async generators)."""

    def __init__(self, *, entities=None, relationships=None, communities=None,
                 documents=None, chunks=None, entity_chunk_map=None):
        self._entities = entities or []
        self._relationships = relationships or []
        self._communities = communities or []
        self._documents = documents or []
        self._chunks = chunks or []
        self._entity_chunk_map = entity_chunk_map or {}

    async def iter_entities(self, tenant_id):
        for e in self._entities:
            yield e

    async def iter_relationships(self, tenant_id):
        for r in self._relationships:
            yield r

    async def iter_communities(self, tenant_id):
        for c in self._communities:
            yield c

    async def iter_documents(self, tenant_id):
        for d in self._documents:
            yield d

    async def iter_chunks(self, tenant_id):
        for c in self._chunks:
            yield c

    async def entity_chunks(self, tenant_id, entity_name):
        for c in self._entity_chunk_map.get(entity_name, []):
            yield c


def test_graphreader_protocol_isinstance():
    from pdf_chat.comprehension.reader import GraphReader

    assert isinstance(_FakeGraphReader(), GraphReader)


def _glossary_row(**kw):
    """A lightweight stand-in for a GlossaryEntry ORM row (attr access only)."""
    from types import SimpleNamespace

    base = dict(
        id="g1", tenant_id="t1", container_id="c1", ontology_version=1,
        term="CAC", expansion="Customer Acquisition Cost",
        definition="spend per new customer", provenance="stated",
        confidence=0.9, variants=["Cust. Acq. Cost"],
        evidence_spans=[{"chunk_id": "ch1", "text": "..."}],
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Records executed statements and returns canned scalar results in order."""

    def __init__(self, results):
        self._results = list(results)
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        return self._results.pop(0) if self._results else _FakeResult([])


@pytest.mark.asyncio
async def test_lookup_glossary_returns_matching_row():
    from pdf_chat.comprehension import reader

    row = _glossary_row()
    session = _FakeSession([_FakeResult([row])])
    out = await reader.lookup_glossary(session, "t1", "CAC")
    assert out is row
    assert session.executed, "lookup_glossary must issue a scoped query"


@pytest.mark.asyncio
async def test_lookup_glossary_miss_returns_none():
    from pdf_chat.comprehension import reader

    session = _FakeSession([_FakeResult([])])
    out = await reader.lookup_glossary(session, "t1", "UNKNOWN")
    assert out is None


@pytest.mark.asyncio
async def test_list_glossary_returns_rows():
    from pdf_chat.comprehension import reader

    rows = [_glossary_row(term="CAC"), _glossary_row(id="g2", term="NRR")]
    session = _FakeSession([_FakeResult(rows)])
    out = await reader.list_glossary(session, "t1")
    assert [r.term for r in out] == ["CAC", "NRR"]


@pytest.mark.asyncio
async def test_current_ontology_version_reads_max():
    from pdf_chat.comprehension import reader

    session = _FakeSession([_FakeResult([3])])
    v = await reader.current_ontology_version(session, "t1")
    assert v == 3


@pytest.mark.asyncio
async def test_current_ontology_version_none_when_no_artifact():
    from pdf_chat.comprehension import reader

    session = _FakeSession([_FakeResult([])])
    v = await reader.current_ontology_version(session, "t1")
    assert v is None


@pytest.mark.asyncio
async def test_list_entities_scoped_to_ontology():
    from pdf_chat.comprehension import reader
    from types import SimpleNamespace

    ents = [SimpleNamespace(name="Acme", entity_type="org"),
            SimpleNamespace(name="Q3", entity_type="period")]
    session = _FakeSession([_FakeResult(ents)])
    out = await reader.list_entities(session, "onto1")
    assert [e.name for e in out] == ["Acme", "Q3"]


@pytest.mark.asyncio
async def test_list_doc_taxonomy_scoped_to_ontology():
    from pdf_chat.comprehension import reader
    from types import SimpleNamespace

    classes = [SimpleNamespace(doc_class="quarterly board deck", confidence=0.8)]
    session = _FakeSession([_FakeResult(classes)])
    out = await reader.list_doc_taxonomy(session, "onto1")
    assert out[0].doc_class == "quarterly board deck"


@pytest.mark.asyncio
async def test_list_temporal_coverage_scoped_to_ontology():
    """list_temporal_coverage is a pure read scoped to ONE ontology version."""
    from pdf_chat.comprehension import reader
    from types import SimpleNamespace

    rows = [SimpleNamespace(subject="Acme", subject_kind="entity",
                            last_mention_date=None, density=1.0)]
    session = _FakeSession([_FakeResult(rows)])
    out = await reader.list_temporal_coverage(session, "onto1")
    assert [c.subject for c in out] == ["Acme"]
    assert session.executed, "list_temporal_coverage must issue a scoped query"


@pytest.mark.asyncio
async def test_topic_map_projects_community_reports_with_citations():
    """topic_map projects (:Community) reports as the company's table-of-contents,
    each carrying citations (invariant 1)."""
    from pdf_chat.comprehension import reader

    communities = [
        {"community_id": "cm1", "report": "Revenue operations overview",
         "level": 0, "citations": ["ch1", "ch2"]},
        {"community_id": "cm2", "report": "Hiring and headcount",
         "level": 0, "citations": ["ch9"]},
    ]
    rdr = _FakeGraphReader(communities=communities)
    topics = await reader.topic_map(rdr, "t1")
    assert len(topics) == 2
    assert topics[0]["report"] == "Revenue operations overview"
    assert topics[0]["citations"] == ["ch1", "ch2"]
    # Every topic carries citations (grounded table-of-contents).
    assert all("citations" in t for t in topics)


@pytest.mark.asyncio
async def test_topic_map_accepts_object_communities():
    """Communities may arrive as objects (Neo4jSearcher rows) not just dicts."""
    from pdf_chat.comprehension import reader
    from types import SimpleNamespace

    communities = [
        SimpleNamespace(community_id="cm1", report="Topic A", level=0,
                        citations=["ch1"]),
    ]
    rdr = _FakeGraphReader(communities=communities)
    topics = await reader.topic_map(rdr, "t1")
    assert topics[0]["community_id"] == "cm1"
    assert topics[0]["citations"] == ["ch1"]


# ===========================================================================
# Task 8 — temporal coverage + staleness annotation  (ontology/temporal worker)
# ===========================================================================
from datetime import datetime, timezone


def _dt(s: str) -> datetime:
    """Parse 'YYYY-MM-DD' into a tz-aware UTC datetime (deterministic tests)."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_temporal_coverage_per_subject_min_max_density_last_mention():
    """compute_temporal_coverage yields per-subject min/max/density/last_mention."""
    from pdf_chat.comprehension import temporal

    # Two entities, each mentioned in dated chunks. Acme spans 3 months with 3
    # mentions; Globex is a single mention.
    chunks = [
        {"chunk_id": "c1", "doc_date": "2025-01-15", "entities": ["Acme"]},
        {"chunk_id": "c2", "doc_date": "2025-02-15", "entities": ["Acme"]},
        {"chunk_id": "c3", "doc_date": "2025-03-15", "entities": ["Acme", "Globex"]},
    ]
    rdr = _FakeGraphReader(chunks=chunks)

    cov = await temporal.compute_temporal_coverage(
        rdr, tenant_id="t1", container_id="c1"
    )
    by_subject = {c["subject"]: c for c in cov}

    assert by_subject["Acme"]["min_date"] == _dt("2025-01-15")
    assert by_subject["Acme"]["max_date"] == _dt("2025-03-15")
    assert by_subject["Acme"]["last_mention_date"] == _dt("2025-03-15")
    # 3 mentions over a ~2-month span ⇒ density is mentions/span and positive.
    assert by_subject["Acme"]["density"] is not None
    assert by_subject["Acme"]["density"] > 0
    # Single-mention subject still recorded with min==max.
    assert by_subject["Globex"]["min_date"] == _dt("2025-03-15")
    assert by_subject["Globex"]["max_date"] == _dt("2025-03-15")
    assert by_subject["Globex"]["last_mention_date"] == _dt("2025-03-15")


@pytest.mark.asyncio
async def test_temporal_coverage_falls_back_to_document_dates():
    """When a chunk lacks its own date, the date comes from its (:Document)."""
    from pdf_chat.comprehension import temporal

    documents = [{"doc_id": "d1", "doc_date": "2024-06-01"}]
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "entities": ["Zephyr"]}]
    rdr = _FakeGraphReader(documents=documents, chunks=chunks)

    cov = await temporal.compute_temporal_coverage(
        rdr, tenant_id="t1", container_id="c1"
    )
    by_subject = {c["subject"]: c for c in cov}
    assert by_subject["Zephyr"]["last_mention_date"] == _dt("2024-06-01")


def test_staleness_annotation_old_mention_gets_note():
    """A mention older than the threshold gets a human note (no literal)."""
    from pdf_chat.comprehension import temporal

    last = _dt("2023-09-10")
    now = _dt("2026-06-05")  # fixed clock — never the wall clock
    note = temporal.staleness_annotation(last, now, container_id="c1")
    assert note is not None
    # Human-facing, mentions recency, never a raw confidence number.
    assert "may be outdated" in note
    assert "2023-09" in note


def test_staleness_annotation_fresh_mention_returns_none():
    """A recent mention is NOT annotated (returns None)."""
    from pdf_chat.comprehension import temporal

    last = _dt("2026-05-01")
    now = _dt("2026-06-05")
    assert temporal.staleness_annotation(last, now, container_id="c1") is None


def test_staleness_annotation_none_last_mention_returns_none():
    """No known last-mention date ⇒ no staleness claim (refuse to guess)."""
    from pdf_chat.comprehension import temporal

    now = _dt("2026-06-05")
    assert temporal.staleness_annotation(None, now, container_id="c1") is None


# ===========================================================================
# Task 7 — versioned ontology builder  (ontology/temporal worker)
# ===========================================================================
class _RecordingSession:
    """An AsyncSession fake: records added rows + serves canned execute results.

    ``execute`` pops the next canned result (for the max-version read); ``add``
    captures persisted ORM rows so tests can assert the artifact was projected.
    """

    def __init__(self, results=None):
        self._results = list(results or [])
        self.added: list = []

    async def execute(self, stmt):
        return self._results.pop(0) if self._results else _FakeResult([])

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        return None


class _DocClusterLLM:
    """Injected fake LLM for open-vocab doc-taxonomy clustering.

    Mirrors the ``CommunityReporter`` injected-LLM convention: exposes an ASYNC
    ``synthesize(prompt, *, model_id, container_id, **kw) -> dict`` (awaited by the
    builder, like the glossary miner's seam) returning a list of arbitrary-string
    doc classes (NOT an enumerated list) with confidences. Records the model_id it
    was called with so the test can assert bulk routing.
    """

    def __init__(self, classes):
        self._classes = classes
        self.model_ids: list[str] = []
        self.calls = 0

    async def synthesize(self, prompt, *, model_id, container_id, **kw):
        self.calls += 1
        self.model_ids.append(model_id)
        return {"classes": self._classes}


def _onto_graph():
    """A small grounded graph fixture (entities/edges/communities/docs/chunks)."""
    entities = [
        {"name": "Acme", "normalized_value": "acme", "type": "org",
         "pagerank": 0.4, "mention_count": 5,
         "evidence_chunk_ids": ["c1", "c2"]},
        {"name": "NRR", "normalized_value": "nrr", "type": "metric",
         "pagerank": 0.2, "mention_count": 3, "evidence_chunk_ids": ["c3"]},
    ]
    relationships = [
        {"src_name": "Acme", "dst_name": "NRR", "relation": "reports",
         "confidence": 0.8, "evidence": {"chunk_id": "c1", "span": "Acme NRR"}},
    ]
    communities = [
        {"community_id": "cm1", "report": "Revenue ops", "level": 0,
         "citations": ["c1"]},
    ]
    documents = [
        {"doc_id": "d1", "title": "Q3 Board Deck", "doc_date": "2025-03-15"},
    ]
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "doc_date": "2025-03-15",
         "text": "Acme reports NRR", "entities": ["Acme", "NRR"]},
    ]
    return dict(entities=entities, relationships=relationships,
                communities=communities, documents=documents, chunks=chunks)


@pytest.mark.asyncio
async def test_build_then_rebuild_bumps_version():
    from pdf_chat.comprehension import ontology_builder
    from pdf_chat.models.comprehension import (
        OntologyEntity, OntologyRelationship, TenantOntology,
    )

    rdr = _FakeGraphReader(**_onto_graph())
    llm = _DocClusterLLM([{"doc_class": "quarterly board deck",
                           "confidence": 0.8, "member_doc_ids": ["d1"]}])

    # First build: no prior version ⇒ version 1.
    session = _RecordingSession(results=[_FakeResult([])])  # max(version) is None
    onto1 = await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
    )
    assert isinstance(onto1, TenantOntology)
    assert onto1.version == 1
    assert onto1.source_graph_signature  # recomputed signature present

    added = session.added
    # Entities + relationships + doc-taxonomy projected and persisted.
    entities = [r for r in added if isinstance(r, OntologyEntity)]
    rels = [r for r in added if isinstance(r, OntologyRelationship)]
    assert {e.name for e in entities} == {"Acme", "NRR"}
    # Graph edges land as three-state "asserted".
    assert rels and all(r.state == "asserted" for r in rels)

    # Rebuild: prior max version is 1 ⇒ new row version 2 (old retained).
    session2 = _RecordingSession(results=[_FakeResult([1])])
    onto2 = await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session2, llm=llm,
    )
    assert onto2.version == 2


@pytest.mark.asyncio
async def test_doc_taxonomy_open_vocab_and_bulk_routed():
    """Doc classes are arbitrary LLM strings (open-vocab) and bulk-routed."""
    from pdf_chat.comprehension import ontology_builder
    from pdf_chat.models.comprehension import DocTaxonomyClass

    rdr = _FakeGraphReader(**_onto_graph())
    # An arbitrary, never-enumerated class string proves open-vocab.
    llm = _DocClusterLLM([{"doc_class": "annual sustainability appendix",
                           "confidence": 0.9, "member_doc_ids": ["d1"]}])
    session = _RecordingSession(results=[_FakeResult([])])

    await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
    )
    classes = [r for r in session.added if isinstance(r, DocTaxonomyClass)]
    assert classes
    assert classes[0].doc_class == "annual sustainability appendix"
    # Bulk routing — never the strong/Opus tier for ingestion clustering.
    assert llm.model_ids and all("opus" not in m.lower() for m in llm.model_ids)


@pytest.mark.asyncio
async def test_doc_taxonomy_low_confidence_class_dropped():
    """A low-confidence learned class is gated out (get_tunable + log_gate)."""
    from pdf_chat.comprehension import ontology_builder
    from pdf_chat.models.comprehension import DocTaxonomyClass

    rdr = _FakeGraphReader(**_onto_graph())
    # Confidence below the ontology.doc_taxonomy_min_confidence floor (0.50).
    llm = _DocClusterLLM([{"doc_class": "noise cluster",
                           "confidence": 0.10, "member_doc_ids": ["d1"]}])
    session = _RecordingSession(results=[_FakeResult([])])

    await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
    )
    classes = [r for r in session.added if isinstance(r, DocTaxonomyClass)]
    assert not classes  # dropped below the confidence floor


@pytest.mark.asyncio
async def test_key_metrics_projected_from_metric_entities():
    """Metric-typed graph entities become grounded KeyMetric registry rows."""
    from pdf_chat.comprehension import ontology_builder
    from pdf_chat.models.comprehension import KeyMetric

    rdr = _FakeGraphReader(**_onto_graph())
    llm = _DocClusterLLM([])
    session = _RecordingSession(results=[_FakeResult([])])

    await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
    )
    metrics = [r for r in session.added if isinstance(r, KeyMetric)]
    assert any(m.metric == "NRR" for m in metrics)


@pytest.mark.asyncio
async def test_temporal_coverage_persisted_under_ontology():
    """The builder also persists temporal coverage rows under the new ontology."""
    from pdf_chat.comprehension import ontology_builder
    from pdf_chat.models.comprehension import TemporalCoverage

    rdr = _FakeGraphReader(**_onto_graph())
    llm = _DocClusterLLM([])
    session = _RecordingSession(results=[_FakeResult([])])

    onto = await ontology_builder.build_tenant_ontology(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
    )
    cov = [r for r in session.added if isinstance(r, TemporalCoverage)]
    assert cov
    assert all(r.ontology_id == onto.ontology_id for r in cov)


# ===========================================================================
# Task 11 — finalization orchestrator + EXIT-criteria acceptance (manager)
# ===========================================================================
from pdf_chat.comprehension.provenance import Provenance, label_for


class _FinalizationLLM:
    """One injected bulk LLM seam serving BOTH finalization phases.

    * Doc-taxonomy clustering (ontology builder) calls ``synthesize(prompt, *,
      model_id, container_id, **kw)`` (the ``CommunityReporter`` convention).
    * Glossary mining calls the three async seam methods
      (``confirm_definition`` / ``synthesize_definition`` / ``adjudicate_variants``).

    Production wires one prompt-cached ``gpt-4o-mini`` client behind all four; the
    test injects this fake. Every method records the ``model_id`` it was handed so
    the test can assert BULK routing (never the strong/Opus tier for ingestion).
    """

    def __init__(self, doc_classes):
        self._doc_classes = doc_classes
        self.model_ids: list[str] = []

    # --- doc-taxonomy clustering (async — awaited by the builder, like the
    #     glossary miner's seam; one prompt-cached gpt-4o-mini client behind both) ---
    async def synthesize(self, prompt, *, model_id, container_id, **kw):
        self.model_ids.append(model_id)
        return {"classes": self._doc_classes}

    # --- glossary seam (async) ------------------------------------------------
    async def confirm_definition(self, *, term, expansion, span, model_id,
                                 container_id):
        self.model_ids.append(model_id)
        # Confirm the company-specific explicit definition above the floor.
        return {"confirmed": True, "expansion": expansion,
                "definition": f"{expansion} — defined in the corpus",
                "confidence": 0.95}

    async def synthesize_definition(self, *, term, contexts, model_id,
                                    container_id):
        self.model_ids.append(model_id)
        return {"definition": f"usage-inferred meaning of {term}",
                "confidence": 0.9}

    async def adjudicate_variants(self, *, term, candidates, model_id,
                                  container_id):
        self.model_ids.append(model_id)
        return {"same": True}


class _ExitSession:
    """A finalization-grade AsyncSession fake.

    Records every ``add``ed ORM row (so the test can inspect the persisted
    artifact), assigns deterministic ``ontology_id`` values on ``flush`` (the real
    DB does this), and serves the read queries the orchestrator + reader issue:

      * ``current_ontology_version`` / idempotency-signature / latest-header reads
        are answered from the rows captured so far (so a SECOND finalize is a
        signature-equal no-op);
      * ``lookup_glossary`` (post-finalization) returns the captured GlossaryEntry
        rows for the looked-up term, newest version first.

    The fake is intentionally dumb: it pattern-matches on the compiled SQL text to
    decide which canned projection to serve, which is enough for the exit flow.
    """

    def __init__(self):
        self.added: list = []
        self._onto_seq = 0

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        from pdf_chat.models.comprehension import TenantOntology
        for row in self.added:
            if isinstance(row, TenantOntology) and getattr(row, "ontology_id", None) is None:
                self._onto_seq += 1
                row.ontology_id = f"onto-{self._onto_seq}"

    # --- read side -----------------------------------------------------------
    def _ontologies(self):
        from pdf_chat.models.comprehension import TenantOntology
        return [r for r in self.added if isinstance(r, TenantOntology)]

    def _glossary(self):
        from pdf_chat.models.comprehension import GlossaryEntry
        return [r for r in self.added if isinstance(r, GlossaryEntry)]

    async def execute(self, stmt):
        text = str(stmt).lower()
        ontos = self._ontologies()
        latest = max(ontos, key=lambda o: o.version) if ontos else None

        # glossary lookup by term (table is pdf_glossary_entry).
        if "glossary" in text:
            rows = sorted(self._glossary(), key=lambda g: g.ontology_version,
                          reverse=True)
            return _FakeResult(rows)
        # max(version) — current_ontology_version
        if "max(" in text and "version" in text:
            return _FakeResult([latest.version] if latest else [])
        # Full-header select (latest TenantOntology row) — its column list carries
        # ``ontology_id``. Checked BEFORE the scalar-signature branch, because the
        # full header's column list ALSO contains ``source_graph_signature``.
        if "ontology_id" in text and "tenant_ontology" in text:
            return _FakeResult([latest] if latest else [])
        # source_graph_signature SCALAR select — idempotency probe.
        if "source_graph_signature" in text:
            return _FakeResult([latest.source_graph_signature] if latest else [])
        return _FakeResult([])


def _exit_graph():
    """A grounded graph whose corpus states a COMPANY-SPECIFIC explicit definition.

    ``ZephyrMetric`` is a coined, company-specific term defined in-line in a
    chunk — the exit flow asks "what does ZephyrMetric mean here" and must get a
    CITED, company-specific answer.
    """
    entities = [
        {"name": "Acme", "normalized_value": "acme", "type": "org",
         "pagerank": 0.5, "mention_count": 4, "evidence_chunk_ids": ["c1"]},
        {"name": "ZephyrMetric", "normalized_value": "zephyrmetric",
         "type": "metric", "pagerank": 0.3, "mention_count": 2,
         "evidence_chunk_ids": ["c1"]},
    ]
    relationships = [
        {"src_name": "Acme", "dst_name": "ZephyrMetric", "relation": "tracks",
         "confidence": 0.8, "evidence": {"chunk_id": "c1"}},
    ]
    communities = [
        {"community_id": "cm1", "report": "Revenue operations overview",
         "level": 0, "citations": ["c1"]},
    ]
    documents = [
        {"doc_id": "d1", "title": "Q3 Board Deck", "doc_date": "2025-03-15"},
    ]
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "doc_date": "2025-03-15",
         "page_num": 2, "bbox": [1, 2, 3, 4],
         "text": "The Zephyr Metric (ZephyrMetric) measures pipeline velocity.",
         "entities": ["Acme", "ZephyrMetric"]},
    ]
    return dict(entities=entities, relationships=relationships,
                communities=communities, documents=documents, chunks=chunks)


@pytest.mark.asyncio
async def test_finalization_builds_artifact_and_exit_flow(monkeypatch):
    """End-to-end exit: finalize → topic map + cited glossary answer + version.

    Drives the EXIT scenario with fakes: finalization builds the ontology then
    mines the glossary; then we (1) browse the topic map, (2) ask
    ``glossary_lookup`` for a company-specific term and assert a CITED,
    company-specific answer with a provenance LABEL, and (3) assert
    ``ontology/version`` is queryable. Also asserts glossary build routed bulk
    (``select_model(task=synthesis)`` ⇒ gpt-4o-mini, never strong/Opus).
    """
    from pdf_chat.comprehension import finalize, ontology_builder, glossary_miner
    from pdf_chat.comprehension import reader as comp_reader
    from pdf_chat.model_router import TaskClass, select_model

    # Spy on select_model in BOTH bulk modules: record the task it was called with
    # and the choice it returned, while delegating to the real router.
    tasks_seen: list = []
    choices: list = []

    def _spy(*, task, container_id, signals, store=None):
        tasks_seen.append(TaskClass(task))
        choice = select_model(task=task, container_id=container_id,
                              signals=signals, store=store)
        choices.append(choice)
        return choice

    monkeypatch.setattr(ontology_builder, "select_model", _spy)
    monkeypatch.setattr(glossary_miner, "select_model", _spy)

    rdr = _FakeGraphReader(**_exit_graph())
    llm = _FinalizationLLM(
        [{"doc_class": "quarterly board deck", "confidence": 0.9,
          "member_doc_ids": ["d1"]}]
    )
    session = _ExitSession()
    # A background table that does NOT contain the coined term (open-vocab signal).
    background = {"the": 12.0, "measures": 6.0, "pipeline": 5.0, "velocity": 4.0}

    onto = await finalize.finalize_comprehension(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
        background_freq=background,
    )
    assert onto.version == 1

    # The glossary was mined + persisted, stamped with the ontology's version.
    from pdf_chat.models.comprehension import GlossaryEntry
    glossary_rows = [r for r in session.added if isinstance(r, GlossaryEntry)]
    assert glossary_rows, "finalization must persist mined glossary rows"
    assert all(g.ontology_version == onto.version for g in glossary_rows)

    # ── EXIT (1): browse the topic map — grounded table-of-contents w/ citations ─
    topics = await comp_reader.topic_map(rdr, "t1")
    assert topics and topics[0]["report"] == "Revenue operations overview"
    assert topics[0]["citations"] == ["c1"]

    # ── EXIT (2): "what does ZephyrMetric mean here?" → CITED company-specific ───
    from pdf_chat.agent.tools_glossary import GlossaryLookupTool
    from pdf_chat.agent.state import PdfChatState

    class _Deps:
        def __init__(self, reader, session):
            self.reader = reader
            self.session = session

    tool = GlossaryLookupTool()
    hits = await tool.run(
        PdfChatState(query="what does ZephyrMetric mean here", tenant_id="t1"),
        _Deps(comp_reader, session), term="ZephyrMetric",
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit["term"] == "ZephyrMetric"
    # Company-specific expansion learned from THIS corpus (not generic English):
    # the corpus-stated long form for the coined term surfaces verbatim.
    assert "Zephyr Metric" in (hit["expansion"] or "")
    assert hit["definition"]  # a grounded definition, not a refusal
    # A human provenance LABEL — never a raw confidence number.
    assert hit["provenance"] == label_for(Provenance.STATED)
    assert "confidence" not in hit
    # CITED: the grounding chunk surfaces.
    assert hit["citations"] and hit["citations"][0]["chunk_id"] == "c1"

    # ── EXIT (3): ontology version is queryable ─────────────────────────────────
    version = await comp_reader.current_ontology_version(session, "t1")
    assert version == 1

    # ── C7: glossary + ontology build routed BULK (synthesis), never strong/Opus ─
    assert tasks_seen and all(t is TaskClass.SYNTHESIS for t in tasks_seen)
    assert choices and all(c.is_strong is False for c in choices)
    assert all("opus" not in c.model_id.lower() for c in choices)


def test_signature_detects_chunk_content_change():
    """_compute_signature folds in a chunk-content digest: two graphs identical
    except one chunk's TEXT differs ⇒ DIFFERENT signatures (forces a re-version,
    since the glossary is mined from chunk text). FIX G."""
    from pdf_chat.comprehension.ontology_builder import _compute_signature

    entities = [{"name": "Acme"}]
    relationships = [{"src_name": "Acme", "dst_name": "NRR", "relation": "reports"}]
    documents = [{"doc_id": "d1"}]
    chunks_a = [{"chunk_id": "c1", "text": "Acme reports NRR."}]
    chunks_b = [{"chunk_id": "c1", "text": "Acme reports a DIFFERENT metric."}]

    sig_a = _compute_signature(entities, relationships, documents, chunks_a)
    sig_b = _compute_signature(entities, relationships, documents, chunks_b)
    assert sig_a != sig_b, "a chunk text change must change the signature"

    # A TRULY unchanged chunk substrate yields the SAME signature (idempotency).
    sig_a2 = _compute_signature(entities, relationships, documents,
                                [{"chunk_id": "c1", "text": "Acme reports NRR."}])
    assert sig_a == sig_a2


@pytest.mark.asyncio
async def test_finalization_idempotent_on_unchanged_graph(monkeypatch):
    """A re-finalize on an UNCHANGED graph is a no-op (signature-equal): no new
    version, the existing header is returned."""
    from pdf_chat.comprehension import finalize

    rdr = _FakeGraphReader(**_exit_graph())
    llm = _FinalizationLLM([])
    session = _ExitSession()
    background = {"the": 12.0}

    onto1 = await finalize.finalize_comprehension(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
        background_freq=background,
    )
    from pdf_chat.models.comprehension import TenantOntology
    headers_after_first = [r for r in session.added if isinstance(r, TenantOntology)]
    assert len(headers_after_first) == 1
    assert onto1.version == 1

    # Re-finalize the SAME graph — signature unchanged ⇒ no new header built.
    onto2 = await finalize.finalize_comprehension(
        rdr, tenant_id="t1", container_id="c1", session=session, llm=llm,
        background_freq=background,
    )
    headers_after_second = [r for r in session.added if isinstance(r, TenantOntology)]
    assert len(headers_after_second) == 1  # idempotent: no churn
    assert onto2.version == 1
    assert onto2.source_graph_signature == onto1.source_graph_signature

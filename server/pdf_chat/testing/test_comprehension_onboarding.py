"""Phase 5 — Task 10 tests: the read-only onboarding projection API.

FastAPI ``TestClient`` with a fake principal / reader / session injected via
``app.dependency_overrides`` (no live Neo4j / Postgres). The onboarding surface
is the domain-naive engineer's entry point:

  GET /api/pdf/onboarding/topic-map      → community reports as a cited TOC
  GET /api/pdf/onboarding/entities       → paged entity browse
  GET /api/pdf/onboarding/glossary       → glossary browse with provenance LABELS
  GET /api/pdf/onboarding/doc-taxonomy   → open-vocab learned doc classes
  GET /api/pdf/onboarding/ontology/version → current version int

Guarantees under test:
  * every response is tenant-scoped (tenant from the JWT principal, never client
    input) and read-only (no write verb mounted on the router);
  * the topic map carries citations (grounded TOC — invariant 1);
  * glossary rows surface human provenance LABELS, NOT raw confidence numbers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdf_chat.api import onboarding as onboarding_mod
from pdf_chat.api.onboarding import (
    get_onboarding_reader,
    get_onboarding_session,
    onboarding_router,
)
from pdf_chat.api.routes import Principal, get_principal
from pdf_chat.comprehension.provenance import Provenance, label_for


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeReader:
    """Stands in for pdf_chat.comprehension.reader — records tenant scoping."""

    def __init__(self):
        self.topic_calls: list[str] = []
        self.version_calls: list[str] = []
        self.glossary_version_args: list = []
        self.temporal_rows: list = []

    async def topic_map(self, reader, tenant_id):  # signature mirrors reader.topic_map
        self.topic_calls.append(tenant_id)
        return [
            {"community_id": "cm1", "report": "Revenue operations overview",
             "level": 0, "citations": ["ch1", "ch2"]},
        ]

    async def current_ontology_version(self, session, tenant_id):
        self.version_calls.append(tenant_id)
        return 2

    async def list_entities(self, session, ontology_id, *, limit=None, offset=None):
        return [SimpleNamespace(name="Acme", entity_type="org", normalized_value="acme",
                                pagerank=0.9, mention_count=12)]

    async def list_glossary(self, session, tenant_id, *, ontology_version=None,
                            limit=None, offset=None):
        # Two versions exist for this tenant; the browse MUST be pinned to the
        # latest (v2) — an unpinned (ontology_version=None) call would leak v1 rows.
        self.glossary_version_args.append(ontology_version)
        all_rows = {
            1: [
                SimpleNamespace(term="OLDTERM", expansion="Legacy Term",
                                definition="from an old version",
                                provenance=Provenance.STATED.value, confidence=0.7,
                                variants=None, evidence_spans=[{"chunk_id": "old"}]),
            ],
            2: [
                SimpleNamespace(term="CAC", expansion="Customer Acquisition Cost",
                                definition="spend per new customer",
                                provenance=Provenance.STATED.value, confidence=0.92,
                                variants=["Cust. Acq. Cost"],
                                evidence_spans=[{"chunk_id": "ch1", "bbox": [1, 2, 3, 4]}]),
                SimpleNamespace(term="ZephyrFlow", expansion=None,
                                definition="internal tool (inferred)",
                                provenance=Provenance.INFERRED.value, confidence=0.6,
                                variants=None, evidence_spans=[{"chunk_id": "ch9"}]),
            ],
        }
        if ontology_version is None:
            # Unpinned ⇒ ALL versions (the bug this fix prevents).
            return [r for rows in all_rows.values() for r in rows]
        return all_rows.get(ontology_version, [])

    async def list_doc_taxonomy(self, session, ontology_id):
        return [SimpleNamespace(doc_class="quarterly board deck", confidence=0.8,
                                member_doc_ids=["d1", "d2"])]

    async def latest_ontology_id(self, session, tenant_id):
        return "onto-v2"

    async def list_temporal_coverage(self, session, ontology_id):
        return list(self.temporal_rows)


def _build_client(reader: _FakeReader, *, tenant_id: str = "t1") -> TestClient:
    app = FastAPI()
    app.include_router(onboarding_router)

    async def _fake_principal() -> Principal:
        return Principal(user_id="u1", tenant_id=tenant_id, groups=[])

    app.dependency_overrides[get_principal] = _fake_principal
    app.dependency_overrides[get_onboarding_reader] = lambda: reader
    app.dependency_overrides[get_onboarding_session] = lambda: object()
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Router shape — self-prefixed + read-only (no write verb)
# --------------------------------------------------------------------------- #
def test_router_is_self_prefixed():
    assert onboarding_router.prefix == "/api/pdf/onboarding"


def test_router_is_read_only_no_write_verbs():
    for route in onboarding_router.routes:
        methods = getattr(route, "methods", set()) or set()
        assert methods <= {"GET", "HEAD"}, f"{route.path} exposes a write verb: {methods}"


# --------------------------------------------------------------------------- #
# topic-map — cited table-of-contents, tenant-scoped (invariant 1)
# --------------------------------------------------------------------------- #
def test_topic_map_returns_cited_toc_tenant_scoped():
    reader = _FakeReader()
    client = _build_client(reader, tenant_id="t1")

    resp = client.get("/api/pdf/onboarding/topic-map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    topics = body["topics"]
    assert topics[0]["report"] == "Revenue operations overview"
    assert topics[0]["citations"] == ["ch1", "ch2"]  # grounded TOC
    # tenant was threaded to the reader (never client input).
    assert reader.topic_calls == ["t1"]


# --------------------------------------------------------------------------- #
# entities — paged browse
# --------------------------------------------------------------------------- #
def test_entities_paged_browse():
    reader = _FakeReader()
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/entities?limit=10&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    assert body["entities"][0]["name"] == "Acme"


def test_entities_empty_when_no_ontology():
    """No built ontology → empty entity browse, not a 500."""
    reader = _FakeReader()

    async def _no_onto(session, tenant_id):
        return None

    reader.latest_ontology_id = _no_onto  # type: ignore[assignment]
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/entities")
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


# --------------------------------------------------------------------------- #
# glossary — provenance LABELS, not raw confidence
# --------------------------------------------------------------------------- #
def test_glossary_browse_surfaces_provenance_labels_not_confidence():
    reader = _FakeReader()
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/glossary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    entries = {e["term"]: e for e in body["entries"]}
    assert entries["CAC"]["provenance"] == label_for(Provenance.STATED)
    assert entries["ZephyrFlow"]["provenance"] == label_for(Provenance.INFERRED)
    # Raw confidence number is NEVER surfaced to the user.
    for e in body["entries"]:
        assert "confidence" not in e
    # inferred entry never mislabeled as stated.
    assert entries["ZephyrFlow"]["provenance"] != label_for(Provenance.STATED)


def test_glossary_browse_pins_to_latest_version():
    """When two versions exist, the browse returns ONLY the latest version's rows
    (the route resolves current_ontology_version and pins list_glossary to it)."""
    reader = _FakeReader()  # current_ontology_version → 2; v1 has OLDTERM
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/glossary")
    assert resp.status_code == 200
    terms = {e["term"] for e in resp.json()["entries"]}
    # Latest (v2) terms present; the stale v1 term must NOT leak in.
    assert {"CAC", "ZephyrFlow"} <= terms
    assert "OLDTERM" not in terms
    # The route pinned list_glossary to the resolved latest version (2), never None.
    assert reader.glossary_version_args == [2]


def test_glossary_browse_empty_when_no_ontology():
    """No ontology built yet → empty glossary browse, not a 500 and not all rows."""
    reader = _FakeReader()

    async def _none(session, tenant_id):
        return None

    reader.current_ontology_version = _none  # type: ignore[assignment]
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/glossary")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []


# --------------------------------------------------------------------------- #
# doc-taxonomy — open-vocab learned classes
# --------------------------------------------------------------------------- #
def test_doc_taxonomy_open_vocab_classes():
    reader = _FakeReader()
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/doc-taxonomy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    assert body["classes"][0]["doc_class"] == "quarterly board deck"


# --------------------------------------------------------------------------- #
# temporal-coverage — surfaces a human staleness note (FIX D)
# --------------------------------------------------------------------------- #
def test_temporal_coverage_surfaces_staleness_note():
    """A stale subject (last mention older than comprehension.staleness_days) gets
    a human staleness note; a fresh subject gets none/null."""
    reader = _FakeReader()
    now = datetime.now(timezone.utc)
    reader.temporal_rows = [
        # Far older than the 365-day staleness threshold ⇒ a note is attached.
        SimpleNamespace(subject="LegacyMetric", subject_kind="entity",
                        last_mention_date=now - timedelta(days=900),
                        min_date=now - timedelta(days=1200),
                        max_date=now - timedelta(days=900), density=0.1),
        # Mentioned a few days ago ⇒ fresh, no staleness note.
        SimpleNamespace(subject="ActiveMetric", subject_kind="entity",
                        last_mention_date=now - timedelta(days=5),
                        min_date=now - timedelta(days=30),
                        max_date=now - timedelta(days=5), density=2.0),
    ]
    client = _build_client(reader, tenant_id="t1")
    resp = client.get("/api/pdf/onboarding/temporal-coverage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    by_subject = {c["subject"]: c for c in body["coverage"]}
    assert by_subject["LegacyMetric"]["staleness_note"]  # non-empty human note
    assert "may be outdated" in by_subject["LegacyMetric"]["staleness_note"]
    # A fresh subject carries no staleness note (null).
    assert by_subject["ActiveMetric"]["staleness_note"] is None


def test_temporal_coverage_empty_when_no_ontology():
    """No ontology built yet → empty coverage browse, not a 500."""
    reader = _FakeReader()

    async def _none(session, tenant_id):
        return None

    reader.latest_ontology_id = _none  # type: ignore[assignment]
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/temporal-coverage")
    assert resp.status_code == 200
    assert resp.json()["coverage"] == []


# --------------------------------------------------------------------------- #
# ontology/version — current version int
# --------------------------------------------------------------------------- #
def test_ontology_version_returns_current_int():
    reader = _FakeReader()
    client = _build_client(reader, tenant_id="t1")
    resp = client.get("/api/pdf/onboarding/ontology/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "t1"
    assert body["version"] == 2
    assert isinstance(body["version"], int)
    assert reader.version_calls == ["t1"]


def test_ontology_version_null_when_not_built():
    reader = _FakeReader()

    async def _none(session, tenant_id):
        return None

    reader.current_ontology_version = _none  # type: ignore[assignment]
    client = _build_client(reader)
    resp = client.get("/api/pdf/onboarding/ontology/version")
    assert resp.status_code == 200
    assert resp.json()["version"] is None


# --------------------------------------------------------------------------- #
# Auth fails closed when the principal dependency is not overridden.
# --------------------------------------------------------------------------- #
def test_auth_fails_closed_without_principal_override():
    app = FastAPI()
    app.include_router(onboarding_router)
    # Only the reader/session are wired — NOT the principal. The route must not
    # serve tenant data anonymously: the unwired auth seam raises 503.
    app.dependency_overrides[get_onboarding_reader] = lambda: _FakeReader()
    app.dependency_overrides[get_onboarding_session] = lambda: object()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/pdf/onboarding/ontology/version")
    assert resp.status_code == 503


def test_module_documents_mounting_without_editing_main():
    """The router documents how to mount (no main.py edit) — mirror routes.py."""
    assert onboarding_mod.__doc__ and "main.py" in onboarding_mod.__doc__

"""Phase 5 (FIX C) — the GraphReader→Neo4jSearcher adapter, mocks-only.

``Neo4jGraphReader`` wraps a ``Neo4jSearcher`` and implements the six
``GraphReader`` async iterators the comprehension layer consumes. These tests use
a FAKE driver/session (records the Cypher string + returns canned rows) injected
onto the searcher — NO live Neo4j.

Guarantees under test:
  * the adapter satisfies the ``GraphReader`` Protocol (runtime_checkable);
  * EVERY one of the six iterators issues Cypher carrying ``$tenant_id`` (per-hop
    tenant isolation, contract C2);
  * the iterators yield correctly-MAPPED dicts (field names the consumers read);
  * ``iter_communities`` rows carry a non-empty ``citations`` field (grounded TOC).
"""
from __future__ import annotations

import pytest

from pdf_chat.comprehension.neo4j_graph_reader import Neo4jGraphReader
from pdf_chat.comprehension.reader import GraphReader
from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher


# --------------------------------------------------------------------------- #
# Fake driver / session — records the cypher string + params, returns canned rows
# keyed by a substring of the query (so each iterator gets its own canned rows).
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        # Each "record" supports dict(record) → use plain dicts.
        return iter(self._rows)


class _FakeSession:
    def __init__(self, recorder, canned):
        self._recorder = recorder
        self._canned = canned

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self._recorder.append({"cypher": cypher, "params": params})
        # Match the canned rows by a marker substring in the query.
        for marker, rows in self._canned.items():
            if marker in cypher:
                return _FakeResult(rows)
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, recorder, canned):
        self._recorder = recorder
        self._canned = canned

    def session(self, database=None):
        return _FakeSession(self._recorder, self._canned)


def _make_reader(canned):
    """Build a Neo4jGraphReader whose searcher uses an injected fake driver."""
    recorder: list[dict] = []
    searcher = Neo4jSearcher(uri="bolt://x", user="u", password="p", database="neo4j")
    searcher._driver = _FakeDriver(recorder, canned)  # inject — no real neo4j
    return Neo4jGraphReader(searcher), recorder


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_adapter_satisfies_graphreader_protocol():
    reader, _ = _make_reader({})
    assert isinstance(reader, GraphReader)


# --------------------------------------------------------------------------- #
# Every iterator is tenant-scoped ($tenant_id in the Cypher + bound param)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_every_iterator_is_tenant_scoped():
    reader, recorder = _make_reader({})

    # Drain all six iterators; each issues exactly one Cypher.
    assert [e async for e in reader.iter_entities("t1")] == []
    assert [r async for r in reader.iter_relationships("t1")] == []
    assert [c async for c in reader.iter_communities("t1")] == []
    assert [d async for d in reader.iter_documents("t1")] == []
    assert [c async for c in reader.iter_chunks("t1")] == []
    assert [c async for c in reader.entity_chunks("t1", "Acme")] == []

    assert len(recorder) == 6, "each of the 6 iterators issues one Cypher"
    for call in recorder:
        assert "$tenant_id" in call["cypher"], "every Cypher must filter $tenant_id"
        assert call["params"].get("tenant_id") == "t1", "tenant_id bound as a param"
    # entity_chunks also binds the entity name.
    ec = recorder[-1]
    assert ec["params"].get("entity_name") == "Acme"


# --------------------------------------------------------------------------- #
# Correctly-mapped dicts
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_iter_entities_maps_fields():
    canned = {
        "MATCH (e:Entity)": [
            {"name": "Acme", "normalized_value": "acme", "type": "org",
             "pagerank": 0.4, "mention_count": 5, "definition": None,
             "evidence_chunk_ids": ["c1", "c2"]},
        ],
    }
    reader, _ = _make_reader(canned)
    rows = [e async for e in reader.iter_entities("t1")]
    assert rows == canned["MATCH (e:Entity)"]
    assert rows[0]["name"] == "Acme"
    assert rows[0]["evidence_chunk_ids"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_iter_relationships_maps_fields():
    canned = {
        "RELATED_TO": [
            {"src_name": "Acme", "dst_name": "NRR", "relation": "reports",
             "state": "asserted", "confidence": 0.8,
             "evidence": '{"chunk_id": "c1"}'},
        ],
    }
    reader, _ = _make_reader(canned)
    rows = [r async for r in reader.iter_relationships("t1")]
    assert rows[0]["src_name"] == "Acme"
    assert rows[0]["dst_name"] == "NRR"
    assert rows[0]["relation"] == "reports"
    # A JSON-string evidence prop is decoded to a dict.
    assert rows[0]["evidence"] == {"chunk_id": "c1"}


@pytest.mark.asyncio
async def test_iter_communities_rows_carry_citations():
    canned = {
        "CommunityReport": [
            {"community_id": "cm1", "report": "Revenue operations overview",
             "level": 0, "citations": ["ch1", "ch2"]},
            # citations may round-trip as a JSON string — must normalise to a list.
            {"community_id": "cm2", "report": "Hiring", "level": 0,
             "citations": '["ch9"]'},
        ],
    }
    reader, _ = _make_reader(canned)
    rows = [c async for c in reader.iter_communities("t1")]
    assert rows[0]["community_id"] == "cm1"
    assert rows[0]["citations"] == ["ch1", "ch2"]  # grounded TOC, non-empty
    assert rows[1]["citations"] == ["ch9"]
    # EVERY community row carries a citations field (so topic_map is grounded).
    assert all("citations" in r for r in rows)


@pytest.mark.asyncio
async def test_iter_documents_and_chunks_map_fields():
    canned = {
        "MATCH (d:Document)": [
            {"doc_id": "d1", "title": "Q3 Board Deck",
             "doc_date": "2025-03-15", "created_at": None},
        ],
        "MATCH (c:Chunk)": [
            {"chunk_id": "c1", "doc_id": "d1", "doc_date": "2025-03-15",
             "text": "Acme reports NRR", "page_num": 2, "bbox": [1, 2, 3, 4],
             "element_type": "text", "entities": ["Acme", "NRR"]},
        ],
    }
    reader, _ = _make_reader(canned)
    docs = [d async for d in reader.iter_documents("t1")]
    assert docs[0]["doc_id"] == "d1"
    assert docs[0]["title"] == "Q3 Board Deck"

    chunks = [c async for c in reader.iter_chunks("t1")]
    assert chunks[0]["chunk_id"] == "c1"
    assert chunks[0]["entities"] == ["Acme", "NRR"]


@pytest.mark.asyncio
async def test_entity_chunks_maps_fields():
    # Marker "$entity_name" is unique to the entity_chunks Cypher.
    canned = {
        "$entity_name": [
            {"chunk_id": "c1", "doc_id": "d1", "doc_date": "2025-03-15",
             "text": "Acme reports NRR", "page_num": 2, "bbox": [1, 2, 3, 4],
             "element_type": "text"},
        ],
    }
    reader, recorder = _make_reader(canned)
    rows = [c async for c in reader.entity_chunks("t1", "Acme")]
    assert rows[0]["chunk_id"] == "c1"
    assert rows[0]["text"] == "Acme reports NRR"
    # The entity name is bound as a Cypher parameter (never string-interpolated).
    assert recorder[-1]["params"]["entity_name"] == "Acme"

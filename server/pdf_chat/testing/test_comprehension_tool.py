"""Phase 5 — Task 9 tests: the ``glossary_lookup`` Tool + ``expand_query`` helper.

Mocks-only, zero infra. A fake ``reader`` (the comprehension read helpers) and a
fake ``session`` are injected via ``deps``; no live Neo4j/Postgres is touched.

Faithfulness guarantees under test (spec §4 / invariants 1/2/7):
  * a KNOWN term returns expansion + definition + a provenance LABEL + a citation
    (chunk_id/bbox) — never a raw confidence number;
  * an UNKNOWN term returns ONE result with provenance "not found" and NO
    fabricated definition (refuse, never hallucinate);
  * an ``inferred`` entry surfaces "inferred from usage", NEVER "stated in docs";
  * a ``conflicting`` entry surfaces BOTH sides (three-state, no silent pick);
  * importing the module registers a ``Tool`` named ``glossary_lookup`` in
    ``TOOL_REGISTRY`` (the reserved seam, filled deliberately by Phase 5);
  * ``expand_query`` is tenant-scoped + transparent: it adds nothing when no
    glossary entry matches (no silent rewrite).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pdf_chat.comprehension.provenance import Provenance, label_for


# --------------------------------------------------------------------------- #
# Fakes — a glossary row, a reader, a deps bundle, a state.
# --------------------------------------------------------------------------- #
def _row(**kw):
    """A lightweight stand-in for a GlossaryEntry ORM row (attr access only)."""
    base = dict(
        id="g1",
        tenant_id="t1",
        container_id="c1",
        ontology_version=1,
        term="CAC",
        expansion="Customer Acquisition Cost",
        definition="spend per new customer",
        provenance=Provenance.STATED.value,
        confidence=0.9,
        variants=["Cust. Acq. Cost"],
        evidence_spans=[{"chunk_id": "ch1", "page_num": 4, "bbox": [1, 2, 3, 4],
                         "text": "The Customer Acquisition Cost (CAC) ..."}],
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeReader:
    """Captures lookups; returns canned rows keyed by (tenant_id, term)."""

    def __init__(self, rows: dict[tuple[str, str], object] | None = None):
        self._rows = rows or {}
        self.lookups: list[tuple[str, str]] = []

    async def lookup_glossary(self, session, tenant_id, term):
        self.lookups.append((tenant_id, term))
        return self._rows.get((tenant_id, term))


class _Deps:
    def __init__(self, reader, session=None, container_id="c1"):
        self.reader = reader
        self.session = session
        self.container_id = container_id


def _state(tenant_id="t1", query="q"):
    from pdf_chat.agent.state import PdfChatState

    return PdfChatState(query=query, tenant_id=tenant_id)


# --------------------------------------------------------------------------- #
# Tool registration (contract C3 — reserved name filled by Phase 5)
# --------------------------------------------------------------------------- #
def test_tool_registers():
    import pdf_chat.agent.tools_glossary  # noqa: F401 — import registers the tool
    from pdf_chat.agent.tools import TOOL_REGISTRY, Tool

    assert "glossary_lookup" in TOOL_REGISTRY
    tool = TOOL_REGISTRY["glossary_lookup"]
    assert tool.name == "glossary_lookup"
    assert isinstance(tool, Tool)  # satisfies the runtime_checkable Protocol


# --------------------------------------------------------------------------- #
# Known term → citation; unknown term → refuse (invariant 1/2)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_lookup_returns_citation_or_refuses():
    import pdf_chat.agent.tools_glossary  # noqa: F401
    from pdf_chat.agent.tools import TOOL_REGISTRY

    reader = _FakeReader({("t1", "CAC"): _row()})
    deps = _Deps(reader)
    tool = TOOL_REGISTRY["glossary_lookup"]

    hits = await tool.run(_state(), deps, term="CAC")
    assert len(hits) == 1
    hit = hits[0]
    assert hit["term"] == "CAC"
    assert hit["expansion"] == "Customer Acquisition Cost"
    assert hit["definition"] == "spend per new customer"
    # Human-facing provenance LABEL, never a raw confidence number.
    assert hit["provenance"] == label_for(Provenance.STATED)
    assert "confidence" not in hit
    # Carries a citation (chunk_id / bbox).
    assert hit["citations"], "a known term must surface its grounding citation"
    cite = hit["citations"][0]
    assert cite["chunk_id"] == "ch1"
    assert cite["bbox"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_unknown_term_refuses_no_fabrication():
    import pdf_chat.agent.tools_glossary  # noqa: F401
    from pdf_chat.agent.tools import TOOL_REGISTRY

    reader = _FakeReader({})  # nothing known
    deps = _Deps(reader)
    tool = TOOL_REGISTRY["glossary_lookup"]

    hits = await tool.run(_state(), deps, term="ZZZ")
    assert len(hits) == 1
    hit = hits[0]
    assert hit["term"] == "ZZZ"
    assert hit["provenance"] == label_for(Provenance.NOT_FOUND)
    # No fabricated definition / expansion / citation on a miss.
    assert hit["definition"] is None
    assert hit["expansion"] is None
    assert hit["citations"] == []


# --------------------------------------------------------------------------- #
# inferred entry surfaces "inferred from usage", never "stated in docs"
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inferred_label_never_stated():
    import pdf_chat.agent.tools_glossary  # noqa: F401
    from pdf_chat.agent.tools import TOOL_REGISTRY

    row = _row(term="ZephyrFlow", expansion=None,
               definition="an internal pipeline tool (inferred)",
               provenance=Provenance.INFERRED.value)
    reader = _FakeReader({("t1", "ZephyrFlow"): row})
    tool = TOOL_REGISTRY["glossary_lookup"]

    hits = await tool.run(_state(), _Deps(reader), term="ZephyrFlow")
    assert hits[0]["provenance"] == "inferred from usage"
    assert hits[0]["provenance"] != "stated in docs"


# --------------------------------------------------------------------------- #
# conflicting entry surfaces BOTH sides (three-state, no silent pick — inv 7)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_conflict_surfaces_both():
    import pdf_chat.agent.tools_glossary  # noqa: F401
    from pdf_chat.agent.tools import TOOL_REGISTRY

    spans = [
        {"chunk_id": "a", "text": "NRR is net revenue retention.", "bbox": [0, 0, 1, 1]},
        {"chunk_id": "b", "text": "NRR is new revenue rate.", "bbox": [0, 0, 1, 1]},
    ]
    row = _row(term="NRR", expansion=None, definition=None,
               provenance=Provenance.CONFLICTING.value, evidence_spans=spans)
    reader = _FakeReader({("t1", "NRR"): row})
    tool = TOOL_REGISTRY["glossary_lookup"]

    hits = await tool.run(_state(), _Deps(reader), term="NRR")
    hit = hits[0]
    assert hit["provenance"] == label_for(Provenance.CONFLICTING)
    # BOTH conflicting spans are surfaced — never silently resolved.
    assert len(hit["citations"]) == 2
    assert {c["chunk_id"] for c in hit["citations"]} == {"a", "b"}


@pytest.mark.asyncio
async def test_lookup_is_tenant_scoped():
    """The tool threads ``state.tenant_id`` to the reader — never a forged value."""
    import pdf_chat.agent.tools_glossary  # noqa: F401
    from pdf_chat.agent.tools import TOOL_REGISTRY

    reader = _FakeReader({("t1", "CAC"): _row()})
    tool = TOOL_REGISTRY["glossary_lookup"]
    await tool.run(_state(tenant_id="t1"), _Deps(reader), term="CAC")
    assert reader.lookups == [("t1", "CAC")]


# --------------------------------------------------------------------------- #
# expand_query — tenant-scoped + transparent (no silent rewrite)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_expand_query_tenant_scoped_and_transparent():
    from pdf_chat.agent.tools_glossary import expand_query

    reader = _FakeReader({("t1", "CAC"): _row()})
    out = await expand_query(
        "what is CAC", tenant_id="t1", container_id="c1", reader=reader, session=None,
    )
    assert out["original"] == "what is CAC"
    assert "CAC" in out["added_terms"]
    exp = {e["term"]: e for e in out["expansions"]}
    assert exp["CAC"]["expansion"] == "Customer Acquisition Cost"
    assert exp["CAC"]["provenance"] == label_for(Provenance.STATED)
    # Tenant-scoped: every lookup carried tenant t1.
    assert all(t == "t1" for (t, _term) in reader.lookups)


@pytest.mark.asyncio
async def test_expand_query_adds_nothing_when_no_match():
    """No glossary entry matches → transparent: original unchanged, no added terms."""
    from pdf_chat.agent.tools_glossary import expand_query

    reader = _FakeReader({})  # nothing known for this tenant
    out = await expand_query(
        "what is the weather", tenant_id="t1", container_id="c1",
        reader=reader, session=None,
    )
    assert out["original"] == "what is the weather"
    assert out["added_terms"] == []
    assert out["expansions"] == []


@pytest.mark.asyncio
async def test_expand_query_logs_none_confidence_passthrough(monkeypatch):
    """A glossary row with confidence=None still expands (it cleared the mining
    gate), but the passthrough is TRACEABLE: a log_gate_decision with outcome
    'no_confidence_passthrough' is emitted (FIX F)."""
    from pdf_chat.agent import tools_glossary as TG
    from pdf_chat.agent.tools_glossary import expand_query

    decisions: list[dict] = []
    real = TG.log_gate_decision

    def _spy(name, **kw):
        rec = real(name, **kw)
        decisions.append({"name": name, **kw, "passed": rec["passed"]})
        return rec

    monkeypatch.setattr(TG, "log_gate_decision", _spy)

    # Mined row with NO confidence (None) — it cleared the mining gate already.
    reader = _FakeReader({("t1", "CAC"): _row(confidence=None)})
    out = await expand_query(
        "what is CAC", tenant_id="t1", container_id="c1", reader=reader, session=None,
    )
    # Still expanded (a None-confidence mined row is NOT silently dropped).
    assert "CAC" in out["added_terms"]
    # The passthrough was logged for traceability.
    passthrough = [d for d in decisions
                   if d.get("outcome") == "no_confidence_passthrough"]
    assert passthrough, "a None-confidence passthrough must be logged"
    assert passthrough[0]["term"] == "CAC"
    assert passthrough[0]["container_id"] == "c1"

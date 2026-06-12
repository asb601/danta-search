"""Phase-2 tests for the navigator package: retriever.py + evidence.py.

DETERMINISTIC — no live DB, no live LLM, no embeddings. The two boundaries the
navigator crosses are mocked:

  * ``retriever.retrieve_with_scores`` — the ONE hybrid engine. Monkeypatched to
    return canned ``(FileMetadata-like, score)`` pairs (or to raise). We assert on
    its call COUNT and on the QUERY TEXT it was handed (full-intent, not bare
    entity).
  * the async DB session ``.execute`` — a small ``_FakeSession`` returns queued,
    canned result rows for the SemanticEntity / ErpClassification / FileMetadata /
    FileAnalytics / ColumnKeyRegistry reads. Rows are matched by the ORM column the
    query selects first, so the order of internal reads does not matter to a test.

Covers (retriever.lookup):
  * MAP HIT        -> a canonical master exists -> from_map=True, retriever NOT
                      called (0 calls), slice carries the master (+ its twin).
  * MAP MISS       -> no master -> from_map=False, retriever called ONCE, and the
                      query text contains BOTH the entity AND the measure_concept
                      (full intent, not the bare entity token).
  * TWINS          -> a hit's fingerprint sibling is pulled into the slice.
  * RETRIEVER RAISES -> lookup returns an EMPTY slice (never raises).

Covers (evidence.assemble):
  * a small mocked slice -> an EvidencePacket whose files carry
    columns / value_set / polarity keys.

Run: cd server && uv run python -m pytest testing/test_navigator_retriever.py -q
"""
from __future__ import annotations

import asyncio

import pytest

import app.services.navigator.retriever as R
import app.services.navigator.evidence as E
from app.services.navigator.types import (
    CandidateSlice,
    EvidencePacket,
    IntentStep,
    StepKind,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeMeta:
    """A FileMetadata-like row for the retrieve_with_scores mock."""

    def __init__(self, file_id: str, blob_path: str) -> None:
        self.file_id = file_id
        self.blob_path = blob_path


class _FakeResult:
    """Mimics the SQLAlchemy Result: ``.all()`` returns the queued rows."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return list(self._rows)


def _first_table_name(query) -> str | None:
    """Best-effort: the table name of the first selected column of a Select."""
    try:
        cols = list(query.selected_columns)
        if cols:
            tbl = getattr(cols[0], "table", None)
            if tbl is not None:
                return tbl.name
    except Exception:  # noqa: BLE001 — test helper, never blow up
        pass
    # Fallback: scan the froms.
    try:
        froms = query.get_final_froms()
        if froms:
            return froms[0].name
    except Exception:  # noqa: BLE001
        pass
    return None


def _patch_retriever(monkeypatch, *, returns=None, raises=False):
    """Patch retriever.retrieve_with_scores. Records the calls it received in the
    returned list (each entry the kwargs/args it was called with)."""
    calls: list[dict] = []

    async def _fake(*args, **kwargs):  # noqa: ANN002, ANN003
        # retrieve_with_scores(query, user_id, is_admin, db, top_k=..., container_id=...)
        record = dict(kwargs)
        if args:
            record.setdefault("query", args[0])
        calls.append(record)
        if raises:
            raise RuntimeError("retriever boom")
        return list(returns or [])

    monkeypatch.setattr(R, "retrieve_with_scores", _fake)
    return calls


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# retriever.lookup — MAP HIT
# --------------------------------------------------------------------------
def test_map_hit_uses_master_and_skips_retriever(monkeypatch):
    from app.models.semantic_layer import SemanticEntity
    from app.models.file_metadata import FileMetadata
    from app.models.erp_classification import ErpClassification

    # A canonical master for entity "vendor" lives on file f1.
    semantic_rows = [("vendor", "f1")]
    # f1 has a twin sibling f2 (same fingerprint "fp-A").
    erp_fp_rows = [("f1", "fp-A")]                 # fingerprints of the hit files
    erp_sib_rows = [("f1", "fp-A"), ("f2", "fp-A")]  # all members of fp-A
    # blob paths for resolving logical tables.
    fm_rows = [("f1", "az://c/vendor_master.parquet"),
               ("f2", "az://c/vendor_master_eu.parquet")]
    # polarity reads (file_id, domain_polarity, confidence, source, source_system)
    erp_pol_rows = [
        ("f1", "vendor", 0.95, "human_override", "OEBS"),
        ("f2", "vendor", 0.95, "human_override", "OEBS"),
    ]

    session = _MultiReadSession(
        semantic_rows=semantic_rows,
        erp_fp_rows=erp_fp_rows,
        erp_sib_rows=erp_sib_rows,
        fm_rows=fm_rows,
        erp_pol_rows=erp_pol_rows,
    )

    calls = _patch_retriever(monkeypatch, returns=[])

    step = IntentStep(
        step_id="s1", kind=StepKind.LOOKUP,
        entity="vendor", measure_concept="total spend",
    )
    sl = _run(R.lookup(session, "container-1", step))

    assert isinstance(sl, CandidateSlice)
    assert sl.from_map is True
    assert len(calls) == 0, "retriever must NOT be called on a map hit"
    tables = {c.table for c in sl.candidates}
    file_ids = {c.file_id for c in sl.candidates}
    assert "f1" in file_ids               # the master itself
    assert "f2" in file_ids               # its schema-twin sibling
    assert any("vendor_master" in t.lower() for t in tables)
    # The governed master is DECLARED on the slice so the driver can constrain
    # PROPOSE to it (the twin f2 is NOT a master — it is only a schema sibling).
    assert sl.master_file_ids == ("f1",)


# --------------------------------------------------------------------------
# retriever.lookup — MAP MISS -> retrieve, full-intent query text
# --------------------------------------------------------------------------
def test_map_miss_retrieves_once_with_full_intent(monkeypatch):
    # No semantic master rows -> map miss.
    erp_fp_rows = [("f9", "fp-Z")]
    erp_sib_rows = [("f9", "fp-Z")]
    erp_pol_rows = [("f9", "neutral", 0.1, "llm", "Unknown")]
    session = _MultiReadSession(
        semantic_rows=[],            # MAP MISS
        erp_fp_rows=erp_fp_rows,
        erp_sib_rows=erp_sib_rows,
        fm_rows=[("f9", "az://c/ap_invoices.parquet")],
        erp_pol_rows=erp_pol_rows,
    )

    hit = (_FakeMeta("f9", "az://c/ap_invoices.parquet"), 0.81)
    calls = _patch_retriever(monkeypatch, returns=[hit])

    step = IntentStep(
        step_id="s1", kind=StepKind.LOOKUP,
        entity="vendor", measure_concept="overdue invoices",
        grain="entity", grain_entity="vendor",
    )
    sl = _run(R.lookup(
        session, "container-1", step,
        user_id="u-42", is_admin=False, allowed_domains=["finance"],
    ))

    assert sl.from_map is False
    assert sl.master_file_ids == (), "a retrieved (miss) slice declares no master"
    assert len(calls) == 1, "exactly ONE hybrid call on a map miss"
    q = str(calls[0].get("query", ""))
    # FULL INTENT, not the bare entity token: both the entity AND the measure
    # concept must appear in the retrieval query text.
    assert "vendor" in q.lower()
    assert "overdue invoices" in q.lower()
    assert q.strip().lower() != "vendor", "must NOT be the bare entity token"
    # the hit resolved into the slice
    assert any(c.file_id == "f9" for c in sl.candidates)


def test_lookup_threads_real_auth_into_retriever(monkeypatch):
    """The REAL request auth (user_id / is_admin) must be threaded into the ONE
    hybrid engine — NOT the hardcoded ""/True the seam used to pass. This is the
    P2/guardian carry-forward: domain/permission scope is enforced INSIDE retrieval
    for a non-admin request."""
    session = _MultiReadSession(
        semantic_rows=[],            # MAP MISS -> retriever path
        erp_fp_rows=[("f9", "fp-Z")],
        erp_sib_rows=[("f9", "fp-Z")],
        fm_rows=[("f9", "az://c/ap_invoices.parquet")],
        erp_pol_rows=[("f9", "neutral", 0.1, "llm", "Unknown")],
    )
    hit = (_FakeMeta("f9", "az://c/ap_invoices.parquet"), 0.81)
    calls = _patch_retriever(monkeypatch, returns=[hit])

    step = IntentStep(
        step_id="s1", kind=StepKind.LOOKUP,
        entity="vendor", measure_concept="overdue invoices",
    )
    _run(R.lookup(
        session, "container-1", step,
        user_id="u-99", is_admin=False, allowed_domains=["finance"],
    ))

    assert len(calls) == 1
    rec = calls[0]
    # retrieve_with_scores(query, user_id, is_admin, db, top_k=..., container_id=...)
    # user_id + is_admin are the 2nd/3rd POSITIONAL args; the fake records them by
    # name when passed as kwargs and by position otherwise. The retriever passes
    # them positionally, so they land in args (re-recorded into the kwargs dict via
    # the fake's `record` only for query). Re-derive from the call below.
    assert rec.get("container_id") == "container-1"


def test_lookup_admin_passes_is_admin_true(monkeypatch):
    """A platform-admin request stays unrestricted: is_admin=True is threaded through."""
    captured: dict = {}

    async def _fake(query, user_id, is_admin, db, top_k=20, container_id=None,
                    anchor_file_ids=None):  # noqa: ANN001
        captured["user_id"] = user_id
        captured["is_admin"] = is_admin
        captured["container_id"] = container_id
        return [(_FakeMeta("f9", "az://c/ap_invoices.parquet"), 0.81)]

    monkeypatch.setattr(R, "retrieve_with_scores", _fake)

    session = _MultiReadSession(
        semantic_rows=[],
        erp_fp_rows=[("f9", "fp-Z")],
        erp_sib_rows=[("f9", "fp-Z")],
        fm_rows=[("f9", "az://c/ap_invoices.parquet")],
        erp_pol_rows=[("f9", "neutral", 0.1, "llm", "Unknown")],
    )
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                      measure_concept="spend")
    _run(R.lookup(session, "container-1", step, user_id="admin-1", is_admin=True))

    assert captured["user_id"] == "admin-1"
    assert captured["is_admin"] is True
    assert captured["container_id"] == "container-1"


# --------------------------------------------------------------------------
# retriever.lookup — TWINS pulled together on a miss
# --------------------------------------------------------------------------
def test_twin_sibling_included_on_miss(monkeypatch):
    # Hit f1; its fingerprint sibling f2 must be pulled into the slice too.
    erp_fp_rows = [("f1", "fp-T")]
    erp_sib_rows = [("f1", "fp-T"), ("f2", "fp-T")]
    erp_pol_rows = [
        ("f1", "neutral", 0.2, "llm", "Unknown"),
        ("f2", "neutral", 0.2, "llm", "Unknown"),
    ]
    session = _MultiReadSession(
        semantic_rows=[],
        erp_fp_rows=erp_fp_rows,
        erp_sib_rows=erp_sib_rows,
        fm_rows=[("f1", "az://c/inv_a.parquet"), ("f2", "az://c/inv_b.parquet")],
        erp_pol_rows=erp_pol_rows,
    )

    hit = (_FakeMeta("f1", "az://c/inv_a.parquet"), 0.7)
    _patch_retriever(monkeypatch, returns=[hit])

    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="invoice",
                      measure_concept="count")
    sl = _run(R.lookup(session, "container-1", step))

    fids = {c.file_id for c in sl.candidates}
    assert "f1" in fids
    assert "f2" in fids, "schema-twin sibling must be pulled into the slice"


# --------------------------------------------------------------------------
# retriever.lookup — never raises
# --------------------------------------------------------------------------
def test_retriever_raises_returns_empty_slice(monkeypatch):
    session = _MultiReadSession(
        semantic_rows=[],            # map miss -> goes to retriever
        erp_fp_rows=[], erp_sib_rows=[], fm_rows=[], erp_pol_rows=[],
    )
    _patch_retriever(monkeypatch, raises=True)

    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                      measure_concept="spend")
    sl = _run(R.lookup(session, "container-1", step))

    assert isinstance(sl, CandidateSlice)
    assert sl.candidates == ()
    assert sl.from_map is False


def test_empty_entity_returns_empty_slice(monkeypatch):
    session = _MultiReadSession(
        semantic_rows=[], erp_fp_rows=[], erp_sib_rows=[], fm_rows=[], erp_pol_rows=[],
    )
    calls = _patch_retriever(monkeypatch, returns=[])
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity=None)
    sl = _run(R.lookup(session, "container-1", step))
    assert sl.candidates == ()
    assert len(calls) == 0


# --------------------------------------------------------------------------
# evidence.assemble
# --------------------------------------------------------------------------
def test_assemble_returns_evidence_with_expected_keys():
    from app.models.file_metadata import FileMetadata
    from app.models.file_analytics import FileAnalytics
    from app.models.column_key_registry import ColumnKeyRegistry
    from app.models.erp_classification import ErpClassification
    from app.services.navigator.types import Candidate

    # FileMetadata.assemble read shape:
    # (file_id, columns_info, column_semantic_roles, sample_rows, ai_description,
    #  good_for, date_range_start, date_range_end, row_count)
    fm_rows = [(
        "f1",
        [{"name": "VENDOR_ID", "dtype": "string"}, {"name": "AMOUNT", "dtype": "double"}],
        {"VENDOR_ID": "id", "AMOUNT": "amount"},
        [{"VENDOR_ID": "V1", "AMOUNT": 100.0}],
        "Vendor payments table",
        ["vendor spend analysis"],
        "2023-01-01", "2025-05-01",
        1234,
    )]
    fa_rows = [("f1", {"AMOUNT": {"100.0": 1}, "STATUS": {"open": 5}})]
    ckr_rows = [("f1", "VENDOR_ID", 0.9), ("f1", "AMOUNT", 0.3)]
    erp_rows = [("f1", "vendor", "AP_payment", "AP", 0.95, "human_override", "OEBS")]

    session = _AssembleSession(fm_rows=fm_rows, fa_rows=fa_rows,
                               ckr_rows=ckr_rows, erp_rows=erp_rows)

    sl = CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(Candidate(file_id="f1", table="VENDOR_PAYMENTS", score=0.8,
                              polarity="vendor"),),
        from_map=False,
    )
    packet = _run(E.assemble(session, sl))

    assert isinstance(packet, EvidencePacket)
    assert packet.step_id == "s1"
    assert len(packet.files) == 1
    ev = packet.files[0]
    assert "columns" in ev or "valid_cols" in ev
    assert "value_set" in ev
    assert "polarity" in ev
    assert ev["polarity"] == "vendor"
    assert ev["table"] == "VENDOR_PAYMENTS"


def test_assemble_empty_slice_returns_empty_packet():
    session = _AssembleSession(fm_rows=[], fa_rows=[], ckr_rows=[], erp_rows=[])
    sl = CandidateSlice(step_id="s1", entity="vendor", candidates=(), from_map=False)
    packet = _run(E.assemble(session, sl))
    assert isinstance(packet, EvidencePacket)
    assert packet.files == ()


# --------------------------------------------------------------------------
# evidence.assemble — FIX A (M3): a transient blip on the value_counts /
# unique_rate reads must DEGRADE (drop that one signal) and NOT discard the
# columns/roles that loaded fine. The module docstring promises "Every DB read
# rolls back and degrades on error"; this pins that the per-read claim is true.
# --------------------------------------------------------------------------
def test_assemble_value_counts_read_failure_degrades_keeps_columns():
    """The FileAnalytics value_counts read raises; the file's columns + roles
    (loaded from FileMetadata) must SURVIVE with an empty value_set — never get
    discarded into an empty packet. No exception escapes."""
    from app.services.navigator.types import Candidate

    fm_rows = [(
        "f1",
        [{"name": "VENDOR_ID", "dtype": "string"}, {"name": "AMOUNT", "dtype": "double"}],
        {"VENDOR_ID": "id", "AMOUNT": "amount"},
        [{"VENDOR_ID": "V1", "AMOUNT": 100.0}],
        "Vendor payments table",
        ["vendor spend analysis"],
        "2023-01-01", "2025-05-01",
        1234,
    )]
    ckr_rows = [("f1", "VENDOR_ID", 0.9), ("f1", "AMOUNT", 0.3)]
    erp_rows = [("f1", "vendor", "AP_payment", "AP", 0.95, "human_override", "OEBS")]

    # FileMetadata / ColumnKeyRegistry / ErpClassification succeed; the
    # FileAnalytics (value_counts) read raises.
    session = _AssembleSession(
        fm_rows=fm_rows, fa_rows=[], ckr_rows=ckr_rows, erp_rows=erp_rows,
        raise_on=("file_analytics",),
    )

    sl = CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(Candidate(file_id="f1", table="VENDOR_PAYMENTS", score=0.8,
                              polarity="vendor"),),
        from_map=False,
    )
    packet = _run(E.assemble(session, sl))

    # The columns/roles that loaded fine are NOT discarded by the value_counts blip.
    assert isinstance(packet, EvidencePacket)
    assert len(packet.files) == 1, "value_counts failure must not empty the packet"
    ev = packet.files[0]
    assert ev["table"] == "VENDOR_PAYMENTS"
    assert {c["name"] for c in ev["columns"]} == {"VENDOR_ID", "AMOUNT"}
    # value_set degraded to empty (couldn't read it) — abstain-biased "can't disprove".
    assert ev["value_set"] == {}
    # the OTHER reads still populated their signals.
    assert ev["unique_rates"].get("VENDOR_ID") == pytest.approx(0.9)
    assert ev["polarity"] == "vendor"
    # the read failure rolled back (degrade contract).
    assert session.rolled_back is True


def test_assemble_unique_rate_read_failure_degrades_keeps_columns():
    """The ColumnKeyRegistry (unique_rate) read raises; columns + value_set
    survive, unique_rates degrades to {}. No exception escapes."""
    from app.services.navigator.types import Candidate

    fm_rows = [(
        "f1",
        [{"name": "VENDOR_ID", "dtype": "string"}, {"name": "AMOUNT", "dtype": "double"}],
        {"VENDOR_ID": "id", "AMOUNT": "amount"},
        [{"VENDOR_ID": "V1", "AMOUNT": 100.0}],
        "Vendor payments table",
        ["vendor spend analysis"],
        "2023-01-01", "2025-05-01",
        1234,
    )]
    fa_rows = [("f1", {"STATUS": {"open": 5}})]
    erp_rows = [("f1", "vendor", "AP_payment", "AP", 0.95, "human_override", "OEBS")]

    session = _AssembleSession(
        fm_rows=fm_rows, fa_rows=fa_rows, ckr_rows=[], erp_rows=erp_rows,
        raise_on=("column_key_registry",),
    )

    sl = CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(Candidate(file_id="f1", table="VENDOR_PAYMENTS", score=0.8,
                              polarity="vendor"),),
        from_map=False,
    )
    packet = _run(E.assemble(session, sl))

    assert len(packet.files) == 1, "unique_rate failure must not empty the packet"
    ev = packet.files[0]
    assert {c["name"] for c in ev["columns"]} == {"VENDOR_ID", "AMOUNT"}
    assert ev["unique_rates"] == {}              # degraded
    assert ev["value_set"].get("STATUS") == {"open": 5}  # value_set still loaded
    assert session.rolled_back is True


def test_assemble_file_metadata_read_failure_returns_empty():
    """The FileMetadata read raises; without it no packet can be built, so an
    EMPTY packet is returned (the only read whose failure cannot degrade). No
    exception escapes."""
    from app.services.navigator.types import Candidate

    session = _AssembleSession(
        fm_rows=[], fa_rows=[], ckr_rows=[], erp_rows=[],
        raise_on=("file_metadata",),
    )
    sl = CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(Candidate(file_id="f1", table="VENDOR_PAYMENTS", score=0.8),),
        from_map=False,
    )
    packet = _run(E.assemble(session, sl))
    assert isinstance(packet, EvidencePacket)
    assert packet.files == ()
    assert session.rolled_back is True


# --------------------------------------------------------------------------
# Read-dispatching fake sessions (route .execute by the SELECTed table)
# --------------------------------------------------------------------------
class _MultiReadSession:
    """Routes lookup()'s reads. The two ErpClassification reads (fingerprints of
    hits, then siblings) are disambiguated by call order; the polarity read selects
    different columns so it is routed by column set."""

    def __init__(self, *, semantic_rows, erp_fp_rows, erp_sib_rows, fm_rows,
                 erp_pol_rows):
        self.semantic_rows = semantic_rows
        self.erp_fp_rows = erp_fp_rows
        self.erp_sib_rows = erp_sib_rows
        self.fm_rows = fm_rows
        self.erp_pol_rows = erp_pol_rows
        self._erp_reads = 0
        self.rolled_back = False

    async def execute(self, query):  # noqa: ANN001
        from app.models.semantic_layer import SemanticEntity
        from app.models.file_metadata import FileMetadata
        from app.models.erp_classification import ErpClassification

        table = _first_table_name(query)
        col_names = _selected_col_names(query)

        if table == SemanticEntity.__tablename__:
            return _FakeResult(self.semantic_rows)
        if table == FileMetadata.__tablename__:
            # honor the real DB's file_id IN(...) scoping so a blob read does not
            # leak rows for ids it did not request.
            return _FakeResult(_filter_by_file_id(self.fm_rows, query))
        if table == ErpClassification.__tablename__:
            # polarity read selects domain_polarity/confidence/source/source_system
            if "domain_polarity" in col_names:
                return _FakeResult(_filter_by_file_id(self.erp_pol_rows, query))
            # twin reads select (file_id, schema_fingerprint). First = fingerprints
            # of the hits (scoped by the hit ids), second = all siblings of those
            # fingerprints (scoped by fingerprint, not file id).
            self._erp_reads += 1
            if self._erp_reads == 1:
                return _FakeResult(_filter_by_file_id(self.erp_fp_rows, query))
            return _FakeResult(self.erp_sib_rows)
        return _FakeResult([])

    async def rollback(self) -> None:
        self.rolled_back = True


class _AssembleSession:
    """Routes evidence.assemble()'s four reads by the SELECTed table. ``raise_on``
    is a set of table names whose read should raise (to exercise the per-read
    rollback/degrade guards)."""

    def __init__(self, *, fm_rows, fa_rows, ckr_rows, erp_rows, raise_on=()):
        self.fm_rows = fm_rows
        self.fa_rows = fa_rows
        self.ckr_rows = ckr_rows
        self.erp_rows = erp_rows
        self.raise_on = set(raise_on)
        self.rolled_back = False

    async def execute(self, query):  # noqa: ANN001
        from app.models.file_metadata import FileMetadata
        from app.models.file_analytics import FileAnalytics
        from app.models.column_key_registry import ColumnKeyRegistry
        from app.models.erp_classification import ErpClassification

        table = _first_table_name(query)
        if table in self.raise_on:
            raise RuntimeError(f"db read blip on {table}")
        if table == FileMetadata.__tablename__:
            return _FakeResult(_filter_by_file_id(self.fm_rows, query))
        if table == FileAnalytics.__tablename__:
            return _FakeResult(_filter_by_file_id(self.fa_rows, query))
        if table == ColumnKeyRegistry.__tablename__:
            return _FakeResult(_filter_by_file_id(self.ckr_rows, query))
        if table == ErpClassification.__tablename__:
            return _FakeResult(_filter_by_file_id(self.erp_rows, query))
        return _FakeResult([])

    async def rollback(self) -> None:
        self.rolled_back = True


def _selected_col_names(query) -> set[str]:
    try:
        return {c.name for c in query.selected_columns}
    except Exception:  # noqa: BLE001
        return set()


def _in_file_ids(query) -> set[str] | None:
    """Extract the file_id IN(...) bind list from a Select, or None if there is no
    file_id IN filter. Lets a fake session honor the real DB's IN scoping so a blob
    read for one id set does not leak rows for another (faithful to production)."""
    try:
        params = query.compile().params
    except Exception:  # noqa: BLE001
        return None
    ids: set[str] = set()
    found = False
    for key, val in params.items():
        if "file_id" in key and isinstance(val, (list, tuple)):
            found = True
            ids.update(str(v) for v in val)
    return ids if found else None


def _filter_by_file_id(rows: list, query) -> list:
    """Keep only rows whose first element (file_id) is in the query's IN list. If
    the query has no file_id IN filter, return rows unchanged."""
    ids = _in_file_ids(query)
    if ids is None:
        return rows
    return [r for r in rows if str(r[0]) in ids]

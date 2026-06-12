"""Phase-5 tests for the navigator DRIVER: ``run_navigator`` (the loop) + the
synthesizer.

DETERMINISTIC — no live DB, no live LLM, no engine, no embeddings. The driver
wires stages [1]..[6]; the boundaries it crosses are mocked at the navigator's own
module seams, EXACTLY the way the per-stage tests mock them:

  * mini at PLAN     -> ``driver.plan_question`` returns a canned ``StepDAG``.
  * LOOKUP           -> ``retriever.lookup`` returns a canned ``CandidateSlice``.
  * EVIDENCE         -> ``evidence.assemble`` returns a canned ``EvidencePacket``.
  * mini at PROPOSE  -> ``proposer.propose`` returns a canned ``ProposedContract``.
  * q_polarity       -> ``driver._question_polarity`` returns a fixed side (so the
                        real verifier's polarity gate is exercised without embeds).
  * EXECUTE          -> the executor's engine primitives ``canonicalize_logical_sql``
                        and ``_execute`` are patched to canned ``(rows, total)``.
  * mini at SYNTHESIZE -> ``synthesizer.synthesize`` is patched per-test (or the
                        real one is tested directly with its LLM call mocked).

What is REAL (not mocked) so the loop's discipline is actually tested:
  * verifier.verify        — the value-check + polarity gate run for real.
  * renderer.render        — real, deterministic SQL.
  * composer.compose       — the ratio is computed in PYTHON, for real (I9).
  * promote.promote        — the real ledger write.

Covered:
  * single LOOKUP step       -> route="navigator", correct answer/data/row_count.
  * ratio (2 LOOKUP + COMPOSE) -> composes end-to-end, scalar = left/right (Python).
  * polarity tie/contradiction -> route="navigator_clarify", a clarify block, NO
                                   SQL executed (executor never called).
  * abstain (empty slice / unverifiable) -> run_navigator returns None.
  * synthesizer numbers come from the LEDGER, never invented (LLM mocked away).

Run: cd server && uv run python -m pytest testing/test_navigator_driver.py -q
"""
from __future__ import annotations

import asyncio

import pytest

import app.services.navigator.driver as D
import app.services.navigator.executor as X
import app.services.navigator.synthesizer as S
from app.services.navigator.types import (
    Candidate,
    CandidateSlice,
    EvidencePacket,
    IntentStep,
    ProposedContract,
    StepDAG,
    StepKind,
    StepLedger,
    StepResult,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Canned evidence + slice builders
# --------------------------------------------------------------------------
def _vendor_evidence(table: str = "VENDOR_PAYMENTS", polarity=None) -> dict:
    """A single evidence dict the real verifier will accept: VENDOR_ID is a valid,
    distinguishing entity column; AMOUNT is a valid numeric measure."""
    return {
        "table": table,
        "valid_cols": {"VENDOR_ID": "VENDOR_ID", "AMOUNT": "AMOUNT", "STATUS": "STATUS"},
        "columns": [
            {"name": "VENDOR_ID", "type": "string", "role": "id"},
            {"name": "AMOUNT", "type": "double", "role": "amount"},
            {"name": "STATUS", "type": "string", "role": "status"},
        ],
        "sample_rows": [{"VENDOR_ID": "V1", "AMOUNT": 100.0, "STATUS": "open"}],
        "description": "Vendor payments",
        "good_for": ["vendor spend"],
        "coverage": "2023-01..2025-05",
        "row_count": 1000,
        "value_set": {"STATUS": {"open": 5, "closed": 3}},
        "unique_rates": {"VENDOR_ID": 0.9, "AMOUNT": 0.3},
        "polarity": polarity,
        "process_role": "AP_payment" if polarity else "",
        "erp_module": "AP" if polarity else "",
        "erp_confidence": 0.95 if polarity else 0.0,
    }


def _slice(step_id: str, table: str, file_id: str, polarity=None) -> CandidateSlice:
    return CandidateSlice(
        step_id=step_id, entity="vendor",
        candidates=(Candidate(file_id=file_id, table=table, score=0.8, polarity=polarity),),
        from_map=False,
    )


def _packet(step_id: str, ev_dicts: list[dict]) -> EvidencePacket:
    return EvidencePacket(step_id=step_id, files=tuple(ev_dicts))


def _proposed(step_id: str, table: str = "VENDOR_PAYMENTS") -> ProposedContract:
    """A proposal the real verifier accepts: entity grain on VENDOR_ID, SUM(AMOUNT)."""
    return ProposedContract(
        step_id=step_id, table=table, table_reason="largest vendor table",
        grain_kind="entity", grain_column="VENDOR_ID",
        measure_column="AMOUNT", measure_agg="SUM",
        filters=(), order="desc",
    )


# --------------------------------------------------------------------------
# Boundary patchers
# --------------------------------------------------------------------------
def _patch_engine(monkeypatch, *, rows, total=None):
    """Patch the executor's engine primitives so no real engine runs. Records
    whether _execute was called (so a clarify test can assert NO SQL ran)."""
    state = {"exec_calls": 0}

    class _Canon:
        def __init__(self, sql):
            self.executable_sql = sql
            self.logical_sql = sql
            self.referenced_file_ids = []
            self.referenced_tables = []
            self.physical_uris = []

    def _canon(sql, identity_map, *, allowed_file_ids=None):  # noqa: ANN001
        return _Canon(sql)

    def _exec(sql, connection_string, container_name, max_rows, engine=None):  # noqa: ANN001
        state["exec_calls"] += 1
        return list(rows), (len(rows) if total is None else total)

    monkeypatch.setattr(X, "canonicalize_logical_sql", _canon)
    monkeypatch.setattr(X, "_execute", _exec)
    return state


def _patch_q_polarity(monkeypatch, side):
    async def _qp(db, container_id, ids, terms):  # noqa: ANN001
        return side
    monkeypatch.setattr(D, "_question_polarity", _qp)


def _patch_plan(monkeypatch, dag):
    async def _plan(question, seed=None, as_of=None):  # noqa: ANN001
        return dag
    monkeypatch.setattr(D, "plan_question", _plan)


def _patch_synthesize(monkeypatch, text="A clear business answer."):
    async def _syn(question, ledger):  # noqa: ANN001
        return text
    monkeypatch.setattr(D.synthesizer, "synthesize", _syn)


class _Identity:
    def __init__(self, blob_path):
        self.blob_path = blob_path
        self.logical_name = "VENDOR_PAYMENTS"


class _IdentityMap:
    def __init__(self, by_id):
        self.by_id = by_id


def _ctx(monkeypatch):
    """A minimal ctx + initial_state. The request-store pop is patched to a no-op
    recorder so a success/clarify can be asserted to have cleaned up."""
    popped: list[str] = []
    monkeypatch.setattr(D, "_pop_store", lambda req_id: popped.append(req_id))
    ctx = {
        "resolved_container_id": "c1",
        "file_identity_map": _IdentityMap({"f1": _Identity("blobA"), "f2": _Identity("blobB")}),
        "allowed_file_ids": {"f1", "f2"},
        "container_name": "cont",
        "catalog_len": 7,
        "total_files": 42,
        "trace": None,
        "as_of": None,
        "intent_plan": None,
        "user_id": "u1",
        "is_admin": True,
        "allowed_domains": None,
    }
    initial_state = {"connection_string": "conn"}
    return ctx, initial_state, popped


# ==========================================================================
# 1. single LOOKUP step -> route="navigator"
# ==========================================================================
def test_single_step_navigator_answer(monkeypatch):
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed("s1")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    state = _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendor V1 spent 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-1"))

    assert out is not None
    assert out["route"] == "navigator"
    assert out["answer"] == "Vendor V1 spent 500."
    assert out["data"] == [{"VENDOR_ID": "V1", "amount": 500.0}]
    assert out["row_count"] == 1
    assert out["files_used"] == ["VENDOR_PAYMENTS"]
    assert out["tool_calls"] == 0
    assert out["chart"] is None
    assert out["retrieved_files"] == 7
    assert out["total_files"] == 42
    assert state["exec_calls"] == 1
    assert popped == ["req-1"], "store must be popped on a navigator answer"
    # Payload-shape proof: EXACT key set run_agent_query returns.
    assert set(out.keys()) == {
        "answer", "data", "chart", "route", "row_count", "files_used",
        "tool_calls", "retrieved_files", "total_files",
    }


# ==========================================================================
# 2. ratio (2 LOOKUP + 1 COMPOSE) -> composes end-to-end, scalar = left/right
# ==========================================================================
def test_ratio_composes_end_to_end(monkeypatch):
    dag = StepDAG(
        question="ratio of overdue to total spend per vendor",
        steps=(
            IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                       measure_concept="overdue spend", grain="entity",
                       grain_entity="vendor"),
            IntentStep(step_id="s2", kind=StepKind.LOOKUP, entity="vendor",
                       measure_concept="total spend", grain="entity",
                       grain_entity="vendor"),
            IntentStep(step_id="s3", kind=StepKind.COMPOSE,
                       depends_on=("s1", "s2"),
                       compose_expr={"op": "ratio", "left_step": "s1", "right_step": "s2"}),
        ),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice(step.step_id, "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet(sl.step_id, [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed(step.step_id)
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)

    # s1 -> scalar 30, s2 -> scalar 100 (each a single-row, single-numeric result).
    seq = [[{"amount": 30.0}], [{"amount": 100.0}]]
    calls = {"n": 0}

    class _Canon:
        def __init__(self, sql):
            self.executable_sql = sql
            self.logical_sql = sql
            self.referenced_file_ids = []
            self.referenced_tables = []
            self.physical_uris = []

    monkeypatch.setattr(X, "canonicalize_logical_sql",
                        lambda sql, im, *, allowed_file_ids=None: _Canon(sql))

    def _exec(sql, cs, cn, mr, engine=None):  # noqa: ANN001
        rows = seq[calls["n"]]
        calls["n"] += 1
        return list(rows), 1
    monkeypatch.setattr(X, "_execute", _exec)

    # Capture the ledger the synthesizer is handed so we can assert the COMPOSE
    # scalar was computed in Python (0.3), not by any LLM.
    captured = {}

    async def _syn(question, ledger):  # noqa: ANN001
        captured["ledger"] = ledger
        return "The ratio is 0.3."
    monkeypatch.setattr(D.synthesizer, "synthesize", _syn)

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator(dag.question, None, "c1", ctx, st, "req-2"))

    assert out is not None
    assert out["route"] == "navigator"
    led: StepLedger = captured["ledger"]
    # The COMPOSE result is stored under the PLAN step_id (s3) so downstream steps
    # that depend_on s3 read it back; the ratio 30/100 = 0.3 is computed in PYTHON.
    assert led.get_scalar("s1") == 30.0
    assert led.get_scalar("s2") == 100.0
    assert led.get_scalar("s3") == pytest.approx(0.3)  # 30/100 in PYTHON, not LLM
    assert calls["n"] == 2, "exactly two LOOKUP executions; COMPOSE runs no SQL"
    # The COMPOSE-final scalar is surfaced as a one-row dataset.
    assert out["row_count"] == 1
    assert out["data"] and pytest.approx(0.3) == list(out["data"][0].values())[0]


# ==========================================================================
# 2b. multi-branch (2 independent LOOKUP, NO compose) -> data has BOTH branches
# ==========================================================================
def test_multi_branch_payload_includes_all_sinks(monkeypatch):
    """"customer receipts vs vendor payments" decomposes into TWO independent
    LOOKUP steps (no depends_on, no COMPOSE). BOTH are sinks. The structured
    payload must carry rows from BOTH branches (the synthesizer prose describes
    both), each row tagged with its branch — never silently drop one branch the
    way ``list(ledger.results.values())[-1]`` did."""
    dag = StepDAG(
        question="customer receipts vs vendor payments",
        steps=(
            IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="customer",
                       measure_concept="receipts", grain="entity",
                       grain_entity="customer"),
            IntentStep(step_id="s2", kind=StepKind.LOOKUP, entity="vendor",
                       measure_concept="payments", grain="entity",
                       grain_entity="vendor"),
        ),
    )
    _patch_plan(monkeypatch, dag)

    # Each step resolves to its OWN table so the two branches are distinguishable.
    tables = {"s1": "AR_RECEIPTS", "s2": "VENDOR_PAYMENTS"}
    files = {"s1": "f1", "s2": "f2"}

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice(step.step_id, tables[step.step_id], files[step.step_id])
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet(sl.step_id, [_vendor_evidence(table=tables[sl.step_id])])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed(step.step_id, table=tables[step.step_id])
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)

    # s1 -> 2 receipt rows; s2 -> 1 payment row. Distinct row contents per branch.
    seq = {
        "AR_RECEIPTS": [{"VENDOR_ID": "C1", "amount": 10.0},
                        {"VENDOR_ID": "C2", "amount": 20.0}],
        "VENDOR_PAYMENTS": [{"VENDOR_ID": "V1", "amount": 500.0}],
    }
    calls = {"n": 0}

    class _Canon:
        def __init__(self, sql):
            self.executable_sql = sql
            self.logical_sql = sql
            self.referenced_file_ids = []
            self.referenced_tables = []
            self.physical_uris = []

    monkeypatch.setattr(X, "canonicalize_logical_sql",
                        lambda sql, im, *, allowed_file_ids=None: _Canon(sql))

    def _exec(sql, cs, cn, mr, engine=None):  # noqa: ANN001
        # The rendered SQL names the table; pick the right branch deterministically.
        rows = seq["AR_RECEIPTS"] if '"AR_RECEIPTS"' in sql else seq["VENDOR_PAYMENTS"]
        calls["n"] += 1
        return list(rows), len(rows)
    monkeypatch.setattr(X, "_execute", _exec)

    _patch_synthesize(monkeypatch, "Customers received X; vendors were paid Y.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator(dag.question, None, "c1", ctx, st, "req-8"))

    assert out is not None
    assert out["route"] == "navigator"
    assert calls["n"] == 2, "both branches execute"
    # BOTH branches present: 2 receipt rows + 1 payment row = 3.
    assert out["row_count"] == 3
    assert len(out["data"]) == 3
    # Every row is tagged with a branch key so the payload represents both sinks.
    branch_key = "_branch"
    branches = {r.get(branch_key) for r in out["data"]}
    assert len(branches) == 2, f"both branches must be labelled, saw {branches}"
    # The underlying measures from both tables survived.
    amounts = sorted(r.get("amount") for r in out["data"])
    assert amounts == [10.0, 20.0, 500.0]
    # files_used still lists every touched table.
    assert out["files_used"] == ["AR_RECEIPTS", "VENDOR_PAYMENTS"]
    assert popped == ["req-8"]


def test_single_sink_selection_is_last_step_byte_identical(monkeypatch):
    """REGRESSION: a single-step plan selects exactly the step ``[-1]`` would —
    the multi-sink change must not alter the single-sink (common) case at all."""
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed("s1")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendor V1 spent 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-9"))

    assert out is not None
    # Byte-identical to the legacy single-sink payload: NO branch tag, exact rows.
    assert out["data"] == [{"VENDOR_ID": "V1", "amount": 500.0}]
    assert out["row_count"] == 1


def test_compose_terminated_plan_selects_only_compose_sink(monkeypatch):
    """REGRESSION: in a 2-LOOKUP + 1-COMPOSE plan the LOOKUP steps are COMPOSE
    operands (NOT sinks); the lone sink is the COMPOSE step. The payload must be
    that single COMPOSE scalar as a one-row dataset — same as ``[-1]`` picked."""
    dag = StepDAG(
        question="ratio of overdue to total spend per vendor",
        steps=(
            IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                       measure_concept="overdue spend", grain="entity",
                       grain_entity="vendor"),
            IntentStep(step_id="s2", kind=StepKind.LOOKUP, entity="vendor",
                       measure_concept="total spend", grain="entity",
                       grain_entity="vendor"),
            IntentStep(step_id="s3", kind=StepKind.COMPOSE,
                       depends_on=("s1", "s2"),
                       compose_expr={"op": "ratio", "left_step": "s1", "right_step": "s2"}),
        ),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice(step.step_id, "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet(sl.step_id, [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed(step.step_id)
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)

    seq = [[{"amount": 30.0}], [{"amount": 100.0}]]
    calls = {"n": 0}

    class _Canon:
        def __init__(self, sql):
            self.executable_sql = sql
            self.logical_sql = sql
            self.referenced_file_ids = []
            self.referenced_tables = []
            self.physical_uris = []

    monkeypatch.setattr(X, "canonicalize_logical_sql",
                        lambda sql, im, *, allowed_file_ids=None: _Canon(sql))

    def _exec(sql, cs, cn, mr, engine=None):  # noqa: ANN001
        rows = seq[calls["n"]]
        calls["n"] += 1
        return list(rows), 1
    monkeypatch.setattr(X, "_execute", _exec)
    _patch_synthesize(monkeypatch, "The ratio is 0.3.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator(dag.question, None, "c1", ctx, st, "req-10"))

    assert out is not None
    assert out["route"] == "navigator"
    # The COMPOSE sink's scalar (0.3) surfaced as a single one-row dataset — NOT a
    # multi-branch concatenation of the two LOOKUP operands.
    assert out["row_count"] == 1
    assert len(out["data"]) == 1
    assert pytest.approx(0.3) == list(out["data"][0].values())[0]


# ==========================================================================
# 3. polarity tie/contradiction -> route="navigator_clarify", NO SQL executed
# ==========================================================================
def test_polarity_contradiction_clarifies_without_sql(monkeypatch):
    dag = StepDAG(
        question="vendor payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    # The slice has a customer-side and a vendor-side candidate; the chosen pick is
    # customer-side, but the question's polarity is vendor -> contradiction.
    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(
            step_id="s1", entity="vendor",
            candidates=(
                Candidate(file_id="f1", table="AR_RECEIPTS", score=0.8, polarity="customer"),
                Candidate(file_id="f2", table="AP_PAYMENTS", score=0.7, polarity="vendor"),
            ),
            from_map=False,
        )
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [
            _vendor_evidence(table="AR_RECEIPTS", polarity="customer"),
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    # mini proposes the WRONG (customer) side table.
    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed("s1", table="AR_RECEIPTS")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    # The question's polarity is vendor -> verify rejects the customer pick.
    _patch_q_polarity(monkeypatch, "vendor")
    state = _patch_engine(monkeypatch, rows=[{"x": 1}])
    _patch_synthesize(monkeypatch, "(should not be called)")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("vendor payments", None, "c1", ctx, st, "req-3"))

    assert out is not None
    assert out["route"] == "navigator_clarify"
    assert out["answer"] == "vendor payments"   # the question echoed back
    assert out["data"] == []
    assert out["chart"] is None
    assert "clarify" in out
    assert out["clarify"]["reason"] == "polarity_contradicts_question"
    assert len(out["clarify"]["options"]) >= 1
    assert state["exec_calls"] == 0, "NO SQL may run on a clarify"
    assert popped == ["req-3"], "store popped on a clarify"


def test_genuine_polarity_tie_clarifies_before_proposing(monkeypatch):
    """A slice spanning BOTH reliable sides (customer + vendor) with an UNKNOWN
    question polarity is a genuine tie: the driver must clarify BEFORE proposing —
    so mini's propose is never even called and no SQL runs."""
    dag = StepDAG(
        question="payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(
            step_id="s1", entity="vendor",
            candidates=(
                Candidate(file_id="f1", table="AR_RECEIPTS", score=0.8, polarity="customer"),
                Candidate(file_id="f2", table="AP_PAYMENTS", score=0.7, polarity="vendor"),
            ),
            from_map=False,
        )
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [
            _vendor_evidence(table="AR_RECEIPTS", polarity="customer"),
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    propose_calls = {"n": 0}

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        propose_calls["n"] += 1
        return _proposed("s1")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    # UNKNOWN question polarity (None) + two reliable sides -> genuine tie.
    _patch_q_polarity(monkeypatch, None)
    state = _patch_engine(monkeypatch, rows=[{"x": 1}])

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("payments", None, "c1", ctx, st, "req-7"))

    assert out is not None
    assert out["route"] == "navigator_clarify"
    assert out["clarify"]["reason"] == "polarity_tie"
    assert propose_calls["n"] == 0, "the tie is caught BEFORE mini proposes"
    assert state["exec_calls"] == 0
    assert popped == ["req-7"]


# ==========================================================================
# 3b. reliability-gated polarity PRE-FILTER applied to PROPOSE (FIX B / M5)
# ==========================================================================
def _two_side_slice() -> CandidateSlice:
    """A slice with one customer-side and one vendor-side candidate."""
    return CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(
            Candidate(file_id="f1", table="AR_RECEIPTS", score=0.8, polarity="customer"),
            Candidate(file_id="f2", table="AP_PAYMENTS", score=0.7, polarity="vendor"),
        ),
        from_map=False,
    )


def _capture_propose(monkeypatch):
    """Patch proposer.propose to RECORD the tables present in the evidence packet it
    is handed (so a test can assert the pre-filter narrowed the slice), and propose
    the vendor-side table so the run reaches an answer."""
    seen: dict[str, list[str]] = {}

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        seen["tables"] = [f["table"] for f in ev.files]
        return _proposed(step.step_id, table="AP_PAYMENTS")
    monkeypatch.setattr(D.proposer, "propose", _propose)
    return seen


def test_reliable_opposite_side_is_filtered_before_propose(monkeypatch):
    """q_polarity reliably known (vendor); the slice has a reliable-opposite
    (customer) candidate + a same-side (vendor) one. PROPOSE must receive ONLY the
    same-side candidate — the cross-side twin is dropped BEFORE the LLM sees it."""
    dag = StepDAG(
        question="vendor payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _two_side_slice()
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # Evidence carries the RELIABILITY-GATED polarity (as evidence.assemble emits).
        return _packet("s1", [
            _vendor_evidence(table="AR_RECEIPTS", polarity="customer"),
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)
    _patch_q_polarity(monkeypatch, "vendor")
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("vendor payments", None, "c1", ctx, st, "req-m5a"))

    assert out is not None and out["route"] == "navigator"
    assert seen["tables"] == ["AP_PAYMENTS"], (
        "the reliable-opposite (customer) twin must be filtered before propose"
    )


def test_unreliable_opposite_polarity_is_not_filtered(monkeypatch):
    """The opposite-looking candidate carries an UNRELIABLE polarity (evidence
    polarity=None after the reliability gate). It must NOT be filtered — both
    candidates reach propose. A misclassified/guessed side can NEVER drop the right
    table (never-reduce-correctness)."""
    dag = StepDAG(
        question="vendor payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _two_side_slice()
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # The "customer-looking" candidate's reliability gate produced None (unreliable),
        # so its evidence polarity is None — partition must keep it.
        return _packet("s1", [
            _vendor_evidence(table="AR_RECEIPTS", polarity=None),
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)
    _patch_q_polarity(monkeypatch, "vendor")
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("vendor payments", None, "c1", ctx, st, "req-m5b"))

    assert out is not None
    assert sorted(seen["tables"]) == ["AP_PAYMENTS", "AR_RECEIPTS"], (
        "an unreliable (None) polarity must never be filtered out"
    )


def test_filter_keeps_matching_side_and_unconstrained(monkeypatch):
    """The filtered set handed to PROPOSE keeps the matching side AND any
    unconstrained (reliability-gated None) candidate, dropping ONLY the reliable
    opposite — so the right table is never lost. (The hard never-empty guarantee
    itself is unit-tested directly on ``partition_by_polarity``.)"""
    dag = StepDAG(
        question="vendor payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(
            step_id="s1", entity="vendor",
            candidates=(
                Candidate(file_id="f1", table="AR_RECEIPTS", score=0.8, polarity="customer"),
                Candidate(file_id="f2", table="AP_PAYMENTS", score=0.7, polarity="vendor"),
            ),
            from_map=False,
        )
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # Three candidates: reliable opposite (customer), matching (vendor), and an
        # UNCONSTRAINED (None) one. q_polarity=vendor must keep {vendor, None}.
        return _packet("s1", [
            _vendor_evidence(table="AR_RECEIPTS", polarity="customer"),
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
            _vendor_evidence(table="GL_LEDGER", polarity=None),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)
    _patch_q_polarity(monkeypatch, "vendor")
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("vendor payments", None, "c1", ctx, st, "req-m5c"))

    assert out is not None and out["route"] == "navigator"
    assert sorted(seen["tables"]) == ["AP_PAYMENTS", "GL_LEDGER"], (
        "keep matching side + unconstrained; drop ONLY the reliable opposite"
    )


def test_unknown_q_polarity_passes_full_set_unchanged(monkeypatch):
    """When q_polarity is unknown the slice is single-sided (no tie), so the full set
    is passed unchanged (current behavior preserved)."""
    dag = StepDAG(
        question="vendor payments",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="payments", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(
            step_id="s1", entity="vendor",
            candidates=(
                Candidate(file_id="f1", table="AP_PAYMENTS", score=0.8, polarity="vendor"),
                Candidate(file_id="f2", table="AP_PAYMENTS_2", score=0.7, polarity="vendor"),
            ),
            from_map=False,
        )
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # Single reliable side (vendor) -> partition leaves it unchanged regardless.
        return _packet("s1", [
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
            _vendor_evidence(table="AP_PAYMENTS_2", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)
    # q_polarity UNKNOWN (None).
    _patch_q_polarity(monkeypatch, None)
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("vendor payments", None, "c1", ctx, st, "req-m5d"))

    assert out is not None
    assert sorted(seen["tables"]) == ["AP_PAYMENTS", "AP_PAYMENTS_2"], (
        "unknown q_polarity must pass the full set unchanged"
    )


# ==========================================================================
# 3c. MAP-HIT master constraint applied to PROPOSE (I5 strengthened)
# ==========================================================================
def _map_hit_twin_slice(master_id: str = "f2") -> CandidateSlice:
    """A MAP-HIT slice: one governed canonical master (``master_id`` -> AP_PAYMENTS)
    plus two schema-twin siblings. ``from_map=True`` and the master is declared in
    ``master_file_ids`` (exactly one). All three are vendor-side lookalikes."""
    return CandidateSlice(
        step_id="s1", entity="vendor",
        candidates=(
            Candidate(file_id="f2", table="AP_PAYMENTS", score=0.0, polarity="vendor"),
            Candidate(file_id="f3", table="AP_PAYMENTS_2", score=0.0, polarity="vendor"),
            Candidate(file_id="f4", table="AP_PAYMENTS_3", score=0.0, polarity="vendor"),
        ),
        from_map=True,
        master_file_ids=(master_id,),
    )


def test_map_hit_single_master_constrains_propose(monkeypatch):
    """A MAP HIT with EXACTLY ONE canonical master in the slice must CONSTRAIN
    propose to that master — the map already decided, so the noisy ``good_for`` of
    the templated twins can NOT re-litigate the table choice. PROPOSE receives ONLY
    the master's evidence; the twins remain available to verify/join."""
    dag = StepDAG(
        question="invoice amount by vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="invoice amount", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _map_hit_twin_slice(master_id="f2")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # All three twins carry near-identical good_for (the dataset's noise).
        return _packet("s1", [
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
            _vendor_evidence(table="AP_PAYMENTS_2", polarity="vendor"),
            _vendor_evidence(table="AP_PAYMENTS_3", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)  # proposes AP_PAYMENTS (the master) back
    _patch_q_polarity(monkeypatch, "vendor")
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("invoice amount by vendor", None, "c1", ctx, st, "req-i5a"))

    assert out is not None and out["route"] == "navigator"
    assert seen["tables"] == ["AP_PAYMENTS"], (
        "a single-master map hit must constrain propose to the governed master only"
    )


def test_non_map_slice_propose_sees_full_set_unchanged(monkeypatch):
    """REGRESSION: a NON-map slice (``from_map=False``) with the SAME three-twin
    shape must leave the propose evidence UNCHANGED — the master constraint applies
    ONLY to map hits, so retrieval-driven slices keep abstain-bias / full evidence."""
    dag = StepDAG(
        question="invoice amount by vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="invoice amount", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        # Same candidates, but a RETRIEVED slice: from_map=False, no master declared.
        return CandidateSlice(
            step_id="s1", entity="vendor",
            candidates=(
                Candidate(file_id="f2", table="AP_PAYMENTS", score=0.8, polarity="vendor"),
                Candidate(file_id="f3", table="AP_PAYMENTS_2", score=0.7, polarity="vendor"),
                Candidate(file_id="f4", table="AP_PAYMENTS_3", score=0.6, polarity="vendor"),
            ),
            from_map=False,
        )
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [
            _vendor_evidence(table="AP_PAYMENTS", polarity="vendor"),
            _vendor_evidence(table="AP_PAYMENTS_2", polarity="vendor"),
            _vendor_evidence(table="AP_PAYMENTS_3", polarity="vendor"),
        ])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen = _capture_propose(monkeypatch)
    _patch_q_polarity(monkeypatch, "vendor")
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendors were paid 500.")

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("invoice amount by vendor", None, "c1", ctx, st, "req-i5b"))

    assert out is not None
    assert sorted(seen["tables"]) == ["AP_PAYMENTS", "AP_PAYMENTS_2", "AP_PAYMENTS_3"], (
        "a non-map (retrieved) slice must pass the full evidence unchanged"
    )


# ==========================================================================
# 4. abstain (empty slice) -> run_navigator returns None, store NOT popped
# ==========================================================================
def test_empty_slice_abstains_returns_none(monkeypatch):
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(step_id="s1", entity="vendor", candidates=(), from_map=False)
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    state = _patch_engine(monkeypatch, rows=[{"x": 1}])
    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-4"))

    assert out is None, "an empty slice must abstain (None) -> agent fall-through"
    assert state["exec_calls"] == 0
    assert popped == [], "store must NOT be popped on abstain (agent still needs it)"


def test_plan_none_abstains(monkeypatch):
    _patch_plan(monkeypatch, None)
    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("???", None, "c1", ctx, st, "req-5"))
    assert out is None
    assert popped == []


# ==========================================================================
# 4b. FIX B (M4): every abstain emits ONE standardized ``navigator_abstain``
# event with its reason — so a SYSTEMIC verify regression (every step failing)
# is distinguishable from honest "no plan fits" in production. Control flow is
# unchanged: it still returns None and the caller still falls through.
# ==========================================================================
def _spy_logger(monkeypatch):
    """Record every (event, fields) the driver's structlog logger emits, so a test
    can assert the standardized ``navigator_abstain`` event fired. Mirrors how the
    suite monkeypatches ``D.*`` seams."""
    events: list[tuple[str, dict]] = []

    class _Spy:
        def info(self, event, **fields):  # noqa: ANN001
            events.append((event, fields))

        def warning(self, event, **fields):  # noqa: ANN001
            events.append((event, fields))

    monkeypatch.setattr(D, "logger", _Spy())
    return events


def _abstain_events(events: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    return [(e, f) for (e, f) in events if e == "navigator_abstain"]


def test_abstain_emits_standardized_event_with_reason(monkeypatch):
    """A plan-abstain (planner returns None) returns None AND emits exactly one
    ``navigator_abstain`` event carrying the site's reason."""
    events = _spy_logger(monkeypatch)
    _patch_plan(monkeypatch, None)
    ctx, st, popped = _ctx(monkeypatch)

    out = _run(D.run_navigator("???", None, "c1", ctx, st, "req-m4a"))

    assert out is None, "abstain control flow intact (still returns None)"
    abstains = _abstain_events(events)
    assert len(abstains) == 1, f"exactly one standardized abstain event, saw {events}"
    _evt, fields = abstains[0]
    assert fields.get("reason") == "navigator_plan_abstain"
    assert popped == [], "store NOT popped on abstain"


def test_step_abstain_emits_event_with_step_reason(monkeypatch):
    """An empty slice abstains at the STEP level: one ``navigator_abstain`` event
    fires with the empty-slice reason (preserving the existing reason text) and
    surfaces the step_id field for queryability."""
    events = _spy_logger(monkeypatch)
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return CandidateSlice(step_id="s1", entity="vendor", candidates=(), from_map=False)
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    _patch_engine(monkeypatch, rows=[{"x": 1}])
    ctx, st, popped = _ctx(monkeypatch)

    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-m4b"))

    assert out is None
    abstains = _abstain_events(events)
    assert len(abstains) == 1, f"exactly one abstain event, saw {events}"
    _evt, fields = abstains[0]
    # The step-resolution reason for an empty slice is preserved verbatim.
    assert fields.get("reason") == "empty_slice"
    assert fields.get("step_id") == "s1"


def test_seam_error_abstains_with_seam_error_reason(monkeypatch):
    """The catch-all ``except`` abstains with reason ``navigator_seam_error`` (still
    returns None) so an unexpected crash is also a queryable abstain, not a silent one."""
    events = _spy_logger(monkeypatch)

    async def _boom(question, seed=None, as_of=None):  # noqa: ANN001
        raise RuntimeError("unexpected")
    monkeypatch.setattr(D, "plan_question", _boom)

    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("anything", None, "c1", ctx, st, "req-m4c"))

    assert out is None
    abstains = _abstain_events(events)
    assert len(abstains) == 1
    _evt, fields = abstains[0]
    assert fields.get("reason") == "navigator_seam_error"


def test_unverifiable_pick_abstains(monkeypatch):
    """A proposed table not in the slice -> verify rejects -> abstain (None)."""
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    # mini hallucinates a table that is NOT in the slice -> verify "table_not_in_slice".
    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed("s1", table="NONEXISTENT_TABLE")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    state = _patch_engine(monkeypatch, rows=[{"x": 1}])
    ctx, st, popped = _ctx(monkeypatch)
    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-6"))

    assert out is None, "an unverifiable pick must abstain (None)"
    assert state["exec_calls"] == 0
    assert popped == []


# ==========================================================================
# 4b. TIME-WINDOW threading + abstain-safety (L8). The driver resolves the
# question's time window once (temporal.parse_temporal anchored on ctx["as_of"])
# and threads it through propose -> verify -> render. A time-scoped question
# must NEVER be answered with an ALL-TIME number.
# ==========================================================================
def _dated_vendor_evidence(table: str = "VENDOR_PAYMENTS") -> dict:
    """Vendor evidence WITH a real date column (PAYMENT_DATE, type/role=date) so a
    time window can be applied to it."""
    return {
        "table": table,
        "valid_cols": {"VENDOR_ID": "VENDOR_ID", "AMOUNT": "AMOUNT",
                       "STATUS": "STATUS", "PAYMENT_DATE": "PAYMENT_DATE"},
        "columns": [
            {"name": "VENDOR_ID", "type": "string", "role": "id"},
            {"name": "AMOUNT", "type": "double", "role": "amount"},
            {"name": "STATUS", "type": "string", "role": "status"},
            {"name": "PAYMENT_DATE", "type": "date", "role": "date"},
        ],
        "sample_rows": [{"VENDOR_ID": "V1", "AMOUNT": 100.0, "PAYMENT_DATE": "2025-02-01"}],
        "description": "Vendor payments with a payment date",
        "good_for": ["vendor spend"],
        "coverage": "2023-01..2025-05",
        "row_count": 1000,
        "value_set": {"STATUS": {"open": 5, "closed": 3}},
        "unique_rates": {"VENDOR_ID": 0.9, "AMOUNT": 0.3},
        "polarity": None,
        "process_role": "",
        "erp_module": "",
        "erp_confidence": 0.0,
    }


def _capture_rendered_sql(monkeypatch, *, rows, total=None):
    """Patch the executor's engine primitives, RECORDING the (deterministically
    rendered) SQL string the executor was handed — so a test can assert the date
    bounds are (or are not) present. The driver renders REAL SQL from the verified
    contract; only the engine is mocked."""
    state = {"exec_calls": 0, "sql": []}

    class _Canon:
        def __init__(self, sql):
            self.executable_sql = sql
            self.logical_sql = sql
            self.referenced_file_ids = []
            self.referenced_tables = []
            self.physical_uris = []

    monkeypatch.setattr(X, "canonicalize_logical_sql",
                        lambda sql, im, *, allowed_file_ids=None: _Canon(sql))

    def _exec(sql, cs, cn, mr, engine=None):  # noqa: ANN001
        state["exec_calls"] += 1
        state["sql"].append(sql)
        return list(rows), (len(rows) if total is None else total)
    monkeypatch.setattr(X, "_execute", _exec)
    return state


def test_time_scoped_question_renders_date_bounds(monkeypatch):
    """(a) A time-scoped question ("last quarter") + a verified contract whose table
    HAS a date column -> the rendered SQL carries the >= / <= date bounds on that
    column. The window is resolved INSIDE run_navigator from the question text +
    ctx["as_of"] (the data's latest date), anchoring the relative window."""
    import datetime as _dt

    dag = StepDAG(
        question="total invoice amount per vendor last quarter",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="invoice amount", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        return _packet("s1", [_dated_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    # PROPOSE must RECEIVE a non-None window (the driver resolved it) AND name the
    # date column so the verifier can bind it. Record the window it was handed.
    seen: dict = {}

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        seen["time_window"] = time_window
        pc = _proposed("s1")
        return ProposedContract(**{**pc.__dict__, "time_filter_column": "PAYMENT_DATE"})
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    state = _capture_rendered_sql(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendor V1 spent 500 last quarter.")

    ctx, st, popped = _ctx(monkeypatch)
    # Anchor 'now' to the data's latest date so "last quarter" resolves to a window
    # the data covers (2025-04-30 -> last calendar quarter = 2025-01-01..2025-03-31).
    ctx["as_of"] = _dt.date(2025, 4, 30)

    out = _run(D.run_navigator(dag.question, None, "c1", ctx, st, "req-tw1"))

    assert out is not None and out["route"] == "navigator"
    # The driver resolved a NON-None window from the question + as_of and threaded it.
    assert seen["time_window"] is not None
    assert seen["time_window"][0] == _dt.date(2025, 1, 1)
    assert seen["time_window"][1] == _dt.date(2025, 3, 31)
    # The rendered SQL carries the date bounds on the quoted date column.
    assert state["exec_calls"] == 1
    sql = state["sql"][0]
    assert '"PAYMENT_DATE" >= \'2025-01-01\'' in sql
    assert '"PAYMENT_DATE" <= \'2025-03-31\'' in sql


def test_time_scoped_question_without_date_column_abstains(monkeypatch):
    """(b) A time-scoped question + a chosen table with NO usable date column ->
    verify FAILS ``no_time_column_for_window`` -> run_navigator ABSTAINS (None) and
    NO SQL runs. The whole point: never answer a time-scoped question with an
    all-time number."""
    import datetime as _dt

    events = _spy_logger(monkeypatch)

    dag = StepDAG(
        question="total invoice amount per vendor last quarter",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="invoice amount", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # The default _vendor_evidence() has NO date column at all.
        return _packet("s1", [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        return _proposed("s1")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    state = _capture_rendered_sql(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}])
    _patch_synthesize(monkeypatch, "(should not be called)")

    ctx, st, popped = _ctx(monkeypatch)
    ctx["as_of"] = _dt.date(2025, 4, 30)

    out = _run(D.run_navigator(dag.question, None, "c1", ctx, st, "req-tw2"))

    assert out is None, "a windowed question with no date column must ABSTAIN"
    assert state["exec_calls"] == 0, "NO all-time SQL may run for a windowed question"
    assert popped == [], "store left in place on an abstain (agent fall-through)"
    # The standardized abstain reason is the new no_time_column_for_window signal.
    abstains = _abstain_events(events)
    assert any(f.get("reason") == "no_time_column_for_window" for _e, f in abstains)


def test_non_time_scoped_question_renders_no_date_bounds(monkeypatch):
    """(c) A NON-time-scoped question -> window is None -> behaviour byte-identical
    to today: the rendered SQL adds NO date WHERE clause, and a date-less table is
    NOT required to carry a date column. Proves the no-window path is unchanged."""
    dag = StepDAG(
        question="total spend per vendor",
        steps=(IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor",
                          measure_concept="total spend", grain="entity",
                          grain_entity="vendor"),),
    )
    _patch_plan(monkeypatch, dag)

    async def _lookup(db, cid, step, **kw):  # noqa: ANN001
        return _slice("s1", "VENDOR_PAYMENTS", "f1")
    monkeypatch.setattr(D.retriever, "lookup", _lookup)

    async def _assemble(db, sl):  # noqa: ANN001
        # Date-less evidence; with NO window this verifies fine (today's behaviour).
        return _packet("s1", [_vendor_evidence()])
    monkeypatch.setattr(D.evidence, "assemble", _assemble)

    seen: dict = {}

    async def _propose(q, step, ev, time_window=None, prior_failure=None):  # noqa: ANN001
        seen["time_window"] = time_window
        return _proposed("s1")
    monkeypatch.setattr(D.proposer, "propose", _propose)

    _patch_q_polarity(monkeypatch, None)
    state = _capture_rendered_sql(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 500.0}], total=1)
    _patch_synthesize(monkeypatch, "Vendor V1 spent 500.")

    ctx, st, popped = _ctx(monkeypatch)
    ctx["as_of"] = None   # also exercises the wall-clock-anchor fallback path

    out = _run(D.run_navigator("total spend per vendor", None, "c1", ctx, st, "req-tw3"))

    assert out is not None and out["route"] == "navigator"
    # No time scope detected -> the driver passes window=None to propose.
    assert seen["time_window"] is None
    # The rendered SQL is byte-identical to the no-window entity-grain SQL: no date WHERE.
    assert state["exec_calls"] == 1
    sql = state["sql"][0]
    assert "PAYMENT_DATE" not in sql
    assert ">=" not in sql and "<=" not in sql
    assert sql == (
        'SELECT "VENDOR_ID", SUM("AMOUNT") AS "amount"\n'
        'FROM "VENDOR_PAYMENTS"\n'
        'GROUP BY "VENDOR_ID"\n'
        'ORDER BY "amount" DESC'
    )


# ==========================================================================
# 5. synthesizer: numbers in the prose come from the LEDGER, not invented
# ==========================================================================
def test_synthesizer_uses_ledger_numbers_not_invented(monkeypatch):
    """The mini call is mocked to ECHO back the digest it was given. We then assert
    the ledger's number (12345) reaches the prompt — i.e. the figure originates in
    the ledger (I2), never from the model."""
    captured = {}

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    class _Resp:
        def __init__(self, content):
            self.choices = [_Msg(content)]

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0, max_completion_tokens=400):
                    captured["prompt"] = messages[0]["content"]
                    # A faithful synthesizer would only USE the supplied numbers.
                    return _Resp("The verified total is 12,345.")

    monkeypatch.setattr(S, "get_client", lambda: (_Client(), None))

    ledger = StepLedger()
    ledger.results["s1"] = StepResult(
        step_id="s1", sql="SELECT ...", rows=({"amount": 12345.0},), total=1,
        table="VENDOR_PAYMENTS", measure_label="SUM(AMOUNT)", grain="entity",
        scalar=12345.0,
    )

    prose = _run(S.synthesize("total spend", ledger))

    # The number the model was ALLOWED to use was injected from the ledger digest.
    assert "12345" in captured["prompt"] or "12345.0" in captured["prompt"]
    assert "12,345" in prose


def test_synthesizer_falls_back_to_templated_on_llm_error(monkeypatch):
    """When the LLM errors, the answer is a deterministic templated sentence built
    from the SAME ledger — so a number can NEVER originate in the synthesizer."""
    def _boom():
        raise RuntimeError("llm down")
    monkeypatch.setattr(S, "get_client", _boom)

    ledger = StepLedger()
    ledger.results["s1"] = StepResult(
        step_id="s1", rows=(), total=7, table="VENDOR_PAYMENTS",
        measure_label="SUM(AMOUNT)", grain="entity", scalar=None,
    )
    prose = _run(S.synthesize("total spend", ledger))
    assert "7" in prose
    assert "SUM(AMOUNT)" in prose


def test_synthesizer_empty_ledger_safe_line(monkeypatch):
    prose = _run(S.synthesize("anything", StepLedger()))
    assert isinstance(prose, str) and prose


# ==========================================================================
# 6. synthesizer NUMBER-GROUNDING GUARD (FIX A / M6)
# ==========================================================================
def _patch_synthesizer_mini(monkeypatch, prose_text):
    """Patch synthesizer.get_client so its ONE call returns ``prose_text`` verbatim,
    EXACTLY the way the existing synthesizer tests mock the LLM."""
    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    class _Resp:
        def __init__(self, content):
            self.choices = [_Msg(content)]

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature=0, max_completion_tokens=400):
                    return _Resp(prose_text)

    monkeypatch.setattr(S, "get_client", lambda: (_Client(), None))


def _scalar_ledger(scalar=12345.0, table="VENDOR_PAYMENTS",
                   measure_label="SUM(AMOUNT)") -> StepLedger:
    ledger = StepLedger()
    ledger.results["s1"] = StepResult(
        step_id="s1", sql="SELECT ...", rows=({"amount": scalar},), total=1,
        table=table, measure_label=measure_label, grain="entity", scalar=scalar,
    )
    return ledger


def test_synthesizer_fabricated_magnitude_falls_back_to_template(monkeypatch):
    """A hallucinated magnitude NOT in the ledger trips the grounding guard, so the
    LLM prose is DISCARDED and the deterministic ``_fallback_prose`` is returned —
    the one place a wrong number could otherwise surface verbatim (M6)."""
    ledger = _scalar_ledger(scalar=12345.0)
    # The ledger's only verified number is 12,345; 98,765.43 is fabricated.
    _patch_synthesizer_mini(monkeypatch, "The verified total is 98,765.43.")

    prose = _run(S.synthesize("total spend", ledger))

    expected = S._fallback_prose("total spend", ledger)
    assert prose == expected, "ungrounded magnitude must fall back to the template"
    # The fabricated figure must NOT survive; the grounded one (from the template) does.
    assert "98,765" not in prose
    assert "12,345" in prose


def test_synthesizer_grounded_scalar_prose_is_kept(monkeypatch):
    """Prose containing ONLY the grounded (formatted) scalar is kept verbatim — the
    guard is TOLERANT and must never reject a faithful answer."""
    ledger = _scalar_ledger(scalar=12345.0)
    _patch_synthesizer_mini(monkeypatch, "The total verified spend is 12,345.")

    prose = _run(S.synthesize("total spend", ledger))
    assert prose == "The total verified spend is 12,345."


def test_synthesizer_small_counts_and_years_not_false_rejected(monkeypatch):
    """Tolerance: small bare integers ("top 5", "3 results"), ordinals, and years
    appearing in the rows must NEVER trip the guard alongside grounded magnitudes."""
    ledger = StepLedger()
    # A top-3 result over rows that carry a year (2024) and grounded magnitudes.
    ledger.results["s1"] = StepResult(
        step_id="s1", sql="SELECT ...",
        rows=(
            {"VENDOR_ID": "V1", "year": 2024, "amount": 12345.0},
            {"VENDOR_ID": "V2", "year": 2024, "amount": 6789.0},
            {"VENDOR_ID": "V3", "year": 2024, "amount": 4321.0},
        ),
        total=3, table="VENDOR_PAYMENTS", measure_label="SUM(AMOUNT)",
        grain="entity", scalar=None,
    )
    _patch_synthesizer_mini(
        monkeypatch,
        "Across the top 3 vendors in 2024, the leader spent 12,345, followed by "
        "6,789 and 4,321.",
    )

    prose = _run(S.synthesize("top vendors", ledger))
    # Kept verbatim: "3", "2024" (year in rows), the grounded magnitudes all pass.
    assert prose == (
        "Across the top 3 vendors in 2024, the leader spent 12,345, followed by "
        "6,789 and 4,321."
    )


def test_prose_is_grounded_pure_helper_rejects_only_unknown_magnitude():
    """The PURE helper: a grounded magnitude + small ints pass; an unknown magnitude
    fails. A year-like int is grounded ONLY when it appears in the rows. Exercises the
    helper directly (no LLM)."""
    ledger = _scalar_ledger(scalar=12345.0)
    # Grounded scalar + a small count -> grounded (no year claimed).
    assert S._prose_is_grounded("Top 5 vendors spent 12,345.", ledger) is True
    # An unknown magnitude (decimal / thousands / >=1000) -> NOT grounded.
    assert S._prose_is_grounded("It was 7,654.32 actually.", ledger) is False
    # A bare small integer alone is never a violation.
    assert S._prose_is_grounded("There are 3 vendors.", ledger) is True
    # A year NOT present in the rows is treated like any other ungrounded magnitude.
    assert S._prose_is_grounded("In 2024 it was 12,345.", ledger) is False

    # When the year DOES appear in the rows it is grounded (structurally allowed).
    led_with_year = StepLedger()
    led_with_year.results["s1"] = StepResult(
        step_id="s1", rows=({"year": 2024, "amount": 12345.0},), total=1,
        table="T", measure_label="SUM(AMOUNT)", grain="entity", scalar=None,
    )
    assert S._prose_is_grounded("In 2024 spend was 12,345.", led_with_year) is True

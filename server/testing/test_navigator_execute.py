"""Phase-4 tests for the navigator package: executor.py + promote.py + composer.py.

DETERMINISTIC — no live engine, no live DB, no live LLM. The loop's stages [4]
EXECUTE, [5] PROMOTE and [5] COMPOSE are all pure-of-network here:

  * ``executor.execute`` runs already-rendered logical SQL through the SAME path
    the coordinator/graph use: ``canonicalize_logical_sql`` then ``_execute``. We
    monkeypatch BOTH navigator-imported names so no engine is touched: the
    canonicalizer returns a canned ``CanonicalSQL`` and ``_execute`` returns canned
    ``(rows, total)``. We assert the ``StepResult`` shape, that ``scalar`` is set
    ONLY for a 1-row × 1-numeric-measure result, and that an engine/canonicalize
    failure yields an empty-rows ``StepResult`` (NEVER raises) so the driver can
    abstain (I12).
  * ``promote.promote`` writes a verified, executed conclusion into the ledger
    keyed by ``step.step_id`` and returns the (mutated) ledger so downstream steps
    read it back via ``ledger.get_scalar(step_id)``.
  * ``composer.compose`` does ALL cross-step arithmetic in PYTHON (I9/I2): the
    ratio 30/100 = 0.3 is computed here, never by an LLM. Divide-by-zero and a
    missing upstream scalar yield a safe ``StepResult`` (scalar=None), never raise.

I11: numbers come from the engine via ``_execute`` (mocked here), never invented.
I9 : cross-step math lives ONLY in ``compose`` (Python), never the LLM.

Run: cd server && uv run python -m pytest testing/test_navigator_execute.py -q
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

import app.services.navigator.executor as X
import app.services.navigator.promote as PR
import app.services.navigator.composer as C
from app.services.navigator.types import (
    ComposePlan,
    IntentStep,
    StepKind,
    StepLedger,
    StepResult,
    VerifiedContract,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Mock helpers: canonicalize + _execute are the navigator-imported names.
# --------------------------------------------------------------------------
class _FakeCanonical:
    """The shape executor consumes off canonicalize_logical_sql's return."""

    def __init__(self, executable_sql: str, referenced_file_ids=(), referenced_tables=()):
        self.logical_sql = executable_sql
        self.executable_sql = executable_sql
        self.referenced_file_ids = list(referenced_file_ids)
        self.referenced_tables = list(referenced_tables)
        self.physical_uris = []


def _patch_engine(monkeypatch, *, rows, total=None, canon_raises=False, exec_raises=False):
    """Patch the navigator-imported ``canonicalize_logical_sql`` and ``_execute`` so
    no real engine/canonicalizer runs. ``rows`` is the canned row list; ``total``
    defaults to len(rows)."""

    def _canon(sql, identity_map, *, allowed_file_ids=None):  # noqa: ANN001
        if canon_raises:
            raise ValueError("boom-canonicalize")
        return _FakeCanonical(sql)

    def _exec(sql, connection_string, container_name, max_rows, engine=None):  # noqa: ANN001
        if exec_raises:
            raise RuntimeError("boom-engine")
        return list(rows), (len(rows) if total is None else total)

    monkeypatch.setattr(X, "canonicalize_logical_sql", _canon)
    monkeypatch.setattr(X, "_execute", _exec)


_LOGICAL_SQL = (
    'SELECT SUM("AMOUNT") AS "amount"\nFROM "VENDOR_PAYMENTS"'
)


def _execute_kwargs(**overrides):
    base = dict(
        identity_map=object(),
        allowed_file_ids={"fa"},
        connection_string="conn",
        container_name="cont",
        step_id="s1",
        table="VENDOR_PAYMENTS",
        measure_label="amount",
        grain="entity",
    )
    base.update(overrides)
    return base


# ==========================================================================
# executor.execute — shape + scalar derivation + never-raises
# ==========================================================================
def test_execute_single_row_single_measure_sets_scalar(monkeypatch):
    # one row, one numeric column -> scalar is that value.
    _patch_engine(monkeypatch, rows=[{"amount": 100.0}])
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert isinstance(res, StepResult)
    assert res.step_id == "s1"
    assert res.table == "VENDOR_PAYMENTS"
    assert res.measure_label == "amount"
    assert res.grain == "entity"
    assert res.total == 1
    assert res.rows == ({"amount": 100.0},)
    assert res.scalar == 100.0
    assert res.sql == _LOGICAL_SQL


def test_execute_single_row_integer_measure_sets_scalar(monkeypatch):
    _patch_engine(monkeypatch, rows=[{"match_count": 42}])
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs(measure_label="match_count")))
    assert res.scalar == 42.0


def test_execute_multi_row_no_scalar(monkeypatch):
    # entity-grain result: id + measure across many rows -> NOT a scalar.
    _patch_engine(
        monkeypatch,
        rows=[
            {"VENDOR_ID": "V1", "amount": 100.0},
            {"VENDOR_ID": "V2", "amount": 50.0},
        ],
        total=2,
    )
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert res.scalar is None
    assert res.total == 2
    assert len(res.rows) == 2


def test_execute_single_row_multi_numeric_no_scalar(monkeypatch):
    # one row but TWO numeric columns -> ambiguous, not a single measure -> None.
    _patch_engine(monkeypatch, rows=[{"a": 1.0, "b": 2.0}])
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert res.scalar is None


def test_execute_single_row_non_numeric_no_scalar(monkeypatch):
    # one row, one column, but the value is not numeric -> no scalar.
    _patch_engine(monkeypatch, rows=[{"name": "VendorCo"}])
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert res.scalar is None


def test_execute_single_row_id_plus_measure_picks_measure_scalar(monkeypatch):
    # one row with an id column + one numeric measure: a single grain value with one
    # numeric measure column IS a scalar (the measure). e.g. one vendor's total.
    _patch_engine(monkeypatch, rows=[{"VENDOR_ID": "V1", "amount": 250.0}])
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert res.scalar == 250.0


def test_execute_caps_rows_at_max_rows(monkeypatch):
    big = [{"VENDOR_ID": f"V{i}", "amount": float(i)} for i in range(50)]
    _patch_engine(monkeypatch, rows=big, total=50)
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs(max_rows=5)))
    assert len(res.rows) == 5
    assert res.total == 50


def test_execute_engine_failure_returns_empty_never_raises(monkeypatch):
    _patch_engine(monkeypatch, rows=[], exec_raises=True)
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert isinstance(res, StepResult)
    assert res.rows == ()
    assert res.scalar is None
    assert res.total in (0, None)


def test_execute_canonicalize_failure_returns_empty_never_raises(monkeypatch):
    _patch_engine(monkeypatch, rows=[], canon_raises=True)
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert isinstance(res, StepResult)
    assert res.rows == ()
    assert res.scalar is None


def test_execute_zero_rows_no_scalar(monkeypatch):
    _patch_engine(monkeypatch, rows=[], total=0)
    res = _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    assert res.rows == ()
    assert res.scalar is None
    assert res.total == 0


# ==========================================================================
# promote.promote — write verified conclusion into the ledger
# ==========================================================================
def _vc(step_id: str) -> VerifiedContract:
    return VerifiedContract(
        step_id=step_id,
        table="VENDOR_PAYMENTS",
        grain_kind="aggregate",
        grain_col=None,
        measure_col="AMOUNT",
        agg="SUM",
        filters=(),
        order="DESC",
        reason="ok",
    )


def test_promote_writes_scalar_into_ledger_and_returns_it():
    ledger = StepLedger()
    step1 = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="overdue")
    step2 = IntentStep(step_id="s2", kind=StepKind.LOOKUP, entity="total")
    r1 = StepResult(step_id="s1", scalar=30.0, table="T", measure_label="overdue")
    r2 = StepResult(step_id="s2", scalar=100.0, table="T", measure_label="total")

    out = PR.promote(ledger, step1, _vc("s1"), r1)
    out = PR.promote(out, step2, _vc("s2"), r2)

    # returns the (same, mutated) ledger -> downstream can read it.
    assert out is ledger
    assert out.get_scalar("s1") == 30.0
    assert out.get_scalar("s2") == 100.0
    assert out.results["s1"] is r1
    assert out.results["s2"] is r2


def test_promote_absent_step_scalar_is_none():
    ledger = StepLedger()
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP)
    PR.promote(ledger, step, _vc("s1"), StepResult(step_id="s1", scalar=5.0))
    assert ledger.get_scalar("never") is None


def test_promote_persist_to_map_is_a_documented_noop_seam():
    # I8: persistent cross-query map write-back is a tracked follow-up, present as a
    # clearly-named no-op seam that must NOT touch any semantic table.
    assert hasattr(PR, "_persist_to_map")
    # calling the seam returns None and does nothing observable.
    assert PR._persist_to_map(StepLedger(), _vc("s1"), StepResult(step_id="s1")) is None


# ==========================================================================
# composer.compose — the ONLY cross-step math, in PYTHON (I9/I2)
# ==========================================================================
def _ledger_with(*pairs) -> StepLedger:
    ledger = StepLedger()
    for sid, scalar in pairs:
        ledger.results[sid] = StepResult(step_id=sid, scalar=scalar)
    return ledger


def test_compose_ratio_python_math():
    # THE ratio test: 30 / 100 = 0.3, computed in Python by compose (not an LLM).
    ledger = _ledger_with(("s1", 30.0), ("s2", 100.0))
    plan = ComposePlan(op="ratio", left_step="s1", right_step="s2")
    res = C.compose(ledger, plan)
    assert isinstance(res, StepResult)
    assert res.scalar == pytest.approx(0.3)


def test_compose_diff_python_math():
    ledger = _ledger_with(("s1", 100.0), ("s2", 30.0))
    res = C.compose(ledger, ComposePlan(op="diff", left_step="s1", right_step="s2"))
    assert res.scalar == pytest.approx(70.0)


def test_compose_growth_python_math():
    # growth = (left - right) / right ; (120 - 100)/100 = 0.2
    ledger = _ledger_with(("s1", 120.0), ("s2", 100.0))
    res = C.compose(ledger, ComposePlan(op="growth", left_step="s1", right_step="s2"))
    assert res.scalar == pytest.approx(0.2)


def test_compose_share_python_math():
    # share = left / (left + right) ; 25 / (25 + 75) = 0.25
    ledger = _ledger_with(("s1", 25.0), ("s2", 75.0))
    res = C.compose(ledger, ComposePlan(op="share", left_step="s1", right_step="s2"))
    assert res.scalar == pytest.approx(0.25)


def test_compose_ratio_divide_by_zero_is_none_no_raise():
    ledger = _ledger_with(("s1", 30.0), ("s2", 0.0))
    res = C.compose(ledger, ComposePlan(op="ratio", left_step="s1", right_step="s2"))
    assert res.scalar is None


def test_compose_growth_divide_by_zero_is_none_no_raise():
    ledger = _ledger_with(("s1", 30.0), ("s2", 0.0))
    res = C.compose(ledger, ComposePlan(op="growth", left_step="s1", right_step="s2"))
    assert res.scalar is None


def test_compose_share_zero_denominator_is_none_no_raise():
    # left + right == 0 -> undefined share.
    ledger = _ledger_with(("s1", 0.0), ("s2", 0.0))
    res = C.compose(ledger, ComposePlan(op="share", left_step="s1", right_step="s2"))
    assert res.scalar is None


def test_compose_missing_upstream_step_is_none_no_raise():
    ledger = _ledger_with(("s1", 30.0))  # s2 absent
    res = C.compose(ledger, ComposePlan(op="ratio", left_step="s1", right_step="s2"))
    assert isinstance(res, StepResult)
    assert res.scalar is None


def test_compose_none_upstream_scalar_is_none_no_raise():
    ledger = StepLedger()
    ledger.results["s1"] = StepResult(step_id="s1", scalar=30.0)
    ledger.results["s2"] = StepResult(step_id="s2", scalar=None)  # not a scalar step
    res = C.compose(ledger, ComposePlan(op="ratio", left_step="s1", right_step="s2"))
    assert res.scalar is None


def test_compose_unknown_op_is_none_no_raise():
    ledger = _ledger_with(("s1", 30.0), ("s2", 100.0))
    res = C.compose(ledger, ComposePlan(op="frobnicate", left_step="s1", right_step="s2"))
    assert res.scalar is None


def test_compose_measure_label_describes_computation():
    ledger = _ledger_with(("s1", 30.0), ("s2", 100.0))
    res = C.compose(ledger, ComposePlan(op="ratio", left_step="s1", right_step="s2"))
    assert res.measure_label is not None
    assert "ratio" in res.measure_label.lower()
    # the step_id is a synthetic compose id derived from the plan operands.
    assert res.step_id


# ==========================================================================
# the LLM is NEVER called in executor / promote / compose
# ==========================================================================
def test_no_llm_client_used_at_runtime(monkeypatch):
    """compose / promote / executor must NOT call an LLM client. We sentinel every
    plausible client factory so any accidental call raises; then exercise all three
    stages and assert none tripped the sentinel."""
    tripped: list[str] = []

    def _boom(*a, **k):  # noqa: ANN002, ANN003
        tripped.append("called")
        raise AssertionError("LLM client must not be called in execute/promote/compose")

    for modname in ("app.core.ai_client", "app.core.openai_client", "app.agent.llm"):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for attr in ("get_client", "get_async_client", "get_openai_client",
                     "build_llm", "get_llm"):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, _boom, raising=False)

    # executor
    _patch_engine(monkeypatch, rows=[{"amount": 10.0}])
    _run(X.execute(_LOGICAL_SQL, **_execute_kwargs()))
    # promote
    ledger = StepLedger()
    PR.promote(ledger, IntentStep(step_id="s1", kind=StepKind.LOOKUP), _vc("s1"),
               StepResult(step_id="s1", scalar=10.0))
    # compose
    C.compose(_ledger_with(("s1", 30.0), ("s2", 100.0)),
              ComposePlan(op="ratio", left_step="s1", right_step="s2"))

    assert tripped == []


# ==========================================================================
# self-containment: no resolve.* imports in the new navigator modules
# ==========================================================================
def test_phase4_modules_have_no_resolve_imports():
    pkg = pathlib.Path(X.__file__).parent
    offenders: list[str] = []
    for name in ("executor.py", "promote.py", "composer.py"):
        text = (pkg / name).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from app.services.resolve") or stripped.startswith(
                "import app.services.resolve"
            ):
                offenders.append(f"{name}: {stripped}")
    assert offenders == [], f"navigator phase-4 must not import resolve.*: {offenders}"

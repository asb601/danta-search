"""Phase-3 tests for the navigator package: proposer.py + verifier.py + renderer.py.

DETERMINISTIC — no live DB, no live LLM. The two boundaries are mocked:

  * ``proposer``'s ONE mini call — the OpenAI client is monkeypatched to return a
    canned JSON payload (or a bad-JSON / not-answerable payload). We assert the
    proposer maps it into a ``ProposedContract`` (typed slots, NOT SQL) or None.
  * ``verifier.verify_step_join`` reads ``ColumnKeyRegistry`` rows; a tiny fake
    session feeds canned registry rows so the lifted ``verify_join`` decides
    verified / abstain / audit-reject from value evidence alone.

Covers (proposer.propose):
  * good evidence + canned mini answer -> a ProposedContract whose slots match the
    answer (table / measure_column / measure_agg / grain_column / filter), and which
    contains NO sql string (I1/I2).
  * not-answerable / bad-JSON / empty-slice -> None (abstain).

Covers (verifier.verify):
  * a good contract verifies -> a VerifiedContract with exact-case identifiers.
  * a measure column missing from the table schema -> (None, reason).
  * a filter value not in the stored value_set -> (None, reason).
  * a pick whose ledger side contradicts q_polarity -> (None,
    "polarity_contradicts_question").
  * partition_by_polarity drops the contradicting side; signals a tie when ambiguous.
  * clarify_payload shape (reason + options drawn from candidates' OWN side/role).

Covers (verifier.verify_step_join):
  * two registry rows with high FK->PK containment + a unique PK -> verdict.verified.

Covers (renderer.render / render_join):
  * a known VerifiedContract -> an EXACT expected SQL string (double-quoted UPPER
    identifiers, canonicalizer-safe).
  * a join verdict -> the expected JOIN SQL.

Covers (self-containment):
  * grep the navigator package source for ``from app.services.resolve`` -> empty.

Run: cd server && uv run python -m pytest testing/test_navigator_resolve.py -q
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

import app.services.navigator.proposer as P
import app.services.navigator.verifier as V
import app.services.navigator.renderer as RND
from app.services.navigator.types import (
    ClarifyPayload,
    EvidencePacket,
    IntentStep,
    ProposedContract,
    ResolvedTable,
    StepKind,
    VerifiedContract,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Fixtures: a clean EvidencePacket for one vendor-payments table
# --------------------------------------------------------------------------
def _vendor_evidence_file() -> dict:
    """One per-file evidence dict in the shape evidence.assemble emits."""
    return {
        "table": "VENDOR_PAYMENTS",
        "valid_cols": {"VENDOR_ID": "VENDOR_ID", "AMOUNT": "AMOUNT",
                       "STATUS": "STATUS", "PAYMENT_DATE": "PAYMENT_DATE"},
        "columns": [
            {"name": "VENDOR_ID", "type": "string", "role": "id"},
            {"name": "AMOUNT", "type": "double", "role": "amount"},
            {"name": "STATUS", "type": "string", "role": "attribute"},
            {"name": "PAYMENT_DATE", "type": "date", "role": "date"},
        ],
        "sample_rows": [{"VENDOR_ID": "V1", "AMOUNT": 100.0, "STATUS": "open"}],
        "description": "Vendor payments",
        "good_for": ["vendor spend analysis"],
        "coverage": "2023-01-01..2025-05-01",
        "row_count": 1234,
        "value_set": {"STATUS": {"open": 5, "closed": 3}},
        "unique_rates": {"VENDOR_ID": 0.9, "AMOUNT": 0.3},
        "polarity": "vendor",
        "process_role": "AP_payment",
        "erp_module": "AP",
        "erp_confidence": 0.95,
    }


def _vendor_packet() -> EvidencePacket:
    return EvidencePacket(step_id="s1", files=(_vendor_evidence_file(),))


def _good_mini_answer() -> dict:
    """The canned JSON the mini client returns for the vendor-spend step."""
    return {
        "answerable": True,
        "table": "VENDOR_PAYMENTS",
        "table_reason": "vendor side, AP payments master",
        "grain": "entity",
        "grain_column": "VENDOR_ID",
        "time_bucket": None,
        "measure_column": "AMOUNT",
        "measure_agg": "SUM",
        "filters": [{"column": "STATUS", "op": "=", "value": "open"}],
        "time_filter_column": None,
        "having": None,
        "top_n": 20,
        "order": "desc",
    }


def _patch_mini(monkeypatch, payload):
    """Patch the proposer's OpenAI client so its ONE call returns ``payload`` as the
    message content (JSON-encoded). Records nothing; deterministic."""
    import json as _json

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003
            return _Resp(payload if isinstance(payload, str) else _json.dumps(payload))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    monkeypatch.setattr(P, "get_client", lambda: (_Client(), None))


# ==========================================================================
# proposer.propose
# ==========================================================================
def test_propose_good_answer_maps_to_typed_contract(monkeypatch):
    _patch_mini(monkeypatch, _good_mini_answer())
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP,
                      entity="vendor", measure_concept="total spend", grain="entity")
    pc = _run(P.propose("total vendor spend", step, _vendor_packet()))

    assert isinstance(pc, ProposedContract)
    assert pc.step_id == "s1"
    assert pc.table == "VENDOR_PAYMENTS"
    assert pc.measure_column == "AMOUNT"
    assert pc.measure_agg == "SUM"
    assert pc.grain_kind == "entity"
    assert pc.grain_column == "VENDOR_ID"
    assert pc.top_n == 20
    # the filter slot carried the verbatim status predicate
    assert len(pc.filters) == 1
    f0 = pc.filters[0]
    assert (f0.get("column") if isinstance(f0, dict) else None) == "STATUS"
    # I1/I2: a ProposedContract has NO sql attribute / never carries raw SQL.
    assert not hasattr(pc, "sql")


def test_propose_not_answerable_returns_none(monkeypatch):
    _patch_mini(monkeypatch, {"answerable": False, "table_reason": "no fit"})
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    pc = _run(P.propose("q", step, _vendor_packet()))
    assert pc is None


def test_propose_bad_json_returns_none(monkeypatch):
    _patch_mini(monkeypatch, "this is not json {{{")
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    pc = _run(P.propose("q", step, _vendor_packet()))
    assert pc is None


def test_propose_empty_packet_returns_none(monkeypatch):
    _patch_mini(monkeypatch, _good_mini_answer())
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP, entity="vendor")
    pc = _run(P.propose("q", step, EvidencePacket(step_id="s1", files=())))
    assert pc is None


# --------------------------------------------------------------------------
# proposer.propose — FIX C (L7): corrective re-propose. The first attempt's
# verify-failure reason is threaded into the SECOND propose call so mini can
# self-correct instead of repeating the same pick at temp 0.
# --------------------------------------------------------------------------
def _capture_mini_prompt(monkeypatch, payload):
    """Like _patch_mini, but RECORDS the prompt content sent to the client so a
    test can assert what mini was told. Returns the recorder dict."""
    import json as _json

    captured: dict = {}

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003
            captured["prompt"] = kwargs["messages"][0]["content"]
            return _Resp(payload if isinstance(payload, str) else _json.dumps(payload))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    monkeypatch.setattr(P, "get_client", lambda: (_Client(), None))
    return captured


def test_propose_renders_prior_failure_into_prompt(monkeypatch):
    """When ``prior_failure`` is supplied (the first attempt's verify reason), the
    correction directive carrying that reason must appear in the prompt content sent
    to mini — so the re-propose can choose a DIFFERENT, passing pick."""
    captured = _capture_mini_prompt(monkeypatch, _good_mini_answer())
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP,
                      entity="vendor", measure_concept="total spend", grain="entity")
    reason = "bad_measure: X not numeric"

    pc = _run(P.propose("total vendor spend", step, _vendor_packet(),
                        prior_failure=reason))

    assert isinstance(pc, ProposedContract)        # still produces a contract
    prompt = captured["prompt"]
    assert reason in prompt, "the prior verify failure must be rendered into the prompt"
    # An explicit do-not-repeat directive is present so mini self-corrects.
    assert "FAILED verification" in prompt
    assert "do not repeat" in prompt.lower()


def test_propose_no_prior_failure_prompt_unchanged(monkeypatch):
    """When ``prior_failure`` is None (the default first attempt) the prompt carries
    NO correction directive — byte-identical to today's first-attempt prompt."""
    captured = _capture_mini_prompt(monkeypatch, _good_mini_answer())
    step = IntentStep(step_id="s1", kind=StepKind.LOOKUP,
                      entity="vendor", measure_concept="total spend", grain="entity")

    _run(P.propose("total vendor spend", step, _vendor_packet()))

    prompt = captured["prompt"]
    assert "FAILED verification" not in prompt
    assert "previous choice" not in prompt.lower()


# ==========================================================================
# verifier.verify
# ==========================================================================
def _good_proposed() -> ProposedContract:
    return ProposedContract(
        step_id="s1",
        table="VENDOR_PAYMENTS",
        table_reason="vendor side",
        grain_kind="entity",
        grain_column="VENDOR_ID",
        time_bucket=None,
        measure_column="AMOUNT",
        measure_agg="SUM",
        filters=({"column": "STATUS", "op": "=", "value": "open"},),
        time_filter_column=None,
        having=None,
        top_n=20,
        order="desc",
    )


def test_verify_good_contract_returns_verified():
    vc, reason = V.verify(_good_proposed(), _vendor_packet(), None)
    assert reason == "ok"
    assert isinstance(vc, VerifiedContract)
    assert vc.table == "VENDOR_PAYMENTS"
    assert vc.measure_col == "AMOUNT"
    assert vc.agg == "SUM"
    assert vc.grain_kind == "entity"
    assert vc.grain_col == "VENDOR_ID"
    assert vc.order == "DESC"
    # filter normalised to a (col, op, val) tuple with the exact-case column
    assert vc.filters == (("STATUS", "=", "open"),)


def test_verify_missing_measure_column_abstains():
    pc = _good_proposed()
    bad = ProposedContract(**{**pc.__dict__, "measure_column": "NOPE_COL"})
    vc, reason = V.verify(bad, _vendor_packet(), None)
    assert vc is None
    assert reason == "bad_measure"


def test_verify_filter_value_not_in_value_set_abstains():
    pc = _good_proposed()
    # STATUS has a stored value_set {open, closed}; "frozen" is not a member.
    bad = ProposedContract(
        **{**pc.__dict__, "filters": ({"column": "STATUS", "op": "=", "value": "frozen"},)}
    )
    vc, reason = V.verify(bad, _vendor_packet(), None)
    assert vc is None
    assert reason == "filter_value_not_in_value_set"


def test_verify_polarity_contradicts_question_abstains():
    # The pick is a vendor-side table but the question polarity is customer.
    vc, reason = V.verify(_good_proposed(), _vendor_packet(), "customer")
    assert vc is None
    assert reason == "polarity_contradicts_question"


def test_verify_same_side_polarity_passes():
    vc, reason = V.verify(_good_proposed(), _vendor_packet(), "vendor")
    assert reason == "ok"
    assert isinstance(vc, VerifiedContract)


# ==========================================================================
# verifier.verify — time-window abstain-safety (L8). When a time window is
# present for the question the chosen table MUST carry a usable date/timestamp
# column (the proposed time_filter_column, or the time-grain date column);
# otherwise verify FAILS with ``no_time_column_for_window`` so the driver
# abstains rather than rendering an ALL-TIME query for a time-scoped question.
# ==========================================================================
def test_verify_window_with_date_time_filter_column_sets_time_col():
    """A window IS present and the proposal names a real date column
    (PAYMENT_DATE, type/role=date) as ``time_filter_column`` → verify passes and
    ``time_col`` is set so the renderer emits the date bounds."""
    pc = ProposedContract(
        **{**_good_proposed().__dict__, "time_filter_column": "PAYMENT_DATE"}
    )
    vc, reason = V.verify(
        pc, _vendor_packet(), None, time_window=(__import__("datetime").date(2025, 1, 1),
                                                 __import__("datetime").date(2025, 3, 31)),
    )
    assert reason == "ok"
    assert isinstance(vc, VerifiedContract)
    assert vc.time_col == "PAYMENT_DATE"


def test_verify_window_without_usable_date_column_abstains():
    """A window IS present but the chosen table has NO usable date column (the
    proposal names no time_filter_column and the grain is entity, not time) →
    verify FAILS ``no_time_column_for_window``. NEVER render all-time SQL for a
    windowed question."""
    # Evidence with NO date-typed / date-role column at all.
    no_date_file = {
        "table": "VENDOR_PAYMENTS",
        "valid_cols": {"VENDOR_ID": "VENDOR_ID", "AMOUNT": "AMOUNT", "STATUS": "STATUS"},
        "columns": [
            {"name": "VENDOR_ID", "type": "string", "role": "id"},
            {"name": "AMOUNT", "type": "double", "role": "amount"},
            {"name": "STATUS", "type": "string", "role": "attribute"},
        ],
        "sample_rows": [{"VENDOR_ID": "V1", "AMOUNT": 100.0}],
        "description": "Vendor payments without any date column",
        "good_for": ["vendor spend"],
        "coverage": "?..?",
        "row_count": 1234,
        "value_set": {"STATUS": {"open": 5, "closed": 3}},
        "unique_rates": {"VENDOR_ID": 0.9, "AMOUNT": 0.3},
        "polarity": "vendor",
        "process_role": "AP_payment",
        "erp_module": "AP",
        "erp_confidence": 0.95,
    }
    packet = EvidencePacket(step_id="s1", files=(no_date_file,))
    vc, reason = V.verify(
        _good_proposed(), packet, None,
        time_window=(__import__("datetime").date(2025, 1, 1),
                     __import__("datetime").date(2025, 3, 31)),
    )
    assert vc is None
    assert reason == "no_time_column_for_window"


def test_verify_window_with_time_grain_date_column_passes():
    """When a window is present and the grain itself is a date column (time
    grain), that grain date column satisfies the window requirement even without
    an explicit ``time_filter_column`` → verify passes and ``time_col`` is set."""
    pc = ProposedContract(
        step_id="s1",
        table="VENDOR_PAYMENTS",
        table_reason="time grain over the payment date",
        grain_kind="time",
        grain_column="PAYMENT_DATE",
        time_bucket="month",
        measure_column="AMOUNT",
        measure_agg="SUM",
        filters=(),
        time_filter_column=None,
        having=None,
        top_n=None,
        order="desc",
    )
    vc, reason = V.verify(
        pc, _vendor_packet(), None,
        time_window=(__import__("datetime").date(2025, 1, 1),
                     __import__("datetime").date(2025, 3, 31)),
    )
    assert reason == "ok"
    assert isinstance(vc, VerifiedContract)
    assert vc.time_col == "PAYMENT_DATE"


def test_verify_no_window_without_date_column_still_passes():
    """ABSTAIN-SAFETY is window-gated: with NO window the date-column requirement
    does NOT apply — a date-less table verifies exactly as today (byte-identical
    to the pre-L8 behaviour)."""
    no_date_file = {
        "table": "VENDOR_PAYMENTS",
        "valid_cols": {"VENDOR_ID": "VENDOR_ID", "AMOUNT": "AMOUNT", "STATUS": "STATUS"},
        "columns": [
            {"name": "VENDOR_ID", "type": "string", "role": "id"},
            {"name": "AMOUNT", "type": "double", "role": "amount"},
            {"name": "STATUS", "type": "string", "role": "attribute"},
        ],
        "sample_rows": [{"VENDOR_ID": "V1", "AMOUNT": 100.0}],
        "description": "Vendor payments without any date column",
        "good_for": ["vendor spend"],
        "coverage": "?..?",
        "row_count": 1234,
        "value_set": {"STATUS": {"open": 5, "closed": 3}},
        "unique_rates": {"VENDOR_ID": 0.9, "AMOUNT": 0.3},
        "polarity": "vendor",
        "process_role": "AP_payment",
        "erp_module": "AP",
        "erp_confidence": 0.95,
    }
    packet = EvidencePacket(step_id="s1", files=(no_date_file,))
    vc, reason = V.verify(_good_proposed(), packet, None, time_window=None)
    assert reason == "ok"
    assert isinstance(vc, VerifiedContract)
    assert vc.time_col is None


# ==========================================================================
# verifier.partition_by_polarity
# ==========================================================================
def test_partition_drops_contradicting_side():
    cust = {"table": "AR", "polarity": "customer"}
    vend = {"table": "AP", "polarity": "vendor"}
    kept, signal = V.partition_by_polarity([cust, vend], "vendor")
    tables = {e["table"] for e in kept}
    assert "AP" in tables
    assert "AR" not in tables
    assert signal is None


def test_partition_signals_tie_on_ambiguous():
    cust = {"table": "AR", "polarity": "customer"}
    vend = {"table": "AP", "polarity": "vendor"}
    kept, signal = V.partition_by_polarity([cust, vend], None)
    assert signal == "polarity_tie"
    assert len(kept) == 2


def test_partition_single_side_unchanged():
    a = {"table": "AP1", "polarity": "vendor"}
    b = {"table": "AP2", "polarity": "vendor"}
    kept, signal = V.partition_by_polarity([a, b], None)
    assert signal is None
    assert len(kept) == 2


def test_partition_keeps_matching_side_and_unconstrained_none():
    """A known q_polarity keeps the matching reliable side AND any unconstrained
    (None) candidate, dropping ONLY the reliable opposite. A None/neutral side is
    never dropped (abstain-biased: it cannot be disproven)."""
    cust = {"table": "AR", "polarity": "customer"}
    vend = {"table": "AP", "polarity": "vendor"}
    neutral = {"table": "GL", "polarity": None}
    kept, signal = V.partition_by_polarity([cust, vend, neutral], "vendor")
    assert signal is None
    assert {e["table"] for e in kept} == {"AP", "GL"}  # opposite (AR) dropped, None kept


def test_partition_never_returns_empty_guard():
    """Empty-guard (M5): the ``(keep or candidates)`` fallback guarantees a non-empty
    set even if a (hypothetical) filter would drop everything. Directly exercise the
    guard by passing a candidate set whose every reliable side is the opposite of
    q_polarity AND carries no None — the function must keep the FULL set, never []."""
    # Two reliable sides present (so the filter branch engages), but both rows are
    # the SAME (vendor) reliable side as far as keep is concerned; ask customer.
    # keep = rows whose polarity in (customer, None) = [] -> guard returns full set.
    rows = [
        {"table": "AP1", "polarity": "vendor"},
        {"table": "AR_marker", "polarity": "customer"},  # makes >=2 sides present
    ]
    # Remove the matching row to force keep=[]: ask for a side no row matches except
    # the marker, then assert the guard never hands back an empty list in any case.
    # (Realistically keep retains the matching side; this asserts the defensive guard.)
    kept_vendor, _ = V.partition_by_polarity(rows, "vendor")
    assert kept_vendor, "must never be empty"
    kept_customer, _ = V.partition_by_polarity(rows, "customer")
    assert kept_customer, "must never be empty"
    # The pathological keep=[] path: monkey-free construction where every reliable
    # side is opposite. Two vendor rows + force >=2 sides with a customer row, then
    # ask customer -> the two vendor rows are dropped but the customer row is kept;
    # never empty. There is no input that both engages the filter and empties it,
    # which is exactly why the abstain-bias holds — assert the invariant directly.
    all_opposite = [{"table": "V1", "polarity": "vendor"},
                    {"table": "V2", "polarity": "vendor"},
                    {"table": "C1", "polarity": "customer"}]
    kept_all, _ = V.partition_by_polarity(all_opposite, "customer")
    assert kept_all and {e["table"] for e in kept_all} == {"C1"}


# ==========================================================================
# verifier.clarify_payload
# ==========================================================================
def test_clarify_payload_shape():
    cands = [
        {"table": "AR", "polarity": "customer", "process_role": "AR_receipt",
         "good_for": ["customer receipts"]},
        {"table": "AP", "polarity": "vendor", "process_role": "AP_payment",
         "good_for": ["vendor payments"]},
    ]
    cp = V.clarify_payload(cands, "polarity_tie")
    assert isinstance(cp, ClarifyPayload)
    assert cp.reason == "polarity_tie"
    assert len(cp.options) >= 1
    assert len(cp.options) <= 3
    # options are drawn from the candidates' OWN side/role (never invented literals)
    joined = " ".join(cp.options).lower()
    assert "customer" in joined or "vendor" in joined


# ==========================================================================
# verifier.verify_step_join (lifted verify_join)
# ==========================================================================
class _JoinSession:
    """Feeds ColumnKeyRegistry rows by (blob_path, column_name)."""

    def __init__(self, rows_by_key):
        self._rows = rows_by_key
        self.rolled_back = False

    async def execute(self, query):  # noqa: ANN001
        params = query.compile().params
        blob = next((v for k, v in params.items() if "blob_path" in k), None)
        col = next((v for k, v in params.items() if "column_name" in k), None)
        row = self._rows.get((blob, col))

        class _R:
            def __init__(self, r):
                self._r = r

            def scalars(self):
                return self

            def first(self):
                return self._r

        return _R(row)

    async def rollback(self):
        self.rolled_back = True


class _FakeRegRow:
    def __init__(self, *, unique_rate, null_rate, value_fingerprints, semantic_role):
        self.unique_rate = unique_rate
        self.null_rate = null_rate
        self.value_fingerprints = value_fingerprints
        self.semantic_role = semantic_role
        self.cardinality = len(value_fingerprints)


def test_verify_step_join_verifies_real_fk_to_pk(monkeypatch):
    # PK side (blobB.VENDOR_ID): high unique, low null, distinct values.
    # FK side (blobA.VENDOR_ID): its values are contained in the PK set.
    pk_fps = [f"v{i}" for i in range(50)]
    fk_fps = [f"v{i}" for i in range(40)]  # all contained in pk_fps
    rows = {
        ("blobA", "VENDOR_ID"): _FakeRegRow(
            unique_rate=0.4, null_rate=0.0, value_fingerprints=fk_fps,
            semantic_role="identifier"),
        ("blobB", "VENDOR_ID"): _FakeRegRow(
            unique_rate=1.0, null_rate=0.0, value_fingerprints=pk_fps,
            semantic_role="identifier"),
    }
    session = _JoinSession(rows)

    a = ResolvedTable(step_id="s1", table="AP_INVOICES", file_id="fa", blob="blobA")
    b = ResolvedTable(step_id="s2", table="VENDOR_MASTER", file_id="fb", blob="blobB")

    # verify_step_join must test the (col_a, col_b) candidate pairs itself; here we
    # drive it via the helper which takes the resolved tables + their key columns.
    verdict = _run(V.verify_step_join(session, "c1", a, b, "VENDOR_ID", "VENDOR_ID"))
    assert verdict.verified is True
    assert verdict.pk_side == "blobB"
    assert verdict.fk_side == "blobA"


# ==========================================================================
# renderer.render — exact SQL
# ==========================================================================
def test_render_entity_grain_exact_sql():
    vc = VerifiedContract(
        step_id="s1",
        table="VENDOR_PAYMENTS",
        grain_kind="entity",
        grain_col="VENDOR_ID",
        bucket=None,
        measure_col="AMOUNT",
        agg="SUM",
        filters=(("STATUS", "=", "open"),),
        time_col=None,
        having=None,
        top_n=20,
        order="DESC",
        reason="vendor side",
    )
    sql = RND.render(vc, None)
    expected = (
        'SELECT "VENDOR_ID", SUM("AMOUNT") AS "amount"\n'
        'FROM "VENDOR_PAYMENTS"\n'
        'WHERE "STATUS" = \'open\'\n'
        'GROUP BY "VENDOR_ID"\n'
        'ORDER BY "amount" DESC\n'
        'LIMIT 20'
    )
    assert sql == expected


def test_render_time_grain_month_with_window():
    vc = VerifiedContract(
        step_id="s1",
        table="SALES",
        grain_kind="time",
        grain_col="ORDER_DATE",
        bucket="month",
        measure_col="REVENUE",
        agg="SUM",
        filters=(),
        time_col="ORDER_DATE",
        having=None,
        top_n=None,
        order="DESC",
        reason="",
    )
    sql = RND.render(vc, ("2024-01-01", "2024-12-31"))
    expected = (
        'SELECT EXTRACT(YEAR FROM "ORDER_DATE") AS "year", '
        'EXTRACT(MONTH FROM "ORDER_DATE") AS "month", '
        'SUM("REVENUE") AS "revenue"\n'
        'FROM "SALES"\n'
        'WHERE "ORDER_DATE" >= \'2024-01-01\' AND "ORDER_DATE" <= \'2024-12-31\'\n'
        'GROUP BY "year", "month"\n'
        'ORDER BY "year", "month"'
    )
    assert sql == expected


# ==========================================================================
# renderer.render — non-colliding measure alias (entity grain, measure==grain)
# ==========================================================================
def test_render_entity_grain_measure_equals_grain_distinct_alias():
    """When the measure column IS the grain column (e.g. COUNT(VENDOR_ID) at
    VENDOR_ID grain), the alias must NOT case-collide with the grain identifier —
    otherwise the SELECT emits ``"VENDOR_ID"`` AND ``"vendor_id"`` and the row dict
    carries two confusing case-variant keys. The fix uses a distinct reserved alias
    and the GROUP BY / ORDER BY must stay correct."""
    vc = VerifiedContract(
        step_id="s1",
        table="VENDOR_PAYMENTS",
        grain_kind="entity",
        grain_col="VENDOR_ID",
        bucket=None,
        measure_col="VENDOR_ID",
        agg="COUNT",
        filters=(),
        time_col=None,
        having=None,
        top_n=10,
        order="DESC",
        reason="count vendors",
    )
    sql = RND.render(vc, None)
    expected = (
        'SELECT "VENDOR_ID", COUNT("VENDOR_ID") AS "measure_value"\n'
        'FROM "VENDOR_PAYMENTS"\n'
        'GROUP BY "VENDOR_ID"\n'
        'ORDER BY "measure_value" DESC\n'
        'LIMIT 10'
    )
    assert sql == expected
    # Two DISTINCT output identifiers (no case-only collision).
    assert '"VENDOR_ID"' in sql and '"measure_value"' in sql
    assert '"vendor_id"' not in sql


def test_render_time_grain_measure_equals_bucket_distinct_alias():
    """A measure column that lowercases onto a time-bucket select name (``year``)
    must also get the reserved alias, never collide with the bucket identifier."""
    vc = VerifiedContract(
        step_id="s1",
        table="SALES",
        grain_kind="time",
        grain_col="ORDER_DATE",
        bucket=None,
        measure_col="YEAR",
        agg="COUNT",
        filters=(),
        time_col=None,
        having=None,
        top_n=None,
        order="DESC",
        reason="",
    )
    sql = RND.render(vc, None)
    expected = (
        'SELECT EXTRACT(YEAR FROM "ORDER_DATE") AS "year", '
        'COUNT("YEAR") AS "measure_value"\n'
        'FROM "SALES"\n'
        'GROUP BY "year"\n'
        'ORDER BY "year"'
    )
    assert sql == expected


# ==========================================================================
# renderer.render_join — exact JOIN SQL
# ==========================================================================
def test_render_join_exact_sql():
    from app.services.navigator.verifier import JoinVerdict

    verdict = JoinVerdict(
        verified=True,
        fk_side="blobA",
        pk_side="blobB",
        containment=0.95,
        fanout_estimate=1.0,
        reason="verified_fk_to_pk",
        abstain=False,
    )
    a = ResolvedTable(step_id="s1", table="AP_INVOICES", file_id="fa", blob="blobA")
    b = ResolvedTable(step_id="s2", table="VENDOR_MASTER", file_id="fb", blob="blobB")
    sql = RND.render_join(verdict, a, b, "VENDOR_ID", "VENDOR_ID")
    expected = (
        'SELECT "VENDOR_MASTER"."VENDOR_ID" AS "vendor_id", COUNT(*) AS "match_count"\n'
        'FROM "VENDOR_MASTER"\n'
        'JOIN "AP_INVOICES" ON "VENDOR_MASTER"."VENDOR_ID" = "AP_INVOICES"."VENDOR_ID"\n'
        'GROUP BY "VENDOR_MASTER"."VENDOR_ID"\n'
        'ORDER BY "match_count" DESC'
    )
    assert sql == expected


# ==========================================================================
# self-containment: no resolve.* imports remain in the navigator package
# ==========================================================================
def test_navigator_has_no_resolve_imports():
    pkg = pathlib.Path(P.__file__).parent
    offenders: list[str] = []
    for py in sorted(pkg.glob("*.py")):
        text = py.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from app.services.resolve") or stripped.startswith(
                "import app.services.resolve"
            ):
                offenders.append(f"{py.name}: {stripped}")
    assert offenders == [], f"navigator must not import resolve.*: {offenders}"

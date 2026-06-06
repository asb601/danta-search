"""Pure tests for the tiered model router (contract C7).

No live infra: the per-tenant budget counter is exercised through an injected
fake ``BudgetStore`` (a dict-backed double), and the LLM is never called — the
router returns a ``ModelChoice`` only. Covers: bulk→mini, escalation gate fires
above/below threshold, per-tenant budget exhaustion, embeddings fixed, and the
structural invariant that an ingestion task can NEVER select Opus.
"""
from __future__ import annotations

from app.core.config import get_settings

from pdf_chat.model_router import (
    BudgetStore,
    ModelChoice,
    RedisBudgetStore,
    TaskClass,
    embedding_model,
    escalation_allowed,
    select_model,
)
from pdf_chat.tunables import (
    TUN_MODEL_QUERYTIME_STRONG,
    TUN_MODEL_STRONG_ID,
)


# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeBudgetStore:
    """Dict-backed per-tenant escalation counter (a pure BudgetStore double)."""

    def __init__(self, *, used: int = 0, total: int = 100) -> None:
        self._used: dict[str, int] = {}
        self._total = total
        self._seed = used

    def used(self, container_id: str) -> int:
        return self._used.get(container_id, self._seed)

    def total(self, container_id: str) -> int:
        return self._total

    def reserve(self, container_id: str) -> None:
        self._used[container_id] = self.used(container_id) + 1


def _open_budget() -> FakeBudgetStore:
    """A store with plenty of headroom (used=0, large window)."""
    return FakeBudgetStore(used=0, total=1000)


def _fire_signal() -> dict:
    """A signal guaranteed to fire (confidence well below the floor)."""
    return {"extract_confidence": 0.20}


# ── Bulk selection ───────────────────────────────────────────────────────────


def test_bulk_task_selects_gpt4o_mini():
    choice = select_model(task="extraction", container_id="t1", signals={})
    assert choice.model_id == get_settings().chat_deployment()
    assert choice.is_strong is False
    assert choice.provider == "azure"


def test_bulk_when_no_budget_store_wired_fails_safe():
    # Firing signal but no store → fail safe to bulk (uncapped spend impossible).
    choice = select_model(task="extraction", container_id="t1", signals=_fire_signal())
    assert choice.is_strong is False
    assert choice.model_id == get_settings().chat_deployment()


def test_model_choice_is_frozen():
    choice = ModelChoice("azure", "x", False)
    try:
        choice.model_id = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ModelChoice must be frozen/immutable")


# ── Escalation gate ──────────────────────────────────────────────────────────


def test_escalation_fires_above_threshold():
    store = _open_budget()
    assert escalation_allowed("t1", {"extract_confidence": 0.20}, store=store) is True
    choice = select_model(
        task="extraction", container_id="t1", signals={"extract_confidence": 0.20}, store=_open_budget()
    )
    assert choice.is_strong is True


def test_no_escalation_below_threshold():
    store = _open_budget()
    assert (
        escalation_allowed("t1", {"extract_confidence": 0.95, "figure_ratio": 0.0}, store=store)
        is False
    )
    choice = select_model(
        task="extraction",
        container_id="t1",
        signals={"extract_confidence": 0.95, "figure_ratio": 0.0},
        store=_open_budget(),
    )
    assert choice.is_strong is False
    assert choice.model_id == get_settings().chat_deployment()


def test_figure_ratio_signal_fires():
    store = _open_budget()
    assert escalation_allowed("t1", {"figure_ratio": 0.90}, store=store) is True


def test_intent_signals_fire_cross_domain_and_definitional():
    assert escalation_allowed("t1", {"cross_domain": True}, store=_open_budget()) is True
    assert escalation_allowed("t1", {"definitional": True}, store=_open_budget()) is True


def test_empty_signals_never_escalate():
    assert escalation_allowed("t1", {}, store=_open_budget()) is False


# ── Per-tenant budget cap ────────────────────────────────────────────────────


def test_budget_exhaustion_blocks_escalation():
    # Cap = int(0.05 * 100) = 5; seed used at 5 → exhausted.
    store = FakeBudgetStore(used=5, total=100)
    assert escalation_allowed("t1", _fire_signal(), store=store) is False
    choice = select_model(
        task="extraction", container_id="t1", signals=_fire_signal(), store=store
    )
    assert choice.is_strong is False


def test_budget_exhausted_logs_decision():
    from pdf_chat import model_router

    captured = {}

    def fake_log(name, *, score, threshold, outcome, **ctx):
        captured[name] = {"score": score, "threshold": threshold, "outcome": outcome, **ctx}
        return {"gate": name, "outcome": outcome}

    store = FakeBudgetStore(used=5, total=100)
    orig = model_router.log_gate_decision
    model_router.log_gate_decision = fake_log  # type: ignore[assignment]
    try:
        escalation_allowed("t1", _fire_signal(), store=store)
    finally:
        model_router.log_gate_decision = orig  # type: ignore[assignment]
    assert "model_router.budget_exhausted" in captured
    assert captured["model_router.budget_exhausted"]["outcome"] == "deny"
    assert captured["model_router.budget_exhausted"]["container_id"] == "t1"


def test_budget_reserve_increments_per_tenant():
    store = FakeBudgetStore(used=0, total=1000)
    escalation_allowed("tenant-a", _fire_signal(), store=store)
    # Tenant isolation: reserving for tenant-a does not touch tenant-b.
    assert store.used("tenant-a") == 1
    assert store.used("tenant-b") == 0


def test_budget_cap_at_least_one_with_tiny_window():
    # Even a tiny window yields cap >= 1 so escalation is reachable once.
    store = FakeBudgetStore(used=0, total=1)
    assert escalation_allowed("t1", _fire_signal(), store=store) is True


# ── Opus-never-at-ingestion structural invariant ─────────────────────────────


def test_ingestion_never_selects_opus(monkeypatch):
    opus = "claude-opus-4-8"
    sonnet = "claude-sonnet-4-6"
    monkeypatch.setenv("PDF_TUNABLE_MODEL.QUERYTIME_STRONG_ID", opus)
    monkeypatch.setenv("PDF_TUNABLE_MODEL.STRONG_ID", sonnet)

    for ingestion_task in ("extraction", "synthesis"):
        choice = select_model(
            task=ingestion_task,
            container_id="t1",
            signals=_fire_signal(),
            store=_open_budget(),
        )
        assert choice.is_strong is True
        assert choice.model_id != opus
        assert choice.model_id == sonnet

    # Query-time MAY reach Opus — proves the allowlist branch is the only seam.
    qt = select_model(
        task="query_synthesis",
        container_id="t1",
        signals=_fire_signal(),
        store=_open_budget(),
    )
    assert qt.model_id == opus
    assert qt.provider == "anthropic"


def test_opus_structurally_unreachable_from_ingestion_allowlist():
    """Static guarantee: no ingestion TaskClass can route to the querytime id.

    This asserts the allowlist itself (not a single call), so the invariant
    holds even if defaults / env change the concrete Opus id.
    """
    from pdf_chat import model_router

    for task in (TaskClass.EXTRACTION, TaskClass.SYNTHESIS):
        assert task in model_router._INGESTION_TASKS
    for task in (TaskClass.QUERY_SYNTHESIS, TaskClass.QUERY_PLANNING):
        assert task not in model_router._INGESTION_TASKS

    # And: with an arbitrary distinct querytime id, ingestion never returns it.
    qt_id = "claude-opus-some-future"

    class _Store(FakeBudgetStore):
        pass

    def _tunable(container_id, key, default):
        if key == TUN_MODEL_QUERYTIME_STRONG:
            return qt_id
        if key == TUN_MODEL_STRONG_ID:
            return "claude-sonnet-4-6"
        return default

    orig = model_router.get_tunable
    model_router.get_tunable = _tunable  # type: ignore[assignment]
    try:
        for ingestion_task in ("extraction", "synthesis"):
            choice = select_model(
                task=ingestion_task,
                container_id="t1",
                signals=_fire_signal(),
                store=_Store(used=0, total=1000),
            )
            assert choice.model_id != qt_id
    finally:
        model_router.get_tunable = orig  # type: ignore[assignment]


# ── Embeddings fixed ─────────────────────────────────────────────────────────


def test_embedding_model_is_fixed():
    assert embedding_model("t1") == "text-embedding-3-small"


def test_embedding_model_is_tunable(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_MODEL.EMBEDDING_ID", "text-embedding-3-large")
    assert embedding_model("t1") == "text-embedding-3-large"


# ── RedisBudgetStore (mocked client; no live Redis) ──────────────────────────


class _FakeRedis:
    def __init__(self, data=None):
        self._data = dict(data or {})

    def get(self, key):
        return self._data.get(key)

    def incr(self, key):
        self._data[key] = int(self._data.get(key, 0)) + 1
        return self._data[key]


def test_redis_budget_store_reads_and_reserves():
    client = _FakeRedis(
        {
            "pdf:escbudget:t1": "2",
            "pdf:escbudget:total:t1": "100",
        }
    )
    store: BudgetStore = RedisBudgetStore(client)
    assert store.used("t1") == 2
    assert store.total("t1") == 100
    store.reserve("t1")
    assert store.used("t1") == 3


def test_redis_budget_store_defaults_to_zero_when_unset():
    store = RedisBudgetStore(_FakeRedis())
    assert store.used("brand-new") == 0
    assert store.total("brand-new") == 0


def test_redis_budget_store_keys_are_tenant_scoped():
    client = _FakeRedis()
    store = RedisBudgetStore(client)
    store.reserve("tenant-a")
    assert store.used("tenant-a") == 1
    assert store.used("tenant-b") == 0  # isolation: distinct keys per container


def test_redis_budget_store_reserve_increments_used_key():
    """reserve() INCRs the per-tenant used key; reads target the right keys."""
    client = _FakeRedis()
    store = RedisBudgetStore(client)
    store.reserve("t1")
    store.reserve("t1")
    # The numerator lives under the tenant-scoped used key.
    assert client._data["pdf:escbudget:t1"] == 2
    assert store.used("t1") == 2
    # total() reads the distinct total key (unset → 0).
    assert "pdf:escbudget:total:t1" not in client._data
    assert store.total("t1") == 0


# ── set_default_budget_store install (process-global; teardown-safe) ──────────


def test_set_default_budget_store_enables_reserve():
    """Installing the default store makes escalation reserve WITHOUT an explicit
    store= kwarg; teardown resets the global so the other tests stay green."""
    from pdf_chat import model_router

    store = FakeBudgetStore(used=0, total=1000)
    prev = model_router._DEFAULT_STORE
    model_router.set_default_budget_store(store)
    try:
        choice = select_model(
            task="query_synthesis",
            container_id="t1",
            signals={"definitional": True},  # a firing signal
        )
        assert choice.is_strong is True               # escalated via the default store
        assert store.used("t1") == 1                  # reserve went through the default
    finally:
        model_router.set_default_budget_store(prev)


def test_default_store_none_fails_safe():
    """With the default store explicitly None, a firing signal cannot escalate."""
    from pdf_chat import model_router

    prev = model_router._DEFAULT_STORE
    model_router.set_default_budget_store(None)
    try:
        assert escalation_allowed("t1", {"definitional": True}) is False
    finally:
        model_router.set_default_budget_store(prev)

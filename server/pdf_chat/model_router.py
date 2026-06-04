"""Tiered model router — the single model-selection seam (contract C7).

Bulk work resolves to ``gpt-4o-mini`` (via the main app's
``get_settings().chat_deployment()``, which honors ``DISABLE_GPT4O``). A
strong tier (default ``claude-sonnet-4-6``) is reached ONLY when
``escalation_allowed`` returns True — which requires (a) at least one
data-driven signal above its tunable threshold AND (b) the per-tenant
escalation budget (a tunable cap + a counter store) is not exhausted.

A task→tier allowlist makes Opus structurally unreachable from any ingestion
task: ``EXTRACTION``/``SYNTHESIS`` may only ever reach the standard strong id,
never the query-time strong id (Opus). This invariant is asserted by test.

GOVERNING CRITERIA (cost-at-scale, multi-tenant, per-client tunable):
- every model id / threshold / budget cap resolves via ``get_tunable`` —
  no bare comparison literal lives in this module;
- every gate / escalation / budget decision logs via ``log_gate_decision``;
- everything is scoped by ``container_id`` (the tenant boundary);
- the LLM is never invoked here — the router returns a ``ModelChoice`` only,
  so the whole module is pure-testable with zero infra.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.core.config import get_settings  # type: ignore  # late-safe; pure import

from .tunables import (
    get_tunable,
    log_gate_decision,
    TUN_MODEL_BULK_ID,
    TUN_MODEL_STRONG_ID,
    TUN_MODEL_QUERYTIME_STRONG,
    TUN_MODEL_EMBEDDING_ID,
    TUN_ESC_CONF_FLOOR,
    TUN_ESC_FIGURE_RATIO,
    TUN_ESC_BUDGET_FRACTION,
    TUN_ESC_BUDGET_WINDOW_PAGES,
    TUNABLE_DEFAULTS,
)


class TaskClass(str, Enum):
    """Stable INTENT-layer task kinds (never customer-domain meaning)."""

    EXTRACTION = "extraction"            # ingestion bulk
    SYNTHESIS = "synthesis"             # ingestion bulk (community reports, glossary)
    QUERY_SYNTHESIS = "query_synthesis"  # query-time
    QUERY_PLANNING = "query_planning"   # query-time


# Ingestion allowlist: tasks that may ONLY ever reach the standard strong tier.
# Opus (the query-time strong id) is structurally unreachable from these.
_INGESTION_TASKS = frozenset({TaskClass.EXTRACTION, TaskClass.SYNTHESIS})


@dataclass(frozen=True)
class ModelChoice:
    """An immutable model selection. ``is_strong`` flags the escalated tier."""

    provider: str  # "azure" | "anthropic"
    model_id: str
    is_strong: bool


def _provider_of(model_id: str) -> str:
    """Infer the provider from the model id (Claude → anthropic, else azure)."""
    return "anthropic" if model_id.startswith("claude") else "azure"


# ── Per-tenant escalation budget ────────────────────────────────────────────


class BudgetStore(Protocol):
    """Per-tenant escalation counter. Pure tests inject a fake; prod uses Redis."""

    def used(self, container_id: str) -> int: ...
    def total(self, container_id: str) -> int: ...    # pages seen (window denominator)
    def reserve(self, container_id: str) -> None: ...  # increment used


# Production store, set by the worker bootstrap. Tests inject a fake via ``store=``.
_DEFAULT_STORE: "BudgetStore | None" = None


def set_default_budget_store(store: "BudgetStore | None") -> None:
    """Install the process-wide budget store (called by the worker bootstrap)."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


class RedisBudgetStore:
    """Redis-backed per-tenant escalation counter (production ``_DEFAULT_STORE``).

    Reuses the existing Redis instance (spec §6 Redis-reuse rule); keys are
    namespaced per tenant so one container can never read another's budget.
    The client is injected so this stays import-safe with zero infra; a real
    ``redis.Redis`` (or async wrapper) is supplied by the worker bootstrap.
    """

    _USED_KEY = "pdf:escbudget:{container_id}"
    _TOTAL_KEY = "pdf:escbudget:total:{container_id}"

    def __init__(self, client) -> None:
        self._client = client

    def used(self, container_id: str) -> int:
        raw = self._client.get(self._USED_KEY.format(container_id=container_id))
        return int(raw) if raw is not None else 0

    def total(self, container_id: str) -> int:
        raw = self._client.get(self._TOTAL_KEY.format(container_id=container_id))
        return int(raw) if raw is not None else 0

    def reserve(self, container_id: str) -> None:
        self._client.incr(self._USED_KEY.format(container_id=container_id))


def _signals_fire(container_id: str, signals: dict) -> bool:
    """True if any data-driven escalation signal crosses its tunable threshold.

    Signals are passed through by the extractor (confidence/figure density) and
    the query planner (cross_domain/definitional booleans). Each crossing is
    logged so a score is never compared-and-discarded silently.
    """
    conf = signals.get("extract_confidence")
    if conf is not None:
        floor = get_tunable(container_id, TUN_ESC_CONF_FLOOR, TUNABLE_DEFAULTS[TUN_ESC_CONF_FLOOR])
        if conf < floor:
            log_gate_decision(
                "router.signal.low_confidence",
                score=conf,
                threshold=floor,
                outcome="fire",
                container_id=container_id,
            )
            return True
    fig = signals.get("figure_ratio")
    if fig is not None:
        ratio = get_tunable(container_id, TUN_ESC_FIGURE_RATIO, TUNABLE_DEFAULTS[TUN_ESC_FIGURE_RATIO])
        if fig > ratio:
            log_gate_decision(
                "router.signal.figure_heavy",
                score=fig,
                threshold=ratio,
                outcome="fire",
                container_id=container_id,
            )
            return True
    # cross_domain / definitional are booleans the planner/extractor pass through.
    if signals.get("cross_domain") or signals.get("definitional"):
        log_gate_decision(
            "router.signal.intent",
            score=1.0,
            threshold=0.0,
            outcome="fire",
            container_id=container_id,
            kind="cross_domain" if signals.get("cross_domain") else "definitional",
        )
        return True
    return False


def escalation_allowed(
    container_id: str,
    signals: dict,
    *,
    store: "BudgetStore | None" = None,
) -> bool:
    """True only when a data-driven signal fires AND budget is not exhausted.

    Budget is a per-tenant cap = ``budget_fraction * window`` pages. When no
    budget store is wired we fail safe (deny escalation) so a misconfigured
    deployment can never spend uncapped on the strong tier.
    """
    if not _signals_fire(container_id, signals):
        return False
    store = store or _DEFAULT_STORE
    if store is None:  # no budget tracker wired → fail safe (deny)
        log_gate_decision(
            "router.budget.unavailable",
            score=0.0,
            threshold=0.0,
            outcome="deny",
            container_id=container_id,
        )
        return False
    used, total = store.used(container_id), store.total(container_id)
    fraction = get_tunable(
        container_id, TUN_ESC_BUDGET_FRACTION, TUNABLE_DEFAULTS[TUN_ESC_BUDGET_FRACTION]
    )
    window = get_tunable(container_id, TUN_ESC_BUDGET_WINDOW_PAGES, max(total, 1))
    cap = max(1, int(fraction * window))
    if used >= cap:
        log_gate_decision(
            "model_router.budget_exhausted",
            score=used,
            threshold=cap,
            outcome="deny",
            container_id=container_id,
        )
        return False
    store.reserve(container_id)
    log_gate_decision(
        "model_router.budget_reserve",
        score=used + 1,
        threshold=cap,
        outcome="allow",
        container_id=container_id,
    )
    return True


def select_model(
    *,
    task,
    container_id: str,
    signals: dict,
    store: "BudgetStore | None" = None,
) -> ModelChoice:
    """Return the BULK model unless the data-driven escalation gate fires.

    Opus (the query-time strong id) can NEVER be returned for an ingestion
    task: the per-task allowlist below makes it structurally unreachable from
    ``EXTRACTION``/``SYNTHESIS``.
    """
    task = TaskClass(task)
    bulk_id = get_tunable(container_id, TUN_MODEL_BULK_ID, get_settings().chat_deployment())
    bulk = ModelChoice(_provider_of(bulk_id), bulk_id, is_strong=False)

    if not escalation_allowed(container_id, signals, store=store):
        return bulk

    # Ingestion may only ever reach the standard strong tier (Sonnet/GPT-4-class).
    # Query-time may reach the query-only strong id (Opus, off by default).
    if task in _INGESTION_TASKS:
        strong_id = get_tunable(
            container_id, TUN_MODEL_STRONG_ID, TUNABLE_DEFAULTS[TUN_MODEL_STRONG_ID]
        )
    else:
        strong_id = get_tunable(
            container_id,
            TUN_MODEL_QUERYTIME_STRONG,
            get_tunable(
                container_id, TUN_MODEL_STRONG_ID, TUNABLE_DEFAULTS[TUN_MODEL_STRONG_ID]
            ),
        )
    log_gate_decision(
        "model_router.escalate",
        score=1.0,
        threshold=0.0,
        outcome="strong",
        task=task.value,
        model_id=strong_id,
        container_id=container_id,
    )
    return ModelChoice(_provider_of(strong_id), strong_id, is_strong=True)


def embedding_model(container_id: str) -> str:
    """Embeddings are fixed to text-embedding-3-small (configurable id only)."""
    return get_tunable(
        container_id, TUN_MODEL_EMBEDDING_ID, TUNABLE_DEFAULTS[TUN_MODEL_EMBEDDING_ID]
    )

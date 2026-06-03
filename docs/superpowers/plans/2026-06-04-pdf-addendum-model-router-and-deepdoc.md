# PDF Agentic Graph RAG — ADDENDUM: Tiered Model Router & DeepDoc ONNX Micro-Components

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this addendum task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is an ADDENDUM — it presumes `2026-06-03-pdf-phase0-1-foundations.md` is implemented (esp. `pdf_chat/tunables.py` with `get_tunable` + `log_gate_decision`, contract C1).

**Goal:** Deliver the two cross-cutting additions deferred in the foundations plan:
- **(A) Tiered model router** (`pdf_chat/model_router.py`) — implements INDEX contract **C7**. Bulk = `gpt-4o-mini`; a data-driven, **budget-capped** escalation gate selects a strong tier (default `claude-sonnet-4-6`); **Opus can never be selected by an ingestion task**. Extends **Phase 0**.
- **(B) DeepDoc ONNX micro-components** (`pdf_chat/ingestion/deepdoc/`) — vendored, **optional** enhancers (multi-column/reading-order, table-structure for spanning cells, table-rotation) gated on **hard pages only**, degrading gracefully to PyMuPDF/OCR when ONNX/cv2/xgboost are absent. Extends **Phase 1**.

**Architecture:** The router is the single seam every later phase calls to pick a model: extraction (Phase 2), synthesis/glossary/community reports (Phases 3/5), and query-time agent steps (Phase 3) all call `select_model(task=..., container_id=..., signals=...)` instead of reading a deployment name directly. Bulk work resolves to `gpt-4o-mini` via the main app's `get_settings().chat_deployment()` (`server/app/core/config.py:130`, honors `DISABLE_GPT4O`). Strong-tier escalation only fires when `escalation_allowed(...)` returns True — which requires (a) at least one data-driven signal above its tunable threshold AND (b) the per-tenant escalation budget (a tunable cap + a Redis counter) is not exhausted. A **task→tier allowlist** makes Opus structurally unreachable from any `ingestion.*` task (asserted by test). The DeepDoc layer mirrors the existing `parser_router.py` pattern (`server/pdf_chat/ingestion/parser_router.py:33`): a pure gate decides per page whether to invoke an ONNX enhancer; if invoked, the enhancer re-emits `UnifiedElement[]` conforming to `ingestion/ton_schema.py:32` (text/table/figure/formula + `bbox` + `reading_order` + `confidence`). All ONNX/cv2/xgboost imports are guarded so the system degrades to the Phase-1 PyMuPDF/OCR fast path when models or deps are absent. We **port concepts** from RAGFlow (Apache-2.0, spec §6) — `_assign_column` KMeans column clustering (`/Users/bharath/Desktop/projects/ragflow-main/deepdoc/parser/pdf_parser.py:804`), spanning-cell TSR (`deepdoc/vision/table_structure_recognizer.py:30`), table-rotation (`deepdoc/parser/pdf_parser.py:318`) — we do **not** vendor the DeepDoc monolith or its type-specific chunkers (spec §6, invariant 4).

**Tech Stack:** Python 3.12 · `uv` · pytest (run via `uv run --with pytest --with pytest-asyncio pytest`) · `onnxruntime` (guarded, optional) · `opencv-python-headless` (guarded) · `xgboost` (guarded) · `scikit-learn` KMeans (guarded; concept ported from RAGFlow) · Azure OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · Redis (escalation budget counter) · structlog (via `log_gate_decision`).

---

## Test & Convention Notes (read before Task 1)

- **Run tests** from `server/`: `uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q`.
- **Test layout:** router tests → new `pdf_chat/testing/test_model_router.py`; DeepDoc tests → new `pdf_chat/testing/test_deepdoc.py`. Both are pure (no infra) by default.
- **Async tests** use `asyncio.run(...)`, NOT `@pytest.mark.asyncio` (match `test_ingestion.py:353-367`). The budget counter is tested through an injected fake (a dict-backed `BudgetStore`), so the router's pure logic needs no live Redis.
- **Infra-gated tests** (real `onnxruntime` session, real Redis budget) use `@pytest.mark.infra` (registered in `pdf_chat/testing/conftest.py`, Phase-0 Task 1.5) and are excluded by the default `-m "not infra"` run.
- **No magic literals:** every model id, gate threshold, budget cap, and DeepDoc complexity threshold resolves via `get_tunable(container_id, key, default)` from `pdf_chat/tunables.py`. Every gate/skip/escalation/budget-exhaustion decision logs via `log_gate_decision(name, score=..., threshold=..., outcome=..., **ctx)`. **No bare score-comparison literal in any `.py` under `pdf_chat/`.** Types/tiers are never hardcoded dictionaries of customer-domain meaning.
- **Deterministic seams:** the LLM is never called in these tasks; the router returns a `ModelChoice`, it does not invoke a model. The ONNX session is injected (`session_factory`) so tests pass a mock returning fixed tensors. Both seams are pure-testable.
- **LLM policy:** bulk = `gpt-4o-mini` only. Strong tier is the only place a non-mini id appears, and it is a tunable, gated, capped. **Opus is query-time-only and off by default.**
- **Commits:** conventional commits, each ending with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Create / Modify | Responsibility | Phase |
|---|---|---|---|
| `server/pdf_chat/model_router.py` | Create | `ModelChoice`, `select_model`, `escalation_allowed`, `BudgetStore` protocol + `RedisBudgetStore`. The single model-selection seam. | **0** |
| `server/pdf_chat/testing/test_model_router.py` | Create | Pure tests: bulk→mini, escalation gate, budget exhaustion, Opus-never-at-ingestion, embeddings fixed. | **0** |
| `server/pdf_chat/tunables.py` | Modify | Add router + DeepDoc tunable-key constants and defaults (no literals elsewhere). | **0/1** |
| `server/pdf_chat/ingestion/deepdoc/__init__.py` | Create | Public exports: `enhance_page`, `deepdoc_available`, `DeepDocUnavailable`. | **1** |
| `server/pdf_chat/ingestion/deepdoc/_deps.py` | Create | Guarded imports (onnxruntime, cv2, xgboost, sklearn) → capability flags. | **1** |
| `server/pdf_chat/ingestion/deepdoc/column_order.py` | Create | `assign_columns()` — KMeans reading-order detection (concept ported from RAGFlow `_assign_column`). | **1** |
| `server/pdf_chat/ingestion/deepdoc/table_structure.py` | Create | `recognize_table_structure()` — ONNX TSR + spanning-cell span calc (concept ported from RAGFlow TSR). | **1** |
| `server/pdf_chat/ingestion/deepdoc/rotation.py` | Create | `best_rotation()` — table-rotation evaluation (concept ported from `_evaluate_table_orientation`). | **1** |
| `server/pdf_chat/ingestion/deepdoc/enhancer.py` | Create | `should_enhance()` gate + `enhance_page()` orchestrator → `UnifiedElement[]`. | **1** |
| `server/pdf_chat/ingestion/deepdoc/models/.gitkeep` | Create | Pre-downloaded ONNX model dir (populated at container build, NOT committed). | **1** |
| `server/pdf_chat/testing/test_deepdoc.py` | Create | Pure tests: skip on simple page, use on flagged page, graceful degradation, schema conformance, KMeans column ordering. | **1** |
| `server/Dockerfile` (or build script) | Modify | Add ONNX model pre-download step (build-time, not first-run). | **1** |

---

# PART A — TIERED MODEL ROUTER (extends Phase 0)

## Task A1 — Router tunable keys (RED → GREEN)

- [ ] **A1.1** Add the router tunable-key constants + defaults to `pdf_chat/tunables.py`. No id/threshold/cap is a literal anywhere else.

```python
# pdf_chat/tunables.py  (append to existing constants block)

# ── Model router (contract C7) ──────────────────────────────────────────────
TUN_MODEL_BULK_ID            = "model.bulk_id"               # default gpt-4o-mini
TUN_MODEL_STRONG_ID          = "model.strong_id"             # default claude-sonnet-4-6
TUN_MODEL_QUERYTIME_STRONG   = "model.querytime_strong_id"   # default claude-opus-4-* (query-only)
TUN_MODEL_EMBEDDING_ID       = "model.embedding_id"          # default text-embedding-3-small
TUN_ESC_CONF_FLOOR           = "router.escalation.conf_floor"        # extract_confidence below → signal
TUN_ESC_FIGURE_RATIO         = "router.escalation.figure_ratio"      # figure/formula density above → signal
TUN_ESC_BUDGET_FRACTION      = "router.escalation.budget_fraction"   # per-tenant cap (frac of pages)
TUN_ESC_BUDGET_WINDOW_PAGES  = "router.escalation.budget_window"     # denominator for the fraction

# Defaults live in the get_tunable default-table only (single source); examples:
#   model.bulk_id            -> get_settings().chat_deployment()  (resolved, not literal)
#   model.strong_id          -> "claude-sonnet-4-6"
#   model.querytime_strong_id-> "claude-opus-4-8"
#   router.escalation.conf_floor   -> 0.55
#   router.escalation.figure_ratio -> 0.40
#   router.escalation.budget_fraction -> 0.05   (≤3–5% per spec §8)
```

- [ ] **A1.2** Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q` — must stay green.
- [ ] **A1.3** Commit: `feat(pdf): add model-router tunable keys (C7)`.

## Task A2 — `ModelChoice` + bulk selection (RED first)

- [ ] **A2.1** RED — write `test_model_router.py::test_bulk_task_selects_gpt4o_mini`: assert `select_model(task="extraction", container_id="t1", signals={}).model_id == get_settings().chat_deployment()` and `.is_strong is False`.
- [ ] **A2.2** GREEN — implement the dataclass + bulk path:

```python
# pdf_chat/model_router.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from app.core.config import get_settings  # type: ignore  # late-safe; pure import
from .tunables import (
    get_tunable, log_gate_decision,
    TUN_MODEL_BULK_ID, TUN_MODEL_STRONG_ID, TUN_MODEL_QUERYTIME_STRONG,
    TUN_MODEL_EMBEDDING_ID, TUN_ESC_CONF_FLOOR, TUN_ESC_FIGURE_RATIO,
    TUN_ESC_BUDGET_FRACTION, TUN_ESC_BUDGET_WINDOW_PAGES,
)


class TaskClass(str, Enum):
    """Stable INTENT-layer task kinds (never customer-domain meaning)."""
    EXTRACTION        = "extraction"          # ingestion bulk
    SYNTHESIS         = "synthesis"           # ingestion bulk (community reports, glossary)
    QUERY_SYNTHESIS   = "query_synthesis"     # query-time
    QUERY_PLANNING    = "query_planning"      # query-time


_INGESTION_TASKS = frozenset({TaskClass.EXTRACTION, TaskClass.SYNTHESIS})


@dataclass(frozen=True)
class ModelChoice:
    provider: str          # "azure" | "anthropic"
    model_id: str
    is_strong: bool


def _provider_of(model_id: str) -> str:
    return "anthropic" if model_id.startswith("claude") else "azure"


def select_model(*, task, container_id: str, signals: dict) -> ModelChoice:
    """Return the BULK model unless the data-driven escalation gate fires.

    Opus (query-time strong id) can NEVER be returned for an ingestion task:
    the per-task allowlist below makes it structurally unreachable.
    """
    task = TaskClass(task)
    bulk_id = get_tunable(container_id, TUN_MODEL_BULK_ID, get_settings().chat_deployment())
    bulk = ModelChoice(_provider_of(bulk_id), bulk_id, is_strong=False)

    if not escalation_allowed(container_id, signals):
        return bulk

    # Ingestion may only ever reach the standard strong tier (Sonnet/GPT-4-class).
    # Query-time may reach the query-only strong id (Opus, off by default).
    if task in _INGESTION_TASKS:
        strong_id = get_tunable(container_id, TUN_MODEL_STRONG_ID, "claude-sonnet-4-6")
    else:
        strong_id = get_tunable(
            container_id, TUN_MODEL_QUERYTIME_STRONG,
            get_tunable(container_id, TUN_MODEL_STRONG_ID, "claude-sonnet-4-6"),
        )
    log_gate_decision("model_router.escalate", score=1.0, threshold=0.0,
                      outcome="strong", task=task.value, model_id=strong_id,
                      container_id=container_id)
    return ModelChoice(_provider_of(strong_id), strong_id, is_strong=True)


def embedding_model(container_id: str) -> str:
    """Embeddings are fixed to text-embedding-3-small (configurable id only)."""
    return get_tunable(container_id, TUN_MODEL_EMBEDDING_ID, "text-embedding-3-small")
```

- [ ] **A2.3** Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_model_router.py -q`.
- [ ] **A2.4** Commit: `feat(pdf): ModelChoice + bulk model selection (C7)`.

## Task A3 — Escalation gate + per-tenant budget (RED first)

- [ ] **A3.1** RED — write three tests:
  - `test_escalation_fires_above_threshold`: `signals={"extract_confidence": 0.20}` (below `conf_floor`) → `escalation_allowed` True and `select_model(...).is_strong is True`.
  - `test_no_escalation_below_threshold`: `signals={"extract_confidence": 0.95, "figure_ratio": 0.0}` → False, choice is bulk.
  - `test_budget_exhaustion_blocks_escalation`: inject a `BudgetStore` already at the cap → `escalation_allowed` returns False even with a firing signal, and a `model_router.budget_exhausted` decision is logged.

- [ ] **A3.2** GREEN — implement the gate, the budget protocol, and the default Redis-backed store:

```python
# pdf_chat/model_router.py  (continued)
from typing import Protocol


class BudgetStore(Protocol):
    """Per-tenant escalation counter. Pure tests inject a fake; prod uses Redis."""
    def used(self, container_id: str) -> int: ...
    def total(self, container_id: str) -> int: ...     # pages seen (window denominator)
    def reserve(self, container_id: str) -> None: ...  # increment used


_DEFAULT_STORE: "BudgetStore | None" = None  # set by worker bootstrap; tests inject


def _signals_fire(container_id: str, signals: dict) -> bool:
    conf = signals.get("extract_confidence")
    if conf is not None:
        floor = get_tunable(container_id, TUN_ESC_CONF_FLOOR, 0.55)
        if conf < floor:
            log_gate_decision("router.signal.low_confidence", score=conf,
                              threshold=floor, outcome="fire", container_id=container_id)
            return True
    fig = signals.get("figure_ratio")
    if fig is not None:
        ratio = get_tunable(container_id, TUN_ESC_FIGURE_RATIO, 0.40)
        if fig > ratio:
            log_gate_decision("router.signal.figure_heavy", score=fig,
                              threshold=ratio, outcome="fire", container_id=container_id)
            return True
    # cross_domain / definitional are booleans the planner/extractor pass through
    if signals.get("cross_domain") or signals.get("definitional"):
        log_gate_decision("router.signal.intent", score=1.0, threshold=0.0,
                          outcome="fire", container_id=container_id,
                          kind="cross_domain" if signals.get("cross_domain") else "definitional")
        return True
    return False


def escalation_allowed(container_id: str, signals: dict,
                       *, store: "BudgetStore | None" = None) -> bool:
    """True only when a data-driven signal fires AND budget is not exhausted."""
    if not _signals_fire(container_id, signals):
        return False
    store = store or _DEFAULT_STORE
    if store is None:                       # no budget tracker wired → fail safe (deny)
        log_gate_decision("router.budget.unavailable", score=0.0, threshold=0.0,
                          outcome="deny", container_id=container_id)
        return False
    used, total = store.used(container_id), store.total(container_id)
    fraction = get_tunable(container_id, TUN_ESC_BUDGET_FRACTION, 0.05)
    window = get_tunable(container_id, TUN_ESC_BUDGET_WINDOW_PAGES, max(total, 1))
    cap = max(1, int(fraction * window))
    if used >= cap:
        log_gate_decision("model_router.budget_exhausted", score=used, threshold=cap,
                          outcome="deny", container_id=container_id)
        return False
    store.reserve(container_id)
    log_gate_decision("model_router.budget_reserve", score=used + 1, threshold=cap,
                      outcome="allow", container_id=container_id)
    return True
```

Wire `select_model`'s `escalation_allowed(...)` call to forward an optional `store=` (default `None` → `_DEFAULT_STORE`) so tests inject a fake. Add `RedisBudgetStore` (guarded redis import; `INCR`/`GET` keyed `pdf:escbudget:{container_id}` per spec §6 Redis reuse rule) as the production `_DEFAULT_STORE`, set by the worker bootstrap.

- [ ] **A3.3** Run the router test file (all green).
- [ ] **A3.4** Commit: `feat(pdf): escalation gate + per-tenant budget cap (C7)`.

## Task A4 — Opus-never-at-ingestion invariant (RED first)

- [ ] **A4.1** RED — `test_ingestion_never_selects_opus`: set `model.querytime_strong_id="claude-opus-4-8"`, force a firing signal + open budget, then assert that for BOTH `task="extraction"` and `task="synthesis"` the returned `model_id` is NOT the Opus id (it is the Sonnet strong id), while `task="query_synthesis"` MAY return Opus. This proves the `_INGESTION_TASKS` allowlist branch.
- [ ] **A4.2** GREEN — already satisfied by Task A2's branch; if it fails, the allowlist is the bug, not the test.
- [ ] **A4.3** Run full router suite: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_model_router.py -q`.
- [ ] **A4.4** Commit: `test(pdf): assert ingestion can never select Opus (C7 invariant)`.

---

# PART B — DEEPDOC ONNX MICRO-COMPONENTS (extends Phase 1)

## Task B1 — Guarded deps + capability flags (RED first)

- [ ] **B1.1** RED — `test_deepdoc.py::test_degrades_when_onnxruntime_absent`: monkeypatch `_deps.HAS_ONNX = False`, call `enhance_page(...)` on a flagged page, assert it returns the **input** elements unchanged (fast-path) and logs a `deepdoc.unavailable` decision (no exception).
- [ ] **B1.2** GREEN — `pdf_chat/ingestion/deepdoc/_deps.py`:

```python
# pdf_chat/ingestion/deepdoc/_deps.py
from __future__ import annotations

try:
    import onnxruntime  # noqa: F401
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from sklearn.cluster import KMeans  # noqa: F401
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def deepdoc_available() -> bool:
    """Enhancers need ONNX + cv2 at minimum; KMeans-only column order needs sklearn."""
    return HAS_ONNX and HAS_CV2
```

- [ ] **B1.3** Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_deepdoc.py -q`.
- [ ] **B1.4** Commit: `feat(pdf): guarded deepdoc dependency capability flags`.

## Task B2 — KMeans multi-column / reading-order (RED first)

Concept ported from RAGFlow `_assign_column` (`/Users/bharath/Desktop/projects/ragflow-main/deepdoc/parser/pdf_parser.py:804`, KMeans cluster on left-edge `x0` with indent tolerance at lines 821-844). We port the **concept only** — no RAGFlow code is copied; sklearn is guarded.

- [ ] **B2.1** RED — `test_assign_columns_orders_two_column_page`: given boxes for an obvious 2-column page, assert `assign_columns(...)` returns elements with `reading_order` that reads column-1 top-to-bottom then column-2 (not interleaved by raw `y`).
- [ ] **B2.2** GREEN — `pdf_chat/ingestion/deepdoc/column_order.py`: `assign_columns(elements, *, container_id) -> list[UnifiedElement]`. Cluster left edges via KMeans (k chosen by the best-silhouette-up-to-`max_try` loop, concept from RAGFlow:836-844), then sort by `(col_id, y_top)` and overwrite `reading_order`. Indent tolerance is `width * get_tunable(container_id, "deepdoc.indent_tol_frac", 0.12)` — no literal. If `not HAS_SKLEARN`, return input unchanged + `log_gate_decision("deepdoc.column_order.skipped", ...)`.
- [ ] **B2.3** Run the deepdoc test file (green).
- [ ] **B2.4** Commit: `feat(pdf): KMeans multi-column reading-order enhancer`.

## Task B3 — ONNX table-structure (spanning cells) + rotation (RED first)

Concepts ported from `table_structure_recognizer.py:30` (TSR class, spanning-cell labels at :37, span calc `__cal_spans` at :501-550) and `_evaluate_table_orientation` (`pdf_parser.py:318`, 4-angle OCR-confidence rotation).

- [ ] **B3.1** RED — write two tests:
  - `test_table_structure_emits_spanning_cells`: inject a **mock ONNX session** (`session_factory` returns an object whose `.run(...)` yields fixed cell+span boxes); assert `recognize_table_structure(...)` produces a table `UnifiedElement` (markdown content) whose rows reflect a colspan (concept: `__cal_spans` rowspan/colspan, RAGFlow:510-529).
  - `test_rotation_picks_best_angle`: inject a mock OCR scorer returning the highest confidence at 90°; assert `best_rotation(...)` returns `(90, ...)` and writes the chosen angle into the element's metadata.
- [ ] **B3.2** GREEN — `table_structure.py::recognize_table_structure(image, *, container_id, session_factory)` and `rotation.py::best_rotation(table_img, *, container_id, ocr_scorer)`. Both accept the seam (`session_factory`/`ocr_scorer`) so tests stay pure; both guard `HAS_ONNX`/`HAS_CV2` and degrade to passing through the Phase-1 PyMuPDF/OCR table when unavailable. Output is a `UnifiedElement` of `ElementType.TABLE` with `bbox` + `confidence` (the min OCR/TSR confidence), conforming to `ingestion/ton_schema.py:32`.
- [ ] **B3.3** Run the deepdoc test file (green).
- [ ] **B3.4** Commit: `feat(pdf): ONNX table-structure (spanning cells) + table-rotation`.

## Task B4 — Hard-page gate + `enhance_page` orchestrator (RED first)

Gate mirrors the pure routing style of `parser_router.py:33`. The enhancer runs ONLY on a flagged page; simple pages take the Phase-1 fast path untouched.

- [ ] **B4.1** RED — write two tests:
  - `test_enhancer_skipped_on_simple_page`: a page with `complexity_score` below the tunable + high `extract_confidence` → `enhance_page` returns input elements unchanged and logs `deepdoc.gate` outcome `skip`. Assert no session_factory call.
  - `test_enhancer_used_on_flagged_page`: a page flagged complex/low-confidence (above gate) → `enhance_page` invokes column-order + table-structure and returns re-ordered/enriched elements; logs outcome `enhance`.
- [ ] **B4.2** GREEN — `pdf_chat/ingestion/deepdoc/enhancer.py`:

```python
# pdf_chat/ingestion/deepdoc/enhancer.py
from __future__ import annotations

from ..ton_schema import UnifiedElement
from ...tunables import get_tunable, log_gate_decision
from ._deps import deepdoc_available
from .column_order import assign_columns
from .table_structure import recognize_table_structure


def should_enhance(*, container_id: str, complexity_score: float,
                   extract_confidence: float) -> bool:
    """Hard-page gate: enhance only when the page is complex OR low-confidence."""
    cx_thr = get_tunable(container_id, "deepdoc.complexity_threshold", 0.60)
    conf_floor = get_tunable(container_id, "deepdoc.confidence_floor", 0.55)
    fire = complexity_score >= cx_thr or extract_confidence < conf_floor
    log_gate_decision("deepdoc.gate", score=complexity_score, threshold=cx_thr,
                      outcome="enhance" if fire else "skip",
                      container_id=container_id, extract_confidence=extract_confidence)
    return fire


def enhance_page(elements: list[UnifiedElement], *, container_id: str,
                 complexity_score: float, extract_confidence: float,
                 page_image=None, session_factory=None) -> list[UnifiedElement]:
    """Optional ONNX enhancement. Always returns valid UnifiedElement[]; on any
    gate-skip / missing-dep / failure it returns the input fast-path unchanged."""
    if not should_enhance(container_id=container_id, complexity_score=complexity_score,
                          extract_confidence=extract_confidence):
        return elements
    if not deepdoc_available():
        log_gate_decision("deepdoc.unavailable", score=0.0, threshold=0.0,
                          outcome="fastpath", container_id=container_id)
        return elements
    try:
        out = assign_columns(elements, container_id=container_id)
        if page_image is not None and session_factory is not None:
            out = recognize_table_structure(page_image, container_id=container_id,
                                            session_factory=session_factory) or out
        return out
    except Exception as exc:  # never block ingestion on an enhancer failure
        log_gate_decision("deepdoc.error", score=0.0, threshold=0.0,
                          outcome="fastpath", container_id=container_id, error=str(exc))
        return elements
```

- [ ] **B4.3** Wire (modify, do not duplicate) the Phase-1 `ingestion/page_extraction.py::extract_page_elements()` to call `enhance_page(...)` as a post-step after the digital/OCR extractor, passing the page's `complexity_score`/`extract_confidence`. This is the ONLY integration touchpoint.
- [ ] **B4.4** Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_deepdoc.py -q`.
- [ ] **B4.5** Commit: `feat(pdf): hard-page gate + enhance_page orchestrator (Phase 1)`.

## Task B5 — Build-time model pre-download (no first-run block)

- [ ] **B5.1** Add a build step that downloads the ONNX layout/TSR models into `pdf_chat/ingestion/deepdoc/models/` at **container build** (not first run). Document in `server/Dockerfile` (or `scripts/fetch_deepdoc_models.sh`): fetch the RAGFlow-compatible ONNX weights (Apache-2.0) to the models dir; the dir ships in the image. The model path resolves via `get_tunable(container_id, "deepdoc.model_dir", "<image path>")`. If the dir is empty at runtime, `deepdoc_available()`-gated code degrades to the fast path — first-run ingestion is never blocked on a download.
- [ ] **B5.2** Add `models/.gitkeep`; ensure weights are git-ignored.
- [ ] **B5.3** Commit: `chore(pdf): build-time deepdoc ONNX model pre-download`.

---

## Final Verification

- [ ] Run both new suites + the existing ingestion/tunables suites:
  `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_model_router.py pdf_chat/testing/test_deepdoc.py pdf_chat/testing/test_tunables.py pdf_chat/testing/test_ingestion.py -q`
- [ ] Grep guard (no literals introduced): `cd server && rg -n "claude-|gpt-4o|0\.[0-9]{2}" pdf_chat/model_router.py pdf_chat/ingestion/deepdoc/` should show matches ONLY as `get_tunable(...)` defaults, never in a bare comparison.
- [ ] Board gate: router C7 invariants (bulk→mini, gated+capped escalation, Opus-never-at-ingestion) + DeepDoc graceful degradation hold.

---

## Cross-Phase Contracts Exposed

- **C7 (router) — Phase 0 owns; Phases 2/3/5 consume.** Every model-using call site calls `select_model(task=TaskClass.X, container_id=cid, signals={...}) -> ModelChoice` instead of reading a deployment name. **Phase 2** (extraction) passes `task="extraction"` + `signals={"extract_confidence", "figure_ratio"}`. **Phase 5** (community reports / glossary) passes `task="synthesis"`. **Phase 3** (query runtime) passes `task="query_synthesis"`/`"query_planning"` + `signals={"cross_domain", "definitional"}` and is the ONLY caller that may receive the Opus id. Embeddings everywhere call `embedding_model(cid)`.
- **DeepDoc — Phase 1 owns.** Phase 1's `extract_page_elements()` calls `enhance_page(...)` as a post-step; downstream phases consume the same `UnifiedElement[]` schema (`ton_schema.py:32`) regardless of whether enhancement ran — the enhancer is invisible to Phases 2-5.

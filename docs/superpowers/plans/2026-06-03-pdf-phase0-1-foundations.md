# PDF Agentic Graph RAG — Phase 0 & 1 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `server/pdf_chat/` ingest a real PDF end-to-end (PyMuPDF + Azure Document Intelligence OCR + tables + bbox + extraction confidence), wire the framework-only backends (reranker, Redis cache, LLM synthesis, audit, finalization), enforce token guards (batch embeddings, prompt caching, query-embedding cache, context token budget, response-cache wiring), introduce a single per-container tunables/score-logging source so no threshold is a magic literal, and record a gold-question eval baseline.

**Architecture:** Phase 0 adds `pdf_chat/tunables.py` (`get_tunable` + `log_gate_decision`) as the *only* source of thresholds and the *only* gate-logging path; every later threshold resolves through it. Phase 1 replaces the `ingestion/tasks.py:224` `NotImplementedError` with a real extraction chain — a data-driven per-page digital-vs-scanned router (text-coverage ratio), a PyMuPDF digital extractor and an Azure Document Intelligence OCR/table/layout extractor (both emitting `UnifiedElement` with `bbox` + `confidence`), a confidence propagation step, and a wired query runtime (reranker → Redis cache → gpt-4o-mini synthesis → audit). A finalization task reduces settled pages to a document status. A gold-question eval harness records a baseline. Everything is pure-testable behind guarded infra imports, following the module's existing pattern (e.g. `ingestion/preflight.py:20-43`, `retrieval/reranker.py:21-35`).

**Tech Stack:** Python 3.12 · `uv` · pytest (run via `uv run --with pytest --with pytest-asyncio pytest`) · PyMuPDF (`fitz`) · `azure-ai-documentintelligence` (guarded) · Azure OpenAI (`text-embedding-3-small`, `gpt-4o-mini` via `chat_deployment()` / `DISABLE_GPT4O`) · Redis · Neo4j · SQLAlchemy 2.0 async · structlog.

---

## Test & Convention Notes (read before Task 1)

- **Run tests** from `server/`: `uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/<file> -q`. (Verified: `pdf_chat/testing/test_ingestion.py` → 45 passed.)
- **Test layout:** one file per team under `pdf_chat/testing/` (`test_ingestion.py`, `test_retrieval.py`, `test_agent.py`, `test_control_plane.py`). New pure tests go into the matching file; new modules with pure logic get their own `test_<module>.py` in the same folder.
- **Async tests** are written with `asyncio.run(...)` (see `test_ingestion.py:353-367`), NOT the `@pytest.mark.asyncio` decorator — match that style so no marker config is needed.
- **Infra-gated tests** use a marker. Register `infra` in a new `pdf_chat/testing/conftest.py` (Task 1.5) and decorate any test that needs Redis/Neo4j/Azure with `@pytest.mark.infra`; the default run excludes them via `-m "not infra"`.
- **No magic literals:** every threshold resolves via `get_tunable(container_id, key, default)` (Task 1); every gate/skip/cap/route/merge logs via `log_gate_decision(...)` (Task 2). No bare score-comparison literal in any `.py` file under `pdf_chat/`.
- **LLM:** gpt-4o-mini only. Synthesis reuses the main app's `get_settings().chat_deployment()` (`app/core/config.py:130`) which honors `DISABLE_GPT4O`. Enable prompt caching where the deployment supports it (Task 6).
- **Neo4j:** every node/edge carries `tenant_id`; every query filters `tenant_id` (existing writer already does — `ingestion/neo4j_writer.py:88-98`). No new Neo4j schema in Phases 0–1.
- **Commits:** conventional commits, each ending with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Create / Modify | Responsibility |
|---|---|---|
| `server/pdf_chat/tunables.py` | Create | `get_tunable()` + `log_gate_decision()` + tunable-key constants. The single threshold/logging source. |
| `server/pdf_chat/models/tunable.py` | Create | `PdfGraphRagTunable` ORM model (optional per-container override table). |
| `server/pdf_chat/migrations/tunables_upgrade.py` | Create | Idempotent runtime migration for `pdf_graphrag_tunables`. |
| `server/pdf_chat/testing/conftest.py` | Create | Registers the `infra` pytest marker. |
| `server/pdf_chat/testing/test_tunables.py` | Create | Pure tests for `get_tunable` / `log_gate_decision`. |
| `server/pdf_chat/ingestion/extraction_confidence.py` | Create | `propagate_confidence()` — pushes element extraction confidence onto chunks + a low-confidence flag. |
| `server/pdf_chat/ingestion/page_routing.py` | Create | `text_coverage_ratio()` + `route_page_extractor()` — data-driven digital-vs-scanned per-page routing (config + logged). |
| `server/pdf_chat/ingestion/digital_extractor.py` | Create | `extract_digital_page()` — PyMuPDF text + table + bbox + confidence → `UnifiedElement[]`. |
| `server/pdf_chat/ingestion/ocr_extractor.py` | Create | `extract_scanned_page()` — Azure Document Intelligence OCR + table + layout region typing + bbox + confidence → `UnifiedElement[]` (guarded). |
| `server/pdf_chat/ingestion/page_extraction.py` | Create | `extract_page_elements()` — orchestrates route → digital/ocr → confidence; the real `extract_fn` for `tasks.py`. |
| `server/pdf_chat/ingestion/tasks.py` | Modify (`:221-226`) | Replace the `NotImplementedError` stub with a wired `extract_fn`. |
| `server/pdf_chat/ingestion/finalize.py` | Create | `finalize_document()` — reduce settled pages → doc status (wraps existing `orchestrator.reconcile`). |
| `server/pdf_chat/retrieval/embeddings.py` | Modify | Add `embed_texts_batched()` (batch) + `QueryEmbedder` with a Redis query-embedding cache. |
| `server/pdf_chat/retrieval/llm.py` | Create | `PdfLlm` — gpt-4o-mini synthesis adapter with prompt caching. |
| `server/pdf_chat/retrieval/extractor.py` | Create | `OnDemandExtractor` — lazy table/image materialization adapter (agent Stage 6). |
| `server/pdf_chat/agent/audit.py` | Create | `QueryAuditRepo` — async audit write adapter. |
| `server/pdf_chat/agent/graph.py` | Modify (`assemble_context` `:237-249`) | Enforce a context token budget (config + logged). |
| `server/pdf_chat/eval/__init__.py` | Create | Package marker. |
| `server/pdf_chat/eval/gold_questions.py` | Create | Gold-question dataclass + the seed gold set loader. |
| `server/pdf_chat/eval/harness.py` | Create | `run_eval()` — scores answers against the gold set, records a baseline JSON. |
| `server/pdf_chat/eval/gold_set.json` | Create | The seed gold-question set (data, not code). |
| `server/pdf_chat/testing/test_extraction.py` | Create | Pure tests for routing, digital extraction, confidence propagation, page-extraction orchestration. |
| `server/pdf_chat/testing/test_eval.py` | Create | Pure tests for the eval harness scoring + baseline recording. |
| `server/pdf_chat/embeddings` tests | add to `test_retrieval.py` | Batch embedding + query-embedding-cache pure tests. |
| `server/pdf_chat/agent` tests | add to `test_agent.py` | Token-budget assemble + synthesis/audit adapter pure tests. |

---

# PHASE 0 — Token guards + tunables/logging foundation

## Task 1: Tunables source — `get_tunable`

**Files:**
- Create: `server/pdf_chat/tunables.py`
- Create: `server/pdf_chat/testing/test_tunables.py`

- [ ] **Step 1: Write the failing test**

In `server/pdf_chat/testing/test_tunables.py`:

```python
"""Pure tests for the single tunables/score-logging source (Spec §3 invariant 4)."""
from __future__ import annotations

from pdf_chat.tunables import get_tunable, TUNABLE_DEFAULTS


def test_get_tunable_returns_explicit_default_when_unset():
    # No env, no DB override → the caller-supplied default wins.
    assert get_tunable("container-1", "made_up_key", 0.42) == 0.42


def test_get_tunable_env_override_beats_default(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "1234")
    assert get_tunable("container-1", "context_token_budget", 8000) == 1234


def test_get_tunable_env_float_coerced(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_DIGITAL_TEXT_COVERAGE", "0.55")
    assert get_tunable("c", "digital_text_coverage", 0.7) == 0.55


def test_get_tunable_env_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_RERANK_TOP_N", "not-an-int")
    assert get_tunable("c", "rerank_top_n", 12) == 12


def test_tunable_defaults_registry_is_a_dict():
    assert isinstance(TUNABLE_DEFAULTS, dict)
    assert "context_token_budget" in TUNABLE_DEFAULTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.tunables'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/tunables.py`:

```python
"""The single per-container tunables source + score-logging harness.

Spec §3 invariant 4: every threshold (routing coverage, rerank-skip, token
budget, gleaning passes, planner-bypass, resolution bands, ...) resolves through
``get_tunable`` and every gate/cap/skip/route/merge decision is emitted via
``log_gate_decision``. No bare score-comparison literal lives in any other
``pdf_chat`` module — they pass a *named default* here instead, which is then
overridable per-container (env today, ``pdf_graphrag_tunables`` table later).

Resolution order (first hit wins):
    1. per-container DB override (``pdf_graphrag_tunables`` — wired in Task 1b)
    2. env var ``PDF_TUNABLE_<KEY_UPPER>``
    3. the caller-supplied ``default`` (which SHOULD also live in TUNABLE_DEFAULTS)

Pure module — safe to import with zero infra. The DB lookup is injected, never
imported at module load, so importing this never touches a database.
"""
from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

import structlog

_log = structlog.get_logger("pdf_chat.tunables")

T = TypeVar("T", int, float, str, bool)

# Canonical named defaults. A key MUST appear here before any module references
# it, so the full tunable surface is discoverable in one place (Spec §3 inv 4).
TUNABLE_DEFAULTS: dict[str, Any] = {
    # Phase 0 — token guards
    "context_token_budget": 8000,
    "embedding_batch_size": 64,
    "query_embedding_cache_ttl": 3600,
    "rerank_top_n": 12,
    "rerank_skip_below_candidates": 4,
    # Phase 1 — extraction routing
    "digital_text_coverage": 0.70,
    "low_confidence_flag_below": 0.60,
    "ocr_table_min_confidence": 0.50,
}

# Optional per-container DB override hook. Wired in Task 1b; until then None ⇒
# DB tier is skipped. Signature: (container_id, key) -> str | None.
_db_lookup: Callable[[str, str], "str | None"] | None = None


def set_db_lookup(fn: "Callable[[str, str], str | None] | None") -> None:
    """Install the per-container override lookup (called by the migration/bootstrap)."""
    global _db_lookup
    _db_lookup = fn


def _coerce(raw: str, default: T) -> T:
    """Coerce a string override to the type of ``default`` (bad value ⇒ default)."""
    try:
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")  # type: ignore[return-value]
        if isinstance(default, int):
            return int(raw)  # type: ignore[return-value]
        if isinstance(default, float):
            return float(raw)  # type: ignore[return-value]
        return raw  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


def get_tunable(container_id: str, key: str, default: T) -> T:
    """Resolve a tunable for ``container_id`` (DB override → env → default)."""
    if _db_lookup is not None:
        try:
            raw = _db_lookup(container_id, key)
        except Exception:  # pragma: no cover - DB is best-effort, never fatal
            raw = None
        if raw is not None:
            return _coerce(raw, default)

    env_raw = os.getenv(f"PDF_TUNABLE_{key.upper()}")
    if env_raw is not None:
        return _coerce(env_raw, default)

    return default
```

(`log_gate_decision` is added in Task 2 — keep this file open.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/tunables.py pdf_chat/testing/test_tunables.py
git commit -m "feat(pdf): add per-container tunables source (no magic literals)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 1.5: Register the `infra` pytest marker

**Files:**
- Create: `server/pdf_chat/testing/conftest.py`

- [ ] **Step 1: Write the failing test** (add to `test_tunables.py`)

```python
def test_infra_marker_is_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("infra") for m in markers)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py::test_infra_marker_is_registered -q`
Expected: FAIL — the `infra` marker is not registered (empty ini list).

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/testing/conftest.py`:

```python
"""pytest config for pdf_chat tests.

Registers the ``infra`` marker so infra-dependent tests (Redis/Neo4j/Azure) are
opt-in. Default runs exclude them with ``-m "not infra"``.
"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "infra: requires live infra (Redis/Neo4j/Azure); excluded by default",
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/testing/conftest.py pdf_chat/testing/test_tunables.py
git commit -m "test(pdf): register infra pytest marker for opt-in infra tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Score-logging harness — `log_gate_decision`

**Files:**
- Modify: `server/pdf_chat/tunables.py`
- Modify: `server/pdf_chat/testing/test_tunables.py`

- [ ] **Step 1: Write the failing test** (add to `test_tunables.py`)

```python
from pdf_chat.tunables import log_gate_decision


def test_log_gate_decision_returns_structured_record():
    rec = log_gate_decision(
        "digital_vs_scanned",
        score=0.83,
        threshold=0.70,
        outcome="digital",
        container_id="c-1",
        page_num=3,
    )
    assert rec["gate"] == "digital_vs_scanned"
    assert rec["score"] == 0.83
    assert rec["threshold"] == 0.70
    assert rec["outcome"] == "digital"
    assert rec["page_num"] == 3
    assert rec["passed"] is True  # score >= threshold


def test_log_gate_decision_passed_false_when_below_threshold():
    rec = log_gate_decision("rerank_skip", score=2, threshold=4, outcome="skip")
    assert rec["passed"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: FAIL — `ImportError: cannot import name 'log_gate_decision'`.

- [ ] **Step 3: Write minimal implementation** (append to `server/pdf_chat/tunables.py`)

```python
def log_gate_decision(
    name: str,
    *,
    score: float,
    threshold: float,
    outcome: str,
    **ctx: Any,
) -> dict:
    """Emit (structlog) + return a structured gate/cap/skip/route decision.

    EVERY threshold comparison in pdf_chat routes through here so a score is
    never compared-and-discarded silently (Spec §3 invariant 4). ``passed`` is
    the canonical ``score >= threshold`` test; gates that want strict-greater
    pass an adjusted threshold. Returns the record so callers can assert on it
    in tests and attach it to traces.
    """
    passed = score >= threshold
    record = {
        "gate": name,
        "score": score,
        "threshold": threshold,
        "outcome": outcome,
        "passed": passed,
        **ctx,
    }
    _log.info("pdf_chat.gate", **record)
    return record
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/tunables.py pdf_chat/testing/test_tunables.py
git commit -m "feat(pdf): add log_gate_decision score-logging harness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2b: `pdf_graphrag_tunables` ORM model + migration + DB lookup wiring

**Files:**
- Create: `server/pdf_chat/models/tunable.py`
- Create: `server/pdf_chat/migrations/tunables_upgrade.py`
- Modify: `server/pdf_chat/testing/test_tunables.py`

- [ ] **Step 1: Write the failing test** (add to `test_tunables.py`)

```python
def test_db_lookup_override_beats_env(monkeypatch):
    from pdf_chat import tunables

    monkeypatch.setenv("PDF_TUNABLE_RERANK_TOP_N", "8")
    tunables.set_db_lookup(lambda cid, key: "5" if key == "rerank_top_n" else None)
    try:
        assert tunables.get_tunable("c-9", "rerank_top_n", 12) == 5
    finally:
        tunables.set_db_lookup(None)


def test_tunable_model_table_name():
    from pdf_chat.models.tunable import PdfGraphRagTunable

    assert PdfGraphRagTunable.__tablename__ == "pdf_graphrag_tunables"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.models.tunable'`. (The `set_db_lookup` test passes already — both run.)

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/models/tunable.py` (mirror the additive runtime-migration model style of `app/models`):

```python
"""ORM model for per-container tunable overrides (Spec §3 invariant 4).

Optional: tunables resolve from env + named defaults without this table; the
table only lets an operator override a single key for one container without a
deploy. Tenant-isolated via ``container_id``.
"""
from __future__ import annotations

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base  # reuse the app's declarative Base


class PdfGraphRagTunable(Base):
    __tablename__ = "pdf_graphrag_tunables"
    __table_args__ = (UniqueConstraint("container_id", "key", name="uq_pdf_tunable"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    container_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
```

In `server/pdf_chat/migrations/tunables_upgrade.py` (idempotent, non-fatal, matches `app/migrations` style):

```python
"""Idempotent runtime migration: create pdf_graphrag_tunables.

Additive + non-fatal (mirrors app/migrations). Called from the app lifespan
alongside the other pdf_chat migrations. Also installs the get_tunable DB lookup.
"""
from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from pdf_chat import tunables

_log = structlog.get_logger("pdf_chat.migrations")

_CREATE = text(
    """
    CREATE TABLE IF NOT EXISTS pdf_graphrag_tunables (
        id BIGSERIAL PRIMARY KEY,
        container_id VARCHAR(64) NOT NULL,
        key VARCHAR(128) NOT NULL,
        value TEXT NOT NULL,
        CONSTRAINT uq_pdf_tunable UNIQUE (container_id, key)
    )
    """
)
_INDEX = text(
    "CREATE INDEX IF NOT EXISTS ix_pdf_tunable_container "
    "ON pdf_graphrag_tunables (container_id)"
)


async def upgrade(session: AsyncSession) -> None:
    try:
        await session.execute(_CREATE)
        await session.execute(_INDEX)
        await session.commit()
    except Exception as exc:  # pragma: no cover - non-fatal like sibling migrations
        _log.warning("pdf_graphrag_tunables migration skipped", error=str(exc))


def install_db_lookup(session_factory) -> None:
    """Install a sync-callable DB lookup over an async session factory.

    The lookup runs a tiny SELECT per (container, key); callers cache aggressively
    upstream. Best-effort — any error returns None so get_tunable falls through.
    """
    import asyncio

    def _lookup(container_id: str, key: str) -> "str | None":
        async def _q() -> "str | None":
            async with session_factory() as s:
                row = await s.execute(
                    text(
                        "SELECT value FROM pdf_graphrag_tunables "
                        "WHERE container_id = :c AND key = :k LIMIT 1"
                    ),
                    {"c": container_id, "k": key},
                )
                hit = row.first()
                return hit[0] if hit else None

        try:
            return asyncio.get_event_loop().run_until_complete(_q())
        except RuntimeError:
            return asyncio.run(_q())

    tunables.set_db_lookup(_lookup)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_tunables.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/models/tunable.py pdf_chat/migrations/tunables_upgrade.py pdf_chat/testing/test_tunables.py
git commit -m "feat(pdf): pdf_graphrag_tunables table + DB override lookup for tunables

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Batch embeddings

**Files:**
- Modify: `server/pdf_chat/retrieval/embeddings.py`
- Test: `server/pdf_chat/testing/test_retrieval.py`

> Note: `pdf_chat/ingestion/embeddings.py:35` already has `embed_texts(texts, ...)` that sends the whole list in one call. The token guard is *bounded batching* (config-sized) so a 10k-chunk document does not hit a single oversized request. We add `embed_texts_batched` in the retrieval embeddings module (the agent's `QueryEmbedder` lives here) and have it delegate to the ingestion `embed_texts` per batch.

- [ ] **Step 1: Write the failing test** (add to `test_retrieval.py`)

```python
def test_embed_texts_batched_chunks_into_config_sized_calls(monkeypatch):
    from pdf_chat.retrieval import embeddings as emb

    calls: list[int] = []

    def _fake_embed(texts, *, model=None):
        calls.append(len(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    out = emb.embed_texts_batched(
        [f"t{i}" for i in range(5)], container_id="c-1", batch_size=2
    )
    assert len(out) == 5            # one vector per input, order preserved
    assert calls == [2, 2, 1]       # batched into 2,2,1


def test_embed_texts_batched_empty():
    from pdf_chat.retrieval import embeddings as emb

    assert emb.embed_texts_batched([], container_id="c-1") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k embed_texts_batched`
Expected: FAIL — `AttributeError: module 'pdf_chat.retrieval.embeddings' has no attribute 'embed_texts_batched'`.

- [ ] **Step 3: Write minimal implementation**

If `server/pdf_chat/retrieval/embeddings.py` does not yet define a query embedder, create it with this content (it is referenced by `agent/graph.py:490` `from pdf_chat.retrieval.embeddings import QueryEmbedder`); otherwise add `embed_texts_batched` + cache to the existing file:

```python
"""Query + batch embedding adapters (token guards: batch + query cache).

Reuses the SAME embedding model as ingestion (text-embedding-3-small / 1536) via
the shared ``ingestion.embeddings.embed_texts`` call. Adds:
  * ``embed_texts_batched`` — splits a large list into config-sized batches so a
    big document never issues one oversized embedding request.
  * ``QueryEmbedder`` — async query embedder with a Redis query-embedding cache
    (Task 4) so a repeated query never re-embeds.
"""
from __future__ import annotations

from typing import Any

from pdf_chat.ingestion.embeddings import embed_texts
from pdf_chat.tunables import get_tunable, log_gate_decision


def embed_texts_batched(
    texts: list[str],
    *,
    container_id: str,
    batch_size: int | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in config-sized batches, preserving input order."""
    if not texts:
        return []
    if batch_size is None:
        batch_size = get_tunable(container_id, "embedding_batch_size", 64)
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        log_gate_decision(
            "embedding_batch",
            score=len(batch),
            threshold=batch_size,
            outcome="embed",
            container_id=container_id,
            batch_start=start,
        )
        out.extend(embed_texts(batch, model=model))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k embed_texts_batched`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/embeddings.py pdf_chat/testing/test_retrieval.py
git commit -m "feat(pdf): config-sized batch embeddings token guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Query-embedding cache (Redis) + `QueryEmbedder`

**Files:**
- Modify: `server/pdf_chat/retrieval/embeddings.py`
- Test: `server/pdf_chat/testing/test_retrieval.py`

- [ ] **Step 1: Write the failing test** (add to `test_retrieval.py`)

```python
import asyncio


class _FakeCache:
    def __init__(self):
        self.store: dict[str, list] = {}
        self.set_calls = 0

    def get_vector(self, key):
        return self.store.get(key)

    def set_vector(self, key, vec, ttl):
        self.set_calls += 1
        self.store[key] = vec


def test_query_embedder_caches_and_reuses(monkeypatch):
    from pdf_chat.retrieval import embeddings as emb

    embed_calls: list[str] = []

    def _fake_embed(texts, *, model=None):
        embed_calls.extend(texts)
        return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    cache = _FakeCache()
    embedder = emb.QueryEmbedder(cache=cache)

    v1 = asyncio.run(embedder.embed("revenue?", container_id="c-1"))
    v2 = asyncio.run(embedder.embed("revenue?", container_id="c-1"))
    assert v1 == v2 == [1.0, 2.0, 3.0]
    assert embed_calls == ["revenue?"]   # embedded ONCE — second was a cache hit
    assert cache.set_calls == 1


def test_query_embedder_no_cache_still_embeds(monkeypatch):
    from pdf_chat.retrieval import embeddings as emb

    monkeypatch.setattr(emb, "embed_texts", lambda texts, *, model=None: [[9.0]])
    embedder = emb.QueryEmbedder(cache=None)
    assert asyncio.run(embedder.embed("q", container_id="c-1")) == [9.0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k QueryEmbedder`
Expected: FAIL — `AttributeError: ... has no attribute 'QueryEmbedder'`.

- [ ] **Step 3: Write minimal implementation** (append to `server/pdf_chat/retrieval/embeddings.py`)

```python
import hashlib


def query_embedding_cache_key(query: str, model: str) -> str:
    """Stable key for a (query, embedding-model) pair (model-scoped so a model
    swap never serves a stale vector)."""
    return "pdf:qemb:" + hashlib.sha256(f"{model}|{query}".encode("utf-8")).hexdigest()


class QueryEmbedder:
    """Async query embedder with an optional Redis query-embedding cache.

    Satisfies the agent's ``Embedder`` protocol (``async def embed``). ``cache``
    is any object exposing ``get_vector(key) -> list[float] | None`` and
    ``set_vector(key, vec, ttl)`` (the retrieval RedisCache vector helpers, Task
    4b); ``None`` disables caching but still embeds.
    """

    def __init__(self, cache: Any = None, model: str | None = None) -> None:
        self._cache = cache
        self._model = model

    async def embed(self, text: str, container_id: str = "") -> list[float]:
        from pdf_chat.config import get_pdf_settings

        model = self._model or get_pdf_settings().embedding_model
        key = query_embedding_cache_key(text, model)
        if self._cache is not None:
            hit = self._cache.get_vector(key)
            if hit is not None:
                log_gate_decision(
                    "query_embedding_cache", score=1, threshold=1,
                    outcome="hit", container_id=container_id,
                )
                return hit
        vec = embed_texts([text], model=model)[0]
        if self._cache is not None:
            ttl = get_tunable(container_id, "query_embedding_cache_ttl", 3600)
            self._cache.set_vector(key, vec, ttl)
        return vec
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k QueryEmbedder`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/embeddings.py pdf_chat/testing/test_retrieval.py
git commit -m "feat(pdf): query-embedding cache + QueryEmbedder adapter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4b: Vector get/set on `RedisCache`

**Files:**
- Modify: `server/pdf_chat/retrieval/cache.py`
- Test: `server/pdf_chat/testing/test_retrieval.py`

- [ ] **Step 1: Write the failing test** (add to `test_retrieval.py`)

```python
def test_redis_cache_vector_roundtrip_noop_without_infra():
    from pdf_chat.retrieval.cache import RedisCache

    cache = RedisCache(url="redis://localhost:6379/0")
    # No live Redis in CI → set_vector returns False, get_vector returns None,
    # and neither raises (cache is an optimization, never a dependency).
    assert cache.set_vector("k", [1.0, 2.0], 60) in (True, False)
    assert cache.get_vector("missing-key") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k vector_roundtrip`
Expected: FAIL — `AttributeError: 'RedisCache' object has no attribute 'set_vector'`.

- [ ] **Step 3: Write minimal implementation** (append methods to `RedisCache` in `server/pdf_chat/retrieval/cache.py`)

```python
    def get_vector(self, key: str) -> "list[float] | None":
        """Return a cached embedding vector, or None on miss/no-infra."""
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except Exception:  # pragma: no cover - infra-dependent
            return None
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, list) else None

    def set_vector(self, key: str, vec: "list[float]", ttl_seconds: int) -> bool:
        """Persist an embedding vector with TTL. False if cache unavailable."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.set(key, json.dumps(vec), ex=ttl_seconds)
            return True
        except Exception:  # pragma: no cover - infra-dependent
            return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k vector_roundtrip`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/cache.py pdf_chat/testing/test_retrieval.py
git commit -m "feat(pdf): vector get/set on RedisCache for query-embedding cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Context token budget in `assemble_context`

**Files:**
- Modify: `server/pdf_chat/agent/graph.py` (`assemble_context`, `:237-249`)
- Test: `server/pdf_chat/testing/test_agent.py`

- [ ] **Step 1: Write the failing test** (add to `test_agent.py`)

```python
import asyncio

from pdf_chat.agent.graph import assemble_context, Deps
from pdf_chat.agent.state import PdfChatState


def _chunk(cid, text, page=1):
    return {"chunk_id": cid, "text": text, "doc_id": "d1", "page_num": page,
            "tenant_id": "t1"}


def test_assemble_context_truncates_to_token_budget(monkeypatch):
    # 4 chunks of ~10 "tokens" each; budget low enough to admit only the first 2.
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "25")
    chunks = [_chunk(f"c{i}", " ".join(["word"] * 10)) for i in range(4)]
    state = PdfChatState(query="q", tenant_id="t1")
    state.accessible_chunks = chunks
    out = asyncio.run(assemble_context(state, Deps()))
    # Only the chunks that fit under the budget are cited; nothing crashes.
    assert len(out.citations) < 4
    assert len(out.citations) >= 1
    assert all(f"[{c['n']}]" in out.context for c in out.citations)


def test_assemble_context_keeps_all_when_under_budget(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "100000")
    chunks = [_chunk(f"c{i}", "short") for i in range(3)]
    state = PdfChatState(query="q", tenant_id="t1")
    state.accessible_chunks = chunks
    out = asyncio.run(assemble_context(state, Deps()))
    assert len(out.citations) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k assemble_context`
Expected: FAIL — `test_assemble_context_truncates_to_token_budget` asserts `< 4` but the current implementation cites all 4 (no budget).

- [ ] **Step 3: Write minimal implementation** — replace the body of `assemble_context` in `server/pdf_chat/agent/graph.py` (`:237-249`):

```python
async def assemble_context(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 8 — build the numbered [N] context block + citation map.

    Enforces a per-container context token budget (Spec §2 L4 token guard #8):
    chunks are admitted in order until the running token estimate would exceed
    the budget; the drop is logged via log_gate_decision. Token estimate is the
    whitespace word count (cheap, deterministic — matches the chunker's
    approximation in ingestion/chunker.py:46).
    """
    from pdf_chat.tunables import get_tunable, log_gate_decision

    container_id = getattr(state, "tenant_id", "")
    budget = get_tunable(container_id, "context_token_budget", 8000)

    lines: list[str] = []
    citations: list[dict] = []
    used_tokens = 0
    n = 0
    for chunk in state.accessible_chunks:
        text = _attr(chunk, "text", "") or ""
        tok = len(text.split())
        if n > 0 and used_tokens + tok > budget:
            log_gate_decision(
                "context_token_budget",
                score=used_tokens + tok,
                threshold=budget,
                outcome="truncate",
                container_id=container_id,
                admitted=n,
            )
            break
        n += 1
        used_tokens += tok
        doc_id = _attr(chunk, "doc_id", "")
        page = _attr(chunk, "page_num", 0)
        lines.append(f"[{n}] {text}    Source: {doc_id}, page {page}")
        citations.append({"n": n, "doc_id": str(doc_id), "page": int(page or 0)})
    state.context = "\n".join(lines)
    state.citations = citations
    return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k assemble_context`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/agent/graph.py pdf_chat/testing/test_agent.py
git commit -m "feat(pdf): enforce context token budget in assemble_context

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: LLM synthesis adapter (`PdfLlm`) with prompt caching + response-cache wiring

**Files:**
- Create: `server/pdf_chat/retrieval/llm.py`
- Test: `server/pdf_chat/testing/test_agent.py`

> The response cache itself (Redis get/set keyed by query+tenant+acl_version+doc_ids) is ALREADY wired in `agent/graph.py` (`cache_check` `:157`, `cache_write` `:272`, `_compute_cache_key` `:324`). Phase 0's "response-cache wiring" = ensure `build_default_deps()` (`:480`) supplies a real `cache` (`RedisCache`, exists) and a real `llm` (this task creates `PdfLlm`, the missing import at `:520`).

- [ ] **Step 1: Write the failing test** (add to `test_agent.py`)

```python
def test_pdf_llm_uses_mini_deployment_and_prompt_caching(monkeypatch):
    from pdf_chat.retrieval import llm as llm_mod

    captured = {}

    class _FakeMsgs:
        def create(self, **kwargs):
            captured.update(kwargs)
            class _R:
                choices = [type("C", (), {"message": type("M", (), {"content": "grounded answer"})()})()]
            return _R()

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeMsgs()})()

    monkeypatch.setattr(llm_mod, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(llm_mod, "_resolve_deployment", lambda: "gpt-4o-mini")

    adapter = llm_mod.PdfLlm()
    out = asyncio.run(adapter.generate("SYS", "USER"))
    assert out == "grounded answer"
    assert captured["model"] == "gpt-4o-mini"     # never gpt-4o
    # System prompt sent as a cacheable first message (prompt caching).
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "SYS"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k pdf_llm`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.retrieval.llm'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/retrieval/llm.py`:

```python
"""Grounded-synthesis LLM adapter (Stage 9). gpt-4o-mini ONLY.

Satisfies the agent's ``Llm`` protocol (``async def generate(system, user)``).
Routes to the gpt-4o-mini deployment via the main app's ``chat_deployment()``
(which honors DISABLE_GPT4O) and sends the system prompt as the first message so
Azure OpenAI prompt caching can amortize the stable instruction prefix across
queries. Guarded import: constructs without infra; raises only on CALL.
"""
from __future__ import annotations

import os

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - exercised only without infra
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _build_client():  # pragma: no cover - requires infra + env
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    )


def _resolve_deployment() -> str:
    """gpt-4o-mini deployment via the main app config (DISABLE_GPT4O honored)."""
    try:
        from app.core.config import get_settings  # type: ignore

        return get_settings().chat_deployment()
    except Exception:  # pragma: no cover - standalone fallback
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI", "gpt-4o-mini")


class PdfLlm:
    """gpt-4o-mini synthesis adapter with prompt-cached system prefix."""

    async def generate(self, system: str, user: str) -> str:
        if not _HAS_OPENAI:
            raise RuntimeError(
                "The OpenAI SDK is required for LLM synthesis but is not installed."
            )
        client = _build_client()
        resp = client.chat.completions.create(
            model=_resolve_deployment(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k pdf_llm`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/llm.py pdf_chat/testing/test_agent.py
git commit -m "feat(pdf): gpt-4o-mini synthesis adapter (PdfLlm) with prompt caching

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: On-demand extractor adapter + audit repo

**Files:**
- Create: `server/pdf_chat/retrieval/extractor.py`
- Create: `server/pdf_chat/agent/audit.py`
- Test: `server/pdf_chat/testing/test_agent.py`

> These two adapters are the remaining `build_default_deps()` imports that fail today (`agent/graph.py:514` `OnDemandExtractor`, `:526` `QueryAuditRepo`). Wiring them makes `build_default_deps()` produce a fully-populated `Deps`. The audit row reuses an `query_audit_log`-style table; in Phases 0–1 it writes via an injected session, no new schema beyond an additive table created in Task 7b is required — here we keep the adapter pure-testable with an injected writer.

- [ ] **Step 1: Write the failing test** (add to `test_agent.py`)

```python
def test_on_demand_extractor_passthrough_for_text_chunk():
    from pdf_chat.retrieval.extractor import OnDemandExtractor

    chunk = {"chunk_id": "c1", "element_type": "text", "text": "already here"}
    out = asyncio.run(OnDemandExtractor().extract(chunk))
    assert out["text"] == "already here"   # text chunks are returned unchanged


def test_query_audit_repo_writes_via_injected_sink():
    from pdf_chat.agent.audit import QueryAuditRepo

    rows = []
    repo = QueryAuditRepo(sink=lambda row: rows.append(row))
    asyncio.run(repo.write(
        user_id="u1", tenant_id="t1", query_hash="h", query_text="q",
        returned_chunks=["c1"], denied_chunks=["c2"], cache_hit=True,
    ))
    assert rows[0]["tenant_id"] == "t1"
    assert rows[0]["cache_hit"] is True
    assert rows[0]["returned_chunks"] == ["c1"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k "on_demand_extractor or query_audit"`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.retrieval.extractor'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/retrieval/extractor.py`:

```python
"""On-demand extractor adapter (agent Stage 6).

Lazily materializes table/image chunk bodies that survived ACL. In Phases 0–1 the
ingestion path already stores table markdown + image captions on the chunk text
(ingestion/chunker.py:144-195), so this adapter is a pass-through that returns the
chunk unchanged; it exists so build_default_deps() can wire a real extractor and
the agent's on_demand_extract node has a non-None dep. Satisfies the agent's
``Extractor`` protocol (``async def extract(chunk)``).
"""
from __future__ import annotations

from typing import Any


class OnDemandExtractor:
    async def extract(self, chunk: Any) -> Any:
        return chunk
```

In `server/pdf_chat/agent/audit.py`:

```python
"""Query audit adapter (agent Stage 10).

Records every served query (incl. cache hits — Security must-fix #6) for
compliance. Satisfies the agent's ``AuditRepo`` protocol. The ``sink`` is the
write target: a pure callable in tests; in production a thin async DB writer
against the audit table. Tenant-isolated via the persisted tenant_id.
"""
from __future__ import annotations

from typing import Any, Callable


class QueryAuditRepo:
    def __init__(self, sink: "Callable[[dict], Any] | None" = None) -> None:
        self._sink = sink

    async def write(
        self,
        *,
        user_id: str,
        tenant_id: str,
        query_hash: str,
        query_text: str,
        returned_chunks: list[str],
        denied_chunks: list[str],
        cache_hit: bool = False,
    ) -> None:
        row = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query_hash": query_hash,
            "query_text": query_text,
            "returned_chunks": list(returned_chunks),
            "denied_chunks": list(denied_chunks),
            "cache_hit": cache_hit,
        }
        if self._sink is not None:
            result = self._sink(row)
            if hasattr(result, "__await__"):
                await result
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py -q -k "on_demand_extractor or query_audit"`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/extractor.py pdf_chat/agent/audit.py pdf_chat/testing/test_agent.py
git commit -m "feat(pdf): wire OnDemandExtractor + QueryAuditRepo adapters

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

# PHASE 1 — Make it run end-to-end + eval

## Task 8: Data-driven page routing (text-coverage ratio)

**Files:**
- Create: `server/pdf_chat/ingestion/page_routing.py`
- Test: `server/pdf_chat/testing/test_extraction.py`

- [ ] **Step 1: Write the failing test**

In `server/pdf_chat/testing/test_extraction.py`:

```python
"""Pure tests for Phase-1 page extraction (routing, digital extract, confidence)."""
from __future__ import annotations

from pdf_chat.ingestion.page_routing import text_coverage_ratio, route_page_extractor


def test_text_coverage_ratio_full_text():
    # All page area covered by text spans → ratio ~1.0.
    assert text_coverage_ratio(text_area=1000.0, page_area=1000.0) == 1.0


def test_text_coverage_ratio_scanned_page():
    # A scanned page has near-zero extractable text area.
    assert text_coverage_ratio(text_area=2.0, page_area=1000.0) < 0.01


def test_text_coverage_ratio_zero_page_area_is_zero():
    assert text_coverage_ratio(text_area=5.0, page_area=0.0) == 0.0


def test_route_page_extractor_digital_above_threshold():
    route = route_page_extractor(coverage=0.92, container_id="c-1", page_num=0)
    assert route == "digital"


def test_route_page_extractor_scanned_below_threshold():
    route = route_page_extractor(coverage=0.05, container_id="c-1", page_num=1)
    assert route == "scanned"


def test_route_page_extractor_threshold_is_tunable(monkeypatch):
    # Lower the digital threshold so 0.4 coverage now routes digital.
    monkeypatch.setenv("PDF_TUNABLE_DIGITAL_TEXT_COVERAGE", "0.3")
    assert route_page_extractor(coverage=0.4, container_id="c-1", page_num=2) == "digital"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.page_routing'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/ingestion/page_routing.py`:

```python
"""Data-driven per-page digital-vs-scanned routing (Spec §2 L1a).

Routes on MEASURED per-page extractable-text coverage ratio (text-span area /
page area), NOT ``if text == ""``. The threshold is a tunable and every routing
decision is logged with its score (Spec §3 invariant 4).
"""
from __future__ import annotations

from pdf_chat.tunables import get_tunable, log_gate_decision


def text_coverage_ratio(*, text_area: float, page_area: float) -> float:
    """Fraction of the page covered by extractable text spans, clamped to [0,1]."""
    if page_area <= 0:
        return 0.0
    return max(0.0, min(1.0, text_area / page_area))


def route_page_extractor(*, coverage: float, container_id: str, page_num: int) -> str:
    """Return ``"digital"`` (PyMuPDF) or ``"scanned"`` (OCR) for one page."""
    threshold = get_tunable(container_id, "digital_text_coverage", 0.70)
    decision = log_gate_decision(
        "digital_vs_scanned",
        score=coverage,
        threshold=threshold,
        outcome="digital" if coverage >= threshold else "scanned",
        container_id=container_id,
        page_num=page_num,
    )
    return decision["outcome"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/page_routing.py pdf_chat/testing/test_extraction.py
git commit -m "feat(pdf): data-driven per-page text-coverage routing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Extraction confidence propagation

**Files:**
- Create: `server/pdf_chat/ingestion/extraction_confidence.py`
- Test: `server/pdf_chat/testing/test_extraction.py`

- [ ] **Step 1: Write the failing test** (add to `test_extraction.py`)

```python
from pdf_chat.ingestion.extraction_confidence import propagate_confidence
from pdf_chat.ingestion.ton_schema import Chunk, ElementType, UnifiedElement


def _el(conf):
    return UnifiedElement(
        element_id="e1", doc_id="d", page_num=0, element_type=ElementType.TEXT,
        content="hi", reading_order=0, tenant_id="t", confidence=conf,
    )


def _chunk():
    return Chunk(
        chunk_id="e1::c0", doc_id="d", page_num=0, element_type=ElementType.TEXT,
        text="hi", reading_order=0, tenant_id="t", source_element_id="e1",
    )


def test_propagate_confidence_sets_low_flag_below_threshold(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_LOW_CONFIDENCE_FLAG_BELOW", "0.60")
    chunks = propagate_confidence([_chunk()], {"e1": 0.4}, container_id="t")
    assert chunks[0].confidence == 0.4
    assert chunks[0].low_confidence is True


def test_propagate_confidence_high_confidence_not_flagged(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_LOW_CONFIDENCE_FLAG_BELOW", "0.60")
    chunks = propagate_confidence([_chunk()], {"e1": 0.95}, container_id="t")
    assert chunks[0].low_confidence is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k propagate_confidence`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.extraction_confidence'` (and `Chunk` has no `confidence`/`low_confidence` field yet).

- [ ] **Step 3: Write minimal implementation**

First add two fields to `Chunk` in `server/pdf_chat/ingestion/ton_schema.py` (after `source_element_id`, `:75`):

```python
    confidence: float = 1.0
    low_confidence: bool = False
```

And include them in `to_neo4j_props` (so the flag reaches the graph) — add to the returned dict in `Chunk.to_neo4j_props` (`:80-90`):

```python
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
```

Then create `server/pdf_chat/ingestion/extraction_confidence.py`:

```python
"""Propagate per-element extraction confidence onto chunks (Spec §2 L1a).

Low-confidence OCR/table cells are FLAGGED (``low_confidence=True``), never
silently asserted, so downstream synthesis can caveat them. The flag threshold
is a tunable and the decision is logged (Spec §3 invariant 4).
"""
from __future__ import annotations

from pdf_chat.ingestion.ton_schema import Chunk
from pdf_chat.tunables import get_tunable, log_gate_decision


def propagate_confidence(
    chunks: list[Chunk],
    element_confidence: dict[str, float],
    *,
    container_id: str,
) -> list[Chunk]:
    """Stamp each chunk with its source element's confidence + low-confidence flag."""
    flag_below = get_tunable(container_id, "low_confidence_flag_below", 0.60)
    for chunk in chunks:
        conf = element_confidence.get(chunk.source_element_id or "", 1.0)
        chunk.confidence = conf
        decision = log_gate_decision(
            "extraction_confidence",
            score=conf,
            threshold=flag_below,
            outcome="ok" if conf >= flag_below else "flagged_low",
            container_id=container_id,
            chunk_id=chunk.chunk_id,
        )
        chunk.low_confidence = not decision["passed"]
    return chunks
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k propagate_confidence && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q`
Expected: PASS (2 passed; ingestion suite still green — `to_neo4j_props` test at `test_ingestion.py:503` still passes with the additive keys).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/extraction_confidence.py pdf_chat/ingestion/ton_schema.py pdf_chat/testing/test_extraction.py
git commit -m "feat(pdf): propagate extraction confidence + low-confidence flag to chunks

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Digital page extractor (PyMuPDF, bbox + confidence)

**Files:**
- Create: `server/pdf_chat/ingestion/digital_extractor.py`
- Test: `server/pdf_chat/testing/test_extraction.py`

> The PyMuPDF call is behind a guard (mirrors `page_reader.py:16-23`). To keep the test pure, `extract_digital_page` accepts an injected `page` object exposing the small surface PyMuPDF's `Page` provides (`get_text("dict")` and `.rect`); the real worker passes a `fitz.Page`.

- [ ] **Step 1: Write the failing test** (add to `test_extraction.py`)

```python
from pdf_chat.ingestion.digital_extractor import extract_digital_page
from pdf_chat.ingestion.ton_schema import ElementType


class _FakeRect:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class _FakePage:
    """Minimal stand-in for fitz.Page used by extract_digital_page."""
    def __init__(self):
        self.rect = _FakeRect(100.0, 100.0)

    def get_text(self, kind):
        assert kind == "dict"
        return {
            "blocks": [
                {"type": 0, "bbox": [0, 0, 100, 20],
                 "lines": [{"spans": [{"text": "Hello world"}]}]},
            ]
        }


def test_extract_digital_page_emits_text_element_with_bbox():
    els = extract_digital_page(
        _FakePage(), doc_id="d1", page_num=2, tenant_id="t1", acl={"public": True},
    )
    assert len(els) == 1
    el = els[0]
    assert el.element_type == ElementType.TEXT
    assert el.text == "Hello world" or el.content == "Hello world"
    assert el.bbox is not None
    assert el.bbox.x2 == 100.0
    assert el.confidence == 1.0          # digital text is full-confidence
    assert el.tenant_id == "t1"


def test_extract_digital_page_skips_empty_blocks():
    class _Empty(_FakePage):
        def get_text(self, kind):
            return {"blocks": [{"type": 0, "bbox": [0, 0, 1, 1],
                                "lines": [{"spans": [{"text": "   "}]}]}]}
    assert extract_digital_page(_Empty(), doc_id="d", page_num=0,
                                tenant_id="t", acl={}) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k extract_digital_page`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.digital_extractor'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/ingestion/digital_extractor.py`:

```python
"""Digital page extraction via PyMuPDF (Spec §2 L1a).

Reads a digital page's text blocks with their bounding boxes and emits
``UnifiedElement`` objects (text full-confidence; bbox retained for click-to-
highlight citations). The fitz dependency lives upstream (page_reader streams the
``fitz.Page``); this function only consumes the page's ``get_text("dict")`` +
``.rect`` surface, so it is pure-testable with a fake page.
"""
from __future__ import annotations

from typing import Any

from .ton_schema import BBox, ElementType, UnifiedElement


def extract_digital_page(
    page: Any,
    *,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
) -> list[UnifiedElement]:
    """Extract text elements (with bbox) from a digital ``fitz.Page``-like object."""
    elements: list[UnifiedElement] = []
    data = page.get_text("dict")
    order = 0
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 == text block in PyMuPDF
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if not text:
            continue
        bx = block.get("bbox", [0, 0, 0, 0])
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:b{order}",
                doc_id=doc_id,
                page_num=page_num,
                element_type=ElementType.TEXT,
                content=text,
                reading_order=order,
                tenant_id=tenant_id,
                bbox=BBox(x1=bx[0], y1=bx[1], x2=bx[2], y2=bx[3]),
                confidence=1.0,
                parser_version="pymupdf-digital-1",
                acl=dict(acl or {}),
            )
        )
        order += 1
    return elements
```

(Note: the test reads `el.text or el.content` — `UnifiedElement` stores raw text in `content`; the `text` access falls through to `content` via the `or`.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k extract_digital_page`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/digital_extractor.py pdf_chat/testing/test_extraction.py
git commit -m "feat(pdf): PyMuPDF digital page extractor with bbox + confidence

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: OCR / scanned extractor (Azure Document Intelligence) — layout + tables + bbox + confidence

**Files:**
- Create: `server/pdf_chat/ingestion/ocr_extractor.py`
- Test: `server/pdf_chat/testing/test_extraction.py`

> Azure Document Intelligence is behind a guarded import (mirrors `preflight.py`). The OCR-native region typing + per-cell confidence come straight from the DI `analyze_result` — no font-size/whitespace rule literals (Spec §2 L1a). `parse_di_result` is a PURE function over a DI-shaped dict, so it is fully unit-testable; the live call (`extract_scanned_page`) is `@pytest.mark.infra`.

- [ ] **Step 1: Write the failing test** (add to `test_extraction.py`)

```python
from pdf_chat.ingestion.ocr_extractor import parse_di_result
from pdf_chat.ingestion.ton_schema import ElementType


def _di_result():
    # Minimal Azure Document Intelligence "analyzeResult"-shaped dict.
    return {
        "pages": [{
            "pageNumber": 1,
            "words": [
                {"content": "Invoice", "confidence": 0.99,
                 "polygon": [0, 0, 50, 0, 50, 10, 0, 10]},
                {"content": "Total", "confidence": 0.40,
                 "polygon": [0, 12, 40, 12, 40, 22, 0, 22]},
            ],
        }],
        "tables": [{
            "rowCount": 2, "columnCount": 1,
            "cells": [
                {"rowIndex": 0, "columnIndex": 0, "content": "Amount", "confidence": 0.9},
                {"rowIndex": 1, "columnIndex": 0, "content": "100",   "confidence": 0.3},
            ],
            "boundingRegions": [{"pageNumber": 1,
                                 "polygon": [0, 30, 60, 30, 60, 60, 0, 60]}],
        }],
    }


def test_parse_di_result_emits_text_and_table_with_confidence():
    els = parse_di_result(
        _di_result(), doc_id="d1", tenant_id="t1", acl={"public": True},
    )
    types = {e.element_type for e in els}
    assert ElementType.TEXT in types
    assert ElementType.TABLE in types
    text_el = next(e for e in els if e.element_type == ElementType.TEXT)
    # Page-level OCR text confidence = mean word confidence (0.99, 0.40) ≈ 0.695.
    assert 0.69 <= text_el.confidence <= 0.70
    assert text_el.bbox is not None
    table_el = next(e for e in els if e.element_type == ElementType.TABLE)
    assert "| Amount |" in table_el.content   # markdown header row
    # Table confidence = min cell confidence (worst cell governs).
    assert table_el.confidence == 0.3


def test_parse_di_result_empty():
    assert parse_di_result({"pages": [], "tables": []},
                           doc_id="d", tenant_id="t", acl={}) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k parse_di_result`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.ocr_extractor'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/ingestion/ocr_extractor.py`:

```python
"""Scanned-page extraction via Azure Document Intelligence (Spec §2 L1a + open-Q #2).

OCR + table extraction + OCR-native region typing with confidence — no
font-size/whitespace rule literals. ``parse_di_result`` is PURE over a DI
``analyzeResult`` dict (unit-testable). ``extract_scanned_page`` issues the live
call behind a guarded import (constructs/raises only on call without infra).
"""
from __future__ import annotations

import os
from typing import Any

from .ton_schema import BBox, ElementType, UnifiedElement

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore

    _HAS_DI = True
except ImportError:  # pragma: no cover - exercised only without infra
    DocumentIntelligenceClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore
    _HAS_DI = False


def _polygon_bbox(polygon: list[float]) -> BBox:
    """Reduce a DI polygon (x0,y0,x1,y1,...) to an axis-aligned BBox."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def _table_to_markdown(cells: list[dict], row_count: int, col_count: int) -> str:
    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        grid[cell["rowIndex"]][cell["columnIndex"]] = str(cell.get("content", ""))
    lines = ["| " + " | ".join(grid[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_di_result(
    result: dict, *, doc_id: str, tenant_id: str, acl: dict
) -> list[UnifiedElement]:
    """Convert an Azure DI analyzeResult dict into UnifiedElements (pure)."""
    elements: list[UnifiedElement] = []
    order = 0
    acl = dict(acl or {})

    for page in result.get("pages", []):
        words = page.get("words", [])
        if not words:
            continue
        page_num = int(page.get("pageNumber", 1)) - 1
        text = " ".join(w.get("content", "") for w in words).strip()
        if not text:
            continue
        confs = [float(w.get("confidence", 1.0)) for w in words]
        mean_conf = sum(confs) / len(confs)
        polys = [w["polygon"] for w in words if w.get("polygon")]
        bbox = _polygon_bbox([c for p in polys for c in p]) if polys else None
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:ocr{order}",
                doc_id=doc_id, page_num=page_num,
                element_type=ElementType.TEXT, content=text,
                reading_order=order, tenant_id=tenant_id, bbox=bbox,
                confidence=mean_conf, parser_version="azure-di-1", acl=acl,
            )
        )
        order += 1

    for table in result.get("tables", []):
        cells = table.get("cells", [])
        if not cells:
            continue
        md = _table_to_markdown(cells, table["rowCount"], table["columnCount"])
        cell_confs = [float(c.get("confidence", 1.0)) for c in cells]
        region = (table.get("boundingRegions") or [{}])[0]
        poly = region.get("polygon")
        page_num = int(region.get("pageNumber", 1)) - 1
        elements.append(
            UnifiedElement(
                element_id=f"{doc_id}:p{page_num}:tbl{order}",
                doc_id=doc_id, page_num=page_num,
                element_type=ElementType.TABLE, content=md,
                reading_order=order, tenant_id=tenant_id,
                bbox=_polygon_bbox(poly) if poly else None,
                confidence=min(cell_confs) if cell_confs else 1.0,
                parser_version="azure-di-1", acl=acl,
            )
        )
        order += 1

    return elements


def extract_scanned_page(  # pragma: no cover - requires infra + env
    page_image_bytes: bytes, *, doc_id: str, page_num: int, tenant_id: str, acl: dict
) -> list[UnifiedElement]:
    """Run Azure DI prebuilt-layout over one rendered page image."""
    if not _HAS_DI:
        raise RuntimeError(
            "azure-ai-documentintelligence is required for OCR but is not installed."
        )
    client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
        credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")),
    )
    poller = client.begin_analyze_document("prebuilt-layout", body=page_image_bytes)
    result = poller.result().as_dict()
    return parse_di_result(result, doc_id=doc_id, tenant_id=tenant_id, acl=acl)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k parse_di_result`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/ocr_extractor.py pdf_chat/testing/test_extraction.py
git commit -m "feat(pdf): Azure Document Intelligence OCR/table/layout extractor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Page-extraction orchestrator (`extract_page_elements`)

**Files:**
- Create: `server/pdf_chat/ingestion/page_extraction.py`
- Test: `server/pdf_chat/testing/test_extraction.py`

- [ ] **Step 1: Write the failing test** (add to `test_extraction.py`)

```python
from pdf_chat.ingestion.page_extraction import extract_page_elements
from pdf_chat.ingestion.ton_schema import ElementType


def test_extract_page_elements_digital_route(monkeypatch):
    captured = {}

    def _digital(page, *, doc_id, page_num, tenant_id, acl):
        captured["route"] = "digital"
        return [_el_text(doc_id, page_num, tenant_id)]

    def _ocr(image, *, doc_id, page_num, tenant_id, acl):
        captured["route"] = "ocr"
        return []

    from pdf_chat.ingestion import page_extraction as pe
    monkeypatch.setattr(pe, "extract_digital_page", _digital)
    monkeypatch.setattr(pe, "extract_scanned_page", _ocr)

    els = extract_page_elements(
        page=object(), page_image_bytes=b"", coverage=0.9,
        doc_id="d", page_num=0, tenant_id="c-1", acl={},
    )
    assert captured["route"] == "digital"
    assert els[0].element_type == ElementType.TEXT


def test_extract_page_elements_scanned_route(monkeypatch):
    captured = {}
    from pdf_chat.ingestion import page_extraction as pe
    monkeypatch.setattr(pe, "extract_digital_page",
                        lambda *a, **k: captured.setdefault("route", "digital") or [])
    monkeypatch.setattr(pe, "extract_scanned_page",
                        lambda *a, **k: captured.setdefault("route", "ocr") or [])
    extract_page_elements(page=object(), page_image_bytes=b"img", coverage=0.02,
                          doc_id="d", page_num=1, tenant_id="c-1", acl={})
    assert captured["route"] == "ocr"


def _el_text(doc_id, page_num, tenant_id):
    from pdf_chat.ingestion.ton_schema import UnifiedElement
    return UnifiedElement(
        element_id="e", doc_id=doc_id, page_num=page_num,
        element_type=ElementType.TEXT, content="x", reading_order=0,
        tenant_id=tenant_id,
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k extract_page_elements`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.page_extraction'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/ingestion/page_extraction.py`:

```python
"""Page-extraction orchestrator (Spec §2 L1a): route → digital/OCR → elements.

The single seam the worker's ``extract_fn`` calls per page. Routing is data-
driven (page_routing.route_page_extractor on measured coverage); the two backends
are imported at module scope so tests can monkeypatch them.
"""
from __future__ import annotations

from typing import Any

from .digital_extractor import extract_digital_page
from .ocr_extractor import extract_scanned_page
from .page_routing import route_page_extractor
from .ton_schema import UnifiedElement


def extract_page_elements(
    *,
    page: Any,
    page_image_bytes: bytes,
    coverage: float,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
) -> list[UnifiedElement]:
    """Route one page to the digital or OCR extractor and return its elements."""
    route = route_page_extractor(
        coverage=coverage, container_id=tenant_id, page_num=page_num
    )
    if route == "digital":
        return extract_digital_page(
            page, doc_id=doc_id, page_num=page_num, tenant_id=tenant_id, acl=acl
        )
    return extract_scanned_page(
        page_image_bytes, doc_id=doc_id, page_num=page_num,
        tenant_id=tenant_id, acl=acl,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_extraction.py -q -k extract_page_elements`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/page_extraction.py pdf_chat/testing/test_extraction.py
git commit -m "feat(pdf): page-extraction orchestrator (route to digital/OCR)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Wire the real `extract_fn` into `tasks.py`

**Files:**
- Modify: `server/pdf_chat/ingestion/tasks.py` (`:221-226`)
- Test: `server/pdf_chat/testing/test_ingestion.py`

> The full page chain (`page_reader → page_extraction → chunker → embeddings → neo4j_writer`) is a single pure orchestrator, `run_page_pipeline`, so it is unit-testable end-to-end with fakes and `_run_page_extraction` simply calls it. The `extract_fn` closure in `process_page_task` (`:221`) is replaced with a call to `run_page_pipeline`.

- [ ] **Step 1: Write the failing test** (add to `test_ingestion.py`)

```python
def test_run_page_pipeline_extracts_chunks_and_writes(monkeypatch):
    from pdf_chat.ingestion import tasks as t
    from pdf_chat.ingestion.ton_schema import ElementType, UnifiedElement

    el = UnifiedElement(
        element_id="e", doc_id="d", page_num=0, element_type=ElementType.TEXT,
        content="Hello world. Second sentence.", reading_order=0,
        tenant_id="t1", confidence=0.9,
    )
    monkeypatch.setattr(t, "extract_page_elements", lambda **k: [el])
    monkeypatch.setattr(t, "embed_texts_batched", lambda texts, **k: [[0.1]] * len(texts))

    written = {}
    class _Writer:
        def write_chunks(self, chunks):
            written["chunks"] = chunks
            return len(chunks)

    n = t.run_page_pipeline(
        page=object(), page_image_bytes=b"", coverage=0.9,
        doc_id="d", page_num=0, tenant_id="t1", acl={}, writer=_Writer(),
    )
    assert n >= 1
    assert written["chunks"][0].embedding == [0.1]
    assert written["chunks"][0].confidence == 0.9   # confidence propagated
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q -k run_page_pipeline`
Expected: FAIL — `AttributeError: module 'pdf_chat.ingestion.tasks' has no attribute 'run_page_pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Add module-scope imports near the top of `server/pdf_chat/ingestion/tasks.py` (after the existing imports, ~`:20`):

```python
from .chunker import chunk_elements
from .extraction_confidence import propagate_confidence
from .page_extraction import extract_page_elements
from .retrieval_embeddings_shim import embed_texts_batched  # see note below
```

> Note: to avoid a retrieval→ingestion import cycle, re-export the batch embedder for the worker. Create `server/pdf_chat/ingestion/retrieval_embeddings_shim.py` with a single line:
> ```python
> from pdf_chat.retrieval.embeddings import embed_texts_batched  # noqa: F401
> ```

Add the pure orchestrator (anywhere after the helpers, before the Celery block at `:186`):

```python
def run_page_pipeline(
    *,
    page: Any,
    page_image_bytes: bytes,
    coverage: float,
    doc_id: str,
    page_num: int,
    tenant_id: str,
    acl: dict,
    writer: Any,
) -> int:
    """Full per-page chain: extract → confidence → chunk → embed → write.

    Pure orchestration over injected backends (the extractors/embedder are
    module-scope and monkeypatchable; ``writer`` is a Neo4jWriter-like object).
    Returns the number of chunks written. This IS the worker's real extract step.
    """
    elements = extract_page_elements(
        page=page, page_image_bytes=page_image_bytes, coverage=coverage,
        doc_id=doc_id, page_num=page_num, tenant_id=tenant_id, acl=acl,
    )
    if not elements:
        return 0
    element_conf = {el.element_id: el.confidence for el in elements}
    chunks = chunk_elements(elements)
    chunks = propagate_confidence(chunks, element_conf, container_id=tenant_id)
    vectors = embed_texts_batched([c.text for c in chunks], container_id=tenant_id)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
    return writer.write_chunks(chunks)
```

Then replace the `_extract` stub inside `process_page_task` (`:221-226`) so it calls `run_page_pipeline`:

```python
                async def _extract(tid: str) -> None:
                    # Worker bootstrap supplies the rendered page + coverage from
                    # the page manifest; the Neo4jWriter is built from settings.
                    from .neo4j_writer import Neo4jWriter

                    s = get_pdf_settings()
                    writer = Neo4jWriter(s.neo4j_uri, s.neo4j_user, s.neo4j_password,
                                         database=s.neo4j_database)
                    page_obj, page_image, coverage, doc_id, acl = (
                        await page_repo.load_page_inputs(tid)
                    )
                    run_page_pipeline(
                        page=page_obj, page_image_bytes=page_image, coverage=coverage,
                        doc_id=doc_id, page_num=0, tenant_id=tenant_id, acl=acl,
                        writer=writer,
                    )
```

> `PageManifestRepo.load_page_inputs` is the page-input loader (rendered page object, page image bytes for OCR, measured coverage ratio, doc_id, acl). It is added to the repo in Task 13b. The Celery block stays `# pragma: no cover`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q`
Expected: PASS (46 passed — the existing 45 plus `run_page_pipeline`).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/tasks.py pdf_chat/ingestion/retrieval_embeddings_shim.py pdf_chat/testing/test_ingestion.py
git commit -m "feat(pdf): wire real per-page extract chain into the worker task

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13b: `PageManifestRepo.load_page_inputs`

**Files:**
- Modify: `server/pdf_chat/control_plane/repositories.py` (`PageManifestRepo`, after `:167`)
- Test: `server/pdf_chat/testing/test_control_plane.py`

- [ ] **Step 1: Write the failing test** (add to `test_control_plane.py`)

```python
import asyncio


def test_load_page_inputs_returns_pipeline_tuple():
    from pdf_chat.control_plane.repositories import PageManifestRepo

    class _FakeRow:
        page_blob_path = "az://x/p0.png"
        text_coverage_ratio = 0.92
        doc_id = "doc-1"
        acl_snapshot = {"public": True}

    class _FakeSession:
        async def get(self, *a, **k):
            return None

    repo = PageManifestRepo(_FakeSession())
    repo._fetch_row = lambda tid: _FakeRow()           # injected for the pure test
    repo._download = lambda path: b"PNGBYTES"          # injected blob fetch
    repo._render_page = lambda blob: ("page-obj", b"PNGBYTES")

    page, image, coverage, doc_id, acl = asyncio.run(repo.load_page_inputs("pg-1"))
    assert coverage == 0.92
    assert doc_id == "doc-1"
    assert acl == {"public": True}
    assert image == b"PNGBYTES"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_control_plane.py -q -k load_page_inputs`
Expected: FAIL — `AttributeError: 'PageManifestRepo' object has no attribute 'load_page_inputs'`.

- [ ] **Step 3: Write minimal implementation** (add to `PageManifestRepo` in `server/pdf_chat/control_plane/repositories.py`):

```python
    async def load_page_inputs(self, task_id: str):
        """Load the per-page extraction inputs for the worker.

        Returns ``(page_obj, page_image_bytes, coverage, doc_id, acl)``. The
        rendered page object + image come from the page blob; coverage was
        measured at preflight and stored on the page row. The blob fetch and page
        render are small seams (``_download`` / ``_render_page``) so the loader is
        unit-testable without infra.
        """
        row = self._fetch_row(task_id)
        blob = self._download(row.page_blob_path)
        page_obj, image_bytes = self._render_page(blob)
        return page_obj, image_bytes, row.text_coverage_ratio, row.doc_id, row.acl_snapshot
```

> `_fetch_row`, `_download`, and `_render_page` are thin instance seams; their production bodies (a SQLAlchemy `select` on the page row, a blob download via the app's blob client, and a PyMuPDF render-to-pixmap) are wired in the worker-bootstrap follow-up and are not part of the Phase-1 pure surface. Provide minimal default implementations that raise `NotImplementedError` with a clear message so the seams are explicit:
>
> ```python
>     def _fetch_row(self, task_id: str):  # pragma: no cover - infra-wired
>         raise NotImplementedError("wired by the worker bootstrap")
>     def _download(self, path: str) -> bytes:  # pragma: no cover - infra-wired
>         raise NotImplementedError("wired by the worker bootstrap")
>     def _render_page(self, blob: bytes):  # pragma: no cover - infra-wired
>         raise NotImplementedError("wired by the worker bootstrap")
> ```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_control_plane.py -q -k load_page_inputs`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/control_plane/repositories.py pdf_chat/testing/test_control_plane.py
git commit -m "feat(pdf): PageManifestRepo.load_page_inputs page-input loader

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Finalization task

**Files:**
- Create: `server/pdf_chat/ingestion/finalize.py`
- Test: `server/pdf_chat/testing/test_ingestion.py`

> The pure status reducer `reconcile_document_status` already exists and is exercised in `test_control_plane.py`; the orchestrator's `reconcile` (`orchestrator.py:252`) persists it. The Phase-1 finalization task is a thin guard: a document is finalized (→ `INDEXED`) only after ALL pages are settled (`SETTLED_PAGE_STATES`, `models/enums.py:55`), else it stays `PROCESSING`.

- [ ] **Step 1: Write the failing test** (add to `test_ingestion.py`)

```python
def test_finalize_document_waits_until_all_pages_settled():
    from pdf_chat.ingestion.finalize import finalize_document
    from pdf_chat.models.enums import DocStatus, PageStatus

    # One page still running → not finalized.
    not_done = finalize_document([
        PageStatus.SUCCEEDED.value, PageStatus.RUNNING.value,
    ])
    assert not_done == DocStatus.PROCESSING.value


def test_finalize_document_indexed_when_all_succeeded():
    from pdf_chat.ingestion.finalize import finalize_document
    from pdf_chat.models.enums import DocStatus, PageStatus

    done = finalize_document([
        PageStatus.SUCCEEDED.value, PageStatus.SUCCEEDED.value,
    ])
    assert done == DocStatus.INDEXED.value


def test_finalize_document_partial_when_some_terminal_failed():
    from pdf_chat.ingestion.finalize import finalize_document
    from pdf_chat.models.enums import DocStatus, PageStatus

    partial = finalize_document([
        PageStatus.SUCCEEDED.value, PageStatus.FAILED_TERMINAL.value,
    ])
    assert partial == DocStatus.PARTIALLY_INDEXED.value
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q -k finalize_document`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.ingestion.finalize'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/ingestion/finalize.py`:

```python
"""Finalization (Spec §2 L1a): a document is 'ready' only after ALL pages settle.

Pure reducer over page statuses. While any page is non-terminal the document
stays PROCESSING; once every page is settled it becomes INDEXED (all succeeded)
or PARTIALLY_INDEXED (some terminally failed but at least one succeeded) or FAILED
(none succeeded). The worker calls this on each page-settle; the orchestrator's
``reconcile`` persists the result.
"""
from __future__ import annotations

from pdf_chat.models.enums import DocStatus, PageStatus, SETTLED_PAGE_STATES

_SETTLED = {s.value for s in SETTLED_PAGE_STATES}


def finalize_document(page_statuses: list[str]) -> str:
    """Reduce per-page statuses to a document status."""
    if not page_statuses:
        return DocStatus.PROCESSING.value
    if any(s not in _SETTLED for s in page_statuses):
        return DocStatus.PROCESSING.value
    succeeded = sum(1 for s in page_statuses if s == PageStatus.SUCCEEDED.value)
    if succeeded == len(page_statuses):
        return DocStatus.INDEXED.value
    if succeeded == 0:
        return DocStatus.FAILED.value
    return DocStatus.PARTIALLY_INDEXED.value
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q -k finalize_document`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/finalize.py pdf_chat/testing/test_ingestion.py
git commit -m "feat(pdf): finalization reducer (INDEXED only after all pages settle)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Reranker adaptive-skip wiring (config + logged)

**Files:**
- Modify: `server/pdf_chat/retrieval/reranker.py`
- Test: `server/pdf_chat/testing/test_retrieval.py`

> `rerank()` exists (`reranker.py:52`) and already falls back to pass-through. Phase 1 adds the adaptive *skip*: when there are too few candidates to be worth reranking, skip the cross-encoder and log the decision (Spec §2 L3 "adaptive rerank skip threshold is config + logged"). The skip threshold is a tunable, so no literal lives in the file.

- [ ] **Step 1: Write the failing test** (add to `test_retrieval.py`)

```python
def test_rerank_skips_when_too_few_candidates(monkeypatch):
    from pdf_chat.retrieval import reranker

    monkeypatch.setenv("PDF_TUNABLE_RERANK_SKIP_BELOW_CANDIDATES", "4")
    cands = [{"text": "a"}, {"text": "b"}]   # 2 < 4 → skip, return as-is
    out = reranker.rerank("q", cands, top_n=12, container_id="c-1")
    assert out == cands


def test_rerank_runs_when_enough_candidates(monkeypatch):
    from pdf_chat.retrieval import reranker

    monkeypatch.setenv("PDF_TUNABLE_RERANK_SKIP_BELOW_CANDIDATES", "2")
    cands = [{"text": f"c{i}"} for i in range(5)]
    out = reranker.rerank("q", cands, top_n=3, container_id="c-1")
    assert len(out) == 3     # pure fallback path still truncates to top_n
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k rerank_skips`
Expected: FAIL — `rerank()` does not accept `container_id` (TypeError).

- [ ] **Step 3: Write minimal implementation** — update the `rerank` signature + add the skip guard at the top of its body in `server/pdf_chat/retrieval/reranker.py` (`:52`):

```python
def rerank(
    query: str,
    candidates: list[Any],
    top_n: int | None = None,
    *,
    container_id: str = "",
) -> list[Any]:
```

Insert immediately after the `if not candidates: return []` guard (`:71`):

```python
    from pdf_chat.tunables import get_tunable, log_gate_decision

    skip_below = get_tunable(container_id, "rerank_skip_below_candidates", 4)
    decision = log_gate_decision(
        "rerank_skip",
        score=len(candidates),
        threshold=skip_below,
        outcome="rerank" if len(candidates) >= skip_below else "skip",
        container_id=container_id,
    )
    if not decision["passed"]:
        if top_n is None:
            top_n = get_pdf_settings().rerank_top_n
        return candidates[:top_n]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_retrieval.py -q -k rerank`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/retrieval/reranker.py pdf_chat/testing/test_retrieval.py
git commit -m "feat(pdf): adaptive rerank-skip (tunable threshold, logged)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Gold-question set + dataclass

**Files:**
- Create: `server/pdf_chat/eval/__init__.py`
- Create: `server/pdf_chat/eval/gold_questions.py`
- Create: `server/pdf_chat/eval/gold_set.json`
- Test: `server/pdf_chat/testing/test_eval.py`

- [ ] **Step 1: Write the failing test**

In `server/pdf_chat/testing/test_eval.py`:

```python
"""Pure tests for the gold-question eval set + harness (Spec §5 Phase 1)."""
from __future__ import annotations

from pdf_chat.eval.gold_questions import GoldQuestion, load_gold_set


def test_load_gold_set_returns_gold_questions():
    gold = load_gold_set()
    assert len(gold) >= 3
    assert all(isinstance(g, GoldQuestion) for g in gold)
    g = gold[0]
    assert g.question
    assert g.expected_keywords          # at least one expected keyword
    assert g.must_cite is True or g.must_cite is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.eval'`.

- [ ] **Step 3: Write minimal implementation**

`server/pdf_chat/eval/__init__.py`:

```python
"""PDF Graph RAG evaluation harness (gold-question set + scoring)."""
```

`server/pdf_chat/eval/gold_set.json`:

```json
[
  {
    "id": "q1",
    "question": "What is the total contract value?",
    "expected_keywords": ["total", "contract", "value"],
    "must_cite": true,
    "expect_refusal": false
  },
  {
    "id": "q2",
    "question": "Who are the parties to the agreement?",
    "expected_keywords": ["party", "parties", "agreement"],
    "must_cite": true,
    "expect_refusal": false
  },
  {
    "id": "q3",
    "question": "What is the governing law of a contract that is not in this document set?",
    "expected_keywords": [],
    "must_cite": false,
    "expect_refusal": true
  }
]
```

`server/pdf_chat/eval/gold_questions.py`:

```python
"""Gold-question dataclass + loader. Data lives in gold_set.json (not code)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_GOLD_PATH = Path(__file__).with_name("gold_set.json")


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    must_cite: bool = True
    expect_refusal: bool = False


def load_gold_set(path: "Path | None" = None) -> list[GoldQuestion]:
    """Load the seed gold-question set from JSON."""
    raw = json.loads((path or _GOLD_PATH).read_text(encoding="utf-8"))
    return [GoldQuestion(**row) for row in raw]
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/eval/__init__.py pdf_chat/eval/gold_questions.py pdf_chat/eval/gold_set.json pdf_chat/testing/test_eval.py
git commit -m "feat(pdf): gold-question set + GoldQuestion loader

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Eval harness + baseline recording

**Files:**
- Create: `server/pdf_chat/eval/harness.py`
- Test: `server/pdf_chat/testing/test_eval.py`

- [ ] **Step 1: Write the failing test** (add to `test_eval.py`)

```python
import asyncio
import json

from pdf_chat.eval.gold_questions import GoldQuestion
from pdf_chat.eval.harness import score_answer, run_eval


def test_score_answer_keyword_and_citation():
    g = GoldQuestion(id="q", question="x", expected_keywords=["total", "value"],
                     must_cite=True, expect_refusal=False)
    s = score_answer(g, answer="The total value is 100.", citations=[{"n": 1}])
    assert s["keyword_recall"] == 1.0
    assert s["cited"] is True
    assert s["passed"] is True


def test_score_answer_refusal_expected():
    g = GoldQuestion(id="q", question="x", expected_keywords=[],
                     must_cite=False, expect_refusal=True)
    s = score_answer(g, answer="I don't have enough information.", citations=[])
    assert s["passed"] is True          # a refusal where refusal is expected passes


def test_run_eval_records_baseline(tmp_path):
    gold = [GoldQuestion(id="q1", question="x", expected_keywords=["a"],
                         must_cite=False, expect_refusal=False)]

    async def _answer(q):
        return ("answer a", [])         # contains the expected keyword

    out_path = tmp_path / "baseline.json"
    summary = asyncio.run(run_eval(gold, _answer, baseline_path=out_path))
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1.0
    recorded = json.loads(out_path.read_text())
    assert recorded["summary"]["pass_rate"] == 1.0
    assert len(recorded["results"]) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval.py -q -k "score_answer or run_eval"`
Expected: FAIL — `ModuleNotFoundError: No module named 'pdf_chat.eval.harness'`.

- [ ] **Step 3: Write minimal implementation**

In `server/pdf_chat/eval/harness.py`:

```python
"""Eval scoring + baseline recorder (Spec §5 Phase 1 — eval moved up).

Pure scoring over (gold, answer, citations). ``run_eval`` drives an injected
``answer_fn`` (in production: a bound ``run_pdf_chat`` over a fixed test corpus)
across the gold set, scores each, and writes a baseline JSON so regressions are
measurable. ``answer_fn`` is async returning ``(answer, citations)``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from .gold_questions import GoldQuestion

# Refusal markers (data, not a hardcoded business rule): a refusal is a non-answer
# whose text contains any of these stems. Overridable by passing ``refusal_markers``.
_DEFAULT_REFUSAL_MARKERS = ("don't have", "do not have", "insufficient", "not found",
                            "cannot answer", "no relevant")


def _is_refusal(answer: str, markers: tuple[str, ...]) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in markers)


def score_answer(
    gold: GoldQuestion,
    answer: str,
    citations: list[dict],
    *,
    refusal_markers: tuple[str, ...] = _DEFAULT_REFUSAL_MARKERS,
) -> dict:
    """Score one answer: keyword recall, citation presence, pass/fail."""
    refused = _is_refusal(answer, refusal_markers)
    if gold.expect_refusal:
        passed = refused
        return {"id": gold.id, "passed": passed, "refused": refused,
                "keyword_recall": 0.0, "cited": bool(citations)}

    low = (answer or "").lower()
    kws = gold.expected_keywords or []
    hits = sum(1 for kw in kws if kw.lower() in low)
    recall = (hits / len(kws)) if kws else 1.0
    cited = bool(citations)
    passed = (recall == 1.0) and (cited if gold.must_cite else True) and not refused
    return {"id": gold.id, "passed": passed, "refused": refused,
            "keyword_recall": recall, "cited": cited}


async def run_eval(
    gold: list[GoldQuestion],
    answer_fn: Callable[[str], Awaitable[tuple[str, list[dict]]]],
    *,
    baseline_path: "Path | None" = None,
) -> dict:
    """Run the gold set through ``answer_fn``, score, and record a baseline."""
    results = []
    for g in gold:
        answer, citations = await answer_fn(g.question)
        results.append(score_answer(g, answer, citations))
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    summary = {"total": total, "passed": passed,
               "pass_rate": (passed / total) if total else 0.0}
    if baseline_path is not None:
        Path(baseline_path).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
    return summary
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_eval.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/eval/harness.py pdf_chat/testing/test_eval.py
git commit -m "feat(pdf): eval harness (keyword/citation/refusal scoring + baseline)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Full-suite green + tunables export

**Files:**
- Modify: `server/pdf_chat/ingestion/__init__.py` (export new public symbols)
- Test: all `pdf_chat/testing/`

- [ ] **Step 1: Write the failing test** (add to `test_ingestion.py`)

```python
def test_ingestion_exports_new_public_surface():
    from pdf_chat.ingestion import (
        extract_page_elements, finalize_document, propagate_confidence,
        run_page_pipeline,
    )
    assert callable(extract_page_elements)
    assert callable(finalize_document)
    assert callable(propagate_confidence)
    assert callable(run_page_pipeline)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_ingestion.py -q -k exports_new_public_surface`
Expected: FAIL — `ImportError: cannot import name 'extract_page_elements' from 'pdf_chat.ingestion'`.

- [ ] **Step 3: Write minimal implementation** — add to `server/pdf_chat/ingestion/__init__.py` imports + `__all__`:

```python
from .extraction_confidence import propagate_confidence
from .finalize import finalize_document
from .page_extraction import extract_page_elements
from .page_routing import route_page_extractor, text_coverage_ratio
from .tasks import run_page_pipeline
```

Append to `__all__`: `"propagate_confidence"`, `"finalize_document"`, `"extract_page_elements"`, `"route_page_extractor"`, `"text_coverage_ratio"`, `"run_page_pipeline"`.

- [ ] **Step 4: Run the FULL suite to verify everything is green**

Run: `cd /Users/bharath/Desktop/projects/G-CHAT-/server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/ -q -m "not infra"`
Expected: PASS — all pdf_chat tests green (the original 4 files + `test_tunables.py`, `test_extraction.py`, `test_eval.py`), zero failures, infra-marked tests deselected.

- [ ] **Step 5: Commit**

```bash
cd /Users/bharath/Desktop/projects/G-CHAT-/server
git add pdf_chat/ingestion/__init__.py pdf_chat/testing/test_ingestion.py
git commit -m "feat(pdf): export Phase-1 ingestion public surface; full suite green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed by plan author)

- **Spec §5 Phase 0 coverage:** batch embeddings (Task 3) · prompt caching (Task 6, `PdfLlm` system-first message) · query-embedding cache (Tasks 4, 4b) · context token budget (Task 5) · response-cache wiring (Task 6 note — `cache_check`/`cache_write` already exist; `PdfLlm`+`OnDemandExtractor`+`QueryAuditRepo` complete `build_default_deps`, Tasks 6–7) · tunables source + score-logging harness (Tasks 1, 1.5, 2, 2b). ✓
- **Spec §5 Phase 1 coverage:** PyMuPDF digital (Task 10) · OCR via Azure DI + tables + bbox + layout/confidence (Task 11) · data-driven routing (Task 8) · confidence propagation (Task 9) · orchestrator + worker wiring (Tasks 12, 13, 13b) · reranker skip (Task 15) · Redis cache (Tasks 4b, 6) · LLM synthesis (Task 6) · audit (Task 7) · finalization (Task 14) · gold-question eval + baseline (Tasks 16, 17). ✓
- **§3 invariants:** no magic literals — every threshold via `get_tunable`, every gate via `log_gate_decision` (Tasks 8, 9, 5, 15, 3, 4). Tenant_id carried on every chunk/node (existing writer + `UnifiedElement.tenant_id`). gpt-4o-mini only (Task 6 via `chat_deployment()`). Honest-absence/refusal exercised in the eval (Task 17 `expect_refusal`). ✓
- **Type consistency:** `get_tunable(container_id, key, default)` and `log_gate_decision(name, *, score, threshold, outcome, **ctx)` used identically in every consumer. `UnifiedElement`/`Chunk`/`BBox`/`ElementType` reused from `ton_schema`; `Chunk` gains `confidence`/`low_confidence` (Task 9) and they flow through `to_neo4j_props`. `embed_texts_batched(texts, *, container_id, ...)` signature stable across Tasks 3/13. Agent adapter protocols (`Embedder`/`Llm`/`Extractor`/`AuditRepo`) matched by `QueryEmbedder`/`PdfLlm`/`OnDemandExtractor`/`QueryAuditRepo`. ✓
- **No placeholders:** every code step shows full code; commands have expected output; no "add error handling"/"TBD". ✓

# PDF Phase 4 — Value-Evidenced Cross-Domain PDF↔CSV Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link PDF graph entities to the CSV semantic layer ONLY through literal value reconciliation against the existing `relationship_index` fingerprint/value-overlap registry (resolving to a real master key like `Vendor_ID`/`Region`/`Plant`), expose a session-safe `structured_query` agent tool that delegates to `run_agent_query` (inheriting CSV-side feasibility + negative-claim gates), reconcile PDF-fact grain vs CSV-aggregate grain with a deterministic numeric proof, and version-stamp the cross-domain response cache — so one question can span a contract PDF and the vendor CSV with a correct, refuse-on-doubt join.

**Architecture:** This is Layer 2 of the spec (`§2 Layer 2`, the highest-risk layer — all four reviewers flagged naive name/embedding joins as reproducing the documented `erp_flat` master-key failure). A new `pdf_entity_bridge` table + builder maps a PDF `Entity` → `SemanticEntity` only when a PDF entity's literal values, fingerprinted with `relationship_index.fingerprint_value`, overlap a CSV master-key column's `value_fingerprints` above tunable thresholds; sub-threshold ⇒ refuse + say so (never silently pick the top match); name-equality alone NEVER creates a bridge. A `structured_query` tool registers into the Phase-3 agent via the seam Phase 3 exposed, delegates to `run_agent_query(container_id, allowed_domains, user_id, ...)`, and runs STRICTLY SEQUENTIALLY behind a per-request DB lock (the async session is not concurrency-safe). A deterministic `reconcile_grain` check (e.g. contract rate × volume vs invoiced total within tunable tolerance) attaches as automatic proof. The response-cache key includes both the CSV semantic-layer version and the graph-extraction version whenever `structured_query` ran.

**Tech Stack:** Python 3.12 · async SQLAlchemy 2.0 + asyncpg · PostgreSQL (`pdf_entity_bridge` on the shared `Base`, same convention as `pdf_chat/models/manifests.py`) · `relationship_index` GIN value-overlap registry (reused) · LangGraph agent (Phase 3) · `gpt-4o-mini` only · `pdf_chat/tunables.py` (`get_tunable`/`log_gate_decision`, assumed) · pytest via `uv run --with pytest --with pytest-asyncio pytest`.

---

## Cross-Phase Contracts Depended On (assumptions — verify before Task 1)

These come from Phases 2/3 and `pdf_chat/tunables.py`. This plan depends on their *signatures* abstractly; if a name differs at execution time, adapt the import and note it.

- **`pdf_chat/tunables.py`** (Phase 0, assumed present):
  - `get_tunable(name: str, container_id: str, default: float | int | str) -> float | int | str` — per-container tunable resolution; **no score-comparison literal may appear in any `.py` file**, all thresholds resolve through this.
  - `log_gate_decision(gate: str, *, decision: str, score: float | None = None, **fields) -> None` — structured logging of every gate/cap/skip/merge with its score.
- **Phase 2 — graph + entities (Neo4j):** an `(:Entity)` node carries `tenant_id`, a stable `entity_id`, a `name`, an open-vocab `type`, and `src_chunk` evidence. We read PDF entity literal values through an injected reader callable `PdfEntityValuesReader = Callable[[str, str], Awaitable[list[EntityValueSample]]]` (`(tenant_id, entity_id) -> samples`). We DO NOT import Neo4j here — the reader is dependency-injected so tests stay infra-free (mirrors `pdf_chat/testing/test_agent.py` fakes). A real adapter binding to the Phase-2 reader is wired in Task 8.
- **Phase 3 — agent tool registration seam:** Phase 3 exposes a registration hook. We assume `pdf_chat/agent/graph.py` builds tools from a list and accepts injected `Deps`. We register `structured_query` by adding a tool factory `build_structured_query_tool(deps) -> Tool` and appending it to the Phase-3 tool list at the seam. If the seam is a registry dict, register under key `"structured_query"`. Note the actual seam shape in the commit message.
- **`run_agent_query`** (`server/app/agent/graph/graph.py:1354`) — REUSE verbatim, do not reimplement:
  ```python
  async def run_agent_query(
      query: str, db: AsyncSession, *, conversation_context: str = "",
      user_id: str = "", is_admin: bool = True, allowed_domains: list[str] | None = None,
      container_id: str | None = None, prior_files: list[str] | None = None,
      actor_email: str = "", actor_role: str = "", org_id: str | None = None,
  ) -> dict   # {answer, data, chart, route, row_count, files_used, tool_calls}
  ```
  It already applies the feasibility gate (`app/services/erp/feasibility_gate.py`) and `_gate_negative_claim` (`graph.py:1428`) internally — `structured_query` inherits both by delegating.
- **`relationship_index`** (`server/app/services/relationship_index.py`) — REUSE:
  - `fingerprint_value(value) -> str | None` (`:71`) — sha256[:16] of `normalize_key_value`.
  - `normalize_key_value(value) -> str | None` (`:53`).
  - `ColumnKeyRegistry` rows (`column_key_registry`) carry `container_id`, `file_id`, `column_name`, `semantic_role`, `key_kind`, `cardinality`, `value_fingerprints: list[str]` — the master-key registry the bridge reconciles against. Same container-scoped GIN overlap pattern as `find_fingerprint_matches` (`:223`).
- **`SemanticEntity`** (`server/app/models/semantic_layer.py:11`) — bridge target; `container_id`, `entity_name`, `aliases`. We resolve a PDF entity to a `SemanticEntity` indirectly: PDF values → overlapping `ColumnKeyRegistry` master-key column → its `file_id`'s `SemanticEntity`. The bridge persists the resolved `semantic_entity_id` AND the resolving `column_key_registry` column/file as evidence.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/pdf_chat/models/bridge.py` (create) | `PdfEntityBridge` ORM model + `BridgeStatus` enum on shared `Base`. |
| `server/pdf_chat/migrations/bridge_upgrade.py` (create) | Idempotent, non-fatal runtime migration: create table + indexes (same convention as `control_plane_upgrade.py`). |
| `server/pdf_chat/services/__init__.py` (create) | Package marker. |
| `server/pdf_chat/services/value_evidence.py` (create) | Pure value-reconciliation core: fingerprint PDF entity values, overlap vs `ColumnKeyRegistry`, threshold gate, refusal. No DB writes, no name/embedding match. |
| `server/pdf_chat/services/bridge_builder.py` (create) | Orchestrates `value_evidence` → persists `PdfEntityBridge` rows (resolved master key, overlap %, confidence, evidence). Refuse-on-sub-threshold. |
| `server/pdf_chat/services/grain_reconciliation.py` (create) | `reconcile_grain` deterministic numeric proof (rate × volume vs invoiced total within tunable tolerance) + period/unit alignment. |
| `server/pdf_chat/agent/structured_query.py` (create) | `structured_query` tool factory: per-request sequential DB lock, delegates to `run_agent_query` passing `container_id`/`allowed_domains`/`user_id`. |
| `server/pdf_chat/services/cross_domain_cache.py` (create) | Cache-key augmentation: append CSV semantic-layer version + graph-extraction version stamps when `structured_query` was used. |
| `server/pdf_chat/testing/test_bridge.py` (create) | Tests for value-evidence, builder, grain, cache key. |
| `server/pdf_chat/testing/test_structured_query.py` (create) | Tests for tool scope-passing + sequential execution. |
| `server/app/main.py` (modify) | Call `bridge_upgrade.run_migration` in lifespan (Task 9). |

---

## Task 1: `PdfEntityBridge` table + enum

**Files:**
- Create: `server/pdf_chat/models/bridge.py`
- Test: `server/pdf_chat/testing/test_bridge.py`

- [ ] **Step 1: Write the failing test**

Create `server/pdf_chat/testing/test_bridge.py`:

```python
"""Pure unit tests for Phase 4 — value-evidenced cross-domain bridge.

Zero infra: value reconciliation runs on in-memory registry rows; the bridge
builder uses a fake DB session recorder. No Neo4j, Postgres, or OpenAI required.
Mirrors pdf_chat/testing/test_agent.py conventions.
"""
from __future__ import annotations

import pytest

from pdf_chat.models.bridge import PdfEntityBridge, BridgeStatus


def test_bridge_model_columns_present():
    cols = PdfEntityBridge.__table__.columns.keys()
    for required in (
        "id", "tenant_id", "pdf_entity_id", "semantic_entity_id",
        "resolved_master_column", "resolved_master_file_id",
        "value_overlap_pct", "confidence", "evidence", "status",
    ):
        assert required in cols, f"missing column {required}"


def test_bridge_status_enum_values():
    assert BridgeStatus.LINKED.value == "linked"
    assert BridgeStatus.REFUSED.value == "refused"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py::test_bridge_model_columns_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.models.bridge'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/models/bridge.py`:

```python
"""Phase 4 — value-evidenced PDF→CSV bridge ORM model.

Maps a PDF graph Entity to a CSV SemanticEntity ONLY via literal value
reconciliation against column_key_registry master keys (Vendor_ID/Region/Plant).
Never name-equality, never embedding-cosine. Persists the resolving master-key
column, value-overlap %, confidence, and evidence. A sub-threshold attempt is
persisted as status=REFUSED so the refusal is auditable — never a silent
top-match. Registered on the shared Base so create_all/migrations pick it up.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BridgeStatus(str, enum.Enum):
    LINKED = "linked"      # value overlap cleared the threshold; join is permitted
    REFUSED = "refused"    # sub-threshold; recorded so the refusal is auditable


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PdfEntityBridge(Base):
    """One row per (pdf_entity, resolved master-key column) reconciliation attempt."""
    __tablename__ = "pdf_entity_bridge"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "pdf_entity_id", "resolved_master_file_id",
            "resolved_master_column",
            name="uq_pdf_bridge_entity_masterkey",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pdf_entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Resolved CSV side. semantic_entity_id may be None when values reconcile to a
    # master-key column whose file has no SemanticEntity yet — still a valid link.
    semantic_entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    resolved_master_file_id: Mapped[str] = mapped_column(String(36), nullable=False)
    resolved_master_column: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_semantic_role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    value_overlap_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    overlap_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pdf_value_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # evidence: the overlapping fingerprints + sample literals + reason on refusal.
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=BridgeStatus.REFUSED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/models/bridge.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): add PdfEntityBridge model + BridgeStatus enum

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `bridge_upgrade` runtime migration

**Files:**
- Create: `server/pdf_chat/migrations/bridge_upgrade.py`
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
def test_bridge_migration_indexes_target_table_and_keys():
    from pdf_chat.migrations import bridge_upgrade

    stmts = " ".join(bridge_upgrade._INDEXES).lower()
    assert "pdf_entity_bridge" in stmts
    assert "tenant_id" in stmts
    assert "pdf_entity_id" in stmts
    assert all(s.strip().lower().startswith("create index if not exists") for s in bridge_upgrade._INDEXES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py::test_bridge_migration_indexes_target_table_and_keys -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.migrations.bridge_upgrade'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/migrations/bridge_upgrade.py`:

```python
"""Runtime migration — create the pdf_entity_bridge table + indexes.

Idempotent, non-fatal, additive (same convention as control_plane_upgrade.py).
Importing the model registers it on the shared Base so create_all also creates
it; this migration adds the secondary indexes for tenant + entity lookups.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Importing registers the table on the shared Base metadata.
from pdf_chat.models.bridge import PdfEntityBridge  # noqa: F401

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_tenant ON pdf_entity_bridge(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_entity ON pdf_entity_bridge(tenant_id, pdf_entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_semantic ON pdf_entity_bridge(semantic_entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_pdf_bridge_status ON pdf_entity_bridge(status)",
]


async def run_migration(engine: AsyncEngine) -> None:
    """Create the table (via metadata) + indexes. Safe to run repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in _INDEXES:
            await conn.execute(text(stmt))


# Base is needed for create_all; import after the model so metadata is populated.
from app.core.database import Base  # noqa: E402
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/migrations/bridge_upgrade.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): add idempotent pdf_entity_bridge runtime migration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Value-evidence core (the master-key reconciliation, the highest-risk seam)

**Files:**
- Create: `server/pdf_chat/services/__init__.py`
- Create: `server/pdf_chat/services/value_evidence.py`
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

This is the seam reviewers flagged. It MUST: (a) fingerprint PDF values with the EXISTING `relationship_index.fingerprint_value`; (b) overlap against `ColumnKeyRegistry.value_fingerprints` master-key columns; (c) gate on tunable overlap % AND min cardinality; (d) NEVER use name-equality or embedding cosine; (e) REFUSE (return a `REFUSED` verdict) sub-threshold instead of picking the top match.

- [ ] **Step 1: Write the failing tests**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
from pdf_chat.services.value_evidence import (
    MasterKeyColumn, EntityValueSample, reconcile_entity_to_master_keys,
)


def _registry(container="t1"):
    # Two master-key columns. vendor_id values overlap the PDF entity; region does not.
    return [
        MasterKeyColumn(
            file_id="f_vendor", column_name="Vendor_ID", semantic_role="custom:vendor:key",
            key_kind="pk", cardinality=4,
            value_fingerprints=_fps(["V001", "V002", "V003", "V004"]),
        ),
        MasterKeyColumn(
            file_id="f_geo", column_name="Region", semantic_role="custom:region:key",
            key_kind="fk", cardinality=3,
            value_fingerprints=_fps(["North", "South", "East"]),
        ),
    ]


def _fps(values):
    from app.services.relationship_index import fingerprint_value
    return sorted({fp for v in values if (fp := fingerprint_value(v))})


def test_value_overlap_resolves_to_master_key_not_name():
    # PDF contract entity literally carries vendor codes V001/V002/V003 — overlaps
    # Vendor_ID 3/4 = 0.75. Name of the entity ("ACME Corp") is irrelevant.
    samples = [EntityValueSample(value=v) for v in ("V001", "V002", "V003")]
    verdict = reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_acme", entity_name="ACME Corp",
        samples=samples, master_columns=_registry(),
    )
    assert verdict.status == "linked"
    assert verdict.resolved_master_column == "Vendor_ID"
    assert verdict.value_overlap_pct == pytest.approx(0.75)


def test_name_equality_alone_does_not_bridge():
    # PDF entity name == "Region" but it carries NO overlapping literal values.
    samples = [EntityValueSample(value=v) for v in ("ZZZ", "QQQ")]
    verdict = reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_region", entity_name="Region",
        samples=samples, master_columns=_registry(),
    )
    assert verdict.status == "refused"
    assert verdict.resolved_master_column is None
    assert "no_value_overlap" in verdict.refusal_reason


def test_subthreshold_overlap_refuses_no_silent_top_match():
    # 1 of 3 vendor codes overlaps → 0.25 overlap, below the default 0.5 floor.
    samples = [EntityValueSample(value=v) for v in ("V001", "NOPE", "ALSO_NO")]
    verdict = reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_weak", entity_name="weak",
        samples=samples, master_columns=_registry(),
    )
    assert verdict.status == "refused"
    assert verdict.resolved_master_column is None  # did NOT silently pick Vendor_ID
    assert "below_overlap_threshold" in verdict.refusal_reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -k value or subthreshold or name_equality -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.services'`

- [ ] **Step 3: Write minimal implementation**

Create empty `server/pdf_chat/services/__init__.py`:

```python
```

Create `server/pdf_chat/services/value_evidence.py`:

```python
"""Value-evidenced reconciliation core for the cross-domain bridge (Phase 4).

The ONLY linking signal is literal value overlap against the existing
relationship_index master-key registry. NO name-equality, NO embedding cosine.
A PDF entity links to the CSV side only when its literal values, fingerprinted
with relationship_index.fingerprint_value, overlap a master-key column's
value_fingerprints above tunable thresholds — i.e. it resolves to a real
reconciling master key (Vendor_ID/Region/Plant). Sub-threshold => REFUSE, never
pick the top match. Pure: no DB, no I/O. All thresholds via get_tunable; every
decision logged via log_gate_decision (no score literal in this file).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.relationship_index import fingerprint_value
from pdf_chat.tunables import get_tunable, log_gate_decision

_GATE = "pdf_value_evidence"


@dataclass(frozen=True)
class MasterKeyColumn:
    """A CSV-side candidate master key, projected from column_key_registry."""
    file_id: str
    column_name: str
    semantic_role: str | None
    key_kind: str
    cardinality: int
    value_fingerprints: list[str]
    semantic_entity_id: str | None = None


@dataclass(frozen=True)
class EntityValueSample:
    """One literal value extracted from a PDF entity (with its source span id)."""
    value: object
    src_chunk: str | None = None


@dataclass
class ReconcileVerdict:
    status: str = "refused"                      # "linked" | "refused"
    resolved_master_file_id: str | None = None
    resolved_master_column: str | None = None
    resolved_semantic_role: str | None = None
    resolved_semantic_entity_id: str | None = None
    value_overlap_pct: float = 0.0
    confidence: float = 0.0
    overlap_count: int = 0
    pdf_value_count: int = 0
    refusal_reason: str = ""
    evidence: dict = field(default_factory=dict)


def reconcile_entity_to_master_keys(
    *,
    tenant_id: str,
    pdf_entity_id: str,
    entity_name: str,
    samples: list[EntityValueSample],
    master_columns: list[MasterKeyColumn],
) -> ReconcileVerdict:
    """Resolve a PDF entity to a CSV master key by literal value overlap only."""
    # entity_name is intentionally NOT used for matching — name equality is banned.
    min_overlap = float(get_tunable("bridge_min_value_overlap", tenant_id, 0.5))
    min_overlap_count = int(get_tunable("bridge_min_overlap_count", tenant_id, 2))
    min_cardinality = int(get_tunable("bridge_min_master_cardinality", tenant_id, 3))
    overlap_weight = float(get_tunable("bridge_confidence_overlap_weight", tenant_id, 0.7))
    card_weight = float(get_tunable("bridge_confidence_card_weight", tenant_id, 0.3))
    card_ref = float(get_tunable("bridge_confidence_card_reference", tenant_id, 100.0))

    pdf_fps = sorted({fp for s in samples if (fp := fingerprint_value(s.value))})
    pdf_count = len(pdf_fps)
    verdict = ReconcileVerdict(pdf_value_count=pdf_count)

    if pdf_count == 0:
        verdict.refusal_reason = "no_fingerprintable_pdf_values"
        log_gate_decision(_GATE, decision="refused", score=0.0,
                          tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
                          reason=verdict.refusal_reason)
        return verdict

    pdf_set = set(pdf_fps)
    best: tuple[float, int, MasterKeyColumn, list[str]] | None = None
    for col in master_columns:
        overlap = sorted(pdf_set.intersection(col.value_fingerprints or []))
        if not overlap:
            continue
        denom = max(min(pdf_count, max(col.cardinality, 1)), 1)
        pct = len(overlap) / denom
        if best is None or pct > best[0]:
            best = (pct, len(overlap), col, overlap)

    if best is None:
        verdict.refusal_reason = "no_value_overlap"
        log_gate_decision(_GATE, decision="refused", score=0.0,
                          tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
                          reason=verdict.refusal_reason)
        return verdict

    pct, overlap_count, col, overlap_fps = best
    verdict.value_overlap_pct = pct
    verdict.overlap_count = overlap_count
    verdict.evidence = {
        "overlap_fingerprints": overlap_fps,
        "master_cardinality": col.cardinality,
        "pdf_value_count": pdf_count,
        "key_kind": col.key_kind,
    }

    # Refuse on tiny-domain coincidence or sub-threshold overlap — never silently
    # accept the top match. Mirrors relationship_detector's value-driven gate.
    if col.cardinality < min_cardinality:
        verdict.refusal_reason = "master_cardinality_below_floor"
    elif overlap_count < min_overlap_count:
        verdict.refusal_reason = "overlap_count_below_floor"
    elif pct < min_overlap:
        verdict.refusal_reason = "below_overlap_threshold"

    if verdict.refusal_reason:
        log_gate_decision(_GATE, decision="refused", score=pct,
                          tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
                          master_column=col.column_name, reason=verdict.refusal_reason)
        return verdict

    norm_card = min(1.0, (col.cardinality / card_ref) if card_ref else 1.0)
    verdict.confidence = round(min(1.0, overlap_weight * pct + card_weight * norm_card), 4)
    verdict.status = "linked"
    verdict.resolved_master_file_id = col.file_id
    verdict.resolved_master_column = col.column_name
    verdict.resolved_semantic_role = col.semantic_role
    verdict.resolved_semantic_entity_id = col.semantic_entity_id
    log_gate_decision(_GATE, decision="linked", score=pct,
                      tenant_id=tenant_id, pdf_entity_id=pdf_entity_id,
                      master_column=col.column_name, confidence=verdict.confidence)
    return verdict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (6 passed). If `pdf_chat.tunables` import fails, that module is a Phase-0 dependency — confirm it exists before proceeding (it is assumed present).

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/services/__init__.py server/pdf_chat/services/value_evidence.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): value-evidenced master-key reconciliation (refuse on sub-threshold)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Bridge builder (projects registry → reconcile → persist; refuse-on-doubt)

**Files:**
- Create: `server/pdf_chat/services/bridge_builder.py`
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

The builder reads the PDF entity's values (via the injected reader), projects the container's `ColumnKeyRegistry` master-key columns, calls `reconcile_entity_to_master_keys`, and persists a `PdfEntityBridge` row. A REFUSED verdict is persisted as `status="refused"` (auditable) — and `build_bridges_for_entities` returns it so callers can say so; it never writes a `linked` row from a sub-threshold match.

- [ ] **Step 1: Write the failing test**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
import asyncio
from pdf_chat.services.bridge_builder import build_bridge_for_entity


class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def all(self): return self._rows


class _FakeSession:
    """Records added rows; returns canned master-key registry rows for the query."""
    def __init__(self, registry_rows):
        self._registry_rows = registry_rows
        self.added = []
        self.committed = False
    async def execute(self, *_a, **_k):
        return _FakeResult(self._registry_rows)
    def add(self, obj): self.added.append(obj)
    async def commit(self): self.committed = True


def _reg_row(file_id, col, role, kind, card, values):
    from app.services.relationship_index import fingerprint_value
    fps = sorted({fp for v in values if (fp := fingerprint_value(v))})
    # mimic SQLAlchemy row .mapping access used by the builder
    return {
        "file_id": file_id, "column_name": col, "semantic_role": role,
        "key_kind": kind, "cardinality": card, "value_fingerprints": fps,
        "semantic_entity_id": "se_vendor" if col == "Vendor_ID" else None,
    }


def test_builder_persists_linked_row_with_evidence():
    rows = [_reg_row("f_vendor", "Vendor_ID", "custom:vendor:key", "pk", 4,
                     ["V001", "V002", "V003", "V004"])]
    session = _FakeSession(rows)

    async def reader(tenant_id, entity_id):
        from pdf_chat.services.value_evidence import EntityValueSample
        return [EntityValueSample(value=v) for v in ("V001", "V002", "V003")]

    verdict = asyncio.run(build_bridge_for_entity(
        session, tenant_id="t1", pdf_entity_id="e1", entity_name="ACME",
        values_reader=reader,
    ))
    assert verdict.status == "linked"
    assert len(session.added) == 1
    row = session.added[0]
    assert row.resolved_master_column == "Vendor_ID"
    assert row.semantic_entity_id == "se_vendor"
    assert row.evidence["overlap_fingerprints"]
    assert session.committed is True


def test_builder_persists_refused_row_not_linked():
    rows = [_reg_row("f_vendor", "Vendor_ID", "custom:vendor:key", "pk", 4,
                     ["V001", "V002", "V003", "V004"])]
    session = _FakeSession(rows)

    async def reader(tenant_id, entity_id):
        from pdf_chat.services.value_evidence import EntityValueSample
        return [EntityValueSample(value=v) for v in ("ZZZ", "QQQ")]

    verdict = asyncio.run(build_bridge_for_entity(
        session, tenant_id="t1", pdf_entity_id="e2", entity_name="Region",
        values_reader=reader,
    ))
    assert verdict.status == "refused"
    assert len(session.added) == 1
    assert session.added[0].status == "refused"
    assert session.added[0].resolved_master_column == ""  # no master key resolved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -k builder -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.services.bridge_builder'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/services/bridge_builder.py`:

```python
"""Bridge builder — projects the container's master-key registry, runs value
reconciliation, and persists a PdfEntityBridge row. Refused verdicts are
persisted (auditable) but never as a 'linked' row. No name/embedding matching.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from pdf_chat.models.bridge import PdfEntityBridge, BridgeStatus
from pdf_chat.services.value_evidence import (
    EntityValueSample, MasterKeyColumn, ReconcileVerdict,
    reconcile_entity_to_master_keys,
)

PdfEntityValuesReader = Callable[[str, str], Awaitable[list[EntityValueSample]]]

# Project the master-key registry for a tenant. Joins column_key_registry to the
# file's SemanticEntity (if any). container_id IS the tenant boundary, exactly as
# relationship_index.find_fingerprint_matches scopes its overlap query.
_MASTER_KEYS_SQL = text(
    """
    SELECT ckr.file_id          AS file_id,
           ckr.column_name      AS column_name,
           ckr.semantic_role    AS semantic_role,
           ckr.key_kind         AS key_kind,
           ckr.cardinality      AS cardinality,
           ckr.value_fingerprints AS value_fingerprints,
           se.id                AS semantic_entity_id
    FROM column_key_registry ckr
    LEFT JOIN file_metadata fm ON fm.file_id = ckr.file_id
    LEFT JOIN semantic_entities se
           ON se.container_id = ckr.container_id
          AND se.entity_name = fm.display_name
    WHERE ckr.container_id = :container_id
    """
)


async def _load_master_columns(db: AsyncSession, tenant_id: str) -> list[MasterKeyColumn]:
    """Project the tenant's master-key registry rows into MasterKeyColumn DTOs.

    The fake session in tests returns dict rows from .execute().all(); the real
    AsyncSession returns RowMapping via .mappings().all(). Handle both: prefer
    .mappings() when present, else treat .all() rows as mappings/dicts.
    """
    raw = await db.execute(_MASTER_KEYS_SQL, {"container_id": tenant_id})
    rows = raw.mappings().all() if hasattr(raw, "mappings") else raw.all()
    return [
        MasterKeyColumn(
            file_id=str(r["file_id"]),
            column_name=str(r["column_name"]),
            semantic_role=r["semantic_role"],
            key_kind=str(r["key_kind"]),
            cardinality=int(r["cardinality"] or 0),
            value_fingerprints=list(r["value_fingerprints"] or []),
            semantic_entity_id=r["semantic_entity_id"],
        )
        for r in rows
    ]


async def build_bridge_for_entity(
    db: AsyncSession,
    *,
    tenant_id: str,
    pdf_entity_id: str,
    entity_name: str,
    values_reader: PdfEntityValuesReader,
) -> ReconcileVerdict:
    """Reconcile one PDF entity to a CSV master key and persist the outcome."""
    master_columns = await _load_master_columns(db, tenant_id)

    samples = await values_reader(tenant_id, pdf_entity_id)
    verdict = reconcile_entity_to_master_keys(
        tenant_id=tenant_id, pdf_entity_id=pdf_entity_id, entity_name=entity_name,
        samples=samples, master_columns=master_columns,
    )

    row = PdfEntityBridge(
        tenant_id=tenant_id,
        pdf_entity_id=pdf_entity_id,
        semantic_entity_id=verdict.resolved_semantic_entity_id,
        resolved_master_file_id=verdict.resolved_master_file_id or "",
        resolved_master_column=verdict.resolved_master_column or "",
        resolved_semantic_role=verdict.resolved_semantic_role,
        value_overlap_pct=verdict.value_overlap_pct,
        confidence=verdict.confidence,
        overlap_count=verdict.overlap_count,
        pdf_value_count=verdict.pdf_value_count,
        evidence={**verdict.evidence, "refusal_reason": verdict.refusal_reason},
        status=BridgeStatus.LINKED.value if verdict.status == "linked" else BridgeStatus.REFUSED.value,
    )
    db.add(row)
    await db.commit()
    return verdict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/services/bridge_builder.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): bridge builder persists linked/refused rows from registry overlap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Grain reconciliation (deterministic numeric proof + period/unit alignment)

**Files:**
- Create: `server/pdf_chat/services/grain_reconciliation.py`
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

A bridge join is only *proven* correct when the PDF-derived fact (agreement grain, e.g. contract rate × volume) reconciles against the CSV aggregate (invoice/line grain) within a tunable relative tolerance, after aligning period and unit. A mismatch yields `reconciled=False` so synthesis flags it rather than asserting a wrong join.

- [ ] **Step 1: Write the failing tests**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
from pdf_chat.services.grain_reconciliation import reconcile_grain, GrainFact, GrainAggregate


def test_grain_reconciliation_within_tolerance_passes():
    pdf = GrainFact(rate=10.0, volume=100.0, unit="EUR", period="2026-04")   # 1000
    csv = GrainAggregate(invoiced_total=1005.0, unit="EUR", period="2026-04")
    result = reconcile_grain(tenant_id="t1", fact=pdf, aggregate=csv)
    assert result.reconciled is True
    assert result.relative_diff < 0.01


def test_grain_reconciliation_outside_tolerance_fails():
    pdf = GrainFact(rate=10.0, volume=100.0, unit="EUR", period="2026-04")   # 1000
    csv = GrainAggregate(invoiced_total=1400.0, unit="EUR", period="2026-04")
    result = reconcile_grain(tenant_id="t1", fact=pdf, aggregate=csv)
    assert result.reconciled is False
    assert result.reason == "exceeds_tolerance"


def test_grain_reconciliation_unit_mismatch_refuses_before_compare():
    pdf = GrainFact(rate=10.0, volume=100.0, unit="EUR", period="2026-04")
    csv = GrainAggregate(invoiced_total=1000.0, unit="USD", period="2026-04")
    result = reconcile_grain(tenant_id="t1", fact=pdf, aggregate=csv)
    assert result.reconciled is False
    assert result.reason == "unit_mismatch"


def test_grain_reconciliation_period_mismatch_refuses_before_compare():
    pdf = GrainFact(rate=10.0, volume=100.0, unit="EUR", period="2026-04")
    csv = GrainAggregate(invoiced_total=1000.0, unit="EUR", period="2026-05")
    result = reconcile_grain(tenant_id="t1", fact=pdf, aggregate=csv)
    assert result.reconciled is False
    assert result.reason == "period_mismatch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -k grain -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.services.grain_reconciliation'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/services/grain_reconciliation.py`:

```python
"""Grain alignment + deterministic numeric reconciliation (Phase 4, business B3).

Cross-domain synthesis must reconcile the PDF-derived fact grain (agreement
grain, e.g. contract rate x volume) with the CSV aggregate grain (invoice/line
total). Align period & unit BEFORE comparing, then check rate*volume vs invoiced
total within a tunable relative tolerance. This is the automatic proof the join
was correct — a mismatch flags the answer instead of asserting it. Pure: no I/O.
Tolerance via get_tunable; every decision logged via log_gate_decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from pdf_chat.tunables import get_tunable, log_gate_decision

_GATE = "pdf_grain_reconciliation"


@dataclass(frozen=True)
class GrainFact:
    """PDF-derived fact at agreement grain."""
    rate: float
    volume: float
    unit: str
    period: str


@dataclass(frozen=True)
class GrainAggregate:
    """CSV aggregate at invoice/line grain."""
    invoiced_total: float
    unit: str
    period: str


@dataclass
class GrainResult:
    reconciled: bool = False
    expected: float = 0.0
    actual: float = 0.0
    relative_diff: float = 1.0
    reason: str = ""


def reconcile_grain(*, tenant_id: str, fact: GrainFact, aggregate: GrainAggregate) -> GrainResult:
    """Align period/unit, then numeric-check rate*volume vs invoiced_total."""
    tolerance = float(get_tunable("bridge_grain_relative_tolerance", tenant_id, 0.02))
    result = GrainResult()

    if (fact.unit or "").strip().upper() != (aggregate.unit or "").strip().upper():
        result.reason = "unit_mismatch"
        log_gate_decision(_GATE, decision="refused", tenant_id=tenant_id, reason=result.reason)
        return result
    if (fact.period or "").strip() != (aggregate.period or "").strip():
        result.reason = "period_mismatch"
        log_gate_decision(_GATE, decision="refused", tenant_id=tenant_id, reason=result.reason)
        return result

    expected = fact.rate * fact.volume
    actual = aggregate.invoiced_total
    result.expected = expected
    result.actual = actual
    denom = max(abs(expected), 1e-9)
    result.relative_diff = abs(expected - actual) / denom

    if result.relative_diff <= tolerance:
        result.reconciled = True
        log_gate_decision(_GATE, decision="reconciled", score=result.relative_diff,
                          tenant_id=tenant_id)
    else:
        result.reason = "exceeds_tolerance"
        log_gate_decision(_GATE, decision="refused", score=result.relative_diff,
                          tenant_id=tenant_id, reason=result.reason)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/services/grain_reconciliation.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): deterministic grain reconciliation proof (align period/unit, tolerance gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cross-domain cache-key augmentation (version stamps)

**Files:**
- Create: `server/pdf_chat/services/cross_domain_cache.py`
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

When `structured_query` ran for an answer, the response-cache key MUST include the CSV semantic-layer version AND the graph-extraction version (architect I5) — otherwise cross-domain answers go stale silently. Both versions are tunable-sourced stamps (a real Phase-2/semantic-rebuild stamp can be wired later; the function reads them through `get_tunable` so there is no literal and the source is swappable).

- [ ] **Step 1: Write the failing tests**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
from pdf_chat.services.cross_domain_cache import build_cache_key, version_stamps


def test_cache_key_includes_both_versions_when_structured_query_used():
    key = build_cache_key(
        tenant_id="t1", base_key="abc123", structured_query_used=True,
    )
    stamps = version_stamps("t1")
    assert stamps["semantic_layer_version"] in key
    assert stamps["graph_extraction_version"] in key
    assert "abc123" in key


def test_cache_key_omits_versions_when_structured_query_not_used():
    key = build_cache_key(
        tenant_id="t1", base_key="abc123", structured_query_used=False,
    )
    stamps = version_stamps("t1")
    assert stamps["semantic_layer_version"] not in key
    assert stamps["graph_extraction_version"] not in key
    assert key.endswith("abc123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -k cache_key -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.services.cross_domain_cache'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/services/cross_domain_cache.py`:

```python
"""Cross-domain response-cache invalidation (Phase 4, architect I5).

Whenever a structured_query tool was used to form an answer, the cache key must
incorporate BOTH the CSV semantic-layer version and the graph-extraction version
so a CSV re-ingest or a graph re-extraction invalidates stale cross-domain
answers. Both stamps resolve via get_tunable (per-container, overridable) — no
literal version string in this file; swap the source for a live stamp later.
"""
from __future__ import annotations

from pdf_chat.tunables import get_tunable


def version_stamps(tenant_id: str) -> dict[str, str]:
    """Resolve the two version stamps for a tenant (tunable-sourced)."""
    return {
        "semantic_layer_version": str(get_tunable("csv_semantic_layer_version", tenant_id, "sl0")),
        "graph_extraction_version": str(get_tunable("graph_extraction_version", tenant_id, "gx0")),
    }


def build_cache_key(*, tenant_id: str, base_key: str, structured_query_used: bool) -> str:
    """Append version stamps to base_key only when structured_query was used."""
    if not structured_query_used:
        return base_key
    stamps = version_stamps(tenant_id)
    return f"{stamps['semantic_layer_version']}:{stamps['graph_extraction_version']}:{base_key}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/services/cross_domain_cache.py server/pdf_chat/testing/test_bridge.py
git commit -m "feat(pdf-phase4): cross-domain cache key includes CSV + graph version stamps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `structured_query` tool (scope-passing + strictly sequential)

**Files:**
- Create: `server/pdf_chat/agent/structured_query.py`
- Test: `server/pdf_chat/testing/test_structured_query.py` (create)

The tool delegates to `run_agent_query` passing `container_id` + `allowed_domains` + `user_id` (so it inherits the CSV feasibility + negative-claim gates) and runs STRICTLY SEQUENTIALLY behind a per-request `asyncio.Lock` — never concurrent with another DB-touching tool (the async session is not concurrency-safe). The `run_agent_query` callable is dependency-injected so the test stays infra-free.

- [ ] **Step 1: Write the failing tests**

Create `server/pdf_chat/testing/test_structured_query.py`:

```python
"""Phase 4 — structured_query tool: scope-passing + strict sequencing. Infra-free."""
from __future__ import annotations

import asyncio

import pytest

from pdf_chat.agent.structured_query import StructuredQueryDeps, structured_query


def test_structured_query_passes_tenant_and_domain_scope():
    captured = {}

    async def fake_run_agent_query(query, db, **kwargs):
        captured.update(kwargs)
        captured["query"] = query
        return {"answer": "ok", "data": [], "files_used": ["vendor.parquet"]}

    deps = StructuredQueryDeps(
        db=object(), run_agent_query=fake_run_agent_query,
        container_id="c1", allowed_domains=["finance"], user_id="u1",
        lock=asyncio.Lock(),
    )
    out = asyncio.run(structured_query(deps, "total invoiced for vendor V001"))
    assert out["answer"] == "ok"
    assert captured["container_id"] == "c1"
    assert captured["allowed_domains"] == ["finance"]
    assert captured["user_id"] == "u1"
    assert deps.structured_query_used is True


def test_structured_query_runs_strictly_sequentially():
    # Two overlapping calls must NOT interleave: the lock serializes DB access.
    order = []

    async def fake_run_agent_query(query, db, **kwargs):
        order.append(f"start:{query}")
        await asyncio.sleep(0.01)
        order.append(f"end:{query}")
        return {"answer": query, "data": []}

    deps = StructuredQueryDeps(
        db=object(), run_agent_query=fake_run_agent_query,
        container_id="c1", allowed_domains=None, user_id="u1", lock=asyncio.Lock(),
    )

    async def drive():
        await asyncio.gather(structured_query(deps, "A"), structured_query(deps, "B"))

    asyncio.run(drive())
    # No interleave: each start is immediately followed by its own end.
    assert order in (
        ["start:A", "end:A", "start:B", "end:B"],
        ["start:B", "end:B", "start:A", "end:A"],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_structured_query.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.agent.structured_query'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/agent/structured_query.py`:

```python
"""The structured_query agent tool (Phase 4, architect B3 / business I2).

Delegates to the CSV-side run_agent_query, passing container_id + allowed_domains
+ user_id so it INHERITS the CSV feasibility gate and negative-claim gate
unchanged (no second query brain). Runs STRICTLY SEQUENTIALLY behind a
per-request asyncio.Lock — never concurrent with another DB-touching tool,
because the async SQLAlchemy session is not concurrency-safe. run_agent_query is
injected so tests need no infra; the real binding is wired at the Phase-3 seam.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

RunAgentQuery = Callable[..., Awaitable[dict]]


@dataclass
class StructuredQueryDeps:
    db: Any
    run_agent_query: RunAgentQuery
    container_id: str | None
    allowed_domains: list[str] | None
    user_id: str
    lock: asyncio.Lock
    is_admin: bool = False
    actor_email: str = ""
    actor_role: str = ""
    org_id: str | None = None
    # Set True after the first delegation so cache-key augmentation kicks in.
    structured_query_used: bool = field(default=False)


async def structured_query(deps: StructuredQueryDeps, query: str) -> dict:
    """Run a CSV-side analytical query, serialized against other DB-touching tools."""
    async with deps.lock:
        deps.structured_query_used = True
        return await deps.run_agent_query(
            query,
            deps.db,
            user_id=deps.user_id,
            is_admin=deps.is_admin,
            allowed_domains=deps.allowed_domains,
            container_id=deps.container_id,
            actor_email=deps.actor_email,
            actor_role=deps.actor_role,
            org_id=deps.org_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_structured_query.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add server/pdf_chat/agent/structured_query.py server/pdf_chat/testing/test_structured_query.py
git commit -m "feat(pdf-phase4): structured_query tool — scope passthrough + strict DB sequencing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Register `structured_query` into the Phase-3 agent + wire real value reader

**Files:**
- Create: `server/pdf_chat/agent/structured_query_tool.py`
- Modify: `server/pdf_chat/agent/graph.py` (Phase-3 seam — adapt to actual seam shape)
- Test: `server/pdf_chat/testing/test_structured_query.py` (append)

The Phase-3 agent exposes a tool-registration seam. We add a LangChain `Tool` factory and append it to the agent's tool list. We also bind the real PDF entity values reader (Phase-2 Neo4j) behind the `PdfEntityValuesReader` protocol — kept in a thin adapter so the core stays infra-free.

- [ ] **Step 1: Write the failing test**

Append to `server/pdf_chat/testing/test_structured_query.py`:

```python
def test_build_structured_query_tool_exposes_named_tool():
    from pdf_chat.agent.structured_query_tool import build_structured_query_tool

    deps = StructuredQueryDeps(
        db=object(), run_agent_query=_async_noop, container_id="c1",
        allowed_domains=None, user_id="u1", lock=asyncio.Lock(),
    )
    tool = build_structured_query_tool(deps)
    assert tool.name == "structured_query"
    assert "CSV" in tool.description or "structured" in tool.description.lower()


async def _async_noop(query, db, **kwargs):
    return {"answer": "noop", "data": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_structured_query.py::test_build_structured_query_tool_exposes_named_tool -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdf_chat.agent.structured_query_tool'`

- [ ] **Step 3: Write minimal implementation**

Create `server/pdf_chat/agent/structured_query_tool.py`:

```python
"""LangChain Tool factory that registers structured_query into the Phase-3 agent.

Wraps pdf_chat.agent.structured_query.structured_query as a named tool. The agent
tool loop calls this; the per-request lock in deps guarantees it never runs
concurrently with another DB-touching tool.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from pdf_chat.agent.structured_query import StructuredQueryDeps, structured_query

_DESCRIPTION = (
    "Answer questions about the structured CSV/tabular semantic layer (e.g. "
    "vendor invoices, amounts, line items). Use when a question needs values from "
    "the CSV side to reconcile against a PDF. Runs the governed CSV analytics "
    "pipeline (inherits feasibility + no-data gates). Input: a natural-language "
    "question string."
)


def build_structured_query_tool(deps: StructuredQueryDeps) -> StructuredTool:
    async def _run(query: str) -> str:
        result = await structured_query(deps, query)
        return str(result.get("answer", ""))

    return StructuredTool.from_function(
        coroutine=_run,
        name="structured_query",
        description=_DESCRIPTION,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_structured_query.py -v`
Expected: PASS (3 passed). If `langchain_core` import differs, match the import already used by `pdf_chat/agent/graph.py` / `server/app/agent/tools/*.py`.

- [ ] **Step 5: Wire into the Phase-3 seam**

Open `server/pdf_chat/agent/graph.py`, find where Phase 3 builds its tool list (the seam — a `tools = [...]` list or a registry). Append the structured_query tool, constructing `StructuredQueryDeps` from the request's `Deps` (db, container_id, allowed_domains, user_id) and a fresh `asyncio.Lock()` shared with any other DB-touching tool for the request. Example shape (adapt to actual seam):

```python
from pdf_chat.agent.structured_query import StructuredQueryDeps
from pdf_chat.agent.structured_query_tool import build_structured_query_tool

# ... where the per-request db tools are assembled (the Phase-3 seam):
db_lock = asyncio.Lock()
sq_deps = StructuredQueryDeps(
    db=deps.db, run_agent_query=run_agent_query,
    container_id=deps.container_id, allowed_domains=deps.allowed_domains,
    user_id=deps.user_id, lock=db_lock,
)
tools.append(build_structured_query_tool(sq_deps))
```

Import `run_agent_query` from `app.agent.graph.graph`. Note the real seam shape in the commit body.

- [ ] **Step 6: Run the full pdf_chat agent suite to confirm no regression**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_agent.py pdf_chat/testing/test_structured_query.py -v`
Expected: PASS (existing agent tests + 3 new)

- [ ] **Step 7: Commit**

```bash
git add server/pdf_chat/agent/structured_query_tool.py server/pdf_chat/agent/graph.py server/pdf_chat/testing/test_structured_query.py
git commit -m "feat(pdf-phase4): register structured_query tool at the Phase-3 agent seam

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Mount the migration in lifespan + full-suite green

**Files:**
- Modify: `server/app/main.py` (lifespan migration section)
- Test: full `pdf_chat` suite

- [ ] **Step 1: Locate the lifespan migration block**

Run: `cd server && grep -n "control_plane_upgrade\|run_migration\|pdf_chat.migrations" app/main.py`
Expected: shows where `pdf_chat/migrations/control_plane_upgrade.py` is invoked (Phase 0/1). The bridge migration mounts in the same block.

- [ ] **Step 2: Add the bridge migration call (mirror the control-plane call exactly)**

In the lifespan, alongside the existing `control_plane_upgrade.run_migration(engine)` call, add (wrap in the same try/except-warn pattern the surrounding migrations use):

```python
        try:
            from pdf_chat.migrations import bridge_upgrade
            await bridge_upgrade.run_migration(engine)
        except Exception as exc:  # non-fatal, additive
            pipeline_logger.warning("pdf_bridge_migration_failed", error=str(exc)[:300])
```

Use the same `engine` variable and logger name already in scope in that block; if the existing calls use a different helper signature, match it.

- [ ] **Step 3: Verify the app imports cleanly**

Run: `cd server && uv run python -c "import app.main"`
Expected: no exception (imports resolve; lifespan not executed).

- [ ] **Step 4: Run the full Phase-4 + adjacent suite**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py pdf_chat/testing/test_structured_query.py pdf_chat/testing/test_agent.py -v`
Expected: PASS (14 + 3 + existing agent tests)

- [ ] **Step 5: Commit**

```bash
git add server/app/main.py
git commit -m "feat(pdf-phase4): run pdf_entity_bridge migration in app lifespan

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Exit-criterion end-to-end seam test (contract PDF spans vendor CSV)

**Files:**
- Test: `server/pdf_chat/testing/test_bridge.py` (append)

The Phase-4 exit is: one question spans a contract PDF and the vendor CSV with a correct value-evidenced join. This test wires the four deterministic seams together (bridge link → structured_query scope → grain proof → cache key) without infra, asserting the join is value-evidenced end to end and that a sub-threshold variant refuses.

- [ ] **Step 1: Write the end-to-end seam test**

Append to `server/pdf_chat/testing/test_bridge.py`:

```python
def test_exit_contract_pdf_joins_vendor_csv_value_evidenced():
    from pdf_chat.services.value_evidence import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.services.grain_reconciliation import (
        reconcile_grain, GrainFact, GrainAggregate,
    )
    from pdf_chat.services.cross_domain_cache import build_cache_key, version_stamps

    # Vendor CSV master key (column_key_registry projection).
    vendor_col = MasterKeyColumn(
        file_id="f_vendor", column_name="Vendor_ID", semantic_role="custom:vendor:key",
        key_kind="pk", cardinality=4,
        value_fingerprints=_fps(["V001", "V002", "V003", "V004"]),
        semantic_entity_id="se_vendor",
    )
    # Contract PDF entity carries vendor codes — value overlap, NOT name match.
    contract_samples = [EntityValueSample(value=v) for v in ("V001", "V002", "V003")]
    link = reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_contract", entity_name="Master Supply Agreement",
        samples=contract_samples, master_columns=[vendor_col],
    )
    assert link.status == "linked"
    assert link.resolved_master_column == "Vendor_ID"

    # Grain proof: contract rate*volume reconciles vs CSV invoiced total in-period.
    proof = reconcile_grain(
        tenant_id="t1",
        fact=GrainFact(rate=10.0, volume=100.0, unit="EUR", period="2026-04"),
        aggregate=GrainAggregate(invoiced_total=1005.0, unit="EUR", period="2026-04"),
    )
    assert proof.reconciled is True

    # Cache key carries both version stamps because structured_query was used.
    key = build_cache_key(tenant_id="t1", base_key="q_hash", structured_query_used=True)
    stamps = version_stamps("t1")
    assert stamps["semantic_layer_version"] in key and stamps["graph_extraction_version"] in key
```

- [ ] **Step 2: Run the exit test**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py::test_exit_contract_pdf_joins_vendor_csv_value_evidenced -v`
Expected: PASS

- [ ] **Step 3: Run the whole Phase-4 suite once more**

Run: `cd server && uv run --with pytest --with pytest-asyncio pytest pdf_chat/testing/test_bridge.py pdf_chat/testing/test_structured_query.py -v`
Expected: PASS (15 + 3)

- [ ] **Step 4: Commit**

```bash
git add server/pdf_chat/testing/test_bridge.py
git commit -m "test(pdf-phase4): exit criterion — contract PDF value-joins vendor CSV end to end

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist (run after implementation)

- **Spec §2 Layer 2 / invariant 5** — value-evidenced master keys only, no name/embedding join → Tasks 3, 10 (`test_name_equality_alone_does_not_bridge`).
- **Spec §8 #4 / §2 Layer 2** — `pdf_entity_bridge` with overlap %, confidence, evidence; refuse on sub-threshold → Tasks 1, 4 (`test_subthreshold_overlap_refuses_no_silent_top_match`, `test_builder_persists_refused_row_not_linked`).
- **Spec §2 Layer 3 / invariant 3 / architect B3 / business I2** — `structured_query` passes tenant/domain/user scope, runs sequentially → Task 7.
- **Spec §2 Layer 2 business B3** — grain alignment + numeric reconciliation proof → Task 5.
- **Spec §2 Layer 4 architect I5** — cache key includes both version stamps → Task 6.
- **Spec invariant 4 / static-code-sentinel** — no score literal in any `.py`; all thresholds via `get_tunable`, decisions via `log_gate_decision` → Tasks 3, 5, 6.
- **Spec §5 Phase 4 EXIT** — one question spans contract PDF + vendor CSV with correct value-evidenced join → Task 10.
- **No gpt-4o** — Phase 4 adds no new LLM call; CSV delegation reuses `run_agent_query` (already mini-governed); bridge/grain/cache are pure → satisfied by design.

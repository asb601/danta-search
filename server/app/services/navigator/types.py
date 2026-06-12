"""Data contracts for the navigator query runtime.

PURE DATA. No logic, no IO, no imports beyond stdlib (dataclasses / enum /
typing). These are the typed objects that flow through the loop stages:

    PLAN      -> StepDAG (of IntentStep)
    LOOKUP    -> CandidateSlice (of Candidate)
    EVIDENCE  -> EvidencePacket
    PROPOSE   -> ProposedContract
    VERIFY    -> VerifiedContract  (+ ResolvedTable once the blob is bound)
    EXECUTE   -> StepResult
    PROMOTE   -> StepLedger (mutable accumulator across steps)
    COMPOSE   -> ComposePlan -> StepResult
    abstain   -> ClarifyPayload

Every contract is ``@dataclass(frozen=True)`` EXCEPT ``StepLedger`` (the mutable
accumulator the driver threads through the step loop). Collection fields on the
frozen dataclasses are TUPLES (immutable) — never lists.

Field lists are taken verbatim from the IMPLEMENTATION-BLUEPRINT DATA CONTRACTS
section. See docs/superpowers/specs/2026-06-11-merged-query-IMPLEMENTATION-BLUEPRINT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepKind(str, Enum):
    """The role a step plays in the DAG.

    LOOKUP  = a full sub-intent that resolves to exactly one table (the unit of
              decomposition, INVARIANT I3).
    JOIN    = a step that stitches verified tables on value-verified keys.
    COMPOSE = a pure cross-step arithmetic node (ratio / diff / growth / share);
              owns no table, depends on other steps (INVARIANT I9).
    """

    LOOKUP = "LOOKUP"
    JOIN = "JOIN"
    COMPOSE = "COMPOSE"


# ---------------------------------------------------------------------------
# [1] PLAN
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntentStep:
    """One node of the plan DAG — a full sub-intent, NOT a bare entity token.

    ``entity`` and ``measure_concept`` are business-object / business-term
    concepts (resolved to real tables/columns downstream, never here).
    ``compose_expr`` is set only for COMPOSE steps.
    """

    step_id: str
    kind: StepKind
    entity: Optional[str] = None
    measure_concept: Optional[str] = None
    grain: Optional[str] = None
    grain_entity: Optional[str] = None
    time_grain: Optional[str] = None
    filters: tuple[Any, ...] = ()
    threshold: Optional[dict] = None
    depends_on: tuple[str, ...] = ()
    join_entities: tuple[str, ...] = ()
    compose_expr: Optional[dict] = None


@dataclass(frozen=True)
class StepDAG:
    """The ordered DAG of intent-steps for one question plus the raw intent."""

    question: str
    steps: tuple[IntentStep, ...] = ()
    intent: Optional[dict] = None


# ---------------------------------------------------------------------------
# [3a] LOOKUP / RETRIEVE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One retrieved (or map-resolved) candidate table for a step."""

    file_id: str
    table: str
    score: float = 0.0
    polarity: Optional[str] = None


@dataclass(frozen=True)
class CandidateSlice:
    """The twin-aware slice of candidates for one step. ``from_map`` is True when
    the slice came from the MAP (a map hit) rather than the retriever (a miss).

    ``master_file_ids`` are the file ids in this slice that are GOVERNED CANONICAL
    MASTERS for the step's concept (a map-hit declares them; a retrieved slice
    leaves it empty). A schema-twin sibling pulled in for context is NOT a master —
    only the file(s) the map elected. When a map hit carries exactly one master the
    driver constrains PROPOSE to it (the map already decided; the noisy ``good_for``
    of templated twins must not re-litigate the table choice — INVARIANT I5)."""

    step_id: str
    entity: Optional[str]
    candidates: tuple[Candidate, ...] = ()
    from_map: bool = False
    master_file_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# [3b] EVIDENCE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidencePacket:
    """The per-file stored evidence assembled for a step's candidate slice."""

    step_id: str
    files: tuple[dict, ...] = ()


# ---------------------------------------------------------------------------
# [3c] PROPOSE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProposedContract:
    """mini's typed slot proposal for a step — CHOOSES table/column/agg/filter,
    never SQL (INVARIANT I2). Not yet verified against evidence."""

    step_id: str
    table: str
    table_reason: Optional[str] = None
    grain_kind: Optional[str] = None
    grain_column: Optional[str] = None
    time_bucket: Optional[str] = None
    measure_column: Optional[str] = None
    measure_agg: Optional[str] = None
    filters: tuple[Any, ...] = ()
    time_filter_column: Optional[str] = None
    having: Optional[dict] = None
    top_n: Optional[int] = None
    order: Optional[str] = None


# ---------------------------------------------------------------------------
# [3d] VERIFY
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifiedContract:
    """A proposed contract whose every slot passed VERIFY against evidence
    (INVARIANT I6). Only a verified contract may be rendered to SQL."""

    step_id: str
    table: str
    grain_kind: Optional[str] = None
    grain_col: Optional[str] = None
    bucket: Optional[str] = None
    measure_col: Optional[str] = None
    agg: Optional[str] = None
    filters: tuple[Any, ...] = ()
    time_col: Optional[str] = None
    having: Optional[dict] = None
    top_n: Optional[int] = None
    order: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class ResolvedTable:
    """A verified table bound to its physical blob for execution."""

    step_id: str
    table: str
    file_id: str
    blob: str


# ---------------------------------------------------------------------------
# [4]/[5] EXECUTE / RESULT
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepResult:
    """The executed result of a step. ``scalar`` is the single-number answer when
    the step reduces to one value (used by COMPOSE cross-step math).

    ``error_marker`` is set ONLY on a failure path (a canonicalize/auth/engine
    error in EXECUTE, or an undefined COMPOSE arithmetic): rows are empty and
    scalar is None, so the driver can abstain (I12) rather than crash. A
    successful step leaves it None."""

    step_id: str
    sql: Optional[str] = None
    rows: tuple[Any, ...] = ()
    total: Optional[int] = None
    table: Optional[str] = None
    measure_label: Optional[str] = None
    grain: Optional[str] = None
    scalar: Optional[float] = None
    error_marker: Optional[str] = None


# ---------------------------------------------------------------------------
# abstain / clarify
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClarifyPayload:
    """An abstain-and-confirm payload. ``options`` are drawn from the candidates'
    OWN side/role — never invented literals (INVARIANT I12); capped at 3."""

    reason: str
    options: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# [5] COMPOSE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComposePlan:
    """A deterministic cross-step arithmetic plan. ``op`` is one of
    ratio | diff | growth | share; the operands are step ids."""

    op: str
    left_step: str
    right_step: str


# ---------------------------------------------------------------------------
# PROMOTE / accumulator (the ONE mutable contract)
# ---------------------------------------------------------------------------
@dataclass
class StepLedger:
    """Mutable accumulator threaded through the step loop.

    ``results`` maps step_id -> StepResult as each step executes; ``clarify``
    holds an abstain payload (or dict) if any step could not be verified.
    """

    results: dict[str, StepResult] = field(default_factory=dict)
    clarify: Any = None

    def get_scalar(self, step_id: str) -> Optional[float]:
        """The scalar value of a completed step, or None if the step is absent
        or produced no scalar. Pure lookup — COMPOSE does the arithmetic."""
        res = self.results.get(step_id)
        if res is None:
            return None
        return res.scalar

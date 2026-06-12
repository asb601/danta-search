"""The navigator LOOP — ``run_navigator`` wires stages [1]..[6] into ONE driver.

This is the public entry point of the merged query runtime ("Disciplined
Navigator"). It REPLACES the legacy brain seam + the RESOLVE-contract seam at the
graph cutover (P5). It owns the single agentic loop the TARGET architecture
describes:

    [1] PLAN      planner.plan -> StepDAG | None (abstain)
    [3a] LOOKUP   retriever.lookup (map-first, ONE hybrid engine, twin-aware)
    [3b] EVIDENCE evidence.assemble
    [3c] PROPOSE  proposer.propose (ONE mini call, typed slots)
    [3d] VERIFY   verifier.verify (value-check; polarity gate; abstain/clarify)
    [3e] RENDER   renderer.render / render_join (deterministic SQL)
    [4]  EXECUTE  executor.execute (DataFusion/DuckDB on Parquet)
    [5]  PROMOTE  promote.promote (write the verified conclusion into the ledger)
    [5]  COMPOSE  composer.compose (PURE cross-step arithmetic)
    [6]  SYNTHESIZE synthesizer.synthesize (ONE mini call, prose only)

INVARIANT discipline this driver enforces:
  * I1/I2  — mini only PLANs, PROPOSEs, SYNTHESIZEs; numbers come from the engine
             (executor) and cross-step math from the composer, never the LLM.
  * I3     — the unit of work is the INTENT-STEP from the plan, not the whole
             question retrieved as one.
  * I6/I12 — every contract is verified before render; an unverifiable step
             abstains (returns None) or, on a genuine polarity tie, clarifies.
  * I9     — COMPOSE does ALL cross-step arithmetic deterministically.

Returns a ``run_agent_query``-shaped payload (route="navigator" for an answer,
route="navigator_clarify" for an abstain-to-user question) or ``None`` to fall
through to the existing LangGraph agent loop. NEVER raises — any unexpected error
abstains (None).

q_polarity is derived HERE without importing ``app.services.resolve.*``: the
corpus-relative polarity logic (embed the step's intent terms, score cosine
against each candidate's stored evidence embedding, commit to the reliably-
separated side) is LIFTED into ``_question_polarity`` below, reusing the
navigator-local reliability gate ``verifier._polarity_from_row``.

Request-store lifecycle: on a successful answer OR a clarify the per-request store
is popped (the navigator returned BEFORE ``graph.ainvoke`` ever ran, so it must
clean up after itself — mirror of coordinator.py:305-308). On an ABSTAIN the store
is LEFT IN PLACE so the agent fall-through still has it.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics
from app.core.config import get_settings
from app.models.erp_classification import ErpClassification
from app.models.file_metadata import FileMetadata
from app.retrieval.embeddings import embed_text
from app.retrieval.temporal import parse_temporal
from app.services.navigator import composer
from app.services.navigator import evidence
from app.services.navigator import executor
from app.services.navigator import promote
from app.services.navigator import proposer
from app.services.navigator import renderer
from app.services.navigator import retriever
from app.services.navigator import synthesizer
from app.services.navigator import verifier
from app.services.navigator.planner import plan as plan_question
from app.services.navigator.types import (
    CandidateSlice,
    ClarifyPayload,
    ComposePlan,
    EvidencePacket,
    IntentStep,
    ResolvedTable,
    StepKind,
    StepLedger,
    VerifiedContract,
)

logger = structlog.get_logger("navigator.driver")

# Universal ledger axis — the only closed polarity vocabulary (mirrors the
# classifier/model + verifier). A question can only ever commit to a reliable side.
_RELIABLE_SIDES: frozenset[str] = frozenset({"customer", "vendor"})
# Minimum cosine separation between the best customer-side and best vendor-side
# candidate before the question is allowed to commit to a side (lifted from
# resolve.search._Q_POLARITY_MARGIN). Within the margin the sides are too close to
# call -> q_polarity=None (defer to abstain). A tie-breaking guard, not a business
# threshold.
_Q_POLARITY_MARGIN = 0.05
# How many candidate key columns to test per side before giving up on a JOIN edge.
# Bounds the verify_step_join fan-out; a safety cap, not a business threshold
# (mirrors coordinator._MAX_JOIN_COLS_PER_SIDE).
_MAX_JOIN_COLS_PER_SIDE = 12


# ---------------------------------------------------------------------------
# q_polarity — corpus-relative, NO resolve.* import (lifted from search.py)
# ---------------------------------------------------------------------------
def _is_zero_vector(vec) -> bool:
    return not vec or all(v == 0.0 for v in vec)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. Pure; 0.0 on a degenerate
    (zero-norm / mismatched) vector so it never raises."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


async def _embed_terms(terms: str):
    try:
        vec = await embed_text(terms.strip())
    except Exception as exc:  # noqa: BLE001 — never raise
        logger.warning("nav_embed_error", error=str(exc)[:160])
        return None
    if _is_zero_vector(vec):
        return None
    return vec


async def _question_polarity(
    db: AsyncSession,
    container_id: str,
    candidate_file_ids: list[str],
    question_terms: list[str],
) -> str | None:
    """Derive the QUESTION's ledger polarity corpus-relatively (Signal 2), WITHOUT
    importing ``app.services.resolve.*``. LIFTED from ``search.question_polarity``.

    Embed the step's intent terms (entity + measure concept + question), score
    cosine against each candidate's stored ``description_embedding``, and return the
    RELIABLE side (customer|vendor) of the highest-scoring candidate — but only when
    the best customer-side and best vendor-side scores are separated by more than
    ``_Q_POLARITY_MARGIN``. Within the margin -> None (defer to abstain). None on any
    failure / no embedding / no reliable side. Never raises.
    """
    if not getattr(get_settings(), "BRAIN_POLARITY_GATE_ENABLED", True):
        return None
    terms = " ".join(t for t in (question_terms or []) if t and str(t).strip()).strip()
    if not terms or not candidate_file_ids:
        return None
    q_vec = await _embed_terms(terms)
    if q_vec is None:
        return None
    try:
        rows = (
            await db.execute(
                select(
                    FileMetadata.file_id,
                    FileMetadata.description_embedding,
                    ErpClassification.domain_polarity,
                    ErpClassification.confidence,
                    ErpClassification.source,
                    ErpClassification.source_system,
                )
                .join(ErpClassification, ErpClassification.file_id == FileMetadata.file_id)
                .where(FileMetadata.container_id == container_id)
                .where(FileMetadata.file_id.in_(candidate_file_ids))
                .where(FileMetadata.description_embedding.is_not(None))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("nav_q_polarity_query_error", error=str(exc)[:200])
        return None

    best: dict[str, float] = {}
    for _fid, emb, polarity, conf, src, src_sys in rows:
        side = verifier._polarity_from_row(polarity, conf, src, src_sys)
        if side not in _RELIABLE_SIDES or emb is None:
            continue
        score = _cosine(q_vec, list(emb))
        if side not in best or score > best[side]:
            best[side] = score
    if not best:
        return None
    cust = best.get("customer")
    vend = best.get("vendor")
    if cust is not None and vend is not None:
        if abs(cust - vend) < _Q_POLARITY_MARGIN:
            return None
        return "customer" if cust > vend else "vendor"
    return "customer" if cust is not None else "vendor"


def _step_polarity_terms(step: IntentStep, question: str) -> list[str]:
    """The intent terms whose polarity we score: this step's entity + measure
    concept + the raw question (so a step inherits the question's side cues). Pure."""
    terms: list[str] = []
    if step.entity:
        terms.append(step.entity)
    if step.measure_concept:
        terms.append(step.measure_concept)
    if question:
        terms.append(question)
    return terms


# ---------------------------------------------------------------------------
# time-window resolution — navigator-contained (no graph.py plumbing change)
# ---------------------------------------------------------------------------
def _resolve_time_window(question: str, ctx: dict) -> tuple | None:
    """Resolve the question's time window ONCE, INSIDE the navigator (L8). Reuses the
    retrieval ``temporal.parse_temporal`` regex parser, anchored on ``ctx["as_of"]``
    (the data's latest coverage date) so relative windows ("last quarter", "YTD",
    "last month") resolve against the data — not the wall clock — exactly like
    planner.py / feasibility_gate use ``as_of``. Returns ``(date_from, date_to)`` or
    ``None`` when no time scope is detected (so the no-window path is byte-identical
    to today). Pure-ish (one regex parse); never raises — any failure degrades to
    None (no window) so a parse error can only ever LOSE the scope, never invent one."""
    try:
        anchor = ctx.get("as_of")
        # parse_temporal takes a ``date`` anchor (or None -> wall clock); as_of is a
        # date | None already (feasibility_gate.resolve_as_of_date). Pass it through.
        start, end = parse_temporal(question or "", today=anchor)
    except Exception as exc:  # noqa: BLE001 — never raise; degrade to no-window
        logger.warning("nav_time_window_parse_error", error=str(exc)[:160])
        return None
    if start is None or end is None:
        return None
    return (start, end)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _topo_order(steps: tuple[IntentStep, ...]) -> list[IntentStep] | None:
    """Kahn topological sort over ``depends_on``. Returns steps in dependency order
    (a step appears AFTER everything it depends on), or None on a cycle (the planner
    already rejects cycles, so this is a belt-and-braces guard). Pure."""
    by_id = {s.step_id: s for s in steps}
    indeg = {s.step_id: 0 for s in steps}
    for s in steps:
        for dep in s.depends_on:
            if dep in by_id:
                indeg[s.step_id] += 1
    ready = [s for s in steps if indeg[s.step_id] == 0]
    ordered: list[IntentStep] = []
    while ready:
        s = ready.pop(0)
        ordered.append(s)
        for other in steps:
            if s.step_id in other.depends_on:
                indeg[other.step_id] -= 1
                if indeg[other.step_id] == 0:
                    ready.append(other)
    if len(ordered) != len(steps):
        return None
    return ordered


def _sink_steps(steps: tuple[IntentStep, ...]) -> list[IntentStep]:
    """The ANSWER-BEARING sinks of the plan DAG. A sink is a step that NO other step
    lists in ``depends_on`` AND that is not referenced as a COMPOSE operand
    (``compose_expr.left_step`` / ``right_step``). Pure — reads only the DAG shape.

    Rationale (FIX A): topological order does NOT imply "the last-inserted result is
    the answer". A multi-branch plan with no COMPOSE (e.g. "customer receipts vs
    vendor payments" -> two independent LOOKUPs) has TWO sinks; selecting only the
    last-executed one silently drops the other branch. A single LOOKUP, or a
    COMPOSE-terminated plan (the LOOKUP operands are referenced -> NOT sinks, the
    COMPOSE is the lone sink), yields exactly one sink — byte-identical selection to
    the legacy ``list(ledger.results.values())[-1]`` for every plan that works today.
    Order is preserved (DAG order) so multi-sink concatenation is deterministic."""
    referenced: set[str] = set()
    for s in steps:
        referenced.update(s.depends_on)
        if s.compose_expr:
            for key in ("left_step", "right_step"):
                ref = s.compose_expr.get(key)
                if ref:
                    referenced.add(str(ref))
    return [s for s in steps if s.step_id not in referenced]


def _blob_for_file(file_id: str, identity_map) -> str | None:
    """Resolve the representative blob_path for a file_id from the request identity
    map (the same map the executor + canonicalizer trust)."""
    if identity_map is None or not file_id:
        return None
    identity = getattr(identity_map, "by_id", {}).get(file_id)
    return getattr(identity, "blob_path", None) if identity is not None else None


def _measure_label(vc: VerifiedContract) -> str:
    """A business-readable measure label DERIVED from the verified contract — never
    hardcoded. Pure."""
    if vc.measure_col and vc.agg:
        return f"{vc.agg}({vc.measure_col})"
    return vc.measure_col or "result"


# The key each multi-branch row is tagged with so the structured payload identifies
# which sink (branch) it came from. A SQL-output/labelling identifier, not a
# business knob.
_BRANCH_KEY = "_branch"


def _branch_label(result) -> str:
    """A branch identifier for a sink StepResult — its table, else its measure
    label, else its step_id (whatever distinguishes the branch). Pure."""
    return str(result.table or result.measure_label or result.step_id or "result")


def _branch_rows(result) -> list[dict]:
    """The rows of ONE sink, each tagged with its branch label so a multi-sink
    payload represents every branch the prose discusses. A scalar-only sink (no
    rows) emits ONE labelled row carrying its measure + value. Pure; never raises."""
    label = _branch_label(result)
    rows = list(result.rows or ())
    if rows:
        out: list[dict] = []
        for r in rows:
            row = dict(r) if isinstance(r, dict) else {"value": r}
            row.setdefault(_BRANCH_KEY, label)
            out.append(row)
        return out
    if result.scalar is not None:
        return [{
            _BRANCH_KEY: label,
            (result.measure_label or "result"): result.scalar,
        }]
    return []


def _clarify_payload_dict(
    question: str, clarify: ClarifyPayload, ctx: dict,
) -> dict:
    """Build the run_agent_query-shaped ``navigator_clarify`` payload. The answer is
    the question echoed back (a question to the user, not an asserted answer); the
    clarify block carries the reason + the evidence-derived options."""
    return {
        "answer": question,
        "data": [],
        "chart": None,
        "route": "navigator_clarify",
        "row_count": 0,
        "files_used": [],
        "tool_calls": 0,
        "retrieved_files": ctx.get("catalog_len", 0),
        "total_files": ctx.get("total_files", 0),
        "clarify": {
            "reason": clarify.reason,
            "options": list(clarify.options),
        },
    }


def _pop_store(req_id: str) -> None:
    """Pop the per-request store on a navigator success/clarify (the navigator
    returned before graph.ainvoke ever ran). Mirror of coordinator.py:305-308 —
    cleanup must never break the answer."""
    try:
        from app.agent.graph.graph import _request_stores, _stores_lock  # noqa: PLC0415
        with _stores_lock:
            _request_stores.pop(req_id, None)
    except Exception:  # noqa: BLE001 — cleanup must never break the answer
        pass


def _abstain(reason: str, **fields) -> None:
    """FIX B (M4): the SINGLE standardized abstain seam. Emit ONE structured
    ``navigator_abstain`` event (with ``reason`` + any context fields like
    ``step_id``) and bump the per-reason metric, then return None so the navigator
    falls through to the agent. Routing every ``return None`` abstain site through
    here makes a SYSTEMIC regression (every verify failing) distinguishable from
    honest "no plan fits" — without changing control flow (it still returns None and
    the caller still falls through). Never raises."""
    logger.info("navigator_abstain", reason=reason, **fields)
    try:
        metrics.inc_navigator_abstain(reason)
    except Exception:  # noqa: BLE001 — observability must never break the fall-through
        pass
    return None


def _master_constrained(
    kept: list[dict], slice_: CandidateSlice,
) -> list[dict]:
    """Constrain the propose-time evidence to the GOVERNED CANONICAL MASTER on a map
    hit (I5). Returns ``kept`` UNCHANGED unless the slice is a map hit with EXACTLY
    ONE master AND that master's evidence is present in ``kept`` — then returns just
    that one evidence dict. Pure; never widens ``kept``, never returns empty.

    The master is matched by the (case-insensitive) logical table the slice's master
    ``Candidate`` carries, so it lines up with the evidence dict's ``table``. A map
    hit with 0 or ≥2 masters, a non-map slice, or a master that the polarity filter
    already dropped (not in ``kept``) all leave ``kept`` untouched — the constraint
    only ever STRENGTHENS the map's own single, unambiguous decision."""
    if slice_ is None or not slice_.from_map:
        return kept
    masters = slice_.master_file_ids or ()
    if len(masters) != 1:
        return kept
    master_id = masters[0]
    master_tables = {
        str(c.table or "").upper()
        for c in slice_.candidates
        if c.file_id == master_id and c.table
    }
    if not master_tables:
        return kept
    narrowed = [f for f in kept if str(f.get("table") or "").upper() in master_tables]
    # Never starve propose: only narrow when the master actually survived to ``kept``.
    return narrowed or kept


# ---------------------------------------------------------------------------
# per-step resolution: LOOKUP -> EVIDENCE -> PROPOSE -> VERIFY (+ re-propose x1)
# ---------------------------------------------------------------------------
async def _resolve_lookup_step(
    question: str,
    db: AsyncSession,
    container_id: str,
    ctx: dict,
    step: IntentStep,
    time_window: tuple | None = None,
) -> tuple[VerifiedContract | None, EvidencePacket | None, CandidateSlice | None, str]:
    """Run [3a]..[3d] for ONE LOOKUP step. Returns
    ``(verified_contract | None, evidence_packet | None, candidate_slice | None,
    reason)``.

    A verified contract on success; ``(None, ev, slice, reason)`` on a verify
    failure (so the driver can decide clarify-vs-abstain from the reason + the
    evidence). One re-propose is attempted on a verify failure (I6) before giving
    up. NEVER raises.

    ``time_window`` (L8): the question's resolved ``(start, end)`` window (or None).
    Threaded into PROPOSE (so mini is told the window + picks a date column) and into
    VERIFY (which then REQUIRES a usable date column on the chosen table — see
    ``verifier.verify`` — so a windowed question can never render all-time SQL).
    """
    identity_map = ctx.get("file_identity_map")
    slice_ = await retriever.lookup(
        db, container_id, step, identity_map=identity_map,
        user_id=ctx.get("user_id", "") or "",
        is_admin=bool(ctx.get("is_admin", True)),
        allowed_domains=ctx.get("allowed_domains"),
    )
    if not slice_.candidates:
        return None, None, None, "empty_slice"

    ev = await evidence.assemble(db, slice_)
    if not ev.files:
        return None, None, slice_, "empty_evidence"

    # q_polarity for THIS step, corpus-relative over its own candidate slice.
    slice_ids = [c.file_id for c in slice_.candidates if c.file_id]
    q_polarity = await _question_polarity(
        db, container_id, slice_ids, _step_polarity_terms(step, question),
    )

    # PICK pre-filter (I12): a slice spanning ≥2 reliable ledger sides with an
    # UNKNOWN question polarity is a GENUINE tie — never guessed. Surface it as a
    # clarify signal BEFORE proposing, so the driver abstains-to-user instead of
    # letting mini blend opposite-side twins. ``partition_by_polarity`` is the same
    # deterministic guard the legacy seam used (lifted into the verifier).
    files = [dict(f) for f in ev.files]
    kept, tie = verifier.partition_by_polarity(files, q_polarity)
    if tie == "polarity_tie":
        return None, ev, slice_, "polarity_tie"

    # FIX B (M5): APPLY the reliability-gated filter to what PROPOSE sees, so the
    # documented hard guarantee "cross-side (AP-vs-AR) twins are never blended" is
    # enforced BEFORE the LLM, not merely hoped for in the prompt.
    #
    # SAFETY — partition_by_polarity (verifier.py) is the SOLE source of `kept` and
    # already enforces every never-reduce-correctness constraint, so this is safe to
    # apply verbatim:
    #   * it only DROPS a candidate when (a) q_polarity is a RELIABLE side AND (b) the
    #     candidate's OWN ``polarity`` is the reliable OPPOSITE. The evidence packet's
    #     ``polarity`` is itself the reliability-gated value (evidence.assemble passes
    #     every row through verifier._polarity_from_row → None unless reliable), so a
    #     guessed/unreliable polarity is None and is ALWAYS kept (in (q_polarity, None)).
    #     A misclassified side can therefore NEVER drop the right table.
    #   * when q_polarity is unknown (not a reliable side) it returns the FULL set
    #     unchanged → propose sees exactly today's evidence.
    #   * it NEVER returns empty (the ``(keep or candidates)`` guard), so the filter
    #     cannot starve propose; abstain-bias is preserved.
    # VERIFY still reads the FULL ``ev`` (its own polarity cross-check + table lookup
    # are unchanged); only the propose-time evidence is narrowed. The proposed table
    # is always in ``kept`` ⊆ ``ev.files``, so verify still resolves it.
    #
    # MAP-FIRST table constraint (I5): when this slice is a MAP HIT with EXACTLY ONE
    # governed canonical master, the map ALREADY decided the table — the templated
    # schema-twins it pulled in for context have near-identical ``good_for`` (noise
    # on this dataset) and must NOT be allowed to re-litigate the choice via the LLM.
    # Narrow what PROPOSE sees to the master alone (the twins stay in ``ev`` for
    # verify/JOIN). Applied AFTER the polarity filter and only over ``kept``, so it
    # can never resurrect a reliable-opposite-side table or widen the slice; ≥2
    # masters or 0 (or a master not present in ``kept``) leaves ``kept`` unchanged,
    # so abstain-bias and the no-empty guarantee are intact.
    kept = _master_constrained(kept, slice_)
    propose_ev = EvidencePacket(step_id=ev.step_id, files=tuple(kept))

    # PROPOSE -> VERIFY, with ONE re-propose on a verify failure (I6). FIX C (L7):
    # the first attempt's verify-failure reason is threaded into the SECOND propose
    # so mini CORRECTS its pick instead of repeating the same losing choice at temp 0
    # (without it the re-propose was a no-op — identical inputs, identical failure).
    last_reason = "unverified"
    prior_failure: str | None = None
    for _attempt in range(2):
        pc = await proposer.propose(
            question, step, propose_ev, time_window=time_window,
            prior_failure=prior_failure,
        )
        if pc is None:
            last_reason = "propose_abstain"
            break
        vc, reason = verifier.verify(pc, ev, q_polarity, time_window=time_window)
        if vc is not None:
            return vc, ev, slice_, "ok"
        last_reason = reason
        prior_failure = reason  # feed the failure forward so the re-propose corrects
        # A polarity tie/contradiction will not be cured by a re-propose — stop and
        # let the driver decide clarify-vs-abstain.
        if reason in ("polarity_contradicts_question",):
            break
    return None, ev, slice_, last_reason


async def _resolve_join_step(
    db: AsyncSession,
    container_id: str,
    ctx: dict,
    step: IntentStep,
    resolved_tables: dict[str, ResolvedTable],
) -> tuple[ResolvedTable, ResolvedTable, str, str, "verifier.JoinVerdict"] | None:
    """Run [3e]-JOIN for a JOIN step: take the two upstream resolved tables this
    step depends on, enumerate candidate key columns over ColumnKeyRegistry, and
    VERIFY a value-overlap edge (I7). Returns the verified ``(a, b, col_a, col_b,
    verdict)`` or None (abstain). NEVER raises. Mirrors coordinator._resolve_join.
    """
    deps = [resolved_tables[d] for d in step.depends_on if d in resolved_tables]
    if len(deps) < 2:
        logger.info("join_abstain_insufficient_tables", step_id=step.step_id,
                    resolved=len(deps))
        return None
    a, b = deps[0], deps[1]
    if not a.blob or not b.blob or a.blob == b.blob:
        logger.info("join_abstain_same_or_missing_blob", step_id=step.step_id)
        return None

    cols_a = await _candidate_key_columns(db, container_id, a.blob)
    cols_b = await _candidate_key_columns(db, container_id, b.blob)
    if not cols_a or not cols_b:
        logger.info("join_abstain_no_key_columns", step_id=step.step_id)
        return None

    for ca in cols_a:
        for cb in cols_b:
            verdict = await verifier.verify_step_join(db, container_id, a, b, ca, cb)
            if verdict.verified:
                logger.info(
                    "join_verified", step_id=step.step_id,
                    pk_side=verdict.pk_side, containment=round(verdict.containment, 4),
                )
                return a, b, ca, cb, verdict
    logger.info("join_abstain_no_verified_edge", step_id=step.step_id)
    return None


async def _candidate_key_columns(
    db: AsyncSession, container_id: str, blob_path: str,
) -> list[str]:
    """Candidate join-key columns for a blob, from the precomputed ColumnKeyRegistry
    (value-evidence only, no schema probe), ordered by uniqueness (PK-likely first)
    so verify_step_join tests the most promising pairs first. NEVER raises. Mirrors
    coordinator._candidate_key_columns."""
    from app.models.column_key_registry import ColumnKeyRegistry  # noqa: PLC0415
    try:
        rows = (
            await db.execute(
                select(ColumnKeyRegistry.column_name, ColumnKeyRegistry.unique_rate)
                .where(ColumnKeyRegistry.container_id == container_id)
                .where(ColumnKeyRegistry.blob_path == blob_path)
                .order_by(ColumnKeyRegistry.unique_rate.desc())
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("nav_key_columns_query_error", error=str(exc)[:200])
        return []
    return [str(name) for name, _ur in rows[:_MAX_JOIN_COLS_PER_SIDE]]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
async def run_navigator(
    question: str,
    db: AsyncSession,
    container_id: str | None,
    ctx: dict,
    initial_state: dict,
    req_id: str,
) -> dict | None:
    """Resolve ``question`` end-to-end through the navigator loop, or ``None`` to
    fall through to the agent. Returns a run_agent_query-shaped payload
    (route="navigator" | "navigator_clarify"). NEVER raises — any unexpected error
    abstains (None). Pops the request store on success/clarify, NOT on abstain."""
    try:
        effective_container_id = ctx.get("resolved_container_id") or container_id
        if not effective_container_id:
            return _abstain("navigator_no_container")

        # ── [1] PLAN ──────────────────────────────────────────────────────────
        dag = await plan_question(
            question, ctx.get("intent_plan"), ctx.get("as_of"),
        )
        if dag is None or not dag.steps:
            return _abstain("navigator_plan_abstain")

        ordered = _topo_order(dag.steps)
        if ordered is None:
            return _abstain("navigator_plan_cycle")

        # Resolve the question's time window ONCE (L8), navigator-contained, anchored
        # on ctx["as_of"]. None ⇒ no time scope ⇒ behaviour byte-identical to today.
        time_window = _resolve_time_window(question, ctx)

        identity_map = ctx.get("file_identity_map")
        allowed_file_ids = ctx.get("allowed_file_ids")
        connection_string = initial_state.get("connection_string")
        container_name = ctx.get("container_name")

        ledger = StepLedger()
        resolved_tables: dict[str, ResolvedTable] = {}

        # ── [3] STEP LOOP (topological order) ─────────────────────────────────
        for step in ordered:
            if step.kind == StepKind.COMPOSE:
                # [5] COMPOSE — deterministic cross-step arithmetic (I9). No table,
                # no SQL, no LLM: it reads promoted scalars from the ledger.
                if not step.compose_expr:
                    return _abstain("navigator_compose_no_expr", step_id=step.step_id)
                plan = ComposePlan(
                    op=step.compose_expr.get("op"),
                    left_step=step.compose_expr.get("left_step"),
                    right_step=step.compose_expr.get("right_step"),
                )
                result = composer.compose(ledger, plan)
                if result.error_marker is not None or result.scalar is None:
                    return _abstain("navigator_compose_undefined", step_id=step.step_id,
                                    marker=result.error_marker)
                ledger.results[step.step_id] = result
                continue

            if step.kind == StepKind.JOIN:
                # [3e]-JOIN — value-verified relationship edge (I7) then render+exec.
                join = await _resolve_join_step(
                    db, effective_container_id, ctx, step, resolved_tables,
                )
                if join is None:
                    # _resolve_join_step already logged the granular join_abstain_*
                    # reason; surface a uniform queryable abstain on top of it.
                    return _abstain("navigator_join_abstain", step_id=step.step_id)
                a, b, col_a, col_b, verdict = join
                sql = renderer.render_join(verdict, a, b, col_a, col_b)
                result = await executor.execute(
                    sql,
                    identity_map=identity_map,
                    allowed_file_ids=allowed_file_ids,
                    connection_string=connection_string,
                    container_name=container_name,
                    step_id=step.step_id,
                    table=(a.table if verdict.pk_side == a.blob else b.table),
                    measure_label="COUNT(*)",
                    grain="entity",
                    max_rows=20,
                )
                if result.error_marker is not None:
                    return _abstain("navigator_join_exec_error", step_id=step.step_id,
                                    marker=result.error_marker)
                ledger.results[step.step_id] = result
                continue

            # ── LOOKUP step: [3a]..[3e] then [4] EXECUTE ──────────────────────
            vc, ev, slice_, reason = await _resolve_lookup_step(
                question, db, effective_container_id, ctx, step, time_window,
            )
            if vc is None:
                # A genuine polarity tie (≥2 reliable sides, unknown question side)
                # or a wrong-side pick (the chosen ledger side contradicts the
                # question's) with offerable options -> abstain-and-confirm (I12).
                # Otherwise abstain to the agent.
                if (
                    reason in ("polarity_tie", "polarity_contradicts_question")
                    and ev is not None
                ):
                    files = [dict(f) for f in ev.files]
                    clarify = verifier.clarify_payload(files, reason)
                    if clarify.options:
                        _pop_store(req_id)
                        logger.info("navigator_clarify", step_id=step.step_id,
                                    reason=reason)
                        return _clarify_payload_dict(question, clarify, ctx)
                # Preserve the granular step-resolution reason (empty_slice /
                # empty_evidence / bad_measure / propose_abstain / …) as the
                # standardized abstain reason so a systemic verify regression is
                # visible per-reason.
                return _abstain(reason, step_id=step.step_id)

            # [3e] RENDER deterministic SQL from the verified contract (I10). When a
            # window was resolved, verify guaranteed vc.time_col is a usable date
            # column, so the renderer emits the (quoted) >= / <= bounds; otherwise
            # the window is None and no date WHERE is added (byte-identical to today).
            sql = renderer.render(vc, time_window=time_window)
            # [4] EXECUTE — numbers come from the engine (I11).
            result = await executor.execute(
                sql,
                identity_map=identity_map,
                allowed_file_ids=allowed_file_ids,
                connection_string=connection_string,
                container_name=container_name,
                step_id=step.step_id,
                table=vc.table,
                measure_label=_measure_label(vc),
                grain=vc.grain_kind or "",
                max_rows=20,
            )
            if result.error_marker is not None:
                return _abstain("navigator_exec_error", step_id=step.step_id,
                                marker=result.error_marker)

            # [5] PROMOTE — record the verified, executed conclusion in the ledger.
            promote.promote(ledger, step, vc, result)

            # Bind the resolved table (file_id + blob) for any downstream JOIN
            # step, from the SLICE candidate whose logical table the brain chose.
            file_id = _resolved_file_id(vc, slice_)
            blob = _blob_for_file(file_id, identity_map)
            if blob:
                resolved_tables[step.step_id] = ResolvedTable(
                    step_id=step.step_id, table=vc.table,
                    file_id=file_id, blob=blob,
                )

        if not ledger.results:
            return _abstain("navigator_no_results")

        # ── [6] SYNTHESIZE — prose around the VERIFIED numbers ────────────────
        answer = await synthesizer.synthesize(question, ledger)

        # Build the run_agent_query-shaped payload. data = the ANSWER-BEARING SINK
        # step(s) of the plan DAG — NOT "the last-inserted result", which silently
        # drops a branch in a multi-branch (no-COMPOSE) plan (FIX A). files_used =
        # every table the ledger touched.
        #
        #   * exactly ONE sink (single LOOKUP, or a COMPOSE-terminated plan whose
        #     COMPOSE is the lone sink): use it. A COMPOSE-final sink has no rows —
        #     only a scalar (the ratio/diff/…); surface it as a one-row dataset so
        #     data/row_count are never empty for a valid scalar answer. This is
        #     byte-identical to the legacy [-1] selection for every plan that works.
        #   * MULTIPLE sinks (true multi-branch, no COMPOSE): concatenate every
        #     sink's rows, each tagged with its branch label, so the structured
        #     payload represents ALL branches the synthesizer prose describes.
        sinks = [
            s for s in _sink_steps(dag.steps) if s.step_id in ledger.results
        ]
        if len(sinks) <= 1:
            final = (
                ledger.results[sinks[0].step_id]
                if sinks
                else list(ledger.results.values())[-1]
            )
            data = list(final.rows or ())
            if not data and final.scalar is not None:
                data = [{(final.measure_label or "result"): final.scalar}]
                row_count = 1
            else:
                row_count = final.total if final.total is not None else len(data)
        else:
            data = []
            for s in sinks:
                data.extend(_branch_rows(ledger.results[s.step_id]))
            row_count = len(data)
        files_used = sorted({
            r.table for r in ledger.results.values() if r.table
        })

        _pop_store(req_id)

        trace = ctx.get("trace")
        if trace is not None:
            try:
                trace.set_execution_outcome(
                    rows=len(data), total=row_count or 0, duration_ms=0.0,
                )
                trace.emit()
            except Exception:  # noqa: BLE001 — trace must never break the answer
                pass

        logger.info(
            "navigator_answer",
            n_steps=len(ledger.results),
            files_used=files_used,
            row_count=row_count,
        )
        return {
            "answer": answer,
            "data": data,
            "chart": None,
            "route": "navigator",
            "row_count": row_count,
            "files_used": files_used,
            "tool_calls": 0,
            "retrieved_files": ctx.get("catalog_len", 0),
            "total_files": ctx.get("total_files", 0),
        }
    except Exception as exc:  # noqa: BLE001 — NEVER raise; abstain to the agent
        return _abstain("navigator_seam_error", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# resolved-table file binding (for downstream JOIN steps)
# ---------------------------------------------------------------------------
def _resolved_file_id(vc: VerifiedContract, slice_: CandidateSlice | None) -> str:
    """The file_id of the verified table, looked up from the candidate slice by the
    (case-insensitive) logical table name the brain chose. '' when not resolvable.
    The slice — not the evidence packet — is the carrier of file_id. Pure."""
    if slice_ is None:
        return ""
    target = str(vc.table or "").upper()
    for c in slice_.candidates:
        if str(c.table or "").upper() == target:
            return str(c.file_id or "")
    return ""

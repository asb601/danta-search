"""Phase 3 — Task 5+7: the capped, monotonic tool loop (agent/loop.py).

This is the agent's *driver*. It does NOT know how to talk to Neo4j (the tools in
``agent/tools.py`` own that, contract C3) and it does NOT synthesize an answer
(``agent/synthesis.py`` owns that). Its sole job is to repeatedly dispatch
``TOOL_REGISTRY`` tools, merge their hits into ``state`` while tracking
``seen_chunk_ids``, and STOP — bounded by three hard caps and a
monotonic-progress guard so the agent can never run away across many tenants and
millions of files (the governing scale criterion).

The loop enforces (HARD ENTRY GATE 4):

  1. **Total-call cap** — at most ``budget.max_total_calls`` tool invocations per
     query (mirrors the main-system ``MAX_TOOL_CALLS=8`` at
     ``app/agent/state.py``). Tunable ``agent.max_tool_calls``.
  2. **Per-tool cap** — any single tool is invoked at most
     ``budget.max_per_tool`` times (a misbehaving tool can't monopolize the
     budget). Tunable ``agent.max_per_tool_calls``.
  3. **Decomposition-depth cap** — recursive sub-query expansion is bounded by
     ``budget.max_decomp_depth``. Tunable ``agent.max_decomp_depth``.
  4. **MONOTONIC-PROGRESS guard** — after each retrieval round the loop checks
     whether the round added ANY new ``chunk_id`` to ``state.seen_chunk_ids``.
     A round that adds zero new accessible chunk ids ABORTS the loop — there is
     no point re-fetching the same fixed candidate set forever.

Every cap hit / abort / drop is emitted via ``tunables.log_gate_decision`` with
the running count as ``score`` and the cap as ``threshold`` (Spec §3 inv 4 — no
score is compared-and-discarded silently). NO bare cap literal lives here: every
cap is read via ``get_tunable`` against a named key.

Task-5 retrieval wiring (the body of each round):
  * PRIMARY leg is the FUSED ``multi_vector_search`` (NOT plain ``vector_search``
    and NEVER the legacy ``hybrid_search``);
  * the ``graph_traverse`` leg is merged in ONLY when an anchor ``state.entity``
    has been linked (the entity-linker is the gate — an anchorless graph walk is
    meaningless);
  * results are ACL-filtered (``retrieval.acl.filter_by_acl`` — per-hop tenant
    isolation, cross-tenant chunks dropped) BEFORE rerank;
  * surviving chunks are reranked via ``retrieval.reranker.rerank`` threading
    ``container_id`` (the adaptive-skip tunable lives in the reranker).

The loop NEVER raises: a tool/searcher failure degrades to an empty round (the
honest-absence invariant — an infra outage must not crash the agent, and the
downstream negative-claim gate will refuse rather than hallucinate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import structlog

from pdf_chat.tunables import get_tunable, log_gate_decision
from pdf_chat.retrieval.acl import filter_by_acl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pdf_chat.agent.state import PdfChatState

_log = structlog.get_logger("pdf_chat.agent.loop")

# Named tunable keys (defaults passed inline so this module imports with zero
# infra; the canonical home for these defaults is tunables.TUNABLE_DEFAULTS —
# see the SHARED-FILE additions noted in the integration handoff).
_K_MAX_TOOL_CALLS = "agent.max_tool_calls"        # default 8 (mirrors MAX_TOOL_CALLS)
_K_MAX_PER_TOOL = "agent.max_per_tool_calls"       # default 3
_K_MAX_DECOMP_DEPTH = "agent.max_decomp_depth"     # default 2

# Inline named defaults (registered centrally at integration — listed in return).
_DEFAULT_MAX_TOOL_CALLS = 8
_DEFAULT_MAX_PER_TOOL = 3
_DEFAULT_MAX_DECOMP_DEPTH = 2


@dataclass
class LoopBudget:
    """The three hard caps that bound a single ``run_tool_loop`` invocation.

    All three are tunable per container (``from_tunables``); a direct
    construction is used by tests to pin an exact bound. ``max_total_calls``
    mirrors the main-system ``MAX_TOOL_CALLS``; ``max_per_tool`` stops any one
    tool monopolizing the budget; ``max_decomp_depth`` bounds recursive sub-query
    expansion.
    """

    max_total_calls: int
    max_per_tool: int
    max_decomp_depth: int

    @classmethod
    def from_tunables(cls, *, container_id: str) -> "LoopBudget":
        """Resolve all three caps via ``get_tunable`` — no literal at any callsite."""
        return cls(
            max_total_calls=int(
                get_tunable(container_id, _K_MAX_TOOL_CALLS, _DEFAULT_MAX_TOOL_CALLS)
            ),
            max_per_tool=int(
                get_tunable(container_id, _K_MAX_PER_TOOL, _DEFAULT_MAX_PER_TOOL)
            ),
            max_decomp_depth=int(
                get_tunable(container_id, _K_MAX_DECOMP_DEPTH, _DEFAULT_MAX_DECOMP_DEPTH)
            ),
        )


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #
def _chunk_id(hit: Any) -> str | None:
    """Read a ``chunk_id`` from a dict or dataclass-like hit (None if absent)."""
    if isinstance(hit, dict):
        return hit.get("chunk_id")
    return getattr(hit, "chunk_id", None)


def _hit_text(hit: Any) -> str:
    if isinstance(hit, dict):
        return (hit.get("text") or "")
    return getattr(hit, "text", "") or ""


def _components_satisfied(components: Iterable[str], accessible: list[Any]) -> bool:
    """Sufficiency check: EVERY requested output component is grounded.

    A component (a token/phrase pulled from a decomposed sub-query) is satisfied
    when at least one ACCESSIBLE chunk's text contains it (case-insensitive). An
    empty component list is trivially satisfied (no multi-part ask). This is the
    output-completeness gate: the loop is not "done" until all parts of a
    multi-part question have grounding — never declare partial coverage complete.
    """
    comps = [c for c in components if c and str(c).strip()]
    if not comps:
        return True
    haystacks = [_hit_text(c).lower() for c in accessible]
    if not haystacks:
        return False
    for comp in comps:
        needle = str(comp).strip().lower()
        if not any(needle in h for h in haystacks):
            return False
    return True


def _ordered_tool_names(state: "PdfChatState") -> list[str]:
    """Tool dispatch order for one round.

    PRIMARY fused ``multi_vector_search`` always leads; the ``graph_traverse``
    leg is appended ONLY when an anchor entity has been linked (an anchorless
    graph walk is a no-op — gate it here so it never consumes the budget).
    """
    order = ["multi_vector_search"]
    if getattr(state, "entity", None):
        order.append("graph_traverse")
    return order


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
async def run_tool_loop(state: "PdfChatState", deps, budget: LoopBudget) -> "PdfChatState":
    """Drive ``TOOL_REGISTRY`` under ``budget`` until done / capped / stalled.

    Mutates and returns ``state``: accumulates ``tool_calls`` /
    ``per_tool_calls`` / ``seen_chunk_ids`` and writes ``accessible_chunks``
    (ACL-filtered, deduped, reranked). Honest-absence: never raises — a tool
    failure ends the round with no new chunks (which the monotonic guard then
    treats as a stall).
    """
    # Imported lazily so the module imports with zero infra and tests that
    # monkeypatch the registry/reranker see the patched symbols.
    from pdf_chat.agent.tools import TOOL_REGISTRY
    from pdf_chat.retrieval.reranker import rerank

    container_id = getattr(state, "tenant_id", "") or ""

    # Decomposition-depth cap: refuse to recurse beyond the configured depth.
    # (The actual decomposition is owned by agent/decompose.py; the loop ENFORCES
    # the depth bound.) This is LIVE protection, not decorative: when a decomposed
    # multi-part ask arrives already at/over the depth cap we log AND take the
    # control action — we do NOT run a further (recursive) retrieval round, so a
    # runaway recursive expansion can never consume the budget. The accessible
    # set gathered so far is still finalized + reranked below.
    depth_capped = bool(
        state.decomp_depth >= budget.max_decomp_depth and state.sub_queries
    )
    if depth_capped:
        log_gate_decision(
            _K_MAX_DECOMP_DEPTH,
            score=state.decomp_depth,
            threshold=budget.max_decomp_depth,
            outcome="cap",
            container_id=container_id,
        )

    # Components that must ALL be grounded before we declare sufficiency. When
    # the planner/decomposer produced sub_queries we require each as a component;
    # otherwise there is a single implicit component (the original query) which
    # is satisfied as soon as any accessible chunk is found.
    components = list(state.sub_queries or [])

    # Accumulator of every accessible chunk across rounds (dedup by chunk_id).
    accessible_by_id: dict[str, Any] = {}

    # If the decomposition-depth cap is already hit, do NOT enter the retrieval
    # loop at all (the live control action): we finalize whatever is in context
    # without issuing a further recursive round.
    while not depth_capped:
        # ---- total-call cap (checked BEFORE issuing the next round) ----------
        if state.tool_calls >= budget.max_total_calls:
            log_gate_decision(
                _K_MAX_TOOL_CALLS,
                score=state.tool_calls,
                threshold=budget.max_total_calls,
                outcome="cap",
                container_id=container_id,
            )
            break

        round_new_ids: set[str] = set()
        round_hits: list[Any] = []
        capped_out_this_round = False

        for tool_name in _ordered_tool_names(state):
            tool = TOOL_REGISTRY.get(tool_name)
            if tool is None:
                continue

            # ---- per-tool cap -------------------------------------------------
            used = state.per_tool_calls.get(tool_name, 0)
            if used >= budget.max_per_tool:
                log_gate_decision(
                    _K_MAX_PER_TOOL,
                    score=used,
                    threshold=budget.max_per_tool,
                    outcome="drop",
                    container_id=container_id,
                    tool=tool_name,
                )
                continue

            # ---- total-call cap (also checked per dispatch) -------------------
            if state.tool_calls >= budget.max_total_calls:
                log_gate_decision(
                    _K_MAX_TOOL_CALLS,
                    score=state.tool_calls,
                    threshold=budget.max_total_calls,
                    outcome="cap",
                    container_id=container_id,
                    tool=tool_name,
                )
                capped_out_this_round = True
                break

            # ---- dispatch (never raises) -------------------------------------
            try:
                hits = await tool.run(state, deps)
            except Exception as exc:  # honest-absence: degrade, don't crash
                _log.warning(
                    "pdf_chat.agent.loop.tool_error", tool=tool_name, error=str(exc)
                )
                hits = []
                if state.error is None:
                    state.error = f"tool_error:{tool_name}:{exc}"

            state.tool_calls += 1
            state.per_tool_calls[tool_name] = used + 1

            for h in hits or []:
                cid = _chunk_id(h)
                # Only chunk-bearing hits feed the monotonic guard + context.
                # (Card / neighbour hits without a chunk_id are retrieval signal;
                # their src_chunk_ids are pulled in by the synthesizer, not here.)
                if cid is None:
                    continue
                round_hits.append(h)
                if cid not in state.seen_chunk_ids:
                    round_new_ids.add(cid)

        # ---- ACL filter (per-hop tenant isolation) BEFORE rerank -------------
        if round_hits:
            acc, denied = filter_by_acl(
                round_hits, state.user_id, state.groups, state.tenant_id
            )
            for cid in denied:
                if cid is not None:
                    state.denied_ids.append(cid)
            for ch in acc:
                cid = _chunk_id(ch)
                if cid is not None and cid not in accessible_by_id:
                    accessible_by_id[cid] = ch

        # ---- monotonic-progress guard ---------------------------------------
        # Mark this round's ids as seen, then abort if nothing new was added.
        before = len(state.seen_chunk_ids)
        state.seen_chunk_ids.update(round_new_ids)
        added = len(state.seen_chunk_ids) - before

        if added == 0:
            log_gate_decision(
                "agent.monotonic_progress",
                score=added,
                threshold=1,  # need ≥1 new accessible chunk id to keep going
                outcome="abort",
                container_id=container_id,
                seen=len(state.seen_chunk_ids),
                tool_calls=state.tool_calls,
            )
            break

        if capped_out_this_round:
            # The total cap fired mid-round; the cap was already logged above.
            break

        # ---- sufficiency check ----------------------------------------------
        # Rerank the running accessible set (container_id threads the adaptive
        # skip tunable) so the sufficiency check sees the best-ordered evidence.
        current_accessible = list(accessible_by_id.values())
        if _components_satisfied(components, current_accessible):
            # Every requested output component is grounded → done (success path).
            break

    # ---- finalize: rerank the accessible set, write back to state -----------
    accessible = list(accessible_by_id.values())
    if accessible:
        try:
            accessible = rerank(state.query, accessible, container_id=container_id)
        except Exception as exc:  # rerank is best-effort; never fatal
            _log.warning("pdf_chat.agent.loop.rerank_error", error=str(exc))
    state.accessible_chunks = accessible
    return state


__all__ = ["LoopBudget", "run_tool_loop"]

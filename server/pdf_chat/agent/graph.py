"""PDF RAG agent — the query execution state machine (Spec §6).

Design goals:
  * Pure orchestration. Each node is an ``async def node(state, deps) -> state``
    that reads/writes ``PdfChatState`` and depends only on the injected ``deps``
    adapters (searcher, cache, reranker, llm, audit, embedder, extractor). This
    makes the whole pipeline testable with in-memory fakes and zero infra.
  * Guarded langgraph import. If ``langgraph`` is installed we expose a compiled
    StateGraph; otherwise (and for the simple linear flow) ``run_pdf_chat`` runs
    the nodes in sequence directly. Both paths execute the SAME node functions.
  * No top-level imports of other teams' modules. Real adapters are wired via
    ``build_default_deps()`` using late/guarded imports from ``retrieval/``.

Node order (Spec §6):
  embed_query → cache_check → hybrid_retrieve → rrf_rerank → acl_filter
    → on_demand_extract → assemble_context → llm_generate → cache_write → audit

Short-circuits:
  * cache_check hit  → jump straight to the end (answer already populated).
  * acl_filter empty → set the deterministic "insufficient accessible context"
    answer and skip generation (no hallucination).
  * any node setting ``state.error`` halts the remaining pipeline.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

_logger = logging.getLogger("pdf_chat.agent")

from pdf_chat.agent.state import PdfChatState
from pdf_chat.agent.prompts import (
    SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_user_prompt,
)
from pdf_chat.config import get_pdf_settings


# --------------------------------------------------------------------------- #
# Adapter protocols — what each injected dependency must provide. Fakes in the
# tests and the real retrieval/* adapters both satisfy these.
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class Searcher(Protocol):
    def hybrid_search(
        self,
        query_vector: list[float],
        tenant_id: str,
        doc_ids: list[str] | None = None,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        entity: str | None = None,
    ) -> list[Any]:
        """Return candidate chunks with vector + graph legs already RRF-fused.

        Matches the frozen ``Neo4jSearcher.hybrid_search`` signature. May be a
        sync method (the Neo4j driver is sync); the node awaits it only when the
        adapter returns an awaitable, so both sync and async adapters work.
        """
        ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[Any], top_n: int) -> list[Any]: ...


class Cache(Protocol):
    async def get(self, key: str) -> dict | None: ...
    async def set(self, key: str, value: dict, ttl: int) -> None: ...


class Extractor(Protocol):
    async def extract(self, chunk: Any) -> Any:
        """Lazily materialize table/image content for a chunk (Spec §6 Stage 6)."""
        ...


class Llm(Protocol):
    async def generate(
        self, system: str, user: str, *, container_id: str = "", signals: dict | None = None
    ) -> str: ...


class AuditRepo(Protocol):
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
        ...


@dataclass
class Deps:
    """Injected adapters. All optional so partial pipelines / tests can omit some."""

    embedder: Embedder | None = None
    searcher: Searcher | None = None
    reranker: Reranker | None = None
    cache: Cache | None = None
    extractor: Extractor | None = None
    llm: Llm | None = None
    audit_repo: AuditRepo | None = None


# --------------------------------------------------------------------------- #
# Small helpers for reading heterogeneous chunk objects (dicts or dataclasses).
# --------------------------------------------------------------------------- #
def _attr(chunk: Any, name: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _chunk_id(chunk: Any) -> str:
    return str(_attr(chunk, "chunk_id", ""))


def _is_insufficient(accessible: list[Any]) -> bool:
    """True when too few accessible chunks remain to ground an answer.

    Uses retrieval's ``insufficient_context`` against the configured
    ``min_accessible_chunks`` floor, with a local fallback (empty-only) if the
    retrieval module is unavailable.
    """
    min_required = get_pdf_settings().min_accessible_chunks
    try:
        from pdf_chat.retrieval.acl import insufficient_context  # type: ignore

        return insufficient_context(accessible, min_required)
    except Exception:
        return len(accessible) < min_required


# --------------------------------------------------------------------------- #
# Nodes — each pure-ish: (state, deps) -> state. They mutate and return state.
# --------------------------------------------------------------------------- #
async def embed_query(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 2 — embed the query with the SAME model used at ingestion."""
    if deps.embedder is None:
        state.query_vector = []
        return state
    state.query_vector = await deps.embedder.embed(state.query)
    return state


async def cache_check(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 5 — Redis lookup keyed by query + tenant + sorted groups."""
    state.cache_key = _compute_cache_key(state)
    if deps.cache is None or state.cache_key is None:
        return state
    hit = deps.cache.get(state.cache_key)
    if inspect.isawaitable(hit):
        hit = await hit
    if hit:
        state.cached = True
        state.answer = hit.get("answer", "")
        state.citations = hit.get("citations", [])
        # Preserve reported chunk count from the cached payload.
        state.accessible_chunks = hit.get("_chunks_used_marker", [None] * hit.get("chunks_used", 0))
    return state


async def hybrid_retrieve(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 3 — Neo4j hybrid (vector ANN + graph traversal), RRF-fused.

    Delegates fusion to ``searcher.hybrid_search`` (vector + optional graph legs
    fused with ``rrf`` inside the searcher). The searcher may be sync (the real
    Neo4j driver) or async (test fakes) — we await only an awaitable result.
    """
    if deps.searcher is None:
        return state
    settings = get_pdf_settings()
    vector_top_k = state.top_k or settings.vector_top_k
    result = deps.searcher.hybrid_search(
        query_vector=state.query_vector or [],
        tenant_id=state.tenant_id,
        doc_ids=state.doc_ids,
        vector_top_k=vector_top_k,
        graph_top_k=settings.graph_top_k,
        entity=getattr(state, "entity", None),
    )
    if inspect.isawaitable(result):
        result = await result
    state.candidates = result
    return state


async def rrf_rerank(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 4 — cross-encoder rerank of the fused candidate list."""
    settings = get_pdf_settings()
    candidates = state.candidates
    if deps.reranker is not None and candidates:
        state.reranked = await deps.reranker.rerank(
            state.query, candidates, settings.rerank_top_n
        )
    else:
        state.reranked = candidates[: settings.rerank_top_n]
    return state


async def acl_filter(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 7 — drop chunks the user cannot access (late import of retrieval.filter_by_acl)."""
    chunks = state.reranked
    accessible, denied = _apply_acl(chunks, state.user_id, state.groups, state.tenant_id)
    state.accessible_chunks = accessible
    state.denied_ids = denied
    return state


async def on_demand_extract(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 6 — lazily materialize table/image chunks that survived ACL."""
    if deps.extractor is None or not state.accessible_chunks:
        return state
    materialized: list[Any] = []
    for chunk in state.accessible_chunks:
        etype = _attr(chunk, "element_type")
        etype_val = getattr(etype, "value", etype)
        if etype_val in ("table", "image") and not _attr(chunk, "text"):
            materialized.append(await deps.extractor.extract(chunk))
        else:
            materialized.append(chunk)
    state.accessible_chunks = materialized
    return state


async def assemble_context(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 8 — build the numbered [N] context block + citation map.

    Enforces a per-container context token budget (Spec §2 L4 token guard #8):
    chunks are admitted in order until the running token estimate would exceed
    the budget; the drop is logged via log_gate_decision. The token estimate is
    the whitespace word count scaled by a configurable tokens-per-word multiplier
    (``context_tokens_per_word``, ≈1.3) so the guard is CONSERVATIVE vs real BPE
    tokens, and it counts the citation scaffolding (``[N] ... Source: doc, page``)
    too — not just the raw chunk text — so the budget reflects the real prompt.
    """
    from pdf_chat.tunables import get_tunable, log_gate_decision

    container_id = getattr(state, "tenant_id", "")
    budget = get_tunable(container_id, "context_token_budget")
    tokens_per_word = get_tunable(container_id, "context_tokens_per_word")

    def _est_tokens(s: str) -> int:
        return int(len(s.split()) * tokens_per_word)

    lines: list[str] = []
    citations: list[dict] = []
    used_tokens = 0
    n = 0
    for chunk in state.accessible_chunks:
        text = _attr(chunk, "text", "") or ""
        doc_id = _attr(chunk, "doc_id", "")
        page = _attr(chunk, "page_num", 0)
        # Count the full rendered line (citation scaffolding included), not just
        # the raw text, so the budget reflects what actually reaches the model.
        rendered = f"[{n + 1}] {text}    Source: {doc_id}, page {page}"
        tok = _est_tokens(rendered)
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
        lines.append(f"[{n}] {text}    Source: {doc_id}, page {page}")
        citations.append({"n": n, "doc_id": str(doc_id), "page": int(page or 0)})
    state.context = "\n".join(lines)
    state.citations = citations
    return state


async def llm_generate(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 9 — grounded synthesis.

    Refuses (deterministic, no hallucination) when too few accessible chunks
    survived ACL filtering — using ``insufficient_context`` against the
    ``min_accessible_chunks`` floor, NOT merely the empty case (Security
    must-fix #9). A below-floor context cannot ground an answer safely.
    """
    if _is_insufficient(state.accessible_chunks):
        state.answer = INSUFFICIENT_CONTEXT_MESSAGE
        state.citations = []
        return state
    if deps.llm is None:
        state.answer = INSUFFICIENT_CONTEXT_MESSAGE
        return state
    user = build_user_prompt(state.query, state.context)
    # Thread tenant scope + escalation signals through to the model router (the
    # synthesis path routes via model_router.select_model inside the adapter).
    state.answer = await deps.llm.generate(
        SYSTEM_PROMPT,
        user,
        container_id=getattr(state, "tenant_id", "") or "",
        signals=getattr(state, "router_signals", None) or {},
    )
    return state


async def cache_write(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 10 — persist the grounded answer for cache reuse."""
    if deps.cache is None or state.cache_key is None or state.cached:
        return state
    if _is_insufficient(state.accessible_chunks):
        return state  # never cache a refusal (empty or below the floor)
    settings = get_pdf_settings()
    result = deps.cache.set(
        state.cache_key,
        {
            "answer": state.answer,
            "citations": state.citations,
            "chunks_used": state.chunks_used(),
        },
        settings.cache_ttl_seconds,
    )
    if inspect.isawaitable(result):
        await result
    return state


async def audit(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Stage 10 — record the retrieval for compliance (query_audit_log).

    Runs on BOTH the normal path AND the cache-hit short-circuit (Security
    must-fix #6): a cached answer must never bypass the audit trail. The
    ``cache_hit`` marker distinguishes the two. On a hit the accessible-chunk
    list is empty (retrieval was skipped), so ``returned_chunks`` is empty and
    the row records "this principal was served the cached answer for this key".
    """
    if deps.audit_repo is None:
        return state
    returned = [_chunk_id(c) for c in state.accessible_chunks if _chunk_id(c)]
    result = deps.audit_repo.write(
        user_id=state.user_id,
        tenant_id=state.tenant_id,
        query_hash=state.cache_key or "",
        query_text=state.query,
        returned_chunks=returned,
        denied_chunks=list(state.denied_ids),
        cache_hit=state.cached,
    )
    if inspect.isawaitable(result):
        await result
    return state


# --------------------------------------------------------------------------- #
# Guarded helpers — late/guarded imports from the retrieval team. If retrieval/
# is not yet importable, fall back to a local implementation matching the frozen
# contract signature so the agent still runs in isolation.
# --------------------------------------------------------------------------- #
def _compute_cache_key(state: PdfChatState) -> str:
    # Fold the tenant ACL epoch + queried doc-set into the key so revocation /
    # document deletes evict cached answers (Security must-fix #5).
    acl_version = getattr(state, "acl_version", "0") or "0"
    try:
        from pdf_chat.retrieval.cache import cache_key  # type: ignore

        return cache_key(
            state.query,
            state.tenant_id,
            state.groups,
            acl_version=acl_version,
            doc_ids=state.doc_ids,
        )
    except Exception:
        import hashlib

        docs_part = "*" if state.doc_ids is None else ",".join(sorted(state.doc_ids))
        payload = "|".join(
            [
                state.query,
                state.tenant_id,
                ",".join(sorted(state.groups)),
                acl_version,
                docs_part,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_acl(
    chunks: list[Any], user_id: str, groups: list[str], tenant_id: str
) -> tuple[list[Any], list[str]]:
    # FAIL CLOSED: a missing tenant_id can never satisfy a tenant equality check,
    # so we pass a sentinel that no chunk's tenant_id will equal (default None,
    # NOT the chunk's own tenant). Every chunk is then denied. Log the fallback.
    if not tenant_id:
        _logger.warning(
            "pdf_chat.acl.fail_closed: missing tenant_id on query; denying all "
            "%d candidate chunks (no tenant context to authorize against).",
            len(chunks),
        )
        return [], [_chunk_id(c) for c in chunks]
    try:
        from pdf_chat.retrieval.acl import filter_by_acl  # type: ignore

        return filter_by_acl(chunks, user_id, groups, tenant_id)
    except Exception:
        pass
    # Local fallback mirroring the spec's filter_by_acl (Stage 7). FAIL CLOSED:
    # default the chunk tenant to None (not `tenant_id`) so a chunk missing a
    # tenant is denied rather than silently authorized.
    accessible: list[Any] = []
    denied: list[str] = []
    for chunk in chunks:
        acl = _attr(chunk, "acl", {}) or {}
        chunk_tenant = _attr(chunk, "tenant_id", None)
        allowed = chunk_tenant == tenant_id and (
            user_id in acl.get("allowed_users", [])
            or any(g in acl.get("allowed_groups", []) for g in groups)
            or acl.get("public", False)
        )
        if allowed:
            accessible.append(chunk)
        else:
            denied.append(_chunk_id(chunk))
    return accessible, denied


# Ordered pipeline (used by both the langgraph and the plain runner).
_PIPELINE: list[Callable[[PdfChatState, Deps], Awaitable[PdfChatState]]] = [
    embed_query,
    cache_check,
    hybrid_retrieve,
    rrf_rerank,
    acl_filter,
    on_demand_extract,
    assemble_context,
    llm_generate,
    cache_write,
    audit,
]


# --------------------------------------------------------------------------- #
# Plain async runner (no langgraph required).
# --------------------------------------------------------------------------- #
async def run_pdf_chat(state: PdfChatState, deps: Deps) -> PdfChatState:
    """Run the full pipeline in sequence with short-circuit handling.

    * A cache hit (Stage 5) skips retrieval/synthesis entirely.
    * ``state.error`` set by any node halts the remaining nodes.
    """
    state = await embed_query(state, deps)
    if state.error:
        return state

    state = await cache_check(state, deps)
    if state.error:
        return state
    if state.cached:
        # SECURITY: a cache hit serves a stored answer but must NOT bypass the
        # audit trail. Authorization is handled by the version-keyed cache (the
        # key folds in acl_version + doc_ids, so a revoke/delete misses), so we
        # write the audit row (cache_hit=True) and serve the hit.
        state = await audit(state, deps)
        return state

    for node in _PIPELINE[2:]:  # hybrid_retrieve .. audit
        state = await node(state, deps)
        if state.error:
            return state
    return state


# --------------------------------------------------------------------------- #
# Optional compiled LangGraph (guarded import). Same node functions.
# --------------------------------------------------------------------------- #
def build_graph(deps: Deps):
    """Compile a langgraph StateGraph over the nodes, or raise if unavailable.

    Returns an object with ``.ainvoke(state)``. Callers that don't care which
    engine runs should just use ``run_pdf_chat``.
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("langgraph not installed; use run_pdf_chat instead") from exc

    def _wrap(fn):
        async def _node(state: PdfChatState) -> PdfChatState:
            return await fn(state, deps)

        return _node

    sg = StateGraph(PdfChatState)
    names = [fn.__name__ for fn in _PIPELINE]
    for fn in _PIPELINE:
        sg.add_node(fn.__name__, _wrap(fn))
    sg.set_entry_point(names[0])

    # A cache hit short-circuits retrieval/synthesis but STILL routes through
    # `audit` (Security must-fix #6) so the cached answer is recorded.
    def _after_cache(state: PdfChatState) -> str:
        return "audit" if state.cached else "hybrid_retrieve"

    sg.add_edge("embed_query", "cache_check")
    sg.add_conditional_edges(
        "cache_check", _after_cache, {"audit": "audit", "hybrid_retrieve": "hybrid_retrieve"}
    )
    for prev, nxt in zip(names[2:-1], names[3:]):
        sg.add_edge(prev, nxt)
    sg.add_edge(names[-1], END)
    return sg.compile()


def build_default_deps() -> "AgentDeps":
    """Wire the real retrieval/* + infra adapters via late/guarded imports.

    Returns an :class:`AgentDeps` (the Phase-3 superset of ``Deps``) with whatever
    adapters are importable; missing ones stay None so the pipeline degrades
    gracefully rather than failing to import. The API layer calls this lazily
    inside the route, never at module import time.

    AgentDeps is a ``Deps`` subclass, so the legacy state-machine ``run_pdf_chat``
    path keeps working unchanged while the Phase-3 ``run_pdf_query`` agentic path
    reads the additional planner/tools/loop/synthesis seams. The Phase-3 modules
    (planner/entity_linker/loop/synthesis/negative_claim) are PURE functions
    dispatched by ``run_pdf_query`` — they take ``deps`` rather than being stored
    as adapters, so wiring them is a matter of making the underlying ``searcher``
    (tools), ``llm`` (planner+synthesis), and ``embedder`` available here.
    """
    deps = AgentDeps()
    # The cache is built first so the query embedder can reuse it for the
    # model-scoped query-embedding cache (cache is an optimization, never a
    # dependency — embedding still works if it is None).
    try:
        from pdf_chat.retrieval.cache import RedisCache  # type: ignore

        deps.cache = RedisCache()
    except Exception:
        pass
    # Each adapter is wired independently so one missing module doesn't blank the rest.
    try:
        from pdf_chat.retrieval.embeddings import QueryEmbedder  # type: ignore

        deps.embedder = QueryEmbedder(cache=deps.cache)
    except Exception:
        pass
    try:
        from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher  # type: ignore

        deps.searcher = Neo4jSearcher()
    except Exception:
        pass
    try:
        from pdf_chat.retrieval.reranker import CrossEncoderReranker  # type: ignore

        deps.reranker = CrossEncoderReranker()
    except Exception:
        pass
    try:
        from pdf_chat.retrieval.extractor import OnDemandExtractor  # type: ignore

        deps.extractor = OnDemandExtractor()
    except Exception:
        pass
    try:
        from pdf_chat.retrieval.llm import PdfLlm  # type: ignore

        deps.llm = PdfLlm()
    except Exception:
        pass
    try:
        from pdf_chat.agent.audit import QueryAuditRepo  # type: ignore

        deps.audit_repo = QueryAuditRepo()
    except Exception:
        pass
    return deps


# =========================================================================== #
# Phase 3 — Agentic LangGraph runtime (contract C4)
#
# This is the public entry that supersedes the fixed 10-stage state machine for
# query execution. It REUSES every Phase-3 module (planner, entity_linker, the
# capped tool loop, synthesis with the tag_as_answer HARD GATE, and the
# negative-claim/conflict gate) and threads per-hop tenant isolation everywhere.
#
# The legacy ``run_pdf_chat`` path above is intentionally preserved (the API
# route + eval harness + 18 existing agent tests depend on it). The Phase-3
# pipeline is additive: ``run_pdf_query`` is the new C4 surface.
#
# INFRA-BOOTSTRAP NOTE (Phase 6 / ops — NOT implemented here): the PRIMARY
# ``multi_vector_search`` tool and ``community_report_lookup`` read three Neo4j
# vector indexes that must be created at infra bootstrap:
#   * ``section_card_vector_index``
#   * ``doc_card_vector_index``
#   * ``community_report_vector_index``
# No DDL is issued from this module; the searcher consumes the indexes by name.
# =========================================================================== #
from dataclasses import field as _field  # noqa: E402  (local alias, late import ok)


@dataclass
class AgentDeps(Deps):
    """Phase-3 dependency bundle — a superset of the legacy ``Deps``.

    The Phase-3 nodes are pure functions that receive ``deps`` and read its
    adapters: ``searcher`` (the tool registry's backend, per-hop tenant), ``llm``
    (planner classification + grounded synthesis, both routed via the model
    router), ``embedder`` (query vector), ``cache`` (bypass short-circuit), and
    ``audit_repo`` (compliance trail). It inherits every legacy field so an
    ``AgentDeps`` also drives ``run_pdf_chat`` unchanged.
    """

    # No new fields are required — the Phase-3 modules consume the existing
    # adapters. The subclass exists so callers/tests can distinguish the agentic
    # bundle and so future agentic-only seams have a home without touching Deps.


@dataclass
class PdfQueryResult:
    """The C4 public result of ``run_pdf_query`` (plan-locked).

    * ``answer``      — the grounded (or honestly-refused) answer text.
    * ``citations``   — ``[{n, doc_id, page, bbox, ...}]`` for the cited evidence.
    * ``intent``      — the planner's typed intent (local|global|cross_domain|definitional).
    * ``provenance``  — citation index ``n`` → label (stated|inferred|conflicting|not_found).
    * ``conflicts``   — surfaced three-state source conflicts (never silently resolved).
    """

    answer: str
    citations: list[dict] = _field(default_factory=list)
    intent: str = "local"
    provenance: dict = _field(default_factory=dict)
    conflicts: list[dict] = _field(default_factory=list)


def _result_from_state(state: PdfChatState) -> PdfQueryResult:
    """Project the terminal ``PdfChatState`` onto the C4 result surface."""
    return PdfQueryResult(
        answer=state.answer,
        citations=list(state.citations or []),
        intent=getattr(state, "intent", "local") or "local",
        provenance=dict(getattr(state, "provenance", {}) or {}),
        conflicts=list(getattr(state, "conflicts", []) or []),
    )


# --------------------------------------------------------------------------- #
# Phase-3 agent nodes — each (state, deps) -> state, dispatched by run_pdf_query
# and by the guarded LangGraph builder. Pure orchestration over injected deps;
# every Phase-3 module is imported lazily so this file imports with zero infra.
# --------------------------------------------------------------------------- #
async def plan_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Typed-intent classification + bypass decision (planner.py).

    A cached query (or a high-confidence simple query) sets ``state.bypass`` so
    the loop is skipped. The planner NEVER raises — a backend/parse failure
    degrades to a typed ``fallback_reason`` and ``bypass=False`` (the safe,
    fully-grounded loop path). Router signals (cross_domain/definitional) are
    stashed on the state for the synthesis model router.
    """
    from pdf_chat.agent.planner import plan_query

    container_id = state.tenant_id or ""
    plan = await plan_query(
        state.query, container_id=container_id, llm=getattr(deps, "llm", None),
        cached=bool(state.cached),
    )
    state.intent = plan.intent
    state.planner_confidence = plan.confidence
    state.fallback_reason = plan.fallback_reason
    state.bypass = plan.bypass
    # Merge planner signals (don't clobber any caller-seeded signals).
    merged = dict(state.router_signals or {})
    merged.update(plan.signals or {})
    state.router_signals = merged
    return state


async def decompose_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Populate ``state.sub_queries`` / ``state.output_components`` for a
    multi-part ask BEFORE the tool loop runs (closes the orphaned planner signal).

    Only fires when the planner flagged ``multi_part`` (``state.router_signals``).
    The split is data-driven (decompose.py — LLM model-router seam with a
    deterministic clause fallback); a result of ≤1 component means the query is
    effectively single-part and nothing is populated (the loop then gates on the
    single implicit whole-query component, unchanged). Never raises.
    """
    from pdf_chat.agent.decompose import decompose_query

    if not bool((state.router_signals or {}).get("multi_part")):
        return state
    container_id = state.tenant_id or ""
    components = await decompose_query(
        state.query,
        container_id=container_id,
        llm=getattr(deps, "llm", None),
        signals=state.router_signals,
    )
    if len(components) >= 2:
        state.output_components = list(components)
        # The loop gates sufficiency on sub_queries; each component is a required
        # sub-query so the loop is not "done" until all parts have grounding.
        state.sub_queries = list(components)
    return state


async def link_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Resolve a graph anchor into ``state.entity`` before the graph leg runs."""
    from pdf_chat.agent.entity_linker import link_entities

    return await link_entities(state, deps, container_id=state.tenant_id or "")


async def loop_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Run the capped, monotonic tool loop (loop.py) — multi_vector_search PRIMARY.

    The loop reads ``TOOL_REGISTRY`` (so Phase-4/5 tools register without a loop
    change), threads per-hop tenant isolation on every searcher leg, ACL-filters
    before rerank, and aborts on the total/per-tool/decomp caps or a no-new-chunk
    round. It writes ``state.accessible_chunks``.
    """
    from pdf_chat.agent.loop import LoopBudget, run_tool_loop

    budget = LoopBudget.from_tunables(container_id=state.tenant_id or "")
    return await run_tool_loop(state, deps, budget)


async def synthesize_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Grounded synthesis with the tag_as_answer HARD GATE + citation floor.

    Writes ``state.answer`` / ``state.citations`` / ``state.provenance`` (and
    surfaces conflicts onto ``state.conflicts`` via the shared conflict gate).
    Refuses deterministically (no LLM call) on insufficient context. Never raises.
    """
    from pdf_chat.agent.synthesis import synthesize

    result = await synthesize(state, deps, container_id=state.tenant_id or "")
    state.answer = result.answer
    state.citations = result.citations
    state.provenance = result.provenance
    return state


async def negative_claim_node(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Wrap the final answer with the negative-claim + conflict gate (gate 3).

    An UNPROVEN "no data / not found" claim (retrieval-empty, or coverage not
    proven) is replaced by an honest rewrite — retrieval-empty ≠ absent. A PROVEN
    absence (relevant pages in-context + diagnosed) is kept. Surfaced three-state
    conflicts ride ``state.conflicts`` (never silently resolved). Mirrors the
    main system's ``_gate_negative_claim`` wrapping on the final answer.
    """
    from pdf_chat.agent.negative_claim import (
        evaluate_pdf_negative_claim,
        pdf_honest_rewrite,
    )
    from pdf_chat.tunables import log_gate_decision

    container_id = state.tenant_id or ""
    # Reuse the verdict already computed (and memoized) by synthesis to avoid a
    # second O(n²) _detect_conflicts pass per query. Only recompute when synthesis
    # did not run (e.g. a cache-hit path) OR the answer changed since memoization,
    # so behavior is identical to the previous always-recompute path.
    verdict = getattr(state, "neg_verdict", None)
    memo_answer = getattr(verdict, "_for_answer", None) if verdict is not None else None
    if verdict is None or memo_answer != (state.answer or ""):
        verdict = evaluate_pdf_negative_claim(
            answer=state.answer or "",
            accessible_chunks=state.accessible_chunks or [],
            container_id=container_id,
        )
    if verdict.conflicts:
        state.conflicts = verdict.conflicts
    # Only rewrite an UNPROVEN negative claim. A non-negative answer, or a proven
    # absence, is left exactly as synthesized.
    if verdict.is_negative_claim and not verdict.proven:
        state.answer = pdf_honest_rewrite(verdict)
        state.citations = []
        log_gate_decision(
            "agent.negative_claim_wrap",
            score=0.0,
            threshold=1.0,
            outcome="rewrite_unproven",
            container_id=container_id,
        )
    return state


async def _bypass_retrieve(state: PdfChatState, deps: "AgentDeps") -> PdfChatState:
    """Short retrieval path for a bypass that is NOT a cache hit.

    A confident simple query still needs grounded evidence; rather than the full
    capped loop we run a SINGLE retrieval round (ACL-filtered, reranked) via the
    same loop machinery: the PRIMARY ``multi_vector_search`` leg, plus the
    ``graph_traverse`` leg when entity linking resolved an anchor (so a bypassed
    query that hinges on a named entity is not stuck with vector-only retrieval).
    The total cap allows both legs of the one round; the monotonic-progress guard
    then stops any second round. This keeps the bypass fast while never asserting
    an ungrounded answer.
    """
    from pdf_chat.agent.loop import LoopBudget, run_tool_loop

    # 2 total calls = PRIMARY + (optional) graph leg in a single round; 1 per
    # tool so neither leg repeats. (Literals here are an internal bypass budget,
    # not a tunable gate — the capped LOOP path is the tunable one.)
    budget = LoopBudget(max_total_calls=2, max_per_tool=1, max_decomp_depth=0)
    return await run_tool_loop(state, deps, budget)


# --------------------------------------------------------------------------- #
# The public C4 entry
# --------------------------------------------------------------------------- #
async def run_pdf_query(
    query: str,
    *,
    tenant_id: str,
    container_id: str,
    user_id: str = "",
    groups: list[str] | None = None,
    doc_ids: list[str] | None = None,
    deps: "AgentDeps | None" = None,
) -> PdfQueryResult:
    """Execute a query end-to-end and return a grounded :class:`PdfQueryResult`.

    Pipeline (the SAME nodes the guarded LangGraph builder runs):

      embed → cache_check → plan
        ├─ bypass (cache hit)      → audit → return cached answer
        ├─ bypass (no cache hit)   → short multi_vector_search retrieve → acl
        │                            → synthesize → negative_claim → cache → audit
        └─ loop path               → link_entities → run_tool_loop
                                     → synthesize → negative_claim → cache → audit

    Tenant isolation: ``tenant_id`` is threaded onto the state and every searcher
    leg (per-hop). ``container_id`` scopes every tunable + gate-decision log.
    ``deps=None`` wires real adapters via ``build_default_deps`` (guarded).

    Never raises: a backend failure degrades to a deterministic refusal rather
    than a hallucinated answer (the honest-absence invariant).
    """
    if deps is None:
        deps = build_default_deps()

    state = PdfChatState(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        groups=list(groups or []),
        doc_ids=list(doc_ids) if doc_ids is not None else None,
    )

    # Stage 2/5 — embed + cache lookup (reuses the legacy nodes verbatim).
    state = await embed_query(state, deps)
    state = await cache_check(state, deps)

    # Plan: typed intent + bypass (a cache hit forces bypass=True downstream).
    state = await plan_node(state, deps)

    if state.cached:
        # A cache hit serves the stored answer but MUST still be audited
        # (Security must-fix #6) — the answer already populated by cache_check.
        state = await audit(state, deps)
        return _result_from_state(state)

    # Multi-part decomposition: populate components BEFORE the loop so the
    # sufficiency gate (loop._components_satisfied) sees real components and a
    # multi-part ask is never truncated after the first grounded chunk. (A
    # multi-part query never bypasses — the planner forces the loop — but we run
    # this before the branch so a future bypass+multi_part stays correct.)
    state = await decompose_node(state, deps)

    if state.bypass:
        # Confident simple query, no cache hit: a single PRIMARY retrieval round.
        # Entity linking runs here too (it is a cheap no-op when nothing resolves)
        # so a bypassed query that hinges on a named entity still gets the graph
        # leg rather than vector-only retrieval.
        state = await link_node(state, deps)
        state = await _bypass_retrieve(state, deps)
    else:
        # Full agentic path: link a graph anchor, then run the capped loop.
        state = await link_node(state, deps)
        state = await loop_node(state, deps)

    state = await synthesize_node(state, deps)
    state = await negative_claim_node(state, deps)

    # Cache the grounded answer (never a refusal — the legacy cache_write already
    # refuses to cache below-floor contexts) + audit the retrieval.
    state = await cache_write(state, deps)
    state = await audit(state, deps)
    return _result_from_state(state)


# Ordered Phase-3 pipeline (used by the guarded LangGraph builder).
def _agent_nodes():
    """The Phase-3 node sequence as (name, fn) pairs (lazy so import stays clean)."""
    return [
        ("embed_query", embed_query),
        ("cache_check", cache_check),
        ("plan", plan_node),
        ("decompose", decompose_node),
        ("link_entities", link_node),
        ("run_tool_loop", loop_node),
        ("synthesize", synthesize_node),
        ("negative_claim", negative_claim_node),
        ("cache_write", cache_write),
        ("audit", audit),
    ]


def build_agent_graph(deps: "AgentDeps"):
    """Compile a guarded LangGraph over the SAME Phase-3 nodes, or raise.

    Mirrors the legacy ``build_graph`` guard: if ``langgraph`` is importable we
    return a compiled ``StateGraph`` with ``.ainvoke(state)``; otherwise we raise
    a ``RuntimeError`` pointing callers at ``run_pdf_query`` (the plain runner
    that executes the identical node functions). The graph branches after the
    planner: a cache hit jumps to ``audit``; a bypass skips link+loop straight to
    ``synthesize``; otherwise the full link → loop path runs.
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "langgraph not installed; use run_pdf_query instead"
        ) from exc

    def _wrap(fn):
        async def _node(state: PdfChatState) -> PdfChatState:
            return await fn(state, deps)

        return _node

    sg = StateGraph(PdfChatState)
    for name, fn in _agent_nodes():
        sg.add_node(name, _wrap(fn))
    sg.set_entry_point("embed_query")
    sg.add_edge("embed_query", "cache_check")
    sg.add_edge("cache_check", "plan")

    # A cache hit jumps straight to audit; otherwise decompose (populate
    # components for a multi-part ask) then link an entity (on BOTH the bypass
    # and loop paths — the linker is a cheap no-op when nothing resolves).
    def _after_plan(state: PdfChatState) -> str:
        return "audit" if state.cached else "decompose"

    sg.add_conditional_edges(
        "plan", _after_plan, {"audit": "audit", "decompose": "decompose"}
    )
    sg.add_edge("decompose", "link_entities")

    # After entity linking the bypass path skips the loop and goes straight to
    # synthesis; the full agentic path runs the capped tool loop.
    def _after_link(state: PdfChatState) -> str:
        return "synthesize" if state.bypass else "run_tool_loop"

    sg.add_conditional_edges(
        "link_entities",
        _after_link,
        {"synthesize": "synthesize", "run_tool_loop": "run_tool_loop"},
    )
    sg.add_edge("run_tool_loop", "synthesize")
    sg.add_edge("synthesize", "negative_claim")
    sg.add_edge("negative_claim", "cache_write")
    sg.add_edge("cache_write", "audit")
    sg.add_edge("audit", END)
    return sg.compile()

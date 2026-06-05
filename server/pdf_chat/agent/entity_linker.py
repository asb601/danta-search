"""Phase-3 entity linking — resolve a query to a *graph* entity (contract C3).

`link_entities(state, deps)` runs BEFORE the graph tools (`graph_traverse`,
`get_entity_neighbors`) become reachable. It populates `state.entity` with the
canonical name of the highest-confidence entity the query mentions *that
actually exists in this tenant's graph*. If nothing links above a tunable
confidence floor, `state.entity` stays `None` and the graph leg is skipped — the
loop falls back to vector-only retrieval. This gate is what keeps a query that
names nothing in the graph from triggering a wasteful (and potentially
cross-tenant) relationship walk.

Why a gate, not a guess (Spec §3 invariants 2, 3, 4, 7):
  * **Honest absence (2):** an unrecognized entity is left unlinked rather than
    fuzz-matched to "the closest thing" — the graph leg simply doesn't run.
  * **Per-hop tenant isolation (3):** every searcher probe threads `tenant_id`
    (and the optional `doc_ids` subset); resolution is scoped to the tenant's
    graph, never a global entity table.
  * **No magic literal (4):** the confidence floor is read via
    `get_tunable(container_id, "agent.entity_link_min_confidence", ...)` and the
    link/skip decision is emitted via `log_gate_decision`.
  * **Linkage discovered, not assumed (7):** candidate surface forms are derived
    from the query text (data-driven), then *confirmed against the graph* — we
    never carry a hardcoded entity dictionary or dataset-fitted hint list.

Resolution seams (the linker uses whatever the injected searcher exposes):
  1. `searcher.resolve_entity(text, tenant_id, doc_ids=..., limit=...)` →
     `[{name, score, ...}]` (preferred — an explicit, scored resolver, e.g. an
     embedding/ANN lookup over the entity vector space).
  2. else `searcher.entity_neighbors(entity, tenant_id, doc_ids=...)` — a
     candidate that is a real graph entity yields a non-empty neighbourhood; we
     treat existence as confirmation and score by lexical overlap with the query.

Pure async orchestration over the injected `deps` — no infra import at module
load, so this is unit-testable with an in-memory fake searcher. NEVER raises: a
searcher error or a missing searcher degrades to "no entity linked".
"""
from __future__ import annotations

import logging
from typing import Any

from pdf_chat.agent.state import PdfChatState
from pdf_chat.tunables import get_tunable, log_gate_decision

_logger = logging.getLogger("pdf_chat.agent.entity_linker")

# Named tunable for the link confidence floor (registered default lives below so
# the dial is discoverable + per-container overridable per Spec §3 inv 4). We do
# NOT register it in tunables.TUNABLE_DEFAULTS from here (that is a shared file,
# owned by another task) — instead we always pass an explicit named default to
# get_tunable, and list the registry addition in the integration notes.
_ENTITY_LINK_MIN_CONFIDENCE = "agent.entity_link_min_confidence"
_ENTITY_LINK_MIN_CONFIDENCE_DEFAULT = 0.50

# Candidate-extraction dials (data-driven surface-form generation, not a hint
# list). Also passed as explicit named defaults so no bare literal escapes.
_ENTITY_LINK_MIN_TOKEN_LEN = "agent.entity_link_min_token_len"
_ENTITY_LINK_MIN_TOKEN_LEN_DEFAULT = 3
_ENTITY_LINK_MAX_CANDIDATES = "agent.entity_link_max_candidates"
_ENTITY_LINK_MAX_CANDIDATES_DEFAULT = 8

# Closed-class words that are never entity heads. This is grammatical scaffolding
# (stop-words), NOT a domain/dataset dictionary — it carries no business meaning
# and is identical for every tenant, so it does not violate the no-static-
# heuristics rule (which forbids dataset-fitted classification, not tokenization).
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "what", "which", "who", "whom", "whose", "was", "were", "is", "are",
        "be", "been", "being", "did", "does", "do", "how", "when", "where",
        "why", "tell", "me", "about", "show", "give", "list", "summarize",
        "summarise", "please", "report", "this", "that", "these", "those",
        "document", "documents", "file", "files", "page", "pages", "data",
    }
)


def _maybe_await(value: Any) -> Any:
    """Return a coroutine to await, or the value itself (sync searcher seam)."""
    return value


def _candidate_surface_forms(query: str, *, min_token_len: int, max_candidates: int) -> list[str]:
    """Derive ordered candidate entity surface forms from the query text.

    Data-driven (no entity list): we generate
      * contiguous spans of capitalized / numeric-bearing tokens (proper-noun
        runs like "Acme Corporation", "Q3 2026"), longest-first, AND
      * individual content tokens above ``min_token_len`` (lower-cased),
    de-duplicated, capped at ``max_candidates``. Longer, capitalized spans are
    preferred because they are the most specific surface form the graph resolver
    can confirm.
    """
    raw_tokens = [t.strip(".,;:!?()[]{}\"'") for t in query.split()]
    tokens = [t for t in raw_tokens if t]

    spans: list[str] = []
    run: list[str] = []

    def _flush() -> None:
        if run:
            spans.append(" ".join(run))

    for tok in tokens:
        # A "significant" token for a proper-noun run: starts uppercase or
        # contains a digit (e.g. "Q3", "FY26"). Purely structural.
        significant = tok[:1].isupper() or any(ch.isdigit() for ch in tok)
        if significant and tok.lower() not in _STOPWORDS:
            run.append(tok)
        else:
            _flush()
            run = []
    _flush()

    # Order: longest spans first (most specific), then the remaining content
    # tokens (lower-cased) as single-word fallbacks.
    spans.sort(key=lambda s: len(s.split()), reverse=True)

    singles = [
        t.lower()
        for t in tokens
        if len(t) >= min_token_len and t.lower() not in _STOPWORDS
    ]

    ordered: list[str] = []
    seen: set[str] = set()
    for cand in [*spans, *singles]:
        key = cand.lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(cand)
        if len(ordered) >= max_candidates:
            break
    return ordered


def _coerce_hit(hit: Any) -> tuple[str, float] | None:
    """Normalize a resolver hit to ``(name, score)``; tolerate dict / object."""
    if hit is None:
        return None
    if isinstance(hit, dict):
        name = hit.get("name") or hit.get("entity") or hit.get("normalized_value")
        score = hit.get("score")
    else:
        name = getattr(hit, "name", None) or getattr(hit, "entity", None)
        score = getattr(hit, "score", None)
    if not name:
        return None
    try:
        score_f = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        score_f = 0.0
    return str(name), score_f


async def _call_searcher(fn: Any, /, **kwargs: Any) -> Any:
    """Invoke a (sync or async) searcher method, awaiting only if needed."""
    result = fn(**kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result


async def _resolve_candidate(
    searcher: Any,
    text: str,
    *,
    tenant_id: str,
    doc_ids: list[str] | None,
    limit: int,
) -> list[tuple[str, float]]:
    """Resolve one surface form to scored graph entities via the best seam.

    Prefers an explicit ``resolve_entity`` seam (scored). Falls back to
    ``entity_neighbors``: a candidate that IS a graph entity returns a non-empty
    neighbourhood, so we treat the candidate itself as a confirmed entity with a
    score of 1.0 (existence confirmation). Both seams thread tenant + doc scope.
    """
    resolve = getattr(searcher, "resolve_entity", None)
    if callable(resolve):
        hits = await _call_searcher(
            resolve, text=text, tenant_id=tenant_id, doc_ids=doc_ids, limit=limit
        )
        out: list[tuple[str, float]] = []
        for h in hits or []:
            coerced = _coerce_hit(h)
            if coerced is not None:
                out.append(coerced)
        return out

    neighbors = getattr(searcher, "entity_neighbors", None)
    if callable(neighbors):
        rows = await _call_searcher(
            neighbors, entity=text, tenant_id=tenant_id, doc_ids=doc_ids, limit=limit
        )
        if rows:
            # Existence in the graph (≥1 neighbour) confirms the candidate as a
            # real tenant entity → full-confidence link on the candidate itself.
            return [(text, 1.0)]
        return []

    return []


async def link_entities(
    state: PdfChatState,
    deps: Any,
    *,
    container_id: str = "",
) -> PdfChatState:
    """Populate ``state.entity`` with the best graph-confirmed entity, or leave None.

    Args:
        state: the live ``PdfChatState`` (reads ``query``/``tenant_id``/``doc_ids``;
            writes ``entity``). A pre-set ``state.entity`` is respected (the
            caller pinned an anchor) — no re-resolution.
        deps: the agent ``Deps`` (uses ``deps.searcher``; the resolve/neighbors
            seam). ``None`` searcher → no-op.
        container_id: tenant/container for tunable resolution + gate logging.

    Returns:
        The same ``state`` (mutated in place) for reducer-friendly chaining.

    Never raises — any failure degrades to "no entity linked" (graph leg skipped).
    """
    # Idempotent: a caller-supplied anchor wins; never re-resolve over it.
    if state.entity:
        return state

    searcher = getattr(deps, "searcher", None)
    if searcher is None:
        return state

    floor = get_tunable(
        container_id, _ENTITY_LINK_MIN_CONFIDENCE, _ENTITY_LINK_MIN_CONFIDENCE_DEFAULT
    )
    min_token_len = get_tunable(
        container_id, _ENTITY_LINK_MIN_TOKEN_LEN, _ENTITY_LINK_MIN_TOKEN_LEN_DEFAULT
    )
    max_candidates = get_tunable(
        container_id, _ENTITY_LINK_MAX_CANDIDATES, _ENTITY_LINK_MAX_CANDIDATES_DEFAULT
    )

    candidates = _candidate_surface_forms(
        state.query, min_token_len=min_token_len, max_candidates=max_candidates
    )

    best_name: str | None = None
    best_score: float = 0.0

    for cand in candidates:
        try:
            hits = await _resolve_candidate(
                searcher,
                cand,
                tenant_id=state.tenant_id,
                doc_ids=state.doc_ids,
                limit=max_candidates,
            )
        except Exception:  # noqa: BLE001 — honest-absence: never propagate
            _logger.warning("entity_linker.resolve_failed", exc_info=True)
            hits = []
        for name, score in hits:
            if score > best_score:
                best_name, best_score = name, score

    if best_name is None:
        # Nothing in the query resolved to a graph entity at all — log the empty
        # outcome so the skipped graph leg is auditable, then leave entity None.
        log_gate_decision(
            "agent.entity_link",
            score=0.0,
            threshold=float(floor),
            outcome="no_candidate",
            container_id=container_id,
            tenant_id=state.tenant_id,
        )
        return state

    linked = best_score >= float(floor)
    log_gate_decision(
        "agent.entity_link",
        score=best_score,
        threshold=float(floor),
        outcome="linked" if linked else "skip",
        container_id=container_id,
        tenant_id=state.tenant_id,
        entity=best_name,
    )
    if linked:
        state.entity = best_name
    return state

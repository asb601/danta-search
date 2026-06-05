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

# Sentinel so callers can omit ``default`` and fall back to TUNABLE_DEFAULTS,
# keeping the registry the single source of truth (no duplicated inline literal).
_UNSET: Any = object()

# ── Model router (contract C7) ──────────────────────────────────────────────
# Named tunable keys so no model id / threshold / budget literal appears outside
# this module. Defaults live in TUNABLE_DEFAULTS (single source); the bulk id is
# resolved at call time via get_settings().chat_deployment() (DISABLE_GPT4O-aware)
# and is therefore the only key without a stored literal default.
TUN_MODEL_BULK_ID = "model.bulk_id"                       # default: chat_deployment()
TUN_MODEL_STRONG_ID = "model.strong_id"                   # default: claude-sonnet-4-6
TUN_MODEL_QUERYTIME_STRONG = "model.querytime_strong_id"  # default: claude-opus-4-8 (query-only)
TUN_MODEL_EMBEDDING_ID = "model.embedding_id"             # default: text-embedding-3-small
TUN_ESC_CONF_FLOOR = "router.escalation.conf_floor"       # extract_confidence below → signal
TUN_ESC_FIGURE_RATIO = "router.escalation.figure_ratio"   # figure/formula density above → signal
TUN_ESC_BUDGET_FRACTION = "router.escalation.budget_fraction"  # per-tenant cap (frac of pages)
TUN_ESC_BUDGET_WINDOW_PAGES = "router.escalation.budget_window"  # denominator for the fraction

# Canonical named defaults. A key MUST appear here before any module references
# it, so the full tunable surface is discoverable in one place (Spec §3 inv 4).
TUNABLE_DEFAULTS: dict[str, Any] = {
    # Phase 0 — token guards
    "context_token_budget": 8000,
    # Tokens-per-word multiplier so the whitespace word count is conservative vs
    # real BPE tokens (≈1.3 tokens/word for English). Folded into the context
    # budget guard so it trips before the real model token ceiling.
    "context_tokens_per_word": 1.3,
    "embedding_batch_size": 64,
    "query_embedding_cache_ttl": 3600,
    "rerank_top_n": 12,
    "rerank_skip_below_candidates": 4,
    # Phase 1 — extraction routing
    "digital_text_coverage": 0.70,
    "low_confidence_flag_below": 0.60,
    "ocr_table_min_confidence": 0.50,
    # Phase 0 — model router (contract C7). Model ids are tunable so a tenant
    # can pin a different strong tier; bulk default resolves to chat_deployment()
    # at call time (DISABLE_GPT4O-aware) and is therefore NOT stored here.
    "model.strong_id": "claude-sonnet-4-6",
    "model.querytime_strong_id": "claude-opus-4-8",
    "model.embedding_id": "text-embedding-3-small",
    "router.escalation.conf_floor": 0.55,
    "router.escalation.figure_ratio": 0.40,
    "router.escalation.budget_fraction": 0.05,
    # Phase 1 (addendum B) — optional DeepDoc ONNX enhancer. The enhancer code
    # already resolves these via inline named defaults; registering them here
    # keeps the single-source discoverability rule (Spec §3 inv 4) without
    # changing behavior.
    "deepdoc.complexity_threshold": 0.60,
    "deepdoc.confidence_floor": 0.55,
    "deepdoc.indent_tol_frac": 0.12,
    "deepdoc.column_max_k": 4,
    "deepdoc.tsr_min_confidence": 0.50,
    "deepdoc.rotation_min_confidence": 0.50,
    # Phase 2 — Knowledge Graph (single source of truth for every kg.* dial).
    # Each module already passes these as inline named defaults to get_tunable so
    # it stays import-safe pre-integration; registering them here restores the
    # single-source discoverability + per-container override rule (Spec §3 inv 4).
    # Sectionizer (Task 2)
    "kg.extraction.granularity": "section",
    "kg.sectionize.heading_max_words": 8,
    "kg.sectionize.heading_max_chars": 80,
    # NER / value-overlap backbone (Task 3)
    "kg.ner.max_candidates": 64,
    "kg.link.min_token_len": 3,
    "kg.link.max_value_fanout": 8,
    # Section extraction + gleaning (Task 4)
    "kg.gleaning.max_passes": 2,
    "kg.gleaning.new_entity_floor": 1,
    "kg.extraction.section_tag_cap": 5,
    # Grounding gate (Task 6)
    "kg.tag.min_confidence": 0.50,
    # Short tokens (<= this normalized length) must word-boundary match in the
    # cited span so a 2-char name ("HP"/"Q3") can't ground on an incidental
    # substring. Longer tokens use plain substring containment.
    "kg.ground.word_boundary_max_len": 3,
    # Entity resolution (Task 7)
    "kg.resolution.merge_band_quantile": 0.85,
    "kg.resolution.merge_floor": 0.60,
    "kg.resolution.cooccurrence_lift": 0.05,
    # Card builder (Task 9)
    "kg.card.section_tag_cap": 6,
    "kg.card.summary_max_chars": 480,
    # Communities + cited reports + PageRank (Task 10/11/12)
    "kg.community.resolution": 1.0,
    "kg.community.min_size": 3,
    "kg.report.min_grounded_edges": 2,
    # Multi-representation search (Task 1/8)
    "kg.multivec.top_k": 12,
    # Phase 3 — Agentic LangGraph query runtime (single source for every agent.*
    # dial). Each Phase-3 module already passes these as inline named defaults to
    # get_tunable so it stays import-safe pre-integration; registering them here
    # restores the single-source discoverability + per-container override rule
    # (Spec §3 inv 4) alongside the kg.*/model.* keys above.
    # Planner (Task 2) — high-confidence simple/cached queries bypass the loop.
    "agent.planner_bypass_confidence": 0.80,
    # Tool loop (Task 7) — hard caps; max_tool_calls mirrors the main-system
    # MAX_TOOL_CALLS=8. The monotonic-progress guard needs no literal.
    "agent.max_tool_calls": 8,
    "agent.max_per_tool_calls": 3,
    "agent.max_decomp_depth": 2,
    # Entity linking (Task 3) — graph tools are unreachable below the floor.
    "agent.entity_link_min_confidence": 0.50,
    "agent.entity_link_min_token_len": 3,
    "agent.entity_link_max_candidates": 8,
    # Synthesis (Task 8) — citation-density floor; an answer below it is refused.
    "agent.min_citations_per_claim": 1,
    # Negative-claim gate (Task 10) — coverage proof floor (retrieval-empty is
    # NOT absence): at least this many accessible chunks must be in-context before
    # a "no data" claim can be proven.
    "agent.neg_claim.min_coverage_chunks": 1,
    # Negative-claim phrase/negation lists — TUNABLE so a tenant can extend them
    # (e.g. localized phrasing) without a code change. The module passes its
    # canonical English defaults explicitly, so these are listed here purely for
    # discoverability (Spec §3 inv 4); a tenant override (env/DB) is a
    # comma/newline-separated string the gate parses + UNIONs with nothing (it
    # replaces, defaulting back to canonical when empty).
    "agent.neg_claim.phrases": None,
    "agent.neg_claim.negation_tokens": None,
    # Multi-part decomposition (decompose.py) — max output components a single
    # multi-part ask is split into (bounds the sufficiency-gate fan-out).
    "agent.decomp_max_components": 6,
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


def get_tunable(container_id: str, key: str, default: Any = _UNSET) -> Any:
    """Resolve a tunable for ``container_id`` (DB override → env → default).

    When ``default`` is omitted it falls back to ``TUNABLE_DEFAULTS[key]`` so the
    registry stays the single source of truth and no inline literal is duplicated
    at the call site (Spec §3 invariant 4).
    """
    if default is _UNSET:
        default = TUNABLE_DEFAULTS[key]
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

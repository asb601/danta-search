"""PdfChatState — the single object that flows through every agent node.

Per CONTRACTS.md (Team D owns this). Pure dataclass — no infra imports — so the
graph and its unit tests run with zero infra installed. Each retrieval/synthesis
node reads some fields and writes others; the shape is reducer-friendly (every
field has a default, so a node only sets what it produces and partial states are
always valid).

Field lifecycle (maps to Spec §6 stages):
  query/tenant_id/user_id/groups/doc_ids  → inputs (Stage 1)
  query_vector                            → embed_query (Stage 2)
  cached                                  → cache_check (Stage 5; short-circuits)
  candidates                              → hybrid_retrieve (Stage 3)
  reranked                                → rrf_rerank (Stage 4)
  accessible_chunks / denied_ids          → acl_filter (Stage 7)
  context                                 → assemble_context (Stage 8; after lazy extract Stage 6)
  answer / citations                      → llm_generate (Stage 9/10)
  error                                   → any node may set (terminal, surfaced to API)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PdfChatState:
    # --- Inputs (Stage 1) ---
    query: str
    tenant_id: str
    user_id: str = ""
    groups: list[str] = field(default_factory=list)
    doc_ids: list[str] | None = None
    top_k: int | None = None
    # Optional graph-traversal anchor entity (None → vector-only hybrid search).
    entity: str | None = None
    # Tenant ACL epoch folded into the cache key so a revoke/delete bumps it and
    # transparently invalidates every cached answer for the tenant (default "0").
    acl_version: str = "0"

    # --- embed_query (Stage 2) ---
    query_vector: list[float] | None = None

    # --- hybrid_retrieve (Stage 3) ---
    # Raw candidate chunks (each a dict / Chunk-like with chunk_id, text, acl, ...).
    candidates: list[Any] = field(default_factory=list)

    # --- rrf_rerank (Stage 4) ---
    reranked: list[Any] = field(default_factory=list)

    # --- acl_filter (Stage 7) ---
    accessible_chunks: list[Any] = field(default_factory=list)
    denied_ids: list[str] = field(default_factory=list)

    # --- assemble_context (Stage 8, with on_demand_extract Stage 6) ---
    context: str = ""

    # --- llm_generate (Stage 9/10) ---
    answer: str = ""
    citations: list[dict] = field(default_factory=list)

    # --- cache_check / cache_write (Stage 5) ---
    cached: bool = False
    cache_key: str | None = None

    # --- control ---
    error: str | None = None

    # --- Phase 3: agentic runtime (all defaulted; partial states stay valid) ---
    # Typed planner intent: local|global|cross_domain|definitional (planner.py).
    intent: str = "local"
    # Planner confidence in [0,1]; compared against a tunable bypass floor.
    planner_confidence: float = 0.0
    # Typed fallback reason when the loop is taken instead of a bypass
    # (e.g. "low_confidence:<c>", "planner_error:<E>", "ambiguous_intent").
    fallback_reason: str | None = None
    # High-confidence simple/cached query → skip the tool loop entirely.
    bypass: bool = False
    # Decomposed sub-questions for multi-part asks (decompose.py).
    sub_queries: list[str] = field(default_factory=list)
    # Requested output COMPONENTS for a multi-part ask (decompose.py). Each is a
    # short label the answer MUST cover; the loop gates sufficiency on these and
    # synthesis flags any that stay ungrounded (never silently truncated).
    output_components: list[str] = field(default_factory=list)
    # Current decomposition recursion depth (capped by a tunable).
    decomp_depth: int = 0
    # Total tool invocations this query (hard cap mirrors MAX_TOOL_CALLS).
    tool_calls: int = 0
    # Per-tool invocation counts, keyed by Tool.name (per-tool cap).
    per_tool_calls: dict[str, int] = field(default_factory=dict)
    # chunk_ids already pulled into context — drives the monotonic-progress
    # guard (a round adding zero new ids aborts the loop).
    seen_chunk_ids: set[str] = field(default_factory=set)
    # Signals passed to model_router.select_model (cross_domain, definitional, …).
    router_signals: dict = field(default_factory=dict)
    # Citation index n -> provenance label (stated|inferred|conflicting|not_found).
    provenance: dict[int, str] = field(default_factory=dict)
    # Surfaced (never silently resolved) three-state source conflicts.
    conflicts: list[dict] = field(default_factory=list)
    # Memoized negative-claim/conflict verdict (negative_claim.py). Computed ONCE
    # in synthesis and reused by the negative_claim node so the O(n²) conflict
    # detection does not run twice per query. ``Any`` to keep state infra-free.
    neg_verdict: Any | None = None
    # Per-component grounding map written at synthesis time:
    # {component_label: bool grounded}. Drives the partial-answer honesty flag so
    # a multi-part ask never silently drops a requested component.
    component_grounding: dict[str, bool] = field(default_factory=dict)

    def chunks_used(self) -> int:
        """Number of accessible chunks that fed the LLM context."""
        return len(self.accessible_chunks)

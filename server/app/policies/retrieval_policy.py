"""Retrieval orchestration policy.

Governs all candidate caps, score floors, and shortlist sizing for the
9-stage retrieval pipeline (BM25 → fuzzy → vector → graph_expand → RRF).

WHY CAPS EXIST:
  Without per-channel caps, a catalog with 10,000 files could return 10,000
  BM25 candidates and push the RRF fusion step to O(N·log N) work per request.
  Caps make retrieval latency predictable and bounded regardless of catalog size.

WHY FLOORS EXIST:
  RRF is rank-only (ignores raw scores) but very low-score candidates still
  consume rank slots and pollute the fusion. Dropping them before RRF keeps
  the rank lists meaningful.

FUTURE READINESS:
  - Deployment override: high-recall tenants could raise shortlist_top_k → 12.
  - Tenant override: a catalog with only 3 files has no benefit from a 7-item
    shortlist; floor it at 3 dynamically when supported.
  - Telemetry-driven: if retrieval_miss_count grows, raise bm25_candidates as
    a self-calibrating signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RetrievalPolicy:
    """
    All retrieval-stage thresholds in one typed, immutable config.

    Attributes
    ----------
    stage_limit : int
        Legacy generic stage limit — kept for backward compat (OpenSearch path).
        Prevents: unbounded DB result sets before fusion.

    bm25_candidates : int
        Maximum rows fetched from the BM25 keyword search stage.
        BM25 is the highest-precision channel; cap at 50 to avoid bloating RRF.
        Prevents: long tail of keyword hits diluting the fused ranking.

    vector_candidates : int
        Maximum rows from the embedding cosine-similarity search stage.
        Prevents: low-signal semantic near-misses crowding out precise matches.

    fuzzy_candidates : int
        Maximum rows from the pg_trgm trigram fuzzy search stage.
        Deliberately smaller (30) than BM25/vector — fuzzy is noisier and
        lower-precision; fewer candidates improve fusion signal quality.
        Prevents: spurious fuzzy matches dominating RRF on short queries.

    min_score : float
        Score floor applied per-channel before RRF fusion.
        RRF ignores raw scores (rank-only), but very low scores indicate
        near-zero relevance; dropping them keeps rank lists meaningful.
        0.05 is conservative — only drops results with essentially no signal.
        Prevents: noise candidates consuming rank slots in RRF.

    rrf_k : int
        Constant in the RRF formula: score = Σ 1/(k + rank).
        Higher k = less difference between top/bottom ranks in each list.
        60 is the standard value from Cormack, Clarke & Buettcher (SIGIR 2009).
        Prevents: top-1 results from one channel dominating all other channels.

    shortlist_top_k : int
        Final shortlist size after RRF fusion — number of files surfaced in
        the system prompt. 7 was chosen to balance recall vs. prompt token cost.
        At ~2K tokens per file header, 7 files = ~14K tokens, within budget.
        Prevents: context window overflow from over-broad retrieval.

    lookup_reserved_slots : int
        How many shortlist slots are reserved for dimension/master/lookup tables
        (parties, accounts, dim_*) even if they don't rank well on the query.
        Entity-lookup queries fail silently without a master table in scope.
        3 slots covers most ERP schemas (customer, vendor, GL account masters).
        Prevents: entity resolution failures when dimension tables are outranked.
    """
    stage_limit:            int   = 50
    bm25_candidates:        int   = 50
    vector_candidates:      int   = 50
    fuzzy_candidates:       int   = 30
    min_score:              float = 0.05
    rrf_k:                  int   = 60
    shortlist_top_k:        int   = 7
    lookup_reserved_slots:  int   = 3

    # ── Phase 7: Scale hardening — caller-side caps ───────────────────────────
    # max_top_k : hard ceiling on the `top_k` parameter accepted by
    #   retrieve() / retrieve_with_scores().  Prevents callers from requesting
    #   oversized result sets that would stress RRF fusion, graph expansion, and
    #   trust propagation on every request.
    #   30 = 4× the default shortlist (7) with ample room for edge cases.
    max_top_k:              int   = 30

    # max_rrf_candidates : maximum total candidate count fed into rrf_fuse().
    #   After collecting BM25 + fuzzy + vector + graph results, cap the combined
    #   pool before fusion.  Prevents large catalogs from producing O(N²) rank
    #   computation in the fusion step.  Each channel is already bounded by its
    #   own cap; this is the final safety net for the combined pool.
    max_rrf_candidates:     int   = 200


@lru_cache(maxsize=1)
def get_retrieval_policy() -> RetrievalPolicy:
    """Return the module-level singleton RetrievalPolicy.

    lru_cache guarantees one instance per process. Pass a different policy
    to consuming code via dependency injection when tenant-level overrides
    are needed (future).
    """
    return RetrievalPolicy()

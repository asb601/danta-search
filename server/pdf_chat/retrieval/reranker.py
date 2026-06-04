"""Stage 4 (part 2) — Cross-encoder rerank with a pure fallback.

Production reranks the fused RRF candidates (top-70) down to ``top_n`` (12) with
a CROSS-ENCODER — either the Cohere Rerank API (``rerank-english-v3.0``) or a
self-hosted ``sentence-transformers`` CrossEncoder (BGE-Reranker-v2-M3). A cross
-encoder jointly scores (query, candidate) pairs and is far more precise than the
bi-encoder ANN that produced the candidates.

Both backends are GUARDED (Hard rule #6). When neither library is installed the
function falls back to a PURE pass-through that preserves the input order and
truncates to ``top_n`` — so the query pipeline still works (just without the
precision lift) with zero infra. Tests exercise this fallback.
"""
from __future__ import annotations

import os
from typing import Any

from pdf_chat.config import get_pdf_settings

try:
    import cohere  # type: ignore

    _HAS_COHERE = True
except ImportError:  # pragma: no cover - exercised only without infra
    cohere = None  # type: ignore
    _HAS_COHERE = False

try:
    from sentence_transformers import CrossEncoder  # type: ignore

    _HAS_ST = True
except ImportError:  # pragma: no cover - exercised only without infra
    CrossEncoder = None  # type: ignore
    _HAS_ST = False


_COHERE_MODEL = os.getenv("PDF_COHERE_RERANK_MODEL", "rerank-english-v3.0")
_ST_MODEL = os.getenv("PDF_ST_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Lazily built self-hosted cross-encoder (constructing it loads weights).
_st_encoder: Any = None


def _candidate_text(candidate: Any) -> str:
    """Extract the rerankable text from a candidate (dict or dataclass)."""
    if isinstance(candidate, dict):
        return candidate.get("text", "") or ""
    return getattr(candidate, "text", "") or ""


def rerank(
    query: str,
    candidates: list[Any],
    top_n: int | None = None,
    *,
    container_id: str = "",
) -> list[Any]:
    """Rerank candidates by relevance to ``query`` and truncate to ``top_n``.

    Backend priority: Cohere API → self-hosted CrossEncoder → pure fallback
    (input order preserved). The return value is always a list of the SAME
    candidate objects (dict or dataclass), just reordered and truncated.

    Adaptive skip (Spec §2 L3): when there are too few candidates to be worth
    the cross-encoder cost, the rerank is SKIPPED and the input order returned
    (truncated to ``top_n``). The skip threshold is a per-container tunable
    (``rerank_skip_below_candidates``) and the decision is logged via
    ``log_gate_decision`` — no magic literal lives in this file. This matters at
    scale: across many tenants and millions of files, skipping the cross-encoder
    on tiny candidate sets removes needless inference cost per query.

    Args:
        query: the user query.
        candidates: fused RRF candidates (dicts or dataclasses with ``text``).
        top_n: how many to keep (defaults to ``rerank_top_n`` config).
        container_id: tenant scope for tunable resolution + gate logging.

    Returns:
        The top ``top_n`` candidates, most-relevant first. Without a reranker
        installed (or when the skip gate fires): the first ``top_n`` candidates
        in their original order.
    """
    if not candidates:
        return []

    from pdf_chat.tunables import get_tunable, log_gate_decision

    skip_below = get_tunable(container_id, "rerank_skip_below_candidates")
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

    if top_n is None:
        top_n = get_pdf_settings().rerank_top_n

    if _HAS_COHERE:
        try:
            return _rerank_cohere(query, candidates, top_n)
        except Exception:  # pragma: no cover - infra-dependent
            pass

    if _HAS_ST:
        try:
            return _rerank_sentence_transformers(query, candidates, top_n)
        except Exception:  # pragma: no cover - infra-dependent
            pass

    # Pure fallback: preserve input order, truncate. Keeps the pipeline alive.
    return candidates[:top_n]


def _rerank_cohere(query: str, candidates: list[Any], top_n: int) -> list[Any]:  # pragma: no cover - infra-dependent
    api_key = os.getenv("COHERE_API_KEY", "")
    client = cohere.Client(api_key=api_key)  # type: ignore[union-attr]
    docs = [_candidate_text(c) for c in candidates]
    response = client.rerank(
        query=query, documents=docs, model=_COHERE_MODEL, top_n=top_n
    )
    return [candidates[r.index] for r in response.results]


def _rerank_sentence_transformers(query: str, candidates: list[Any], top_n: int) -> list[Any]:  # pragma: no cover - infra-dependent
    global _st_encoder
    if _st_encoder is None:
        _st_encoder = CrossEncoder(_ST_MODEL)  # type: ignore[misc]
    pairs = [(query, _candidate_text(c)) for c in candidates]
    scores = _st_encoder.predict(pairs)
    ranked = sorted(
        range(len(candidates)), key=lambda i: float(scores[i]), reverse=True
    )
    return [candidates[i] for i in ranked[:top_n]]

"""Stage 4 (part 1) — Reciprocal Rank Fusion.

Pure logic. Merges several independently ranked id-lists (e.g. Neo4j vector ANN
top-50 + graph traversal top-20) into a single fused ranking. No infra; runs and
tests with zero dependencies installed.

Spec formula (enterprise-pdf-4 §6, Stage 4):

    score(d) = Σ over lists  1 / (k + rank + 1)

where ``rank`` is the 0-based position of ``d`` within a list and ``k`` defaults
to 60. The fused list is returned ranked by score descending. Ties are broken
deterministically by first-seen order so the function is stable across runs.
"""
from __future__ import annotations


def rrf(results_lists: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked id-lists via Reciprocal Rank Fusion.

    Args:
        results_lists: a list of ranked lists. Each inner list holds ids
            (strings) ordered best-first. Lists may differ in length and may
            share ids.
        k: RRF damping constant (spec default 60). Larger ``k`` flattens the
            contribution of rank position.

    Returns:
        A single list of unique ids ranked by fused score descending. For equal
        scores, the id seen earliest (first list, then earliest rank) wins, so
        the ordering is stable and deterministic.
    """
    scores: dict[str, float] = {}
    # first_seen preserves a deterministic insertion order for stable tie-breaks.
    first_seen: dict[str, int] = {}
    counter = 0

    for results in results_lists:
        for rank, item in enumerate(results):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
            if item not in first_seen:
                first_seen[item] = counter
                counter += 1

    # Sort by score desc, then by first-seen order asc for stable ties.
    return sorted(scores.keys(), key=lambda x: (-scores[x], first_seen[x]))

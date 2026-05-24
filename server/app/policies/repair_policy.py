"""SQL repair policy.

Governs the bounds on the two-tier SQL repair loop.

Tier 1 — Deterministic AST repair (no LLM):
  Column/table aliasing, quote normalisation, dialect coercion.
  No token budget needed — pure string/AST operations.

Tier 2 — Focused LLM repair (targeted rewrite):
  A bounded LLM call that rewrites only the broken clause.
  Bounded by output_tokens and temperature to prevent hallucinated rewrites.

FUTURE READINESS:
  - Deployment override: a faster LLM endpoint could lower tier2_output_tokens
    to 256 for cost savings without impacting quality.
  - Tenant override: a production tenant that needs exact aggregations could
    lower max_attempts to 1 to prefer a clean "I can't repair this" response
    over a second speculative attempt.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RepairPolicy:
    """
    SQL repair loop bounds.

    max_attempts : int
        Total repair iterations across both tiers combined.
        2 = one deterministic attempt + one LLM attempt.
        Prevents: infinite repair loops from LLM generating new errors.
        After max_attempts, the query is returned with the original SQL error.

    tier2_output_tokens : int
        Maximum LLM output tokens for a Tier 2 repair call.
        512 is enough to rewrite a complex FROM/WHERE clause.
        Prevents: hallucinated multi-page rewrites that introduce new errors.

    tier2_temperature : int | float
        Sampling temperature for Tier 2 LLM calls.
        0 = deterministic/greedy: same broken SQL → same repair, always.
        Prevents: random variation in repair output making debugging impossible.
    """
    max_attempts:       int   = 2
    tier2_output_tokens: int  = 512
    tier2_temperature:  int   = 0


@lru_cache(maxsize=1)
def get_repair_policy() -> RepairPolicy:
    """Return the module-level singleton RepairPolicy."""
    return RepairPolicy()

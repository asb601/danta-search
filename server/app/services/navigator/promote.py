"""[5] PROMOTE (in-loop) — record a verified, executed conclusion in the ledger.

There are TWO promotions in the architecture (I8 — "decide once, reuse"):

  1. IN-LOOP promotion (DONE here): write the verified+executed ``StepResult``
     into the per-question ``StepLedger`` keyed by ``step.step_id``, so later
     steps in the SAME question (COMPOSE math, JOIN grain) can read it back via
     ``ledger.get_scalar(step_id)`` / ``ledger.results[step_id]``. This is pure,
     in-memory, single-question state — no governance surface.

  2. PERSISTENT cross-query map promotion (NOT done here): writing the verified
     conclusion back into the semantic MAP (SemanticEntity / canonical-master /
     approved-join / cached-conclusion tables) so the SAME question across
     DIFFERENT requests is answered deterministically without re-deriving it.
     That write is governance-sensitive (numbers must tie out across users; a bad
     promotion poisons every future query) and is a tracked follow-up. It is left
     here as a clearly-named, documented NO-OP seam — ``_persist_to_map`` — so I8
     is honestly partial and VISIBLE, not silently dropped.

``promote`` is PURE-ish: it mutates and returns the SAME ledger (the one mutable
contract in the loop, by design). It performs NO IO and calls NO LLM.

The navigator is self-contained — this module imports nothing from
``app.services.resolve.*``.
"""
from __future__ import annotations

from app.services.navigator.types import (
    IntentStep,
    StepLedger,
    StepResult,
    VerifiedContract,
)


def _persist_to_map(
    ledger: StepLedger, vc: VerifiedContract, result: StepResult
) -> None:
    """NO-OP seam for PERSISTENT cross-query map promotion.

    TODO(P-followup): cross-query promotion; see I8. This is intentionally a
    no-op: writing a verified conclusion back to the semantic MAP
    (SemanticEntity / canonical-master / approved-join / cached-conclusion) is a
    governance-sensitive, separate workstream. It MUST NOT touch any semantic
    table from inside the query loop. Until that workstream lands, in-loop
    promotion (the ledger write below) is the only promotion the navigator does.
    """
    return None


def promote(
    ledger: StepLedger,
    step: IntentStep,
    vc: VerifiedContract,
    result: StepResult,
) -> StepLedger:
    """Write the verified, executed conclusion into the ledger keyed by
    ``step.step_id`` and return the (same, mutated) ledger so downstream steps
    read it back. PURE-ish (mutates the one mutable contract; no IO, no LLM)."""
    ledger.results[step.step_id] = result
    # I8: persistent cross-query promotion is a tracked follow-up (documented
    # no-op seam). In-loop promotion (above) is what makes this question's later
    # steps able to reference this conclusion.
    _persist_to_map(ledger, vc, result)
    return ledger

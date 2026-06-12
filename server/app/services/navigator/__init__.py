"""navigator — the merged query runtime ("Disciplined Navigator").

ONE agentic loop where gpt-4o-mini does ONLY PLAN, PROPOSE, SYNTHESIZE; every
structural decision is VERIFIED against stored per-file evidence before use, and
verified conclusions are PROMOTED into a growing map.

See docs/superpowers/specs/2026-06-11-merged-query-architecture-TARGET.md
(INVARIANTS I1-I13) and the IMPLEMENTATION-BLUEPRINT for the module layout.

Build is complete (P1-P5):
  types.py (data contracts) · planner.py [1] · retriever.py/evidence.py [3a/3b]
  · proposer.py/verifier.py/renderer.py [3c/3d/3e] · executor.py/promote.py/
  composer.py [4/5] · synthesizer.py [6] · driver.py (the loop).

The public entry point is ``run_navigator(...)`` (driver.py), re-exported here so
the graph seam imports it from the package root.
"""
from __future__ import annotations

from app.services.navigator.driver import run_navigator

__all__ = ["run_navigator"]

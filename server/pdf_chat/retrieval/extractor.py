"""On-demand extractor adapter (agent Stage 6).

Lazily materializes table/image chunk bodies that survived ACL. In Phases 0–1 the
ingestion path already stores table markdown + image captions on the chunk text
(ingestion/chunker.py:144-195), so this adapter is a pass-through that returns the
chunk unchanged; it exists so build_default_deps() can wire a real extractor and
the agent's on_demand_extract node has a non-None dep. Satisfies the agent's
``Extractor`` protocol (``async def extract(chunk)``).
"""
from __future__ import annotations

from typing import Any


class OnDemandExtractor:
    async def extract(self, chunk: Any) -> Any:
        return chunk

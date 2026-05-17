"""Ontology layer schema upgrade.

Adds the columns that power role-indexed relationship detection:

1. file_metadata.column_semantic_roles (JSONB)
   Per-column semantic role map resolved at ingest time.
   Schema: {"source_col": "custom:<kind>:<label>", ...}
   Populated by column_role_resolver. The planner and semantic builder read
   this — zero LLM at query time.

2. file_metadata.role_source (VARCHAR 20)
   How the roles were resolved: "glossary" | "heuristic" | "llm".
   Lets ops understand confidence level and flag files for re-resolution
   if a better glossary is uploaded later.

3. file_relationships.semantic_role (VARCHAR 100)
   The typed role that makes this join valid.
   Without this, the planner cannot distinguish a real key relationship from
   a coincidental name collision (both tables have "id").

4. file_relationships.role_source (VARCHAR 20)
   Mirrors file_metadata.role_source for the relationship record.

5. GIN index on file_metadata.column_semantic_roles
   Useful for metadata JSONB existence queries. Relationship discovery does not
   rely on JSONB value lookup; it uses column_key_registry instead.

Idempotent — safe to run on every startup.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    # ── file_metadata ─────────────────────────────────────────────────────────
    "ALTER TABLE file_metadata "
    "ADD COLUMN IF NOT EXISTS column_semantic_roles JSONB",

    "ALTER TABLE file_metadata "
    "ADD COLUMN IF NOT EXISTS role_source VARCHAR(20)",

    # GIN index so ?& operator (key overlap) is O(log N) not O(N²)
    "CREATE INDEX IF NOT EXISTS idx_file_metadata_semantic_roles "
    "ON file_metadata USING GIN (column_semantic_roles)",

    # ── file_relationships ────────────────────────────────────────────────────
    "ALTER TABLE file_relationships "
    "ADD COLUMN IF NOT EXISTS semantic_role VARCHAR(100)",

    "ALTER TABLE file_relationships "
    "ADD COLUMN IF NOT EXISTS role_source VARCHAR(20)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

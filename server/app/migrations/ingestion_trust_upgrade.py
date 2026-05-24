"""Phase 5 — Ingestion Trustworthiness schema upgrade.

Adds ingestion-trust columns to file_metadata and file_relationships.

NEW COLUMNS
===========
file_metadata:
  column_role_evidence     JSONB       — per-column role confidence + signals
                                         {"col": {"confidence": 0.92,
                                                  "signals": ["column_name"],
                                                  "source": "llm"}}
  ingestion_confidence_score  FLOAT    — overall per-file quality score (0.0–1.0)
  ingestion_confidence_signals JSONB   — score breakdown dict for observability

file_relationships:
  evidence_count  INTEGER              — # overlapping fingerprinted key values
  edge_provenance JSONB                — {card_a, card_b, role_a, role_b,
                                          key_kind_a, key_kind_b}

IDEMPOTENCY
===========
All ALTER TABLE statements use IF NOT EXISTS. Safe to re-run.

ROLLBACK
========
To reverse:
  ALTER TABLE file_metadata
    DROP COLUMN IF EXISTS column_role_evidence,
    DROP COLUMN IF EXISTS ingestion_confidence_score,
    DROP COLUMN IF EXISTS ingestion_confidence_signals;

  ALTER TABLE file_relationships
    DROP COLUMN IF EXISTS evidence_count,
    DROP COLUMN IF EXISTS edge_provenance;

USAGE
=====
  cd server
  python3 -m app.migrations.ingestion_trust_upgrade
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.core.logger import ingest_logger

_UPGRADE_STATEMENTS: tuple[str, ...] = (
    # ── file_metadata additions ───────────────────────────────────────────────
    """
    ALTER TABLE file_metadata
      ADD COLUMN IF NOT EXISTS column_role_evidence         JSONB,
      ADD COLUMN IF NOT EXISTS ingestion_confidence_score   FLOAT,
      ADD COLUMN IF NOT EXISTS ingestion_confidence_signals JSONB
    """,
    # ── file_relationships additions ──────────────────────────────────────────
    """
    ALTER TABLE file_relationships
      ADD COLUMN IF NOT EXISTS evidence_count  INTEGER,
      ADD COLUMN IF NOT EXISTS edge_provenance JSONB
    """,
)


async def run_upgrade(db: AsyncSession) -> None:
    for stmt in _UPGRADE_STATEMENTS:
        await db.execute(text(stmt))
    await db.commit()
    ingest_logger.info("ingestion_trust_upgrade", status="done")


async def _main() -> None:
    async with async_session() as db:
        try:
            await run_upgrade(db)
            print("ingestion_trust_upgrade: OK")
        except Exception as exc:
            print(f"ingestion_trust_upgrade: FAILED — {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    asyncio.run(_main())

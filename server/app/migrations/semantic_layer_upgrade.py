"""Semantic layer schema.

Adds business-facing entities and relationships above the technical ER graph.
The ER graph says a join may exist; the semantic layer records what that join
means, its cardinality, approval state, and business-safe join rule.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS semantic_entities (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        file_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        entity_name VARCHAR(120) NOT NULL,
        primary_key VARCHAR(255),
        attributes JSONB NOT NULL DEFAULT '[]'::jsonb,
        metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
        dimensions JSONB NOT NULL DEFAULT '[]'::jsonb,
        grain VARCHAR(255),
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        source VARCHAR(50) NOT NULL DEFAULT 'ingestion',
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_entities_file UNIQUE (file_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_relationships (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        source_relationship_id VARCHAR(36) REFERENCES file_relationships(id) ON DELETE CASCADE,
        file_a_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        file_b_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        from_entity VARCHAR(120) NOT NULL,
        to_entity VARCHAR(120) NOT NULL,
        from_column VARCHAR(255) NOT NULL,
        to_column VARCHAR(255) NOT NULL,
        relationship_type VARCHAR(30) NOT NULL,
        join_rule JSONB NOT NULL DEFAULT '{}'::jsonb,
        approval_status VARCHAR(20) NOT NULL DEFAULT 'candidate',
        risk_reason VARCHAR(500),
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_relationship_source UNIQUE (source_relationship_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_semantic_entities_container ON semantic_entities (container_id, status)",
    "CREATE INDEX IF NOT EXISTS ix_semantic_entities_file ON semantic_entities (file_id)",
    "CREATE INDEX IF NOT EXISTS ix_semantic_relationships_container ON semantic_relationships (container_id, status, approval_status)",
    "CREATE INDEX IF NOT EXISTS ix_semantic_relationships_files ON semantic_relationships (file_a_id, file_b_id)",
    "CREATE INDEX IF NOT EXISTS ix_semantic_relationships_join_rule ON semantic_relationships USING GIN (join_rule)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())

"""Governed semantic memory schema."""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS semantic_memory_records (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        memory_type VARCHAR(40) NOT NULL,
        canonical_key VARCHAR(255) NOT NULL,
        title VARCHAR(255) NOT NULL,
        summary TEXT,
        normalized_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        behaviors JSONB NOT NULL DEFAULT '[]'::jsonb,
        dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
        constraints JSONB NOT NULL DEFAULT '{}'::jsonb,
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        authority_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        governance_status VARCHAR(20) NOT NULL DEFAULT 'candidate',
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        source VARCHAR(50) NOT NULL DEFAULT 'ingestion',
        source_file_id VARCHAR(36) REFERENCES files(id) ON DELETE CASCADE,
        source_entity_id VARCHAR(36) REFERENCES semantic_entities(id) ON DELETE SET NULL,
        source_relationship_id VARCHAR(36) REFERENCES semantic_relationships(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_memory_canonical UNIQUE (container_id, memory_type, canonical_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memory_evidence (
        id VARCHAR(36) PRIMARY KEY,
        memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        file_id VARCHAR(36) REFERENCES files(id) ON DELETE CASCADE,
        source_type VARCHAR(40) NOT NULL,
        source_id VARCHAR(80),
        evidence_key VARCHAR(120) NOT NULL,
        evidence_value JSONB,
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_memory_evidence_source UNIQUE (memory_id, source_type, source_id, evidence_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memory_links (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        source_memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        target_memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        link_type VARCHAR(40) NOT NULL,
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_memory_link UNIQUE (source_memory_id, target_memory_id, link_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memory_asset_index (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        file_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        index_kind VARCHAR(40) NOT NULL,
        score DOUBLE PRECISION NOT NULL DEFAULT 0,
        terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_memory_asset UNIQUE (memory_id, file_id, index_kind)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memory_term_index (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        term VARCHAR(120) NOT NULL,
        token_class VARCHAR(30) NOT NULL DEFAULT 'term',
        weight DOUBLE PRECISION NOT NULL DEFAULT 1,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_memory_term UNIQUE (container_id, term, memory_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_context_traces (
        id VARCHAR(36) PRIMARY KEY,
        request_id VARCHAR(64) NOT NULL,
        container_id VARCHAR(36) REFERENCES container_configs(id) ON DELETE SET NULL,
        user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
        query_hash VARCHAR(64) NOT NULL,
        selected_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        selected_domain_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        ambiguity_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
        retrieval_guidance JSONB NOT NULL DEFAULT '{}'::jsonb,
        execution_envelope JSONB NOT NULL DEFAULT '{}'::jsonb,
        token_estimate INTEGER NOT NULL DEFAULT 0,
        caps JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE brain_context_traces ADD COLUMN IF NOT EXISTS selected_domain_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
    """
    CREATE TABLE IF NOT EXISTS semantic_domain_clusters (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        domain_type VARCHAR(40) NOT NULL,
        domain_key VARCHAR(255) NOT NULL,
        title VARCHAR(255) NOT NULL,
        summary TEXT,
        normalized_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        workflow_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        lifecycle_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        kpi_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        synonym_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        contributor_file_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        contributor_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        conflict_count INTEGER NOT NULL DEFAULT 0,
        conflict_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        authority_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        drift_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        governance_status VARCHAR(20) NOT NULL DEFAULT 'candidate',
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_domain_cluster UNIQUE (container_id, domain_type, domain_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_domain_evidence (
        id VARCHAR(36) PRIMARY KEY,
        domain_id VARCHAR(36) NOT NULL REFERENCES semantic_domain_clusters(id) ON DELETE CASCADE,
        memory_id VARCHAR(36) NOT NULL REFERENCES semantic_memory_records(id) ON DELETE CASCADE,
        file_id VARCHAR(36) REFERENCES files(id) ON DELETE CASCADE,
        evidence_type VARCHAR(40) NOT NULL,
        evidence_key VARCHAR(255) NOT NULL,
        evidence_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        contribution_weight DOUBLE PRECISION NOT NULL DEFAULT 0,
        confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        authority_score DOUBLE PRECISION NOT NULL DEFAULT 0,
        decay_factor DOUBLE PRECISION NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_domain_evidence_memory UNIQUE (domain_id, memory_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_domain_file_index (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        domain_id VARCHAR(36) NOT NULL REFERENCES semantic_domain_clusters(id) ON DELETE CASCADE,
        file_id VARCHAR(36) NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        domain_type VARCHAR(40) NOT NULL,
        score DOUBLE PRECISION NOT NULL DEFAULT 0,
        terms JSONB NOT NULL DEFAULT '[]'::jsonb,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_domain_file UNIQUE (domain_id, file_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_domain_term_index (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        domain_id VARCHAR(36) NOT NULL REFERENCES semantic_domain_clusters(id) ON DELETE CASCADE,
        term VARCHAR(120) NOT NULL,
        domain_type VARCHAR(40) NOT NULL,
        weight DOUBLE PRECISION NOT NULL DEFAULT 1,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_domain_term UNIQUE (container_id, term, domain_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_domain_conflicts (
        id VARCHAR(36) PRIMARY KEY,
        container_id VARCHAR(36) NOT NULL REFERENCES container_configs(id) ON DELETE CASCADE,
        domain_id VARCHAR(36) NOT NULL REFERENCES semantic_domain_clusters(id) ON DELETE CASCADE,
        conflict_type VARCHAR(40) NOT NULL,
        conflict_key VARCHAR(255) NOT NULL,
        severity VARCHAR(20) NOT NULL DEFAULT 'warning',
        file_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        resolution_status VARCHAR(20) NOT NULL DEFAULT 'open',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_semantic_domain_conflict UNIQUE (domain_id, conflict_type, conflict_key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_smr_container_status_type ON semantic_memory_records (container_id, status, governance_status, memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_smr_source_file ON semantic_memory_records (source_file_id)",
    "CREATE INDEX IF NOT EXISTS idx_smr_terms_gin ON semantic_memory_records USING GIN (normalized_terms)",
    "CREATE INDEX IF NOT EXISTS idx_smr_behaviors_gin ON semantic_memory_records USING GIN (behaviors)",
    "CREATE INDEX IF NOT EXISTS idx_sme_memory ON semantic_memory_evidence (memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_sme_file ON semantic_memory_evidence (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_sml_container ON semantic_memory_links (container_id, status, link_type)",
    "CREATE INDEX IF NOT EXISTS idx_smai_file ON semantic_memory_asset_index (file_id, index_kind, score)",
    "CREATE INDEX IF NOT EXISTS idx_smai_container ON semantic_memory_asset_index (container_id, index_kind)",
    "CREATE INDEX IF NOT EXISTS idx_smti_container_term ON semantic_memory_term_index (container_id, term, status)",
    "CREATE INDEX IF NOT EXISTS idx_smti_memory ON semantic_memory_term_index (memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_sdc_container_status_type ON semantic_domain_clusters (container_id, status, governance_status, domain_type)",
    "CREATE INDEX IF NOT EXISTS idx_sdc_terms_gin ON semantic_domain_clusters USING GIN (normalized_terms)",
    "CREATE INDEX IF NOT EXISTS idx_sdc_workflow_terms_gin ON semantic_domain_clusters USING GIN (workflow_terms)",
    "CREATE INDEX IF NOT EXISTS idx_sdc_lifecycle_terms_gin ON semantic_domain_clusters USING GIN (lifecycle_terms)",
    "CREATE INDEX IF NOT EXISTS idx_sdc_kpi_terms_gin ON semantic_domain_clusters USING GIN (kpi_terms)",
    "CREATE INDEX IF NOT EXISTS idx_sde_domain ON semantic_domain_evidence (domain_id)",
    "CREATE INDEX IF NOT EXISTS idx_sde_file ON semantic_domain_evidence (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_sde_memory ON semantic_domain_evidence (memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_sdfi_file ON semantic_domain_file_index (file_id, domain_type, score)",
    "CREATE INDEX IF NOT EXISTS idx_sdfi_container ON semantic_domain_file_index (container_id, domain_type)",
    "CREATE INDEX IF NOT EXISTS idx_sdti_container_term ON semantic_domain_term_index (container_id, term, status)",
    "CREATE INDEX IF NOT EXISTS idx_sdti_domain ON semantic_domain_term_index (domain_id)",
    "CREATE INDEX IF NOT EXISTS idx_sdcf_domain ON semantic_domain_conflicts (domain_id, resolution_status)",
    "CREATE INDEX IF NOT EXISTS idx_sdcf_container ON semantic_domain_conflicts (container_id, severity)",
    "CREATE INDEX IF NOT EXISTS idx_bct_request ON brain_context_traces (request_id)",
    "CREATE INDEX IF NOT EXISTS idx_bct_container_created ON brain_context_traces (container_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_bct_query_hash ON brain_context_traces (query_hash)",
]


async def migrate() -> None:
    async with engine.begin() as conn:
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))


if __name__ == "__main__":
    asyncio.run(migrate())
"""Deterministic VERIFY tier.

Pure, value-evidence-only verification primitives that confirm or abstain on
join and canonical-master decisions using precomputed ingestion artifacts
(ColumnKeyRegistry fingerprints, semantic roles, FileMetadata columns). No LLM,
no runtime schema discovery, no name lists — every threshold comes from
SemanticPolicy. This package is additive and is intentionally NOT wired into any
live query path.
"""

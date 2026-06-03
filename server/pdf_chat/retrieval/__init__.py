"""Team C — Retrieval pipeline (Stages 3,4,5,7,8 of the query path).

Public surface other teams import. Pure logic (``rrf``, ``filter_by_acl``,
``insufficient_context``, ``cache_key``, ``route_by_element_type``,
``assemble_context``, ``rerank`` fallback) imports and runs with ZERO infra.
Infra adapters (``Neo4jSearcher``, ``RedisCache``, real reranker backends) use
guarded imports and only require their library when actually called.
"""
from __future__ import annotations

from pdf_chat.retrieval.acl import filter_by_acl, insufficient_context
from pdf_chat.retrieval.cache import RedisCache, cache_key
from pdf_chat.retrieval.context_assembly import assemble_context
from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher
from pdf_chat.retrieval.reranker import rerank
from pdf_chat.retrieval.router import (
    ROUTE_IMMEDIATE,
    ROUTE_ON_DEMAND_TABLE,
    ROUTE_ON_DEMAND_VISION,
    route_by_element_type,
)
from pdf_chat.retrieval.rrf import rrf

__all__ = [
    "rrf",
    "filter_by_acl",
    "insufficient_context",
    "cache_key",
    "RedisCache",
    "Neo4jSearcher",
    "rerank",
    "route_by_element_type",
    "ROUTE_IMMEDIATE",
    "ROUTE_ON_DEMAND_TABLE",
    "ROUTE_ON_DEMAND_VISION",
    "assemble_context",
]

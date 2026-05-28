"""
Retrieval stage 5 — Fuzzy / trigram similarity search.

Uses PostgreSQL's pg_trgm extension against the `search_text` column
(backed by a GIN gin_trgm_ops index — created in PHASE 1) when the
database supports it. Azure PostgreSQL deployments can block pg_trgm; in
that case this optional channel falls back to bounded metadata token
matching instead of disappearing entirely.

Why trigram fuzzy on top of BM25?
- BM25 catches exact and stemmed keyword matches.
- Trigram fuzzy catches typos ("salaery" → "salary"), abbreviations
    ("acct" matches "account"), partial names ("Q1 rev" matches
  "Q1 revenue forecast"), and camelCase / concatenated words that
  BM25 tokenises differently.
- They produce different result sets — RRF fusion (PHASE 11) combines both.

Strategy
---------
Two complementary searches are run and merged:

  1. word_similarity — query phrase vs each word in search_text.
     Best for short queries against long descriptions.
     PostgreSQL function: word_similarity(query, search_text)
     Index op: gin_trgm_ops supports this.

  2. strict_word_similarity — tighter; query must match a whole word
     boundary. Less noise, higher precision.
     PostgreSQL function: strict_word_similarity(query, search_text)

We use word_similarity as the primary score because it has better recall
for partial abbreviations and typos. A minimum threshold of 0.2 is applied
to drop unrelated rows before they reach RRF fusion.

Public API
----------
    async def fuzzy_search(
        query: str,
        user_id: str,
        is_admin: bool,
        db: AsyncSession,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 50,
        threshold: float = 0.2,
    ) -> list[tuple[FileMetadata, float]]
        Returns list of (FileMetadata, similarity_score) sorted descending.
        similarity_score is in [0.0, 1.0] — higher = more similar.
"""
from __future__ import annotations

from datetime import date

import re

from sqlalchemy import case, func, Float, literal, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import literal_column

from app.models.file_metadata import FileMetadata
from app.retrieval.filters import build_base_query

# Minimum similarity score to include a row in results.
# 0.2 keeps abbreviations and 1-char typos while dropping noise.
_DEFAULT_THRESHOLD = 0.2
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "show", "the", "to", "use", "what",
    "with", "year", "years", "current", "next", "details", "detail",
    "analyze", "analyse", "summary", "summarize", "summarise",
})


def _pg_trgm_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "word_similarity" in message
        or "pg_trgm" in message
        or "gin_trgm_ops" in message
        or "not allow-listed" in message
    )


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if len(token) > 4 and token.endswith("ies"):
            tokens.append(token[:-3] + "y")
        elif len(token) > 3 and token.endswith("s"):
            tokens.append(token[:-1])
    return list(dict.fromkeys(tokens))[:12]


def _token_boundary_pattern(token: str) -> str:
    escaped = re.escape(token)
    return rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)"


async def _metadata_token_fallback(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    *,
    date_from: date | None,
    date_to: date | None,
    limit: int,
    allowed_domains: list[str] | None,
    container_id: str | None,
) -> list[tuple[FileMetadata, float]]:
    """Bounded metadata fallback for databases without pg_trgm.

    This intentionally searches metadata only, never row values. It is a
    degraded recall channel for source names, column names, descriptions, and
    compact search_text while preserving the same permission/date/container
    filters as the normal fuzzy stage.
    """
    tokens = _query_tokens(query)
    if not tokens:
        return []

    search_text = func.coalesce(FileMetadata.search_text, "")
    blob_path = func.coalesce(FileMetadata.blob_path, "")
    score_expr = literal(0.0)
    conditions = []
    for token in tokens:
        pattern = _token_boundary_pattern(token)
        search_match = search_text.op("~*")(pattern)
        path_match = blob_path.op("~*")(pattern)
        conditions.extend([search_match, path_match])
        score_expr = score_expr + case((search_match, 1.0), else_=0.0)
        score_expr = score_expr + case((path_match, 2.0), else_=0.0)

    score_expr = (score_expr / max(float(len(tokens)), 1.0)).label("fuzzy_score")
    q = (
        build_base_query(
            user_id=user_id,
            is_admin=is_admin,
            date_from=date_from,
            date_to=date_to,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
        .add_columns(score_expr)
        .where(or_(*conditions))
        .order_by(score_expr.desc())
        .limit(limit)
    )

    async with db.begin_nested():
        rows = (await db.execute(q)).all()
    return [(row[0], float(row[1])) for row in rows if float(row[1]) > 0.0]


async def fuzzy_search(
    query: str,
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 50,
    threshold: float = _DEFAULT_THRESHOLD,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
) -> list[tuple[FileMetadata, float]]:
    """
    Run trigram (pg_trgm) fuzzy similarity search against file_metadata.search_text.

    Returns
    -------
    list of (FileMetadata row, similarity_score float) sorted descending.
    Empty list if query is blank or no results pass the threshold.
    """
    if not query or not query.strip():
        return []

    query = query.strip()

    # search_text is a plain TEXT column — reference it as a literal column.
    search_text_col = literal_column("file_metadata.search_text")

    # word_similarity(needle, haystack) — pg_trgm function.
    # Returns similarity of the query phrase against the best matching
    # substring (word boundary aware) in search_text.
    similarity_expr = func.word_similarity(query, search_text_col).label("fuzzy_score")

    base_q = build_base_query(
        user_id=user_id,
        is_admin=is_admin,
        date_from=date_from,
        date_to=date_to,
        allowed_domains=allowed_domains,
        container_id=container_id,
    )

    q = (
        base_q
        .add_columns(similarity_expr)
        # pg_trgm's <word_similarity threshold operator — uses the GIN index
        .where(literal_column("file_metadata.search_text").op("%>>")(query))
        # Also apply our explicit threshold to be sure
        .where(similarity_expr >= threshold)
        .order_by(similarity_expr.desc())
        .limit(limit)
    )

    # pg_trgm may be unavailable or blocked in Azure PostgreSQL. Keep that
    # failure inside this optional stage and use a bounded metadata fallback;
    # unrelated database errors still propagate to the orchestrator.
    try:
        async with db.begin_nested():
            rows = (await db.execute(q)).all()
    except Exception as exc:
        if not _pg_trgm_unavailable(exc):
            raise
        return await _metadata_token_fallback(
            query,
            user_id,
            is_admin,
            db,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )

    return [(row[0], float(row[1])) for row in rows]

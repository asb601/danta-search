"""In-process response cache for identical/near-identical queries.

Strategy
--------
* Key: ``(container_id, normalized_query)`` — exact match after lowercasing and
  collapsing whitespace.  No embeddings needed; same question from 10 concurrent
  users hits Azure once, the other 9 get the cached answer in <50ms.
* Storage: plain dict with an asyncio.Lock — no Redis, no extra infrastructure.
* TTL: 10 minutes (configurable via RESPONSE_CACHE_TTL_SECONDS in settings).
* Max entries: 500 (oldest evicted first — simple FIFO).
* Cache key normalisation strips punctuation variance so minor casing and
    punctuation differences map to the same key.
* Fuzzy fallback: if no exact match, check for a key with Levenshtein ratio > 0.92
  among the last 50 entries (fast, bounded).  Handles minor rephrasings.

Thread-safety: all operations are protected by a threading.Lock so this works
correctly under FastAPI's async concurrency model.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Any

from app.core.config import get_settings

# ── Configuration ──────────────────────────────────────────────────────────────
_TTL_SECONDS: int = 600       # 10 minutes — covers a typical meeting/demo burst
_MAX_ENTRIES: int = 500        # ~500 cached Q&A pairs ≈ <5 MB RAM
_FUZZY_THRESHOLD: float = 0.92 # SequenceMatcher ratio to accept a fuzzy hit
_FUZZY_SCAN_LAST_N: int = 50   # only compare against the most-recent N keys

# ── Store ──────────────────────────────────────────────────────────────────────
# OrderedDict preserves insertion order for FIFO eviction.
# Value: {"payload": dict, "ts": float}
_store: OrderedDict[tuple, dict] = OrderedDict()
_lock = threading.Lock()

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

# Phrases that mark a hollow/fallback answer — never worth caching.
# These come from response_helpers.py fallback_answer() and graph.py synthesis fallback.
_HOLLOW_PHRASES: tuple[str, ...] = (
    "wasn't able to find",
    "was not able to find",
    "could not find",
    "couldn't find",
    "no relevant data",
    "no data found",
    "unable to answer",
    "cannot answer",
    "no information",
    "not available",
    "i don't have",
    "i do not have",
    "failed to retrieve",
    "no results",
    "could not be answered",
    "could not retrieve",
    "please try again",
    "i apologize",
    "unfortunately",
    # False-negative "ran but found nothing" answers — risky to cache because a
    # schema/tool bug can cause the agent to incorrectly report no data exists.
    # Re-running these is cheap; caching a wrong "empty" answer is expensive.
    "found no ",
    "no records found",
    "no matching records",
    "returned no results",
    "yielded no results",
    "no open gl",
    "no line items",
    "no transactions found",
    "no documents found",
)
_MIN_ANSWER_TOKENS = 15  # answers shorter than this are too thin to be worth caching


def _normalize(query: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    q = query.lower()
    q = _PUNCT_RE.sub("", q)
    q = _WHITESPACE_RE.sub(" ", q).strip()
    return q


def _is_expired(entry: dict) -> bool:
    return (time.monotonic() - entry["ts"]) > _TTL_SECONDS


def get_cached_response(cache_key: tuple) -> dict | None:
    """Return cached payload for *cache_key*, or None on miss / expiry.

    cache_key is ``(container_id_or_default, raw_query_string)``.
    Internally normalises the query before lookup and tries fuzzy match on miss.
    """
    container_id, raw_query = cache_key
    norm_query = _normalize(raw_query)
    lookup_key = (container_id, norm_query)

    with _lock:
        # ── Exact match ────────────────────────────────────────────────────────
        entry = _store.get(lookup_key)
        if entry is not None:
            if _is_expired(entry):
                del _store[lookup_key]
                return None
            # Move to end (most-recently-used) to delay FIFO eviction
            _store.move_to_end(lookup_key)
            return dict(entry["payload"])  # return a copy

        # ── Fuzzy match — scan last N entries with same container ──────────────
        candidates = [
            (k, v) for k, v in list(_store.items())[-_FUZZY_SCAN_LAST_N:]
            if k[0] == container_id and not _is_expired(v)
        ]
        for k, v in candidates:
            ratio = SequenceMatcher(None, norm_query, k[1]).ratio()
            if ratio >= _FUZZY_THRESHOLD:
                _store.move_to_end(k)
                return dict(v["payload"])

    return None


def _is_hollow_answer(answer: str) -> bool:
    """Return True if the answer is a fallback/error message not worth caching.

    Two checks:
    1. Token length — answers under 15 words are too thin (e.g. "I couldn't find that.")
    2. Phrase match — explicit fallback phrases from the agent's response_helpers.py
    """
    if not answer:
        return True
    words = answer.split()
    if len(words) < _MIN_ANSWER_TOKENS:
        return True
    low = answer.lower()
    return any(phrase in low for phrase in _HOLLOW_PHRASES)


def set_cached_response(cache_key: tuple, payload: dict) -> None:
    """Store *payload* under *cache_key* with current timestamp.

    Payload is a copy — mutations to the caller's dict won't affect the cache.
    Evicts the oldest entry when the store is full.

    Never caches:
      - error responses
      - empty answers
      - hollow fallback answers (agent couldn't find data, retrying would fix it)
    """
    container_id, raw_query = cache_key
    norm_query = _normalize(raw_query)
    lookup_key = (container_id, norm_query)

    answer = payload.get("answer", "")

    # Don't cache error responses or hollow/fallback answers
    if payload.get("error") or _is_hollow_answer(answer):
        return

    with _lock:
        if lookup_key in _store:
            _store.move_to_end(lookup_key)
        elif len(_store) >= _MAX_ENTRIES:
            _store.popitem(last=False)  # evict oldest
        _store[lookup_key] = {"payload": dict(payload), "ts": time.monotonic()}


def cache_stats() -> dict:
    """Return current cache size and approximate hit potential (for /api/metrics)."""
    with _lock:
        total = len(_store)
        live = sum(1 for v in _store.values() if not _is_expired(v))
    return {"cache_total": total, "cache_live": live, "cache_ttl_s": _TTL_SECONDS}


def clear_cache() -> None:
    """Flush the entire cache (useful for testing / admin endpoints)."""
    with _lock:
        _store.clear()

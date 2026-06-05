"""Phase 4 Task 6 — version-stamped cache key for cross-domain answers.

A cross-domain answer is produced when a PDF question is bridged (via a
value-evidenced master key) into the CSV/structured runtime through the
``structured_query`` tool. Such an answer is a function of TWO knowledge bases:

  * the CSV **semantic layer** (entities/metrics/joins built at CSV ingestion), and
  * the PDF **graph extraction** (sections/entities/values built at PDF ingestion).

If either side changes, every cached cross-domain answer for that tenant must
become unreachable. We achieve this WITHOUT touching the existing cache by
folding both side's version stamps into the cache key whenever a cross-domain
path was taken. A pure-PDF answer (``structured_query`` not used) keeps the plain
``base_key`` so its caching behavior is unchanged.

Design notes:
  * No threshold / score comparison happens here, so this module registers no
    tunable — it is pure key derivation + a best-effort stamp reader.
  * ``version_stamps`` is deliberately **read-only toward ``server/app/``**: it
    never imports or mutates anything under ``app.`` (the live CSV system). It
    returns best-effort stamps, defaulting any missing stamp to ``"0"``.
"""
from __future__ import annotations

import hashlib

# The two version-stamp dimensions a cross-domain answer depends on. Kept as
# named constants so the key composition and the reader agree on the keys and
# no bare string literal drifts between them.
_STAMP_CSV_SEMANTIC = "csv_semantic_layer"
_STAMP_GRAPH_EXTRACTION = "graph_extraction"
_MISSING_STAMP = "0"


def build_cross_domain_cache_key(
    *,
    tenant_id: str,
    base_key: str,
    structured_query_used: bool,
    version_stamps: dict,
) -> str:
    """Derive the cache key for an answer, version-stamped iff cross-domain.

    When ``structured_query_used`` is False the answer is pure-PDF and the
    original ``base_key`` is returned verbatim (no behavior change). When True,
    the answer crossed into the CSV runtime, so we fold in BOTH the CSV semantic
    layer stamp and the PDF graph extraction stamp: bumping either side yields a
    different key, transparently evicting stale cross-domain answers.

    Args:
        tenant_id: request tenant; folded in so stamps cannot collide across
            tenants that happen to share a numeric version.
        base_key: the underlying (pure-PDF) cache key to extend.
        structured_query_used: whether the cross-domain ``structured_query`` tool
            participated in producing the answer.
        version_stamps: stamp dict; missing ``csv_semantic_layer`` /
            ``graph_extraction`` entries default to ``"0"`` (matching
            :func:`version_stamps`).

    Returns:
        ``base_key`` unchanged for a pure-PDF answer; otherwise a hex sha256
        digest over the base key, tenant, and both version stamps.
    """
    if not structured_query_used:
        return base_key

    stamps = version_stamps or {}
    csv_v = str(stamps.get(_STAMP_CSV_SEMANTIC, _MISSING_STAMP))
    graph_v = str(stamps.get(_STAMP_GRAPH_EXTRACTION, _MISSING_STAMP))
    payload = (
        f"{base_key}|{tenant_id}|"
        f"{_STAMP_CSV_SEMANTIC}={csv_v}|{_STAMP_GRAPH_EXTRACTION}={graph_v}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def version_stamps(tenant_id: str) -> dict:
    """Best-effort read of the two cross-domain version stamps for ``tenant_id``.

    Read-only toward ``server/app/``: this function NEVER imports or mutates any
    ``app.`` module (the live CSV system). It resolves each stamp from the
    PDF-side tunables surface (which can be overridden per-container via env/DB
    without importing ``app``), defaulting any unresolved stamp to ``"0"`` so a
    missing source can never raise.

    The stamps are intentionally opaque tokens: a CSV semantic-layer rebuild or a
    PDF graph re-extraction bumps the corresponding token (out of band), which
    flows into :func:`build_cross_domain_cache_key` to evict stale answers.

    Args:
        tenant_id: the container/tenant whose stamps to read.

    Returns:
        ``{"csv_semantic_layer": <str>, "graph_extraction": <str>}``; each value
        is ``"0"`` when its source is unavailable.
    """
    csv_v = _read_stamp(tenant_id, _STAMP_CSV_SEMANTIC)
    graph_v = _read_stamp(tenant_id, _STAMP_GRAPH_EXTRACTION)
    return {
        _STAMP_CSV_SEMANTIC: csv_v,
        _STAMP_GRAPH_EXTRACTION: graph_v,
    }


def _read_stamp(tenant_id: str, dimension: str) -> str:
    """Resolve a single stamp via the tunables surface, never touching ``app``.

    The tunables resolver is import-safe (no infra, no ``app`` import) and looks
    up a per-container override (env/DB) before falling back to the default. We
    use the ``"0"`` default so an unconfigured tenant gets the canonical
    "no version yet" token. Any failure is swallowed → ``"0"``.
    """
    try:
        from pdf_chat.tunables import get_tunable

        value = get_tunable(tenant_id, f"cross_domain.stamp.{dimension}", _MISSING_STAMP)
        return str(value) if value is not None else _MISSING_STAMP
    except Exception:  # pragma: no cover - best-effort; never fatal
        return _MISSING_STAMP

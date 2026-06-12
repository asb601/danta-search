"""[3a] LOOKUP — map-first, twin-aware candidate retrieval for one intent-step.

This is the SEARCH half of the navigator loop. Given one ``IntentStep`` it returns
a small, schema-twin-aware ``CandidateSlice`` of candidate tables — the narrow set
the proposer then reasons over. It conforms to INVARIANTS I4 and I5:

  * MAP-FIRST (I5). First consult the semantic MAP: is there a ``SemanticEntity``
    marked ``is_canonical_master`` for this step's entity (scoped to the
    container)? If so, build the slice from that master (plus its schema-twin
    siblings) and return WITHOUT touching the retriever. The map is the cache of
    governed conclusions; a hit is reused deterministically.
  * RETRIEVE ONLY ON A MISS (I4/I5). When the map has no canonical master, make
    exactly ONE call to the ONE hybrid engine
    ``retrieval.orchestrator.retrieve_with_scores`` (BM25 + vector + RRF). NOT a
    pure-vector search, NOT a new retriever. The retrieval query text is built from
    the STEP's FULL INTENT (entity + measure concept + grain context), never the
    bare entity token (I3/I4).
  * TWINS-AWARE. After a hit (map or retrieve), every schema-fingerprint sibling of
    each candidate is pulled in so a canonical-vs-lookalike decision is never lost
    because one twin out-ranked the others. The fingerprint grouping is reused from
    the legacy seam (``resolve.search._twin_siblings``) — a read of a precomputed
    column, NOT an all-pairs comparison.

Each candidate is resolved to its LOGICAL table name (the name the executor +
canonicalizer trust) from the request's ``identity_map`` — same rule as
``resolve.search._logical_table`` — falling back to the lexical key from the blob
path. A reliable ledger polarity (customer|vendor|None) is attached per candidate
(reusing ``resolve.brain._polarity_from_row``) so the later verifier/clarify has it.

Design properties (enforced):
  * Container-scoped, read-only. No cross-file conclusions are computed here.
  * NEVER raises. On any failed DB read the session is rolled back and whatever was
    gathered so far is returned (empty slice in the worst case), so the loop stays
    additive and can abstain or fall through.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.erp_classification import ErpClassification
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticEntity
from app.retrieval.orchestrator import retrieve_with_scores
from app.services.file_identity import FileIdentityMap, logical_table_key
from app.services.navigator.types import Candidate, CandidateSlice, IntentStep
# Self-contained: the polarity-reliability gate is LIFTED into the verifier (the
# navigator does not depend on the legacy resolve package, which is deleted at P5).
# The twin-grouping helper is lifted below into _twin_siblings. Both are pure reads
# of precomputed columns, not the pure-vector path.
from app.services.navigator.verifier import _polarity_from_row

logger = structlog.get_logger("navigator.retriever")


def _logical_table(
    file_id: str, blob_path: str | None, identity_map: FileIdentityMap | None,
) -> str | None:
    """Resolve the logical table name the executor uses for a file.

    Prefer the request's identity map (``.by_id[file_id].logical_name``) so the
    name matches exactly what ``canonicalize_logical_sql`` resolves; fall back to
    the lexical ``logical_table_key`` only when the map lacks the id. Mirrors
    ``resolve.search._logical_table`` (lifted to avoid coupling the navigator to a
    soon-to-be-deleted module)."""
    if identity_map is not None:
        identity = getattr(identity_map, "by_id", {}).get(file_id)
        if identity is not None and getattr(identity, "logical_name", None):
            return identity.logical_name
    if blob_path:
        return logical_table_key(blob_path)
    return None


def _intent_query_text(step: IntentStep) -> str:
    """Build the retrieval query text from the STEP's FULL INTENT (I3/I4).

    NOT the bare ``entity`` token: combine the business object, the measure
    concept, and the grain context into one contextual phrase so the ONE hybrid
    engine ranks on the whole sub-intent. De-duplicated, order-preserving, blanks
    dropped. Pure."""
    parts: list[str] = []
    for value in (
        step.entity,
        step.measure_concept,
        # grain context — "per <grain_entity>" / "by <time_grain>" — only when set.
        (f"per {step.grain_entity}" if step.grain_entity else None),
        (f"by {step.time_grain}" if step.time_grain else None),
    ):
        s = str(value).strip() if value else ""
        if s and s not in parts:
            parts.append(s)
    return " ".join(parts).strip()


async def _twin_siblings(
    db: AsyncSession,
    container_id: str,
    hit_file_ids: list[str],
) -> dict[str, list[str]]:
    """For the hit files, find their schema-twin siblings (lifted from the legacy
    seam so the navigator is self-contained).

    Groups by ``erp_classifications.schema_fingerprint`` so all members of a twin
    cluster are kept together. Returns ``{fingerprint: [file_id, ...]}`` for the
    fingerprints the hits belong to. A read of a precomputed column — NOT an
    all-pairs comparison. Container-scoped, read-only, never raises."""
    if not hit_file_ids:
        return {}
    try:
        # Fingerprints of the hit files.
        fp_rows = (
            await db.execute(
                select(ErpClassification.file_id, ErpClassification.schema_fingerprint)
                .where(ErpClassification.container_id == container_id)
                .where(ErpClassification.file_id.in_(hit_file_ids))
            )
        ).all()
        fingerprints = {fp for _fid, fp in fp_rows if fp}
        if not fingerprints:
            return {}
        # All container files that share those fingerprints (the twin clusters).
        sib_rows = (
            await db.execute(
                select(ErpClassification.file_id, ErpClassification.schema_fingerprint)
                .where(ErpClassification.container_id == container_id)
                .where(ErpClassification.schema_fingerprint.in_(list(fingerprints)))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("twin_query_error", error=str(exc)[:200])
        return {}
    clusters: dict[str, list[str]] = {}
    for fid, fp in sib_rows:
        if fp and fid:
            clusters.setdefault(fp, []).append(str(fid))
    return clusters


# Abstain-biased thresholds for the SEMANTIC master fallback (general, data-driven —
# NOT per-prompt tuning). A master is pinned by meaning only when its description
# embedding clearly wins: absolute cosine floor AND a margin over the 2nd-best, so
# indistinguishable schema-twins fall through to retrieval rather than mis-pin.
_MASTER_SEMANTIC_MIN = 0.42
_MASTER_SEMANTIC_MARGIN = 0.03


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Pure; 0.0 on degenerate/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sa = sb = 0.0
    for x, y in zip(a, b):
        dot += x * y; sa += x * x; sb += y * y
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return dot / ((sa ** 0.5) * (sb ** 0.5))


async def _master_file_ids(
    db: AsyncSession, container_id: str, entity: str,
) -> list[str]:
    """MAP read (I5): file ids of canonical masters whose concept matches ``entity``.

    Two GENERAL matchers (no per-prompt labels, no flags):
    1. High-precision token-subset (master label ⊆ entity tokens) — fires when the
       elected label happens to match the phrasing exactly.
    2. SEMANTIC fallback when (1) is empty: embed the concept and cosine it against
       the masters' stored ``description_embedding``; pin the single best master ONLY
       when it clears an absolute floor AND beats the runner-up by a margin. This
       makes any phrasing ("vendor spend", "what we owe") map to the right master by
       MEANING, while genuine twins (no clear winner) fall through to retrieval —
       abstain-biased, never a guess. Container-scoped, read-only, never raises."""
    ent = (entity or "").strip()
    if not container_id or not ent:
        return []
    entity_tokens = set(ent.lower().replace(" ", "_").split("_"))
    try:
        rows = (
            await db.execute(
                select(SemanticEntity.entity_name, SemanticEntity.file_id)
                .where(SemanticEntity.container_id == container_id)
                .where(SemanticEntity.is_canonical_master.is_(True))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("master_query_error", error=str(exc)[:200])
        return []
    if not rows:
        return []
    # 1) high-precision token-subset match
    token_hits = [
        str(file_id) for entity_name, file_id in rows
        if file_id and (nt := set(str(entity_name or "").lower().split("_"))) and nt <= entity_tokens
    ]
    if token_hits:
        return token_hits
    # 2) general semantic fallback (abstain-biased)
    try:
        from app.retrieval.embeddings import embed_text  # noqa: PLC0415
        master_ids = [str(fid) for _, fid in rows if fid]
        q = await embed_text(ent)
        if not q or not master_ids:
            return []
        erows = (
            await db.execute(
                select(FileMetadata.file_id, FileMetadata.description_embedding)
                .where(FileMetadata.file_id.in_(master_ids))
                .where(FileMetadata.description_embedding.is_not(None))
            )
        ).all()
        scored = sorted(
            (((_cos(q, list(emb))), str(fid)) for fid, emb in erows if emb is not None),
            reverse=True,
        )
        if not scored:
            return []
        top_score, top_id = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if top_score >= _MASTER_SEMANTIC_MIN and (top_score - second) >= _MASTER_SEMANTIC_MARGIN:
            logger.info("master_semantic_pin", entity=ent, score=round(top_score, 3), margin=round(top_score - second, 3))
            return [top_id]
        return []
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("master_semantic_error", error=str(exc)[:200])
        return []


async def _blob_paths_for(
    db: AsyncSession, file_ids: list[str],
) -> dict[str, str | None]:
    """Bulk-load blob_path for file ids (twins / masters may not carry one yet)."""
    if not file_ids:
        return {}
    try:
        rows = (
            await db.execute(
                select(FileMetadata.file_id, FileMetadata.blob_path)
                .where(FileMetadata.file_id.in_(file_ids))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("blob_path_query_error", error=str(exc)[:200])
        return {}
    return {str(fid): bp for fid, bp in rows}


async def _polarity_for(
    db: AsyncSession, container_id: str, file_ids: list[str],
) -> dict[str, str | None]:
    """Per-file reliable ledger polarity (customer|vendor|None).

    Reuses ``resolve.brain._polarity_from_row`` (the classifier's own reliability
    rule) so the verifier/clarify downstream sees the same side this dataset's
    ingestion classified. Defensive: missing/unreliable rows collapse to None
    ("unconstrained"); any error rolls back and returns what was gathered."""
    if not file_ids:
        return {}
    out: dict[str, str | None] = {}
    try:
        rows = (
            await db.execute(
                select(
                    ErpClassification.file_id,
                    ErpClassification.domain_polarity,
                    ErpClassification.confidence,
                    ErpClassification.source,
                    ErpClassification.source_system,
                )
                .where(ErpClassification.container_id == container_id)
                .where(ErpClassification.file_id.in_(file_ids))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("polarity_query_error", error=str(exc)[:200])
        return {}
    for fid, polarity, conf, src, src_sys in rows:
        out[str(fid)] = _polarity_from_row(polarity, conf, src, src_sys)
    return out


async def _build_slice(
    db: AsyncSession,
    container_id: str,
    step: IntentStep,
    identity_map: FileIdentityMap | None,
    *,
    seed_ids: list[str],
    scores: dict[str, float],
    blob_by_id: dict[str, str | None],
    from_map: bool,
    top_k: int,
    master_file_ids: tuple[str, ...] = (),
) -> CandidateSlice:
    """Expand ``seed_ids`` with their schema-twin siblings, resolve each to its
    logical table, attach polarity, and pack into a capped, hits-first slice.

    ``seed_ids`` are best-first (rank order). Twins follow. Shared by both the map
    and the retrieve paths so twin-expansion + logical resolution are identical.

    ``master_file_ids`` are the GOVERNED CANONICAL MASTERS among the seeds (set only
    on the map-hit path). Carried onto the slice so the driver can constrain PROPOSE
    to a lone master (I5). Only ids that survived logical-table resolution are kept,
    so a master that drops out (no table) never appears as a phantom master."""
    # Twins-together: pull every fingerprint sibling of the seeds.
    clusters = await _twin_siblings(db, container_id, seed_ids)
    twin_ids: list[str] = []
    for members in clusters.values():
        for fid in members:
            if fid not in blob_by_id and fid not in twin_ids and fid not in seed_ids:
                twin_ids.append(fid)
    if twin_ids:
        blob_by_id.update(await _blob_paths_for(db, twin_ids))

    # Seeds first (rank order), then twins. Cap so the slice stays small.
    ordered = seed_ids + [fid for fid in twin_ids if fid not in seed_ids]
    cap = top_k + len(twin_ids)
    ordered = ordered[:cap]

    polarity_by_id = await _polarity_for(db, container_id, ordered)

    master_set = set(master_file_ids)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    resolved_masters: list[str] = []
    for fid in ordered:
        if fid in seen:
            continue
        seen.add(fid)
        table = _logical_table(fid, blob_by_id.get(fid), identity_map)
        if not table:
            continue
        if fid in master_set:
            resolved_masters.append(fid)
        candidates.append(
            Candidate(
                file_id=fid,
                table=table,
                score=float(scores.get(fid, 0.0)),
                polarity=polarity_by_id.get(fid),
            )
        )

    logger.info(
        "lookup_slice",
        step_id=step.step_id,
        entity=(step.entity or "")[:80],
        from_map=from_map,
        seeds=len(seed_ids),
        twins=len(twin_ids),
        candidates=len(candidates),
        masters=len(resolved_masters),
    )
    return CandidateSlice(
        step_id=step.step_id,
        entity=step.entity,
        candidates=tuple(candidates),
        from_map=from_map,
        master_file_ids=tuple(resolved_masters),
    )


def _empty_slice(step: IntentStep) -> CandidateSlice:
    return CandidateSlice(step_id=step.step_id, entity=step.entity, candidates=(),
                          from_map=False)


async def lookup(
    db: AsyncSession,
    container_id: str,
    step: IntentStep,
    identity_map: FileIdentityMap | None = None,
    *,
    top_k: int = 9,
    user_id: str = "",
    is_admin: bool = True,
    allowed_domains: list[str] | None = None,
) -> CandidateSlice:
    """Resolve one intent-step to a twin-aware ``CandidateSlice``. Map-first (I5):
    a canonical-master hit skips the retriever; a miss makes ONE hybrid call (I4)
    over the step's full intent. NEVER raises — an empty slice on any failure.

    The REAL request auth (``user_id`` / ``is_admin`` / ``allowed_domains``) is
    threaded into the ONE hybrid engine so domain/permission scoping is enforced
    INSIDE retrieval (the orchestrator derives the user's ``allowed_domains`` from
    ``user_id`` for non-admins; ``allowed_domains`` is accepted here so the caller's
    already-resolved domain list is an explicit, auditable input rather than an
    invisible re-lookup). A platform-admin request (``is_admin=True``) is
    unrestricted, mirroring the rest of the runtime.
    """
    if not container_id or step is None or not (step.entity and step.entity.strip()):
        return _empty_slice(step) if step is not None else CandidateSlice(
            step_id="", entity=None
        )

    try:
        # ── [3a.a] MAP-FIRST (I5) ─────────────────────────────────────────────
        master_ids = await _master_file_ids(db, container_id, step.entity)
        if master_ids:
            blob_by_id = await _blob_paths_for(db, master_ids)
            # Masters carry no retrieval score; they are governed conclusions. They
            # are also DECLARED on the slice (master_file_ids) so the driver can
            # constrain PROPOSE to a lone master — twins added by _build_slice are
            # siblings for context, NOT masters (I5).
            return await _build_slice(
                db, container_id, step, identity_map,
                seed_ids=master_ids, scores={}, blob_by_id=blob_by_id,
                from_map=True, top_k=top_k, master_file_ids=tuple(master_ids),
            )

        # ── [3a.b] RETRIEVE ONLY ON A MISS (I4) ───────────────────────────────
        query_text = _intent_query_text(step)
        if not query_text:
            return _empty_slice(step)
        # The ONE hybrid engine (BM25 + vector + RRF), container-scoped, with the
        # REAL request auth threaded in (P2/guardian carry-forward): a non-admin
        # request's domain scope is enforced INSIDE retrieval (the orchestrator
        # derives allowed_domains from user_id), not just by the executor's
        # downstream allowed_file_ids allowlist. is_admin=True is unrestricted.
        hits = await retrieve_with_scores(
            query_text, user_id, is_admin, db, top_k=top_k,
            container_id=container_id,
        )
        seed_ids: list[str] = []
        scores: dict[str, float] = {}
        blob_by_id: dict[str, str | None] = {}
        for meta, score in (hits or []):
            fid = str(getattr(meta, "file_id", "") or "")
            if not fid or fid in scores:
                continue
            seed_ids.append(fid)
            scores[fid] = float(score)
            blob_by_id[fid] = getattr(meta, "blob_path", None)
        if not seed_ids:
            logger.info("lookup_empty_retrieval", step_id=step.step_id,
                        query=query_text[:120])
            return _empty_slice(step)
        return await _build_slice(
            db, container_id, step, identity_map,
            seed_ids=seed_ids, scores=scores, blob_by_id=blob_by_id,
            from_map=False, top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 — NEVER raise; degrade to empty slice
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("lookup_error", step_id=getattr(step, "step_id", "?"),
                       error=str(exc)[:200])
        return _empty_slice(step)

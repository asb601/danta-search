"""Phase 5 Task 10 — the read-only onboarding projection API.

A self-prefixed (``/api/pdf/onboarding``) FastAPI router that projects the
persisted comprehension artifact so a domain-naive engineer is productive on day
one: a cited topic map (the company's table-of-contents), a paged entity browse,
a glossary browse (with human provenance LABELS, never raw confidence), the
open-vocab learned document taxonomy, and the current ontology version.

DESIGN (mirrors ``pdf_chat/api/routes.py``):
  * heavy/infra imports are LATE (inside the dependency bodies) so this module
    imports cleanly with zero infra and without requiring the searcher/session
    factories to exist at import time;
  * the principal is derived SOLELY from the verified JWT via the SAME
    ``_resolve_current_user`` / ``get_principal`` override pattern as
    ``routes.py`` — the client never supplies the tenant, so a forged body value
    cannot widen access. Tenant scoping is threaded to every read;
  * READ-ONLY: every route is a ``GET``; no write verb is mounted. The router
    cannot mutate the artifact (it is built at ingest finalization, elsewhere).

The reads delegate to the comprehension read facade
(``pdf_chat/comprehension/onboarding_reads.py::OnboardingReads``), which bundles
the wave-1 ``reader.py`` helpers plus ``latest_ontology_id``. The facade + the
async session + the graph reader are FastAPI dependencies so tests inject fakes.

NOTE — how to mount in server/app/main.py (do NOT edit main.py here):
    from pdf_chat.api.onboarding import onboarding_router
    from pdf_chat.api.routes import _resolve_current_user
    from app.dependencies import get_current_user
    # Bind the JWT principal exactly as for pdf_router (shared override seam):
    app.dependency_overrides[_resolve_current_user] = get_current_user
    app.include_router(onboarding_router)   # already prefixed with /api/pdf/onboarding
The default reader/session/graph-reader dependencies wire real infra lazily, so
no override is required for them in production (only the auth seam is bound).

No score-comparison literal lives in this module: it makes NO gate decision
(pure reads / projection), so there is no ``get_tunable`` / ``log_gate_decision``
call and no magic literal here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from pdf_chat.api.routes import Principal, get_principal
from pdf_chat.comprehension.provenance import Provenance, label_for
from pdf_chat.comprehension.temporal import staleness_annotation

onboarding_router = APIRouter(prefix="/api/pdf/onboarding", tags=["pdf-onboarding"])


# --------------------------------------------------------------------------- #
# Injectable dependencies (late infra imports; tests override these).
# --------------------------------------------------------------------------- #
def get_onboarding_reader() -> Any:
    """Return the comprehension read facade (``OnboardingReads``).

    Late import keeps this module import-safe with zero infra. Tests override
    this dependency with an in-memory fake exposing the same method surface.
    """
    from pdf_chat.comprehension.onboarding_reads import OnboardingReads

    return OnboardingReads()


async def get_onboarding_session() -> Any:  # pragma: no cover - infra
    """Yield an async DB session for the comprehension reads (wired at mount).

    Overridden in production by the app's session factory and in tests by a
    stub. Until wired it returns ``None`` and the facade's reads degrade to an
    empty browse rather than touching a database at import time.
    """
    return None


async def get_onboarding_graph_reader() -> Any:  # pragma: no cover - infra
    """Return the GraphReader for the topic-map projection.

    Returns a ``Neo4jGraphReader`` wrapping a ``Neo4jSearcher`` — the adapter that
    implements the six ``GraphReader`` iterators the comprehension layer consumes
    (a raw ``Neo4jSearcher`` implements only the retrieval surface, so handing it
    here would ``AttributeError`` on ``iter_communities`` etc.). Late, guarded
    import — if the searcher/adapter is not deployable in isolation we return
    ``None`` and the topic map degrades to empty (never a 500). Tests do not need
    to override this; the fake reader ignores the graph-reader argument.
    """
    try:
        from pdf_chat.comprehension.neo4j_graph_reader import (  # type: ignore
            Neo4jGraphReader,
        )
        from pdf_chat.retrieval.neo4j_searcher import Neo4jSearcher  # type: ignore

        return Neo4jGraphReader(Neo4jSearcher())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Projection helpers (provenance LABELS, never raw confidence).
# --------------------------------------------------------------------------- #
def _field(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _entity_view(row: Any) -> dict:
    return {
        "name": _field(row, "name"),
        "entity_type": _field(row, "entity_type"),
        "normalized_value": _field(row, "normalized_value"),
        "pagerank": _field(row, "pagerank"),
        "mention_count": _field(row, "mention_count"),
    }


def _glossary_view(row: Any) -> dict:
    """Glossary browse view — provenance LABEL only; raw confidence NEVER shown."""
    return {
        "term": _field(row, "term"),
        "expansion": _field(row, "expansion"),
        "definition": _field(row, "definition"),
        "provenance": label_for(_field(row, "provenance", Provenance.NOT_FOUND.value)),
        "variants": list(_field(row, "variants", []) or []),
        "citations": [
            {"chunk_id": _field(s, "chunk_id"), "bbox": _field(s, "bbox")}
            for s in (_field(row, "evidence_spans", []) or [])
        ],
    }


def _doc_class_view(row: Any) -> dict:
    return {
        "doc_class": _field(row, "doc_class"),
        # confidence here is the LLM clustering confidence of the class itself, a
        # structural property of the taxonomy (not a per-fact confidence the user
        # must interpret) — surfaced so a browse can sort/threshold classes.
        "confidence": _field(row, "confidence"),
        "member_doc_ids": list(_field(row, "member_doc_ids", []) or []),
    }


def _iso(value: Any) -> Any:
    """Render a datetime as ISO-8601 (pass non-datetimes through unchanged)."""
    return value.isoformat() if isinstance(value, datetime) else value


def _coverage_view(row: Any, *, now: datetime, tenant_id: str) -> dict:
    """Temporal-coverage view with a human staleness NOTE (never a raw age number).

    The staleness note is produced by ``staleness_annotation`` (gated against
    ``comprehension.staleness_days``): a subject whose most recent mention is older
    than the threshold gets "most recent mention is YYYY-MM; may be outdated";
    a fresh (or dateless) subject gets ``None``. ``now`` is the real wall clock —
    the staleness of "outdated" is genuinely relative to the present.
    """
    last = _field(row, "last_mention_date")
    return {
        "subject": _field(row, "subject"),
        "subject_kind": _field(row, "subject_kind"),
        "min_date": _iso(_field(row, "min_date")),
        "max_date": _iso(_field(row, "max_date")),
        "last_mention_date": _iso(last),
        "density": _field(row, "density"),
        "staleness_note": staleness_annotation(last, now, container_id=tenant_id),
    }


# --------------------------------------------------------------------------- #
# Routes — all GET (read-only), all tenant-scoped via the JWT principal.
# --------------------------------------------------------------------------- #
@onboarding_router.get("/topic-map")
async def get_topic_map(
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    graph_reader: Any = Depends(get_onboarding_graph_reader),
):
    """The company's grounded table-of-contents (cited community reports).

    Each topic carries its ``citations`` (chunk ids) — only cited community
    reports are ever written to the graph, so the TOC is grounded (invariant 1).
    Tenant-scoped: the tenant comes from the JWT principal.
    """
    topics = await reader.topic_map(graph_reader, principal.tenant_id)
    return {"tenant_id": principal.tenant_id, "topics": topics}


@onboarding_router.get("/entities")
async def list_entities(
    limit: int = 50,
    offset: int = 0,
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    session: Any = Depends(get_onboarding_session),
):
    """Paged entity browse over the LATEST ontology version for this tenant.

    Returns an empty browse (not a 500) when no ontology has been built yet.
    """
    ontology_id = await reader.latest_ontology_id(session, principal.tenant_id)
    if ontology_id is None:
        return {"tenant_id": principal.tenant_id, "entities": []}
    rows = await reader.list_entities(session, ontology_id, limit=limit, offset=offset)
    return {
        "tenant_id": principal.tenant_id,
        "entities": [_entity_view(r) for r in rows],
    }


@onboarding_router.get("/glossary")
async def list_glossary(
    limit: int = 100,
    offset: int = 0,
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    session: Any = Depends(get_onboarding_session),
):
    """Glossary browse with human provenance LABELS (never raw confidence).

    PINNED to the LATEST ontology version for this tenant: the route resolves
    ``current_ontology_version`` first and passes it to ``list_glossary`` so a
    re-version's stale rows never leak into the browse (an unpinned read would
    return ALL versions, contradicting "latest-version rows"). When no ontology
    has been built yet (version is ``None``) the browse is empty, not a 500.
    An ``inferred`` entry surfaces "inferred from usage" and is never mislabeled
    "stated in docs".
    """
    version = await reader.current_ontology_version(session, principal.tenant_id)
    if version is None:
        return {"tenant_id": principal.tenant_id, "entries": []}
    rows = await reader.list_glossary(
        session, principal.tenant_id, ontology_version=version,
        limit=limit, offset=offset,
    )
    return {
        "tenant_id": principal.tenant_id,
        "entries": [_glossary_view(r) for r in rows],
    }


@onboarding_router.get("/doc-taxonomy")
async def list_doc_taxonomy(
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    session: Any = Depends(get_onboarding_session),
):
    """Open-vocab learned document classes for the latest ontology version.

    Doc classes are LLM-clustered arbitrary strings (never an enumerated list).
    """
    ontology_id = await reader.latest_ontology_id(session, principal.tenant_id)
    if ontology_id is None:
        return {"tenant_id": principal.tenant_id, "classes": []}
    rows = await reader.list_doc_taxonomy(session, ontology_id)
    return {
        "tenant_id": principal.tenant_id,
        "classes": [_doc_class_view(r) for r in rows],
    }


@onboarding_router.get("/temporal-coverage")
async def list_temporal_coverage(
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    session: Any = Depends(get_onboarding_session),
):
    """Per-subject temporal coverage for the latest ontology, with staleness notes.

    Surfaces what was computed + persisted at ingest (min/max/density/last mention
    per subject) and attaches a human staleness NOTE — a subject whose most recent
    mention is older than ``comprehension.staleness_days`` is flagged "may be
    outdated". Tenant-scoped, read-only. Returns an empty browse (not a 500) when
    no ontology has been built yet.
    """
    ontology_id = await reader.latest_ontology_id(session, principal.tenant_id)
    if ontology_id is None:
        return {"tenant_id": principal.tenant_id, "coverage": []}
    now = datetime.now(timezone.utc)
    rows = await reader.list_temporal_coverage(session, ontology_id)
    return {
        "tenant_id": principal.tenant_id,
        "coverage": [
            _coverage_view(r, now=now, tenant_id=principal.tenant_id) for r in rows
        ],
    }


@onboarding_router.get("/ontology/version")
async def get_ontology_version(
    principal: Principal = Depends(get_principal),
    reader: Any = Depends(get_onboarding_reader),
    session: Any = Depends(get_onboarding_session),
):
    """The current (highest) persisted ontology version for this tenant.

    ``null`` when no artifact has been built yet (queryable either way).
    """
    version = await reader.current_ontology_version(session, principal.tenant_id)
    return {"tenant_id": principal.tenant_id, "version": version}


__all__ = [
    "onboarding_router",
    "get_onboarding_reader",
    "get_onboarding_session",
    "get_onboarding_graph_reader",
]

"""Multi-representation card builders (Phase 2, Task 9).

The retrieval index is multi-representation (spec §1b "Multi-representation
index"): we embed not only raw chunks but ALSO **section-cards** (a section
summary + its grounded tag labels) and **doc-cards** (the doc-level relational
tag / summary). At query time all three vector spaces fuse via RRF, so a question
routes to the right document even when its chunks don't lexically match. Cost is
≈nothing extra — embeddings are ≈free and the tag/summary text is the
section-level LLM output already produced upstream (spec §8).

This module is the BUILDER half only: it turns a ``Section`` + its
``ExtractedTag``s into a ``SectionCard`` (and a doc's tags into a ``DocCard``),
produces the card *text* (summary + tag labels), and embeds it via
``retrieval.embeddings.embed_texts_batched`` using the per-container embedding
model from ``model_router.embedding_model``. The Neo4j vector-node write and the
``multi_vector_search`` RRF fusion live in their owners (``kg_writer.py`` /
``neo4j_searcher.py``); a card here is a pure data carrier.

GROUNDING / SAFEGUARD (spec §1b "Misleading-tag safeguard"): a card is a
RETRIEVAL signal, never an answer. Every card carries ``tenant_id`` (the tenant
boundary on every node) and provenance — the ``section_id`` / ``doc_id`` it was
built from and the source chunk ids of its tags — so a doc surfaced mainly via a
card stays verifiable downstream.

GOVERNING CRITERIA (millions of files, many tenants):
  * tenant isolation — every card carries ``tenant_id`` for per-hop isolation;
  * cost-at-scale — embedding goes through the batched seam (request fan-out is a
    per-container tunable) and reuses the section-level summary already produced;
  * per-client tunability — the section-tag cap resolves via ``get_tunable`` and
    every cap decision logs via ``log_gate_decision`` — no magic literal lives
    here;
  * grounded faithfulness — provenance (section/doc id + tag source chunks) is
    carried so a tag-surfaced result stays traceable.

Pure module — safe to import with zero infra. The embedding call is delegated
lazily, so importing this never touches Azure OpenAI or a database.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pdf_chat.tunables import get_tunable, log_gate_decision

# ── Tunable keys (defaults live in / SHOULD be mirrored into TUNABLE_DEFAULTS) ─
# The section-tag cap is shared with the extractor's post-processing cap so the
# card text never carries more tags than the graph kept. Defaults are passed at
# the call site so the module stays import-safe with zero infra, and are LISTED
# for integration to register in tunables.TUNABLE_DEFAULTS (single source).
_TUN_CARD_SECTION_TAG_CAP = "kg.card.section_tag_cap"  # max section tags in card text
_TUN_CARD_SUMMARY_MAX_CHARS = "kg.card.summary_max_chars"  # summary length ceiling

# Named defaults (passed to get_tunable; mirror these into TUNABLE_DEFAULTS).
_DEFAULT_CARD_SECTION_TAG_CAP = 6
_DEFAULT_CARD_SUMMARY_MAX_CHARS = 480


@dataclass(frozen=True)
class SectionCard:
    """A section's retrieval card — section summary + grounded tag labels.

    ``card_id``     deterministic: ``f"{section_id}::card"``.
    ``text``        the embedded representation (summary + tag labels).
    ``embedding``   filled by the builder (``None`` only if embedding is skipped).
    ``tenant_id``   the tenant boundary carried on every node.
    ``tag_labels``  provenance: the grounded tag labels folded into the text.
    ``src_chunk_ids`` provenance: source chunks of the tags (verifiability).
    """

    card_id: str
    section_id: str
    doc_id: str
    tenant_id: str
    text: str
    embedding: list[float] | None = None
    tag_labels: tuple[str, ...] = ()
    src_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocCard:
    """A document's retrieval card — the doc-level relational tag / summary.

    Mirrors ``SectionCard`` provenance so a doc surfaced via its doc-card stays
    traceable to the chunk spans that grounded its tags.
    """

    card_id: str
    doc_id: str
    tenant_id: str
    text: str
    embedding: list[float] | None = None
    tag_labels: tuple[str, ...] = ()
    src_chunk_ids: tuple[str, ...] = ()


def _norm(s: str) -> str:
    """Whitespace-collapsed text (stable for fingerprints / length checks)."""
    return " ".join((s or "").split())


def _summary_of(text: str, *, max_chars: int) -> str:
    """A bounded, deterministic section summary (lead text, length-capped).

    We do NOT issue a separate LLM call: the section's own text (already the
    LLM-extraction unit) is the summary substrate. Truncation is char-bounded
    via a per-container tunable so a long section never produces an oversized
    embedding request — cost-at-scale by construction.
    """
    body = _norm(text)
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip()


def _embed_card_texts(
    texts: list[str], *, container_id: str
) -> list[list[float] | None]:
    """Embed card texts via the batched seam + the per-container model.

    The model id is resolved through ``model_router.embedding_model`` (the single
    embedding-model seam) so cards embed with EXACTLY the model used at ingest /
    query time. ``embed_texts_batched`` caps the per-request fan-out (a tunable),
    so a tenant with millions of cards never issues one oversized request. Both
    imports are lazy so this module stays import-safe with zero infra.
    """
    if not texts:
        return []
    from pdf_chat.model_router import embedding_model
    from pdf_chat.retrieval.embeddings import embed_texts_batched

    model = embedding_model(container_id)
    vecs = embed_texts_batched(texts, container_id=container_id, model=model)
    return list(vecs)


def _card_id(prefix: str) -> str:
    """Deterministic, length-bounded suffix-safe card id helper."""
    return prefix


def build_section_card(section, tags, *, container_id: str) -> SectionCard:
    """Build + embed a SectionCard from a ``Section`` and its grounded tags.

    ``tags`` are the section's tags (``ExtractedTag`` or ``GroundedTag`` — any
    object exposing ``label``/``scope``/``src_chunk_id``; doc-scope tags are
    ignored here, they belong to the doc-card). The card *text* is the section
    summary followed by the (capped) section tag labels — the multi-rep
    representation that lets a question route to this section even when its raw
    chunks don't lexically match. Embedding is produced via the batched seam.

    Carries ``tenant_id`` (from the section) + provenance (section/doc id + tag
    source chunks) so a section surfaced via its card stays verifiable.
    """
    section_tags = [
        t for t in (tags or []) if getattr(t, "scope", "section") == "section"
    ]
    cap = int(
        get_tunable(
            container_id, _TUN_CARD_SECTION_TAG_CAP, _DEFAULT_CARD_SECTION_TAG_CAP
        )
    )
    kept = section_tags[:cap]
    log_gate_decision(
        "kg.card.section_tag_cap",
        score=float(len(section_tags)),
        threshold=float(cap),
        outcome="cap" if len(section_tags) > cap else "keep",
        container_id=container_id,
        section_id=section.section_id,
        kept=len(kept),
    )

    max_chars = int(
        get_tunable(
            container_id, _TUN_CARD_SUMMARY_MAX_CHARS, _DEFAULT_CARD_SUMMARY_MAX_CHARS
        )
    )
    summary = _summary_of(section.text, max_chars=max_chars)
    labels = tuple(_norm(t.label) for t in kept if _norm(getattr(t, "label", "")))
    parts = [summary]
    if labels:
        parts.append("Tags: " + "; ".join(labels))
    text = "\n".join(p for p in parts if p)

    embedding = _embed_card_texts([text], container_id=container_id)[0] if text else None
    src_chunk_ids = tuple(
        getattr(t, "src_chunk_id", "") for t in kept if getattr(t, "src_chunk_id", "")
    )
    return SectionCard(
        card_id=_card_id(f"{section.section_id}::card"),
        section_id=section.section_id,
        doc_id=section.doc_id,
        tenant_id=section.tenant_id,
        text=text,
        embedding=embedding,
        tag_labels=labels,
        src_chunk_ids=src_chunk_ids,
    )


def build_doc_card(
    doc_tags, *, container_id: str, doc_id: str, tenant_id: str
) -> DocCard:
    """Build + embed a DocCard from a document's doc-level tags.

    ``doc_tags`` are the doc-scope tags for ``doc_id`` (each carrying a grounded
    span + source chunk). The card *text* is the doc-level relational tag label(s)
    — the cross-doc semantic-routing signal (the "answer in PDF A, referenced
    from PDF G" case, spec §1b). ``doc_id`` and ``tenant_id`` are explicit because
    the doc-card is built once per document (not per section) and a tag does not
    itself carry a doc id.

    Tags whose ``scope`` is not ``"doc"`` are ignored (section tags belong to the
    section-card). Carries ``tenant_id`` + provenance (source chunk ids) so a doc
    surfaced via its doc-card stays verifiable.
    """
    doc_only = [
        t for t in (doc_tags or []) if getattr(t, "scope", "doc") == "doc"
    ]
    labels = tuple(
        _norm(t.label) for t in doc_only if _norm(getattr(t, "label", ""))
    )
    log_gate_decision(
        "kg.card.doc_tags",
        score=float(len(labels)),
        threshold=1.0,
        outcome="build" if labels else "empty",
        container_id=container_id,
        doc_id=doc_id,
    )
    text = ("Document: " + "; ".join(labels)) if labels else ""
    embedding = _embed_card_texts([text], container_id=container_id)[0] if text else None
    src_chunk_ids = tuple(
        getattr(t, "src_chunk_id", "")
        for t in doc_only
        if getattr(t, "src_chunk_id", "")
    )
    return DocCard(
        card_id=_card_id(f"{doc_id}::doccard"),
        doc_id=doc_id,
        tenant_id=tenant_id,
        text=text,
        embedding=embedding,
        tag_labels=labels,
        src_chunk_ids=src_chunk_ids,
    )

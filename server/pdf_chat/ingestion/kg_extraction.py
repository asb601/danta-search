"""Phase-2 Task 4/5 — SECTION-LEVEL extraction (the ``TaskClass.EXTRACTION`` call site).

This module is the one place that consumes a :class:`~pdf_chat.ingestion.sectionizer.Section`
and produces a grounded knowledge-graph payload — open-vocabulary entities,
relations, and tags — via a single structured ``gpt-4o-mini`` call PER SECTION
(spec §1b: SECTION is the default granularity, NOT per-chunk; retrieval fidelity
comes from embeddings ≈free, so coarser extraction keeps the graph useful at
~8× lower cost).

Contract (spec §1b + §3 invariants 1/4/6, §5 router):
  * The LLM is routed through ``model_router.select_model(task=TaskClass.EXTRACTION,
    container_id=..., signals={})``. ``signals={}`` means the data-driven
    escalation gate can NEVER fire for bulk ingestion → the BULK (``gpt-4o-mini``)
    id is always returned and the strong tier is structurally never invoked
    (asserted by test). Escalation is OFF for bulk ingestion, by construction.
  * IDEMPOTENT on ``section_fingerprint(section, prompt_version, model_id)``: a
    re-extract of the same section with the same prompt+model returns the cached
    result with ZERO additional LLM calls.
  * GROUNDED: every entity/relation/tag carries ``confidence`` + verbatim
    ``span`` + ``src_chunk_id``. The blocking :mod:`grounding_gate` rejects any
    whose claim is absent from its cited span downstream — this module produces
    the candidates; the gate admits/rejects.
  * OPEN-VOCABULARY types: the LLM proposes entity/tag types per tenant (mirrors
    the structured side's ``custom:<kind>:<slug>``); no closed dictionary.
  * Emits ONE doc-level relational tag (scope ``"doc"``) + a small, capped set of
    section topic tags (scope ``"section"``).
  * ADAPTIVE CAPPED GLEANING: re-prompt for missed entities up to
    ``kg.gleaning.max_passes`` passes, stopping early when a pass yields fewer
    than ``kg.gleaning.new_entity_floor`` new entities.

No magic literals: every cap/floor resolves via ``get_tunable`` and every
gate/cache/gleaning decision is emitted via ``log_gate_decision`` (spec §3 inv 4).
Pure-testable with zero infra: the LLM is injected (``SectionExtractor(llm)``);
the optional cache is injected. The router import is the only seam, and it never
invokes a model — it returns a :class:`~pdf_chat.model_router.ModelChoice` only.

GOVERNING CRITERIA (many tenants, millions of files): one LLM call per section
(cost-at-scale), bulk-only routing (escalation OFF), idempotent fingerprint
(re-runs are free), per-container caps (per-client tunability), grounded spans
(faithfulness) — all first-class here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..model_router import TaskClass, select_model
from ..tunables import get_tunable, log_gate_decision

# ── Tunable keys (named here; defaults SHOULD live in TUNABLE_DEFAULTS) ───────
# Passed as NAMED defaults at the call site so this module stays import-safe with
# zero infra and never compares against a bare inline literal (spec §3 inv 4).
# LISTED in the integration return so the single-source registry gains them.
TUN_GLEANING_MAX_PASSES = "kg.gleaning.max_passes"      # adaptive gleaning cap
TUN_GLEANING_NEW_FLOOR = "kg.gleaning.new_entity_floor"  # stop when a pass adds < this
TUN_SECTION_TAG_CAP = "kg.extraction.section_tag_cap"   # max section topic tags kept

_DEFAULT_GLEANING_MAX_PASSES = 2
_DEFAULT_GLEANING_NEW_FLOOR = 1
_DEFAULT_SECTION_TAG_CAP = 5

# Gleaning bookkeeping thresholds are CONTRACT invariants (not per-container
# dials), but they still route through log_gate_decision so no comparison is
# silent. A single pass is always run; gleaning passes are the optional extra.
_MIN_PASSES = 1


# ── extracted artifacts (the KG candidates the grounding gate then admits) ────
@dataclass(frozen=True)
class ExtractedEntity:
    """An open-vocabulary entity candidate proposed from one section.

    ``etype`` is LLM-proposed per tenant (open-vocab, mirrors ``custom:<kind>``).
    ``span`` is the verbatim supporting text; ``src_chunk_id`` cites the chunk it
    came from. The grounding gate / entity resolver consume these downstream.
    """

    name: str
    etype: str
    confidence: float
    span: str
    src_chunk_id: str


@dataclass(frozen=True)
class ExtractedRelation:
    """An (subject, predicate, object) relation candidate from one section.

    Carries ``confidence`` + verbatim ``span`` + ``src_chunk_id`` so the blocking
    grounding gate can verify the claim is present in the cited span before the
    edge is ever persisted (spec invariant 1).
    """

    subject: str
    predicate: str
    obj: str
    confidence: float
    span: str
    src_chunk_id: str


@dataclass(frozen=True)
class ExtractedTag:
    """A grounded tag — a RETRIEVAL signal, never an answer (spec §1b safeguard).

    ``scope`` is ``"doc"`` (the single doc-level relational tag) or ``"section"``
    (a section topic tag). Carries ``confidence`` + verbatim ``span`` +
    ``src_chunk_id`` so the grounding gate can reject ungrounded tags.
    """

    label: str
    scope: str
    confidence: float
    span: str
    src_chunk_id: str


# ── idempotency key ──────────────────────────────────────────────────────────
def section_fingerprint(section, prompt_version: str, model_id: str) -> str:
    """Stable idempotency key for a section's extraction.

    Combines the section's own (model-stable) text fingerprint with the prompt
    version and the resolved model id, so re-running with the SAME section +
    prompt + model returns the cached result, while a prompt/model bump
    invalidates it (spec §1b: idempotent on ``unit_fingerprint + prompt/model``).
    """
    raw = f"{section.fingerprint}|{prompt_version}|{model_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


_ExtractResult = tuple[list[ExtractedEntity], list[ExtractedRelation], list[ExtractedTag]]


class SectionExtractor:
    """The SECTION-level extraction call site (one bulk LLM call per section).

    ``llm`` is any object exposing ``extract(prompt, *, section, model_id,
    container_id, known_entities) -> dict`` and returning a JSON-shaped payload
    (``{"entities": [...], "relations": [...], "tags": [...]}``). It is injected
    so the whole module is pure-testable; production wires the prompt-cached
    Azure ``gpt-4o-mini`` client behind this seam.

    ``cache`` is any mapping-like object exposing ``get(fp)`` / ``set(fp, value)``
    (e.g. a dict in tests, Redis in prod). When present, extraction is idempotent
    on ``section_fingerprint`` with ZERO additional LLM calls on a hit.
    """

    PROMPT_VERSION = "p2.v1"

    # The system block is stable across every section so it is PROMPT-CACHED
    # (spec §5 token reduction: prompt caching is the biggest lever). It states
    # the grounding contract: every claim must be backed by a verbatim span +
    # the chunk it came from; types are open-vocabulary.
    _SYSTEM_PROMPT = (
        "You are a grounded knowledge-graph extractor. From the given document "
        "section emit ONLY claims supported verbatim by the text. For each entity, "
        "relation, and tag include a `span` copied verbatim from the section and "
        "the `src_chunk_id` it came from, plus a `confidence` in [0,1]. Entity and "
        "tag types are OPEN-VOCABULARY (propose the type that fits this tenant's "
        "domain; do not use a fixed list). Emit exactly ONE doc-scope relational "
        "tag summarizing the section's primary subject, plus a small set of "
        "section-scope topic tags. Return strict JSON with keys "
        "`entities`, `relations`, `tags`."
    )

    def __init__(self, llm, *, cache=None) -> None:
        self._llm = llm
        self._cache = cache

    def extract(self, section, *, container_id: str) -> _ExtractResult:
        """Extract grounded entities/relations/tags from ONE section.

        Routes the model via ``select_model(task=EXTRACTION, signals={})`` (bulk
        only — escalation OFF), short-circuits on a cached fingerprint, then runs
        a single LLM pass plus adaptive capped gleaning. The result is cached on
        ``section_fingerprint`` so a re-extract is free.
        """
        # Bulk-only routing: signals={} ⇒ the escalation gate can never fire, so
        # the strong tier is structurally never reached from extraction (spec §5).
        choice = select_model(
            task=TaskClass.EXTRACTION, container_id=container_id, signals={}
        )
        # Defensive invariant: an ingestion extraction must never escalate. This
        # holds by construction (signals={}); the assert documents + guards it.
        assert choice.is_strong is False, "extraction must never reach the strong tier"

        fp = section_fingerprint(section, self.PROMPT_VERSION, choice.model_id)

        if self._cache is not None:
            hit = self._cache.get(fp)
            if hit is not None:
                log_gate_decision(
                    "kg.extract.cache",
                    score=1.0,
                    threshold=1.0,
                    outcome="hit",
                    container_id=container_id,
                    section_id=section.section_id,
                    fingerprint=fp,
                )
                return hit

        result = self._run_extraction(section, container_id=container_id, model_id=choice.model_id)

        if self._cache is not None:
            self._cache.set(fp, result)
        return result

    # ── internals ────────────────────────────────────────────────────────────
    def _run_extraction(
        self, section, *, container_id: str, model_id: str
    ) -> _ExtractResult:
        """One LLM pass + adaptive capped gleaning, then post-process tags."""
        max_passes = max(
            _MIN_PASSES,
            int(
                get_tunable(
                    container_id, TUN_GLEANING_MAX_PASSES, _DEFAULT_GLEANING_MAX_PASSES
                )
            ),
        )
        new_floor = int(
            get_tunable(
                container_id, TUN_GLEANING_NEW_FLOOR, _DEFAULT_GLEANING_NEW_FLOOR
            )
        )

        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        tags: list[ExtractedTag] = []
        seen_names: set[str] = set()

        for pass_no in range(max_passes):
            payload = self._call_llm(
                section,
                container_id=container_id,
                model_id=model_id,
                known_entities=sorted(seen_names),
            )
            p_entities = self._parse_entities(payload, section)
            p_relations = self._parse_relations(payload, section)
            p_tags = self._parse_tags(payload, section)

            new_count = 0
            for e in p_entities:
                key = e.name.strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                    entities.append(e)
                    new_count += 1
            relations.extend(p_relations)
            tags.extend(p_tags)

            # Adaptive stop: a gleaning pass that adds fewer than the floor of new
            # entities is wasted spend → stop. The first pass always runs.
            decision = log_gate_decision(
                "kg.extract.gleaning",
                score=float(new_count),
                threshold=float(new_floor),
                outcome="continue" if new_count >= new_floor else "stop",
                container_id=container_id,
                section_id=section.section_id,
                pass_no=pass_no + 1,
                max_passes=max_passes,
            )
            if pass_no + 1 >= max_passes or not decision["passed"]:
                break

        tags = self._shape_tags(tags, container_id=container_id, section=section)
        return entities, relations, tags

    def _call_llm(
        self, section, *, container_id: str, model_id: str, known_entities: list[str]
    ) -> dict:
        """Invoke the injected LLM seam and return its parsed JSON payload.

        The injected ``llm.extract`` returns either a ``dict`` or a JSON string;
        both are normalized here so the seam stays trivially mockable.
        """
        raw = self._llm.extract(
            self._SYSTEM_PROMPT,
            section=section,
            model_id=model_id,
            container_id=container_id,
            known_entities=known_entities,
        )
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return {}
        return raw or {}

    # ── payload parsing (defensive: a malformed item is dropped, never raised) ─
    def _src_chunk(self, item: dict, section) -> str:
        """Resolve the citing chunk id — the LLM's value if present, else the
        section's first chunk (the section's grounding anchor)."""
        cid = (item or {}).get("src_chunk_id")
        if cid:
            return str(cid)
        return section.chunk_ids[0] if section.chunk_ids else ""

    @staticmethod
    def _conf(item: dict) -> float:
        try:
            return float((item or {}).get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _parse_entities(self, payload: dict, section) -> list[ExtractedEntity]:
        out: list[ExtractedEntity] = []
        for item in (payload or {}).get("entities", []) or []:
            name = str((item or {}).get("name", "")).strip()
            if not name:
                continue
            out.append(
                ExtractedEntity(
                    name=name,
                    etype=str((item or {}).get("type", item.get("etype", ""))).strip(),
                    confidence=self._conf(item),
                    span=str((item or {}).get("span", "")),
                    src_chunk_id=self._src_chunk(item, section),
                )
            )
        return out

    def _parse_relations(self, payload: dict, section) -> list[ExtractedRelation]:
        out: list[ExtractedRelation] = []
        for item in (payload or {}).get("relations", []) or []:
            subj = str((item or {}).get("subject", "")).strip()
            obj = str((item or {}).get("object", item.get("obj", ""))).strip()
            pred = str((item or {}).get("predicate", "")).strip()
            if not (subj and obj and pred):
                continue
            out.append(
                ExtractedRelation(
                    subject=subj,
                    predicate=pred,
                    obj=obj,
                    confidence=self._conf(item),
                    span=str((item or {}).get("span", "")),
                    src_chunk_id=self._src_chunk(item, section),
                )
            )
        return out

    def _parse_tags(self, payload: dict, section) -> list[ExtractedTag]:
        out: list[ExtractedTag] = []
        for item in (payload or {}).get("tags", []) or []:
            label = str((item or {}).get("label", "")).strip()
            if not label:
                continue
            scope = str((item or {}).get("scope", "section")).strip().lower()
            scope = scope if scope in ("doc", "section") else "section"
            out.append(
                ExtractedTag(
                    label=label,
                    scope=scope,
                    confidence=self._conf(item),
                    span=str((item or {}).get("span", "")),
                    src_chunk_id=self._src_chunk(item, section),
                )
            )
        return out

    def _shape_tags(
        self, tags: list[ExtractedTag], *, container_id: str, section
    ) -> list[ExtractedTag]:
        """Emit exactly ONE doc-level relational tag + a capped set of section tags.

        The doc-level tag is the highest-confidence ``scope=="doc"`` tag (the
        section's primary relational summary, spec §1b). Section topic tags are
        the highest-confidence ``scope=="section"`` tags, capped per-container.
        """
        cap = int(
            get_tunable(container_id, TUN_SECTION_TAG_CAP, _DEFAULT_SECTION_TAG_CAP)
        )
        doc_tags = sorted(
            (t for t in tags if t.scope == "doc"),
            key=lambda t: t.confidence,
            reverse=True,
        )
        section_tags = sorted(
            (t for t in tags if t.scope == "section"),
            key=lambda t: t.confidence,
            reverse=True,
        )[:cap]

        shaped: list[ExtractedTag] = []
        if doc_tags:
            shaped.append(doc_tags[0])  # exactly one doc-level relational tag
        shaped.extend(section_tags)
        log_gate_decision(
            "kg.extract.tags",
            score=float(len(shaped)),
            threshold=float(cap),
            outcome="shaped",
            container_id=container_id,
            section_id=section.section_id,
            doc_tags=1 if doc_tags else 0,
            section_tags=len(section_tags),
        )
        return shaped

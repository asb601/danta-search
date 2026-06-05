"""Phase 5 — temporal coverage registry + staleness annotation (Task 8).

Two pure, infra-free pieces of the comprehension artifact:

* ``compute_temporal_coverage`` — per-subject date span learned from the grounded
  graph: for every entity, when was it FIRST and LAST mentioned, over how dense a
  window. Dates come from each chunk's own ``doc_date`` and fall back to its
  ``(:Document).doc_date`` (the chunk inherits its document's date). Tenant-scoped
  via the injected ``GraphReader`` (per-hop isolation lives in the searcher).

* ``staleness_annotation`` — a faithfulness signal (spec §4): if a subject's most
  recent mention is older than ``comprehension.staleness_days`` it returns a
  human-facing note ("most recent mention is YYYY-MM; may be outdated"), else
  ``None``. ``now`` is an EXPLICIT parameter — this module NEVER reads the wall
  clock, so the annotation is deterministic and testable.

No score-comparison literal lives here: the staleness threshold resolves through
``get_tunable`` and the keep/drop decision is emitted via ``log_gate_decision``
(spec §3 invariant 4). The day span (a unit conversion, not a tunable gate) is
read from ``datetime`` arithmetic.

Call site (deferred wiring): ``ontology_builder.build_tenant_ontology`` persists
the coverage rows under the new ontology version; the onboarding/agent surfaces
read them back to annotate stale answers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pdf_chat.tunables import get_tunable, log_gate_decision

from .reader import GraphReader, _field

# Tunable keys (registered in tunables.py::TUNABLE_DEFAULTS — single source).
TUN_STALENESS_DAYS = "comprehension.staleness_days"

# Seconds per day — a unit conversion (NOT a tunable gate threshold), used to
# render the day-delta the staleness gate compares.
_SECONDS_PER_DAY = 86400.0


def _coerce_date(value: Any) -> datetime | None:
    """Parse a date value (ISO ``str`` or ``datetime``) to a tz-aware datetime.

    Naive datetimes are assumed UTC. Unparseable / missing values return ``None``
    so a dateless chunk simply contributes nothing to coverage (never crashes).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept a bare date or a full ISO timestamp.
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Last resort: fromisoformat (handles offsets); fall through on failure.
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _chunk_entities(chunk: Any) -> list[str]:
    """Best-effort list of entity names a chunk mentions (open field shapes)."""
    ents = _field(chunk, "entities", None)
    if ents is None:
        ents = _field(chunk, "entity_names", [])
    return list(ents or [])


async def compute_temporal_coverage(
    reader: GraphReader, *, tenant_id: str, container_id: str,
) -> list[dict]:
    """Per-subject (entity) temporal coverage from the grounded graph.

    Returns one record per mentioned subject::

        {"subject_kind": "entity", "subject", "min_date", "max_date",
         "density", "last_mention_date"}

    ``density`` = mentions / span-in-days (a single-mention subject gets the raw
    mention count, since its span is zero). Dates come from each chunk's own
    ``doc_date``; when a chunk has none, it inherits its ``(:Document).doc_date``.
    """
    # Build a doc_id → date map so dateless chunks inherit their document date.
    doc_dates: dict[str, datetime] = {}
    async for doc in reader.iter_documents(tenant_id):
        d = _coerce_date(_field(doc, "doc_date") or _field(doc, "created_at"))
        if d is not None:
            doc_dates[_field(doc, "doc_id")] = d

    # Accumulate the dated mentions per subject.
    dates_by_subject: dict[str, list[datetime]] = {}
    async for chunk in reader.iter_chunks(tenant_id):
        chunk_date = _coerce_date(_field(chunk, "doc_date"))
        if chunk_date is None:
            chunk_date = doc_dates.get(_field(chunk, "doc_id"))
        if chunk_date is None:
            continue  # no usable date → contributes nothing (never guessed)
        for name in _chunk_entities(chunk):
            dates_by_subject.setdefault(name, []).append(chunk_date)

    coverage: list[dict] = []
    for subject, dates in dates_by_subject.items():
        if not dates:
            continue
        min_date = min(dates)
        max_date = max(dates)
        span_days = (max_date - min_date).total_seconds() / _SECONDS_PER_DAY
        # Mentions-per-day over the span; a zero-span (single date) subject keeps
        # its raw mention count so density is always defined and positive. The
        # ``if span_days`` test is a division-by-zero guard (falsy ⇒ zero span),
        # NOT a tunable score gate (no comparison-literal).
        density = (len(dates) / span_days) if span_days else float(len(dates))
        coverage.append(
            {
                "subject_kind": "entity",
                "subject": subject,
                "min_date": min_date,
                "max_date": max_date,
                "density": density,
                "last_mention_date": max_date,
            }
        )
    return coverage


def staleness_annotation(
    last_mention_date: datetime | None,
    now: datetime,
    *,
    container_id: str,
) -> str | None:
    """A human note when the most recent mention is stale, else ``None``.

    ``now`` is supplied by the caller (never the wall clock) so the result is
    deterministic. The age (in days) is compared to the tunable
    ``comprehension.staleness_days`` via ``log_gate_decision`` (no inline
    literal). A note is returned only when the mention is OLDER than the
    threshold; a fresh mention — and an unknown last-mention date (refuse to
    guess) — return ``None``.
    """
    if last_mention_date is None:
        return None
    if last_mention_date.tzinfo is None:
        last_mention_date = last_mention_date.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = (now - last_mention_date).total_seconds() / _SECONDS_PER_DAY
    threshold_days = float(get_tunable(container_id, TUN_STALENESS_DAYS))
    rec = log_gate_decision(
        "comprehension.staleness_days",
        score=age_days,
        threshold=threshold_days,
        outcome="checked",
        container_id=container_id,
        last_mention_date=last_mention_date.isoformat(),
    )
    # ``passed`` is age >= threshold ⇒ the mention is stale ⇒ annotate.
    if not rec["passed"]:
        return None
    return (
        f"most recent mention is {last_mention_date.strftime('%Y-%m')}; "
        "may be outdated"
    )


__all__ = ["compute_temporal_coverage", "staleness_annotation"]

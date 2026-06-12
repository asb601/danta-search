"""Production LLM adapters for Phase-2 knowledge-graph construction.

``construct_knowledge_graph`` injects two LLM seams that, until now, only existed
as in-memory fakes in the tests:

  * ``SectionExtractor`` needs ``extract(system, *, section, model_id,
    container_id, known_entities) -> dict`` returning
    ``{"entities": [...], "relations": [...], "tags": [...]}``.
  * ``CommunityReporter`` needs ``synthesize(prompt, *, model_id, container_id)
    -> dict`` returning ``{"summary": "..."}``.

Both are SYNC (Phase-2 ``construct_knowledge_graph`` is a sync function that runs
inside the Celery graph-build task). :class:`KgIngestionLlm` implements both by
calling the SAME Azure OpenAI deployment the rest of pdf_chat uses, in strict JSON
mode, at temperature 0. The model id is chosen by the CONSUMER via
``model_router.select_model`` and handed in as ``model_id`` — the adapter never
routes (so per-container escalation / the bulk-only allowlist stay authoritative
at the one selection seam, contract C7).

Guarded import (mirrors ``retrieval.llm.PdfLlm``): the adapter constructs with no
infra and raises a clear ``RuntimeError`` only when a method is actually CALLED
without the OpenAI SDK present.
"""
from __future__ import annotations

import json
from typing import Any

from pdf_chat.config import azure_openai_credentials
from pdf_chat.observability.cost_tracker import get_cost_tracker

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - exercised only without infra
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI = False


def _build_client():  # pragma: no cover - requires infra + env
    endpoint, api_key, api_version = azure_openai_credentials()
    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)


def chat_json(
    *,
    system: str,
    user: str,
    model_id: str,
    container_id: str,
    phase: str,
) -> dict:
    """One strict-JSON chat completion → parsed dict (best-effort, never raises on
    a malformed model reply — returns ``{}``).

    ``phase`` is the cost-tracker bucket (``"extraction"`` / ``"synthesis"`` /
    ``"comprehension"``) so GET /api/pdf/metrics reflects ingestion LLM spend and
    flags a gpt-4o regression as a policy violation against the SELECTED model id.
    """
    if not _HAS_OPENAI:
        raise RuntimeError(
            "The OpenAI SDK is required for KG ingestion LLM calls but is not "
            "installed."
        )
    client = _build_client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = getattr(resp, "usage", None)
    get_cost_tracker().track_llm(
        container_id,
        phase,
        model_id,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost_usd=0.0,  # no price table wired in pdf_chat yet (best-effort).
        document_id=None,
        trace_id=None,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _serialize_section(section: Any, known_entities: list[str] | None) -> str:
    """Render a section into the user message the extractor system prompt expects.

    Includes the section's chunk ids so the model can cite a real ``src_chunk_id``
    for every claim (the grounding gate later verifies the span against that
    chunk's text), and any ``known_entities`` so a gleaning pass can extend recall
    without re-proposing duplicates.
    """
    chunk_ids = list(getattr(section, "chunk_ids", []) or [])
    payload = {
        "section_id": getattr(section, "section_id", ""),
        "chunk_ids": chunk_ids,
        "text": getattr(section, "text", "") or "",
    }
    if known_entities:
        payload["known_entities"] = list(known_entities)
    return json.dumps(payload, ensure_ascii=False)


class KgIngestionLlm:
    """Sync Phase-2 LLM seam: section extraction + community-report synthesis."""

    def extract(
        self,
        system: str,
        *,
        section: Any,
        model_id: str,
        container_id: str,
        known_entities: list[str] | None = None,
    ) -> dict:
        """Run ONE grounded section extraction → ``{entities, relations, tags}``."""
        return chat_json(
            system=system,
            user=_serialize_section(section, known_entities),
            model_id=model_id,
            container_id=container_id,
            phase="extraction",
        )

    def synthesize(self, prompt: str, *, model_id: str, container_id: str) -> dict:
        """Summarize a community's grounded relations → ``{"summary": "..."}``.

        The CONSUMER (``CommunityReporter._build_prompt``) already embeds its
        system instructions + the verbatim evidence in ``prompt``; the adapter
        only enforces JSON-mode output.
        """
        return chat_json(
            system="Return STRICT JSON only. Respond with a single object "
            'of the form {"summary": "..."} and nothing else.',
            user=prompt,
            model_id=model_id,
            container_id=container_id,
            phase="synthesis",
        )

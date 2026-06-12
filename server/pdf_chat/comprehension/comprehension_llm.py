"""Production async LLM adapter for Phase-5 comprehension finalization.

``finalize_comprehension`` (ontology build + glossary mining) awaits an ``llm``
seam exposing four methods that previously only existed as test fakes:

  * ``synthesize(prompt, *, model_id, container_id) -> dict`` — doc-taxonomy
    clustering; the consumer (``ontology_builder``) builds the prompt and reads
    ``{"classes": [...]}``.
  * ``confirm_definition(*, term, expansion, span, model_id, container_id) -> dict``
    — confirm an explicitly-stated acronym/definition; reads
    ``{"confirmed", "confidence", "expansion", "definition"}``.
  * ``synthesize_definition(*, term, contexts, model_id, container_id) -> dict``
    — infer a definition from usage contexts; reads ``{"definition", "confidence"}``.
  * ``adjudicate_variants(*, term, candidates, model_id, container_id) -> dict``
    — decide whether competing expansions corefer; reads ``{"same"}``.

The methods are ASYNC (the comprehension orchestrator awaits them). The blocking
Azure call is run off the event loop via ``asyncio.to_thread`` so finalization can
overlap I/O. The model id is chosen by the CONSUMER via ``select_model`` and
handed in — the adapter never routes (contract C7). For the three definition
methods the consumer passes structured kwargs (not a prompt), so the adapter owns
a small, schema-pinned instruction here; ``synthesize`` forwards the consumer's
prompt unchanged in strict-JSON mode.
"""
from __future__ import annotations

import asyncio
import json

from pdf_chat.ingestion.ingest_llm import chat_json

_PHASE = "comprehension"


class ComprehensionLlm:
    """Async Phase-5 LLM seam over the bulk Azure deployment, strict-JSON mode."""

    async def synthesize(self, prompt: str, *, model_id: str, container_id: str) -> dict:
        return await asyncio.to_thread(
            chat_json,
            system="Return STRICT JSON only — a single object — and nothing else.",
            user=prompt,
            model_id=model_id,
            container_id=container_id,
            phase=_PHASE,
        )

    async def confirm_definition(
        self, *, term: str, expansion: str, span: str, model_id: str, container_id: str
    ) -> dict:
        system = (
            "You verify candidate term definitions found verbatim in a document. "
            "Decide whether the EXPANSION is genuinely the definition/expansion of "
            "the TERM as used in the supporting SPAN. Use ONLY the span as evidence "
            "(never outside knowledge). Return STRICT JSON: "
            '{"confirmed": <bool>, "confidence": <number 0..1>, '
            '"expansion": "<corrected expansion or the original>", '
            '"definition": "<one-sentence definition grounded in the span>"}.'
        )
        user = json.dumps(
            {"term": term, "expansion": expansion, "span": span}, ensure_ascii=False
        )
        return await asyncio.to_thread(
            chat_json,
            system=system,
            user=user,
            model_id=model_id,
            container_id=container_id,
            phase=_PHASE,
        )

    async def synthesize_definition(
        self, *, term: str, contexts: list[str], model_id: str, container_id: str
    ) -> dict:
        system = (
            "You infer a concise definition for a TERM strictly from the way it is "
            "used in the supplied CONTEXTS. Do NOT use outside knowledge; if the "
            "contexts do not support a definition, return a low confidence. Return "
            'STRICT JSON: {"definition": "<one sentence, grounded in the contexts>", '
            '"confidence": <number 0..1>}.'
        )
        user = json.dumps({"term": term, "contexts": list(contexts)}, ensure_ascii=False)
        return await asyncio.to_thread(
            chat_json,
            system=system,
            user=user,
            model_id=model_id,
            container_id=container_id,
            phase=_PHASE,
        )

    async def adjudicate_variants(
        self, *, term: str, candidates: list[str], model_id: str, container_id: str
    ) -> dict:
        system = (
            "You decide whether competing expansions of the SAME term are alias "
            "spellings of ONE meaning (coreferent) or genuinely DIFFERENT meanings "
            "(a conflict). Return STRICT JSON: "
            '{"same": <bool>}  — true only when every candidate denotes the same '
            "concept."
        )
        user = json.dumps({"term": term, "candidates": list(candidates)}, ensure_ascii=False)
        return await asyncio.to_thread(
            chat_json,
            system=system,
            user=user,
            model_id=model_id,
            container_id=container_id,
            phase=_PHASE,
        )

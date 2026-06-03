"""Stage 8 — Context Assembly.

Pure, deterministic. Builds the numbered context block sent to the LLM from the
post-ACL, post-extraction pieces (spec §6 Stage 8):

    [1] {text_chunk_1}      Source: doc_id, page N
    [2] {text_chunk_2}      Source: doc_id, page N
    [TABLE-1] {table_as_markdown}
    [IMAGE-1] {image_description_from_vision_model}
    [GRAPH-1] {entity_relationship_summary}

Text chunks get numeric citation tags ([1], [2], ...) with a ``Source: doc_id,
page N`` trailer the LLM cites inline. Tables, images, and graph nodes get their
own prefixed tags. No infra; identical output for identical input.
"""
from __future__ import annotations

from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict OR a dataclass-like object, uniformly."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _source_tag(chunk: Any) -> str:
    """Build the ``Source: doc_id, page N`` trailer for a text chunk."""
    doc_id = _get(chunk, "doc_id")
    page_num = _get(chunk, "page_num")
    parts = []
    if doc_id is not None:
        parts.append(str(doc_id))
    if page_num is not None:
        parts.append(f"page {page_num}")
    if not parts:
        return ""
    return "Source: " + ", ".join(parts)


def assemble_context(
    text_chunks: list[Any],
    table_results: list[Any] | None = None,
    image_descriptions: list[Any] | None = None,
    graph_nodes: list[Any] | None = None,
) -> str:
    """Assemble the numbered, citation-tagged LLM context block.

    Args:
        text_chunks: ordered accessible text chunks (dict or dataclass with
            ``text`` / ``doc_id`` / ``page_num``). Numbered [1], [2], ...
        table_results: optional table renderings. Each may be a markdown string
            or an object/dict exposing ``text``/``markdown`` (+ optional source).
            Tagged [TABLE-1], [TABLE-2], ...
        image_descriptions: optional vision-model captions (strings or objects).
            Tagged [IMAGE-1], ...
        graph_nodes: optional entity-relationship summaries (strings or objects).
            Tagged [GRAPH-1], ...

    Returns:
        A single deterministic string. Sections are separated by blank lines;
        empty sections are omitted entirely.
    """
    lines: list[str] = []

    for i, chunk in enumerate(text_chunks or [], start=1):
        text = _get(chunk, "text", "") or ""
        src = _source_tag(chunk)
        line = f"[{i}] {text}"
        if src:
            line += f"      {src}"
        lines.append(line)

    def _render(items: list[Any], prefix: str) -> None:
        for i, item in enumerate(items, start=1):
            if isinstance(item, str):
                body = item
                src = ""
            else:
                body = (
                    _get(item, "markdown", None)
                    or _get(item, "description", None)
                    or _get(item, "summary", None)
                    or _get(item, "text", "")
                    or ""
                )
                src = _source_tag(item)
            line = f"[{prefix}-{i}] {body}"
            if src:
                line += f"      {src}"
            lines.append(line)

    if table_results:
        _render(table_results, "TABLE")
    if image_descriptions:
        _render(image_descriptions, "IMAGE")
    if graph_nodes:
        _render(graph_nodes, "GRAPH")

    return "\n".join(lines)

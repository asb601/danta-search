"""Prompt templates for the PDF RAG synthesis step (Spec §6 Stage 9).

Pure strings — no infra. The system prompt enforces strict grounding: the model
must answer ONLY from the assembled context and cite sources as [N]. The user
template injects the numbered context block produced by ``assemble_context``.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. "
    "Answer ONLY from the provided context. "
    "Every factual statement must cite its source as [N], where N matches the "
    "numbered context entries. "
    "If the context does not contain enough information to answer, say so "
    "explicitly and do NOT use outside knowledge or guess. Never fabricate "
    "citations or facts."
)

# Shown to the user (and returned as the answer) when ACL filtering leaves no
# accessible chunks. Deterministic so the API never hallucinates an answer.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I don't have sufficient accessible context to answer this question. "
    "You may not have permission to view the relevant documents, or no indexed "
    "content matched your query."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer using only the context above and cite sources as [N]."
)


def build_user_prompt(query: str, context: str) -> str:
    """Render the user message from the assembled context block + question."""
    return USER_TEMPLATE.format(context=context, query=query)

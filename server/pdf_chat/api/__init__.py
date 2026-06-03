"""Public API surface: the FastAPI router for the PDF RAG system."""
from __future__ import annotations

from pdf_chat.api.routes import pdf_router

__all__ = ["pdf_router"]

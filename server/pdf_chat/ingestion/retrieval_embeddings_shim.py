"""Re-export the retrieval batch embedder for the ingestion worker.

The worker's per-page pipeline needs the config-sized ``embed_texts_batched`` token
guard that lives in ``pdf_chat.retrieval.embeddings``. Importing it through this
thin shim keeps the ``tasks.py`` import edge pointing at the ingestion package and
avoids a retrieval→ingestion import cycle at module scope.
"""
from __future__ import annotations

from pdf_chat.retrieval.embeddings import embed_texts_batched  # noqa: F401

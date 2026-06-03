"""Typed Object Notation (TON) — the unified element schema.

Every parser (PyMuPDF / Docling / Unstructured / VLM) normalizes its output into
`UnifiedElement` so that Stages 11–14 (chunk → embed → store → reconcile) operate
identically regardless of which parser ran. Pure dataclasses — no infra imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class UnifiedElement:
    """Canonical element emitted by every parser path.

    `acl` and `tenant_id` are inherited from upload_manifest.acl_snapshot so each
    downstream chunk carries the access policy of its source document.
    """
    element_id: str
    doc_id: str
    page_num: int
    element_type: ElementType
    content: str                      # raw text OR base64 image OR table markdown
    reading_order: int
    tenant_id: str
    bbox: BBox | None = None
    confidence: float = 1.0
    parser_version: str = ""
    acl: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.element_type, ElementType):
            self.element_type = ElementType(self.element_type)
        if self.bbox is not None and not isinstance(self.bbox, BBox):
            self.bbox = BBox(**self.bbox)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["element_type"] = self.element_type.value
        d["bbox"] = self.bbox.as_dict() if self.bbox else None
        return d


@dataclass
class Chunk:
    """A retrievable unit produced from one or more UnifiedElements."""
    chunk_id: str
    doc_id: str
    page_num: int
    element_type: ElementType
    text: str
    reading_order: int
    tenant_id: str
    acl: dict = field(default_factory=dict)
    embedding: list[float] | None = None
    source_element_id: str | None = None

    def to_neo4j_props(self) -> dict:
        """Flatten for a Cypher CREATE (acl serialized to JSON string)."""
        import json
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "element_type": self.element_type.value,
            "text": self.text,
            "reading_order": self.reading_order,
            "tenant_id": self.tenant_id,
            "acl": json.dumps(self.acl or {}),
            "embedding": self.embedding or [],
        }

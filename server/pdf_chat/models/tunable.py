"""ORM model for per-container tunable overrides (Spec §3 invariant 4).

Optional: tunables resolve from env + named defaults without this table; the
table only lets an operator override a single key for one container without a
deploy. Tenant-isolated via ``container_id``.
"""
from __future__ import annotations

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base  # reuse the app's declarative Base


class PdfGraphRagTunable(Base):
    __tablename__ = "pdf_graphrag_tunables"
    __table_args__ = (UniqueConstraint("container_id", "key", name="uq_pdf_tunable"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    container_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

"""Compile + persist + load the Danta Semantic Contract (DSC).

The contract is a PROJECTION — it invents nothing. It assembles, per container:
  models[]        from FileMetadata (+ ErpClassification business layer)
  relationships[] from SemanticRelationship where approval_status='approved'
                  AND status='active' (DECLARED joins only — never coincidental)
  metrics[]       from SemanticEntity.metrics (calculated fields)
  instructions[]  operational guidance derived from the contract itself
  source_systems  the distinct, reliable systems present
  process_chains  derived from process_role assignments (data-driven)

A content hash over the inputs drives invalidation: recompute only when inputs
change. Everything is defensive — a missing table or column yields a smaller
contract, never an exception that breaks the query path.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import pipeline_logger

CONTRACT_SCHEMA_VERSION = 1


# ── Compile ───────────────────────────────────────────────────────────────────

async def build_contract(db: AsyncSession, container_id: str) -> dict[str, Any]:
    """Assemble the contract dict for a container from existing artifacts.

    Never raises: each section is independently guarded so a partial failure
    degrades the contract rather than the query.
    """
    models = await _build_models(db, container_id)
    relationships = await _build_relationships(db, container_id, {m["file_id"] for m in models})
    metrics = await _build_metrics(db, container_id)

    source_systems = sorted({
        m["source_system"] for m in models
        if m.get("source_system") and m["source_system"] != "Unknown"
    })
    process_chains = _derive_process_chains(models)
    instructions = _derive_instructions(models, relationships, source_systems)

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "container_id": container_id,
        "source_systems": source_systems,
        "models": models,
        "relationships": relationships,
        "metrics": metrics,
        "process_chains": process_chains,
        "instructions": instructions,
    }


async def _build_models(db: AsyncSession, container_id: str) -> list[dict[str, Any]]:
    from app.models.file_metadata import FileMetadata

    try:
        rows = (await db.execute(
            select(FileMetadata).where(FileMetadata.container_id == container_id)
        )).scalars().all()
    except Exception as exc:
        pipeline_logger.warning("contract_models_query_error", error=str(exc)[:160])
        return []

    # Pull ERP classifications in one query, keyed by file_id.
    classifications = await _load_classifications(db, container_id)

    models: list[dict[str, Any]] = []
    for md in rows:
        clf = classifications.get(md.file_id, {})
        columns = _exposed_columns(md)
        if not columns:
            continue
        models.append({
            "file_id": md.file_id,
            "blob_path": md.blob_path,
            "name": _model_name(md),
            "description": (md.ai_description or "")[:600],
            "source_system": clf.get("source_system", "Unknown"),
            "erp_module": clf.get("erp_module", "Unknown"),
            "domain_polarity": clf.get("domain_polarity", "neutral"),
            "process_role": clf.get("process_role", "unknown"),
            "grain": clf.get("grain") or "",
            "classification_reliable": bool(clf.get("reliable", False)),
            "date_range_start": _iso(md.date_range_start),
            "date_range_end": _iso(md.date_range_end),
            "columns": columns,
            "key_metrics": list(md.key_metrics or []),
            "key_dimensions": list(md.key_dimensions or []),
            "column_semantic_roles": dict(md.column_semantic_roles or {}),
        })
    return models


async def _load_classifications(db: AsyncSession, container_id: str) -> dict[str, dict]:
    """Load ERP classifications keyed by file_id. Empty dict if table absent."""
    try:
        from app.models.erp_classification import ErpClassification as ErpRow

        rows = (await db.execute(
            select(ErpRow).where(ErpRow.container_id == container_id)
        )).scalars().all()
    except Exception:
        # Table may not exist yet (migration not run) — degrade silently.
        return {}

    try:
        from app.services.erp.classifier import _confidence_floor  # noqa: PLC0415
        floor = _confidence_floor()
    except Exception:
        floor = 0.55

    out: dict[str, dict] = {}
    for r in rows:
        reliable = (r.source == "human_override") or (
            r.source == "llm" and (r.confidence or 0.0) >= floor and r.source_system != "Unknown"
        )
        out[r.file_id] = {
            "source_system": r.source_system,
            "erp_module": r.erp_module,
            "domain_polarity": r.domain_polarity,
            "process_role": r.process_role,
            "grain": r.grain,
            "reliable": reliable,
        }
    return out


def _exposed_columns(md) -> list[dict[str, Any]]:
    """Build the exposed-column list with value semantics where known.

    Exposure = every column we have metadata for. (Selective hiding for access
    control is a future enrich step; today all real columns are exposed.)
    """
    roles = md.column_semantic_roles or {}
    cols: list[dict[str, Any]] = []
    info = md.columns_info or []
    if isinstance(info, list) and info:
        for col in info:
            if isinstance(col, dict):
                name = str(col.get("name") or col.get("column") or "").strip()
                if not name:
                    continue
                samples = col.get("sample_values") or col.get("samples") or []
                value_semantics = ", ".join(str(s) for s in list(samples)[:8]) if samples else ""
                cols.append({
                    "name": name,
                    "type": str(col.get("type") or col.get("dtype") or ""),
                    "role": roles.get(name, ""),
                    "value_semantics": value_semantics,
                })
            elif isinstance(col, str):
                cols.append({"name": col, "type": "", "role": roles.get(col, ""), "value_semantics": ""})
    return cols


async def _build_relationships(
    db: AsyncSession, container_id: str, shortlist_ids: set[str]
) -> list[dict[str, Any]]:
    """DECLARED joins only: approved + active. This is the cure for coincidental
    joins — a key-overlap candidate that was never approved never appears."""
    try:
        from app.models.semantic_layer import SemanticRelationship

        rows = (await db.execute(
            select(SemanticRelationship).where(
                SemanticRelationship.container_id == container_id,
                SemanticRelationship.status == "active",
                SemanticRelationship.approval_status == "approved",
            )
        )).scalars().all()
    except Exception as exc:
        pipeline_logger.warning("contract_rel_query_error", error=str(exc)[:160])
        return []

    rels: list[dict[str, Any]] = []
    for r in rows:
        rels.append({
            "from_file_id": r.file_a_id,
            "to_file_id": r.file_b_id,
            "from_entity": r.from_entity,
            "to_entity": r.to_entity,
            "from_column": r.from_column,
            "to_column": r.to_column,
            "relationship_type": r.relationship_type,
            "confidence": r.confidence_score,
        })
    return rels


async def _build_metrics(db: AsyncSession, container_id: str) -> list[dict[str, Any]]:
    try:
        from app.models.semantic_layer import SemanticEntity

        rows = (await db.execute(
            select(SemanticEntity).where(
                SemanticEntity.container_id == container_id,
                SemanticEntity.status == "active",
            )
        )).scalars().all()
    except Exception:
        return []

    metrics: list[dict[str, Any]] = []
    for ent in rows:
        for metric in (ent.metrics or []):
            if isinstance(metric, dict):
                metrics.append({
                    "entity": ent.entity_name,
                    "file_id": ent.file_id,
                    **{k: v for k, v in metric.items()},
                })
    return metrics


def _derive_process_chains(models: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group reliable process_role assignments by polarity into ordered chains.

    Data-driven: the chain is whatever process roles actually exist in this
    container, ordered by a canonical sequence hint when recognised, otherwise
    appended. No fixed chain is asserted — empty when nothing is classified.
    """
    # A soft ordering hint for common O2C/P2P stages. Roles not listed still
    # appear (appended at the end), so unknown/bespoke processes are preserved.
    order_hint = [
        "requisition", "purchase_order", "po_line", "goods_receipt", "vendor_invoice", "ap_invoice",
        "payment", "customer_master", "vendor_master", "material_master",
        "sales_order", "sales_order_line", "delivery", "goods_issue", "billing",
        "ar_invoice", "cash_receipt",
    ]
    hint_index = {role: i for i, role in enumerate(order_hint)}

    by_polarity: dict[str, set[str]] = {}
    for m in models:
        if not m.get("classification_reliable"):
            continue
        role = m.get("process_role") or "unknown"
        if role in ("unknown", ""):
            continue
        pol = m.get("domain_polarity") or "neutral"
        by_polarity.setdefault(pol, set()).add(role)

    chains: dict[str, list[str]] = {}
    for pol, roles in by_polarity.items():
        chains[pol] = sorted(roles, key=lambda r: hint_index.get(r, 10_000))
    return chains


def _derive_instructions(
    models: list[dict[str, Any]], relationships: list[dict[str, Any]], source_systems: list[str]
) -> list[str]:
    out: list[str] = []
    if len(source_systems) > 1:
        out.append(
            f"This container mixes multiple systems of record ({', '.join(source_systems)}). "
            f"Never join across systems unless a relationship is explicitly declared below."
        )
    out.append(
        "Use ONLY the declared relationships for joins. A join not listed here is "
        "not approved — do not infer one from matching column names or value ranges."
    )
    # Surface polarity guidance when both sides are present.
    polarities = {m.get("domain_polarity") for m in models}
    if "customer" in polarities and "vendor" in polarities:
        out.append(
            "Customer-side (sales/AR) and vendor-side (purchasing/AP) data are both "
            "present. For a customer/sales question use customer-polarity models for "
            "payment status; for a vendor/purchasing question use vendor-polarity models."
        )
    return out


# ── Persist + load + invalidate ────────────────────────────────────────────────

def _hash_inputs(definition: dict[str, Any]) -> str:
    """Stable content hash over the parts that affect query behaviour."""
    salient = {
        "models": [
            {
                "file_id": m["file_id"],
                "source_system": m["source_system"],
                "domain_polarity": m["domain_polarity"],
                "process_role": m["process_role"],
                "grain": m.get("grain", ""),
                # Include type + role so a re-ingest that changes a column's type
                # or reassigns a semantic role actually rebuilds the contract.
                "columns": [
                    {"name": c.get("name"), "type": c.get("type"), "role": c.get("role")}
                    for c in m["columns"]
                ],
                "date_range": [m.get("date_range_start"), m.get("date_range_end")],
            }
            for m in definition.get("models", [])
        ],
        "relationships": definition.get("relationships", []),
        "metrics": definition.get("metrics", []),
        "instructions": definition.get("instructions", []),
    }
    blob = json.dumps(salient, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


async def compile_and_store_contract(db: AsyncSession, container_id: str) -> dict[str, Any]:
    """Build the contract and upsert it, bumping version when content changed."""
    definition = await build_contract(db, container_id)
    content_hash = _hash_inputs(definition)

    from app.models.semantic_contract import SemanticContract

    existing = (await db.execute(
        select(SemanticContract).where(SemanticContract.container_id == container_id)
    )).scalar_one_or_none()

    if existing and existing.content_hash == content_hash:
        return existing.definition or definition  # unchanged — no write

    if existing:
        existing.definition = definition
        existing.content_hash = content_hash
        existing.version = (existing.version or 1) + 1
        existing.computed_at = datetime.now(timezone.utc)
    else:
        db.add(SemanticContract(
            container_id=container_id,
            definition=definition,
            content_hash=content_hash,
            version=1,
        ))
    try:
        await db.commit()
    except Exception as exc:
        # Parallel re-ingest fans out many files; two may try to INSERT the same
        # container's contract row at once → unique-constraint violation. That's
        # benign (another worker wrote an equivalent contract). Roll back and
        # return the freshly-built definition rather than failing the stage.
        await db.rollback()
        pipeline_logger.info("contract_compile_conflict", container_id=container_id, error=str(exc)[:160])
        return definition
    pipeline_logger.info(
        "contract_compiled",
        container_id=container_id,
        models=len(definition["models"]),
        relationships=len(definition["relationships"]),
        source_systems=definition["source_systems"],
    )
    return definition


async def load_contract(db: AsyncSession, container_id: str) -> dict[str, Any] | None:
    """Load the stored contract for a container. None if not yet compiled."""
    try:
        from app.models.semantic_contract import SemanticContract

        row = (await db.execute(
            select(SemanticContract).where(
                SemanticContract.container_id == container_id,
                SemanticContract.status == "active",
            )
        )).scalar_one_or_none()
        return row.definition if row else None
    except Exception as exc:
        pipeline_logger.warning("contract_load_error", error=str(exc)[:160])
        return None


# ── Small helpers ──────────────────────────────────────────────────────────────

def _iso(value) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _model_name(md) -> str:
    """Logical name = filename stem without the upload hash prefix, uppercased.

    Mirrors the file_identity logical-name convention so contract model names
    line up with the names the LLM sees in the prompt and writes in SQL.
    """
    import re

    blob = (md.blob_path or md.file_id or "").rsplit("/", 1)[-1]
    stem = re.sub(r"^[0-9a-f]{8}_", "", blob, flags=re.IGNORECASE)
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return stem.upper()

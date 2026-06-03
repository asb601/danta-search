"""Document deduplication decision — pure logic, no I/O.

Implements the SHA-256 fingerprint decision matrix from enterprise-pdf spec 4.2.
The caller is responsible for the actual lookup (``UploadManifestRepo.find_by_sha256``
scoped to the tenant) and then asks this function what to do with the result.

Decision matrix (spec 4.2):

    | Scenario                                   | Action    |
    |--------------------------------------------|-----------|
    | Same SHA-256, same tenant, status=indexed  | SKIP      |
    | Same SHA-256, same tenant, status=failed   | REPROCESS |
    | Same SHA-256, different tenant             | NEW       |
    | No existing row                            | NEW       |

This module is intentionally dependency-free so it runs in unit tests with zero
infra and can be reasoned about in isolation.
"""
from __future__ import annotations

from pdf_chat.models.enums import DedupDecision, DocStatus

__all__ = ["decide_dedup"]


def decide_dedup(existing_status: str | None, same_tenant: bool) -> DedupDecision:
    """Decide what to do with a newly uploaded file given any prior record.

    Parameters
    ----------
    existing_status:
        ``status`` of the most relevant existing ``UploadManifest`` row that
        shares this file's SHA-256, or ``None`` if no prior row exists. This is
        a raw ``DocStatus`` value string (e.g. ``"indexed"``, ``"failed"``).
    same_tenant:
        Whether the existing row belongs to the *same* tenant as the new upload.
        When the caller found no existing row this is irrelevant and should be
        passed as ``False``.

    Returns
    -------
    DedupDecision
        ``SKIP`` / ``REPROCESS`` / ``NEW``. ``NEW_VERSION`` is not produced here:
        a different-bytes upload of the same logical doc has a *different* SHA-256,
        so it never reaches this function via the hash path — versioning is decided
        upstream by the (name, tenant) lookup, not the fingerprint check.
    """
    # Branch 1: no prior row with this fingerprint anywhere -> brand new document.
    if existing_status is None:
        return DedupDecision.NEW

    # Branch 2: the matching fingerprint belongs to a *different* tenant. Tenants
    # are fully isolated (separate ACLs, separate Neo4j scope), so an identical
    # byte stream owned by tenant B must be ingested fresh for tenant A.
    if not same_tenant:
        return DedupDecision.NEW

    # --- From here: same SHA-256 AND same tenant. We already have this file. ---

    # Branch 3: a previous ingestion attempt of this exact file failed outright.
    # Nothing usable was indexed, so re-run the pipeline rather than skip.
    if existing_status == DocStatus.FAILED.value:
        return DedupDecision.REPROCESS

    # Branch 4: same file, same tenant, and it is already fully indexed. Pure
    # duplicate upload — skip processing and return the existing upload_id.
    if existing_status == DocStatus.INDEXED.value:
        return DedupDecision.SKIP

    # Branch 5: same file, same tenant, but the prior attempt is still in flight
    # (uploaded / splitting / processing) or only partially indexed. Treat as a
    # duplicate-in-progress: skip rather than spawn a competing pipeline that
    # would race on the same pages. The in-flight job (or a reconciler) will
    # settle the existing row. Re-driving a partial doc is a deliberate operator
    # action, not an automatic upload-time decision.
    return DedupDecision.SKIP

"""Stage 2 — SHA-256 fingerprint.

Pure stdlib. Computes the content hash used by the control plane to deduplicate
re-uploads (`decide_dedup(existing_status, same_tenant)`). No infra imports.
"""
from __future__ import annotations

import hashlib


def compute_sha256(file_bytes: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``file_bytes``.

    Deterministic and dependency-free so it runs identically in the upload API,
    a Celery worker, or a unit test. The empty-bytes digest is the well-known
    ``e3b0c442...`` vector.
    """
    if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"file_bytes must be bytes-like, got {type(file_bytes)!r}")
    return hashlib.sha256(bytes(file_bytes)).hexdigest()

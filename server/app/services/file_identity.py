"""Request-local file identity model.

The LLM works with logical table names.  Runtime code owns every physical
storage detail: file IDs, blob names, parquet paths, and Azure URIs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)
_NON_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
_DUP_UNDERSCORE_RE = re.compile(r"_+")


def _basename(path: str) -> str:
    value = (path or "").strip().strip("'\"")
    if value.startswith("az://"):
        parts = value.split("/", 3)
        value = parts[3] if len(parts) > 3 else ""
    return PurePosixPath(value).name


def display_name_from_path(path: str) -> str:
    """Return a user-facing filename with the upload hash removed."""
    return _HASH_PREFIX_RE.sub("", _basename(path))


def logical_name_from_path(path: str) -> str:
    """Return a stable SQL identifier from a blob/display path."""
    name = display_name_from_path(path)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    name = _NON_IDENTIFIER_RE.sub("_", name.strip())
    name = _DUP_UNDERSCORE_RE.sub("_", name).strip("_")
    if not name:
        name = "TABLE"
    if not (name[0].isalpha() or name[0] == "_"):
        name = f"T_{name}"
    return name.upper()


def normalise_identity_key(value: str) -> str:
    """Normalise a user/model reference for identity lookup."""
    key = (value or "").strip().strip("`\"'[]")
    if key.startswith("az://"):
        key = _basename(key)
    if "." in key:
        key = key.rsplit(".", 1)[0]
    key = _HASH_PREFIX_RE.sub("", key)
    key = _NON_IDENTIFIER_RE.sub("_", key)
    key = _DUP_UNDERSCORE_RE.sub("_", key).strip("_")
    return key.lower()


@dataclass(frozen=True)
class FileIdentity:
    canonical_id: str
    logical_name: str
    sql_name: str
    blob_path: str
    container_name: str
    parquet_blob_path: str | None = None
    display_name: str = ""
    aliases: frozenset[str] = field(default_factory=frozenset)

    @property
    def source_uri(self) -> str:
        return f"az://{self.container_name}/{self.blob_path}"

    @property
    def parquet_uri(self) -> str | None:
        if not self.parquet_blob_path:
            return None
        return f"az://{self.container_name}/{self.parquet_blob_path}"

    @property
    def execution_uri(self) -> str:
        return self.parquet_uri or self.source_uri

    @property
    def execution_format(self) -> str:
        return "parquet" if self.execution_uri.lower().endswith(".parquet") else "csv"

    @property
    def authorization_key(self) -> str:
        return self.canonical_id

    def prompt_record(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "logical_name": self.logical_name,
            "sql_name": self.sql_name,
            "display_name": self.display_name,
            "blob_path": self.blob_path,
            "parquet_blob_path": self.parquet_blob_path,
        }


class FileIdentityMap:
    """Request-local resolver for logical names and canonical file IDs."""

    def __init__(self, identities: list[FileIdentity]) -> None:
        self.identities = identities
        self.by_id = {identity.canonical_id: identity for identity in identities}
        self.by_blob = {identity.blob_path: identity for identity in identities}
        self._table_keys: dict[str, list[FileIdentity]] = {}
        self._reference_keys: dict[str, list[FileIdentity]] = {}

        for identity in identities:
            self._add_table_key(identity.sql_name, identity)
            self._add_table_key(identity.logical_name, identity)
            for alias in identity.aliases:
                self._add_reference_key(alias, identity)

    def _add_table_key(self, key: str, identity: FileIdentity) -> None:
        normalised = normalise_identity_key(key)
        if normalised:
            self._table_keys.setdefault(normalised, []).append(identity)

    def _add_reference_key(self, key: str, identity: FileIdentity) -> None:
        normalised = normalise_identity_key(key)
        if normalised:
            self._reference_keys.setdefault(normalised, []).append(identity)

    def allowed_file_ids(self) -> set[str]:
        return set(self.by_id)

    def allowed_physical_uris(self) -> set[str]:
        uris: set[str] = set()
        for identity in self.identities:
            uris.add(identity.source_uri)
            if identity.parquet_uri:
                uris.add(identity.parquet_uri)
        return uris

    def identity_for_blob(self, blob_path: str) -> FileIdentity | None:
        return self.by_blob.get(blob_path)

    def resolve_table(self, table_name: str) -> FileIdentity:
        key = normalise_identity_key(table_name)
        matches = self._table_keys.get(key, [])
        if not matches:
            raise KeyError(f"Unknown logical table '{table_name}'.")
        unique = {m.canonical_id: m for m in matches}
        if len(unique) > 1:
            choices = ", ".join(sorted(i.sql_name for i in unique.values())[:6])
            raise ValueError(
                f"Logical table '{table_name}' is ambiguous. Use one of: {choices}."
            )
        return next(iter(unique.values()))

    def resolve_reference(self, file_ref: str) -> FileIdentity | None:
        if file_ref in self.by_id:
            return self.by_id[file_ref]
        if file_ref in self.by_blob:
            return self.by_blob[file_ref]
        key = normalise_identity_key(file_ref)
        matches = self._reference_keys.get(key) or self._table_keys.get(key) or []
        unique = {m.canonical_id: m for m in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
        return None

    def prompt_identities_for_catalog(self, catalog: list[dict]) -> list[FileIdentity]:
        ids = [e.get("file_id") for e in catalog if e.get("file_id")]
        return [self.by_id[file_id] for file_id in ids if file_id in self.by_id]

    def trace_records(self, limit: int = 40) -> list[dict]:
        return [identity.prompt_record() for identity in self.identities[:limit]]


def build_file_identity_map(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    container_name: str,
) -> FileIdentityMap:
    """Build the canonical identity map for every visible catalog file."""
    raw: list[tuple[dict, str]] = []
    logical_counts: dict[str, int] = {}
    for entry in catalog:
        file_id = str(entry.get("file_id") or "")
        blob_path = str(entry.get("blob_path") or "")
        if not file_id or not blob_path:
            continue
        logical_name = logical_name_from_path(blob_path)
        raw.append((entry, logical_name))
        logical_counts[logical_name] = logical_counts.get(logical_name, 0) + 1

    identities: list[FileIdentity] = []
    for entry, logical_name in raw:
        file_id = str(entry["file_id"])
        blob_path = str(entry["blob_path"])
        display_name = display_name_from_path(blob_path)
        parquet_blob_path = parquet_paths_all.get(blob_path)
        sql_name = logical_name
        if logical_counts.get(logical_name, 0) > 1:
            sql_name = f"F_{file_id.replace('-', '_')[:8].upper()}"

        aliases = {
            sql_name,
            logical_name,
            display_name,
            blob_path,
            _basename(blob_path),
            file_id,
        }
        if parquet_blob_path:
            aliases.update({
                parquet_blob_path,
                _basename(parquet_blob_path),
                f"az://{container_name}/{parquet_blob_path}",
            })
        aliases.add(f"az://{container_name}/{blob_path}")

        identities.append(FileIdentity(
            canonical_id=file_id,
            logical_name=logical_name,
            sql_name=sql_name,
            blob_path=blob_path,
            container_name=container_name,
            parquet_blob_path=parquet_blob_path,
            display_name=display_name,
            aliases=frozenset(alias for alias in aliases if alias),
        ))

    return FileIdentityMap(identities)
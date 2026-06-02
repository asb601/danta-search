"""Request-local file identity model.

The LLM works with logical table names.  Runtime code owns every physical
storage detail: file IDs, blob names, parquet paths, and Azure URIs.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from app.core.logger import chat_logger


_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)
_NON_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
_DUP_UNDERSCORE_RE = re.compile(r"_+")

# Partition-suffix detection for logical-table consolidation.
# A logical table family (e.g. PROC_PurchaseOrders) is spread across many SOURCE
# files that differ only by a date partition (_YYYY, _YYYY_MM, _YYYY_MM_DD) and/or
# a delimiter-format token (_tab, _pipe, _csv ...) in the UPLOADED filename — a
# common data-export convention (e.g. SLT/SAP extracts). Stripping a trailing
# date is generic partition discovery (cf. Hive). The format-token list is only a
# lexical HINT; it is NOT authoritative — consolidation is additionally gated on a
# column-schema fingerprint (see build_file_identity_map) so two files are merged
# only when their schemas actually match. That keeps grouping data-driven rather
# than fitted to one tenant's filename convention.
_PARTITION_DATE_RE = re.compile(r"_(?:19|20)\d{2}(?:_\d{2}){0,2}$")
_FORMAT_TOKEN_RE = re.compile(r"_(?:tab|pipe|csv|tsv|psv|delim)$", re.IGNORECASE)


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


def logical_table_key(path: str) -> str:
    """Return the consolidated logical-table identifier for a partition file.

    All monthly/format partitions of one table family collapse to a single key:
        PROC_PurchaseOrders_2023_06_tab.txt -> "PROC_PURCHASEORDERS"
        PROC_PurchaseOrders_2024_04.csv     -> "PROC_PURCHASEORDERS"

    This is what makes multi-period questions answerable: the runtime scans the
    whole logical table, never a single retrieved month.

    Edge-case safety: this key is purely lexical, so a table whose business name
    legitimately ends in a 4-digit year (or a `tab`/`pipe` word) could in theory
    be mis-grouped. That is bounded downstream — build_file_identity_map gates
    every merge on a COLUMN-SCHEMA fingerprint, so files that share this key but
    differ in schema are split apart (logged as logical_table_schema_split) and
    never unioned. Grouping is therefore lexical-proposes / schema-disposes.
    """
    name = display_name_from_path(path)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    # Strip trailing partition/format tokens in any order until none remain.
    changed = True
    while changed:
        changed = False
        m = _FORMAT_TOKEN_RE.search(name)
        if m:
            name = name[: m.start()]
            changed = True
        m = _PARTITION_DATE_RE.search(name)
        if m:
            name = name[: m.start()]
            changed = True
    name = _NON_IDENTIFIER_RE.sub("_", name.strip())
    name = _DUP_UNDERSCORE_RE.sub("_", name).strip("_")
    if not name:
        name = "TABLE"
    if not (name[0].isalpha() or name[0] == "_"):
        name = f"T_{name}"
    return name.upper()


def partition_period(path: str) -> str:
    """Return the trailing date-partition token of a file, or '' if none.

    Used to dedupe the same period stored in multiple formats (e.g. the same
    month present as both .xlsx and _pipe.txt) so a logical-table scan does not
    double-count that month.
        PROC_PurchaseOrders_2023_06_tab.txt -> "2023_06"
        PROC_PurchaseOrders_2024_04.csv     -> "2024_04"
    """
    name = display_name_from_path(path)
    if "." in name:
        name = name.rsplit(".", 1)[0]
    # Peel a trailing format token first so the date sits at the end.
    m = _FORMAT_TOKEN_RE.search(name)
    if m:
        name = name[: m.start()]
    m = _PARTITION_DATE_RE.search(name)
    return m.group(0).lstrip("_") if m else ""


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
    # Logical-table consolidation: every physical partition (month/format) that
    # belongs to this logical table. member_file_ids drives authorization breadth;
    # partition_uris drives the multi-partition scan. For a single-file logical
    # table these contain exactly one entry and behaviour is unchanged.
    logical_table_id: str = ""
    member_file_ids: tuple[str, ...] = field(default_factory=tuple)
    partition_uris: tuple[str, ...] = field(default_factory=tuple)
    # Aggregate coverage across all partitions (min start … max end) so the
    # model sees the table's TRUE period span, not one month's range. ISO strings
    # or None. partition_count is the deduped partition total.
    coverage_start: str | None = None
    coverage_end: str | None = None
    partition_count: int = 1

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
    def execution_uris(self) -> tuple[str, ...]:
        """Every physical partition URI to scan for this logical table."""
        return self.partition_uris or (self.execution_uri,)

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
        # Map EVERY partition file id to its consolidated identity so that
        # authorization and reference lookups by any member id resolve correctly.
        self.by_id = {}
        for identity in identities:
            for member_id in (identity.member_file_ids or (identity.canonical_id,)):
                self.by_id[member_id] = identity
        self.by_blob = {identity.blob_path: identity for identity in identities}
        self._table_keys: dict[str, list[FileIdentity]] = {}
        self._reference_keys: dict[str, list[FileIdentity]] = {}

        for identity in identities:
            self._add_table_key(identity.sql_name, identity)
            self._add_table_key(identity.logical_name, identity)
            if identity.logical_table_id:
                self._add_table_key(identity.logical_table_id, identity)
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
            uris.update(identity.partition_uris)
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
    """Build the canonical identity map, consolidating partitions into logical tables.

    Every physical file that shares a logical-table key (table family minus its
    date/format partition suffix) is folded into ONE FileIdentity. A query against
    that logical table scans every partition; authorization spans every member.
    """
    # 1. Bucket catalog entries by their lexical logical-table key.
    lexical_groups: dict[str, list[dict]] = {}
    for entry in catalog:
        file_id = str(entry.get("file_id") or "")
        blob_path = str(entry.get("blob_path") or "")
        if not file_id or not blob_path:
            continue
        lexical_groups.setdefault(logical_table_key(blob_path), []).append(entry)

    # 2. Schema-gate each lexical group: only merge partitions that share the same
    #    column fingerprint. The majority schema keeps the clean logical name; a
    #    divergent schema is split off (suffixed) and logged. This prevents a
    #    coincidental name match (or an over-eager suffix strip) from unioning
    #    schema-incompatible files. Entries lacking column info are placed with the
    #    majority (unknown schema is not treated as a conflict).
    def _fingerprint(entry: dict) -> frozenset[str]:
        cols: list[str] = []
        for col in entry.get("columns_info") or []:
            if isinstance(col, dict) and col.get("name"):
                cols.append(str(col["name"]).strip().lower())
            elif isinstance(col, str):
                cols.append(col.strip().lower())
        if not cols:
            cols = [str(c).strip().lower() for c in (entry.get("column_names") or []) if isinstance(c, str)]
        return frozenset(cols)

    groups: dict[str, list[dict]] = {}
    for logical_key, members in lexical_groups.items():
        by_fp: dict[frozenset[str], list[dict]] = {}
        unknown: list[dict] = []
        for e in members:
            fp = _fingerprint(e)
            (unknown if not fp else by_fp.setdefault(fp, [])).append(e)
        if not by_fp:
            groups[logical_key] = members  # no schema info at all — keep lexical group
            continue
        # Majority schema (most partitions) owns the clean logical name.
        majority_fp = max(by_fp, key=lambda fp: len(by_fp[fp]))
        by_fp[majority_fp].extend(unknown)
        for fp, fp_members in by_fp.items():
            if fp is majority_fp:
                groups[logical_key] = fp_members
            else:
                suffix = hashlib.md5("|".join(sorted(fp)).encode()).hexdigest()[:6].upper()
                split_key = f"{logical_key}_S{suffix}"
                groups[split_key] = fp_members
                chat_logger.warning(
                    "logical_table_schema_split",
                    logical_table=logical_key,
                    split_into=split_key,
                    partition_count=len(fp_members),
                    reason="column schema differs from majority partitions",
                )

    def _exec_uri(blob_path: str) -> str:
        parquet_blob_path = parquet_paths_all.get(blob_path)
        if parquet_blob_path:
            return f"az://{container_name}/{parquet_blob_path}"
        return f"az://{container_name}/{blob_path}"

    def _iso(value) -> str | None:
        if value is None or value == "":
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    identities: list[FileIdentity] = []
    for logical_key, members in groups.items():
        # Stable order so the representative partition is deterministic.
        members = sorted(members, key=lambda e: str(e.get("blob_path") or ""))

        # Dedupe partitions that are the SAME period in different formats (e.g.
        # 2023_06 as both .xlsx and _pipe.txt). Unioning both would double-count
        # that month. Keep one per period — preferring a parquet-backed file.
        deduped: list[dict] = []
        seen_periods: dict[str, dict] = {}
        dropped_dupes = 0
        for e in members:
            bp = str(e.get("blob_path") or "")
            period = partition_period(bp)
            if not period:
                deduped.append(e)  # no detectable period — cannot be a format twin
                continue
            if period not in seen_periods:
                seen_periods[period] = e
                deduped.append(e)
            else:
                # Prefer the parquet-backed partition for the period.
                kept = seen_periods[period]
                if not parquet_paths_all.get(str(kept.get("blob_path") or "")) and parquet_paths_all.get(bp):
                    deduped[deduped.index(kept)] = e
                    seen_periods[period] = e
                dropped_dupes += 1
        if dropped_dupes:
            chat_logger.info(
                "logical_table_format_dedupe",
                logical_table=logical_key,
                dropped_duplicate_format_partitions=dropped_dupes,
                kept=len(deduped),
            )
        members = deduped

        rep = members[0]
        rep_blob = str(rep["blob_path"])
        rep_file_id = str(rep["file_id"])
        rep_parquet = parquet_paths_all.get(rep_blob)
        display_name = display_name_from_path(rep_blob)
        logical_name = logical_name_from_path(rep_blob)

        member_file_ids = tuple(str(e["file_id"]) for e in members)
        partition_uris = tuple(_exec_uri(str(e["blob_path"])) for e in members)

        # Aggregate coverage window across all partitions (min start … max end).
        starts = [s for s in (_iso(e.get("date_range_start")) for e in members) if s]
        ends = [s for s in (_iso(e.get("date_range_end")) for e in members) if s]
        coverage_start = min(starts) if starts else None
        coverage_end = max(ends) if ends else None

        # Aliases: the consolidated logical key plus every member's references,
        # so a model/user can still address the table by any partition name.
        aliases = {logical_key, logical_name, display_name}
        for e in members:
            bp = str(e["blob_path"])
            aliases.update({bp, _basename(bp), str(e["file_id"]), f"az://{container_name}/{bp}"})
            pq = parquet_paths_all.get(bp)
            if pq:
                aliases.update({pq, _basename(pq), f"az://{container_name}/{pq}"})

        identities.append(FileIdentity(
            canonical_id=rep_file_id,
            logical_name=logical_name,
            sql_name=logical_key,
            blob_path=rep_blob,
            container_name=container_name,
            parquet_blob_path=rep_parquet,
            display_name=display_name,
            aliases=frozenset(alias for alias in aliases if alias),
            logical_table_id=logical_key,
            member_file_ids=member_file_ids,
            partition_uris=partition_uris,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            partition_count=len(members),
        ))

    chat_logger.info(
        "logical_tables_consolidated",
        source_files=sum(len(m) for m in groups.values()),
        logical_tables=len(identities),
        multi_partition=sum(1 for i in identities if len(i.member_file_ids) > 1),
    )
    return FileIdentityMap(identities)
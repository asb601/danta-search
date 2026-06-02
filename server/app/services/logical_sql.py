"""Canonicalize logical SQL into authorized physical execution SQL."""
from __future__ import annotations

from dataclasses import dataclass

try:
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp
    _SQLGLOT_AVAILABLE = True
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    _SQLGLOT_AVAILABLE = False

from app.services.file_identity import FileIdentity, FileIdentityMap, normalise_identity_key


class SQLCanonicalizationError(ValueError):
    """Raised when logical SQL cannot be resolved to canonical file identities."""


class SQLParseError(SQLCanonicalizationError):
    """The SQL text could not be parsed (syntax / unsupported dialect idiom).

    Distinct from authorization: a parse error is the model's fault, never an
    access-control denial, and must NOT be surfaced to users as an auth error.
    """


class SQLAuthorizationError(SQLCanonicalizationError):
    """A referenced logical table is not authorized for this request."""


# Vendor SQL dialects the model may emit (SAP/ANSI/T-SQL idioms). We parse with
# the executor dialect first; on failure we retry with these tolerant read
# dialects and transpile the result to duckdb, instead of hard-failing on a
# fixable idiom (e.g. DATEDIFF, GROUP_CONCAT).
_TOLERANT_READ_DIALECTS = ("duckdb", None, "tsql", "spark", "mysql")


@dataclass(frozen=True)
class CanonicalSQL:
    logical_sql: str
    executable_sql: str
    referenced_file_ids: list[str]
    referenced_tables: list[str]
    physical_uris: list[str]


def canonicalize_logical_sql(
    sql: str,
    file_identities: FileIdentityMap,
    *,
    allowed_file_ids: set[str] | None = None,
) -> CanonicalSQL:
    """Resolve logical table references and emit physical SQL for the executor.

        Model-facing SQL must reference logical tables, for example:
            SELECT * FROM ORDERS

    This function rewrites table references to read_parquet/read_csv_auto calls
    after resolving every table through FileIdentityMap.  Authorization is by
    canonical file ID; physical URI allowlists are a secondary invariant later.
    """
    if not _SQLGLOT_AVAILABLE:
        raise SQLCanonicalizationError("SQL parser is unavailable; cannot resolve logical table identities.")

    logical_sql = (sql or "").strip()
    if not logical_sql:
        raise SQLCanonicalizationError("Empty SQL query.")

    tree = _parse_tolerant(logical_sql)

    _reject_physical_storage_references(tree)

    cte_names = {
        normalise_identity_key(getattr(cte, "alias_or_name", "") or "")
        for cte in tree.find_all(exp.CTE)
    }
    referenced: dict[str, FileIdentity] = {}

    for table in list(tree.find_all(exp.Table)):
        if _is_table_function(table):
            raise SQLCanonicalizationError(
                "Physical file functions are not allowed in run_sql. Use logical table names in FROM/JOIN clauses."
            )

        table_name = table.name
        if not table_name:
            continue

        if normalise_identity_key(table_name) in cte_names:
            continue

        try:
            identity = file_identities.resolve_table(table_name)
        except KeyError as exc:
            raise SQLCanonicalizationError(str(exc).strip("'")) from exc
        except ValueError as exc:
            raise SQLCanonicalizationError(str(exc)) from exc

        member_ids = identity.member_file_ids or (identity.canonical_id,)
        member_uris = identity.execution_uris
        # Pair each partition's file id with its scan URI (built in the same order)
        # so authorization filters the SCAN, not just table access. A logical table
        # may have partitions outside this request's grant (per-file domain/ACL);
        # we must scan ONLY the authorized partitions, never the whole table.
        pairs = list(zip(member_ids, member_uris))
        if allowed_file_ids is not None:
            pairs = [(fid, uri) for fid, uri in pairs if fid in allowed_file_ids]
            if not pairs:
                raise SQLAuthorizationError(
                    f"Logical table '{table_name}' is not authorised for this request."
                )
        auth_member_ids = tuple(fid for fid, _ in pairs)
        auth_uris = tuple(uri for _, uri in pairs)

        referenced[identity.canonical_id] = (identity, auth_member_ids, auth_uris)
        table.replace(_physical_table_expression(table, auth_uris))

    if not referenced:
        raise SQLCanonicalizationError(
            "SQL must reference at least one logical table from the current catalog."
        )

    executable_sql = tree.sql(dialect="duckdb")
    referenced_file_ids: list[str] = []
    physical_uris: list[str] = []
    referenced_tables: list[str] = []
    for identity, auth_member_ids, auth_uris in referenced.values():
        referenced_file_ids.extend(auth_member_ids)
        physical_uris.extend(auth_uris)
        referenced_tables.append(identity.sql_name)
    return CanonicalSQL(
        logical_sql=logical_sql,
        executable_sql=executable_sql,
        referenced_file_ids=referenced_file_ids,
        referenced_tables=referenced_tables,
        physical_uris=physical_uris,
    )


def _parse_tolerant(logical_sql: str):
    """Parse model SQL, transpiling vendor idioms to the executor dialect.

    Tries the executor dialect first; on failure retries with tolerant read
    dialects so fixable idioms (DATEDIFF, GROUP_CONCAT, ...) become valid duckdb
    rather than a hard crash. A genuine syntax error raises SQLParseError —
    never an authorization error.
    """
    last_exc: Exception | None = None
    for dialect in _TOLERANT_READ_DIALECTS:
        try:
            tree = sqlglot.parse_one(
                logical_sql,
                dialect=dialect,
                error_level=sqlglot.errors.ErrorLevel.RAISE,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        if dialect != "duckdb":
            # Normalise to the executor dialect; if re-parse of the transpiled
            # SQL fails, fall through to the next read dialect.
            try:
                tree = sqlglot.parse_one(tree.sql(dialect="duckdb"), dialect="duckdb")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        return tree
    raise SQLParseError(f"SQL could not be parsed: {str(last_exc)[:240]}") from last_exc


def _scan_expression(uri: str):
    if uri.lower().endswith(".parquet"):
        return exp.ReadParquet(expressions=[exp.Literal.string(uri)])
    return exp.Anonymous(this="READ_CSV_AUTO", expressions=[exp.Literal.string(uri)])


def _physical_table_expression(original_table, uris: tuple[str, ...]):
    alias = original_table.args.get("alias")
    alias_node = alias.copy() if alias else exp.TableAlias(this=exp.to_identifier(original_table.name))

    if len(uris) == 1:
        new_table = exp.Table(this=_scan_expression(uris[0]))
        new_table.set("alias", alias_node)
        return new_table

    # Logical table spanning many partitions: scan them all. DuckDB supports
    # `UNION ALL BY NAME` (column-order tolerant). DataFusion has no BY NAME, so
    # under that engine fall back to positional `UNION ALL` (partitions of one
    # logical table share a schema post-ingestion, so positional is safe; the
    # schema-fingerprint gate in build_file_identity_map guarantees it). Each
    # branch is a single-path read the executor's regex registers individually.
    union_kw = "UNION ALL BY NAME" if _engine_supports_union_by_name() else "UNION ALL"
    union_sql = f" {union_kw} ".join(
        f"SELECT * FROM {_scan_expression(uri).sql(dialect='duckdb')}" for uri in uris
    )
    inner = sqlglot.parse_one(union_sql, dialect="duckdb")
    return exp.Subquery(this=inner, alias=alias_node)


def _engine_supports_union_by_name() -> bool:
    """DuckDB supports UNION ALL BY NAME; DataFusion does not."""
    try:
        from app.core.config import get_settings
        return str(getattr(get_settings(), "QUERY_ENGINE", "duckdb")).lower() != "datafusion"
    except Exception:
        return True


def _is_table_function(table) -> bool:
    table_expr = table.this
    if isinstance(table_expr, (exp.ReadParquet, exp.ReadCSV)):
        return True
    if isinstance(table_expr, exp.Anonymous):
        name = (table_expr.name or "").lower()
        return name in {"read_parquet", "read_csv", "read_csv_auto"}
    return False


def _reject_physical_storage_references(tree) -> None:
    for literal in tree.find_all(exp.Literal):
        value = str(literal.this or "")
        if "az://" in value:
            raise SQLCanonicalizationError(
                "Physical Azure blob paths are not allowed in run_sql. Use logical table names only."
            )

    for func in tree.find_all(exp.Func):
        if isinstance(func, (exp.ReadParquet, exp.ReadCSV)):
            raise SQLCanonicalizationError(
                "Physical file scan functions are not allowed in run_sql. Use logical table names only."
            )
        if isinstance(func, exp.Anonymous):
            name = (func.name or "").lower()
            if name in {"read_parquet", "read_csv", "read_csv_auto"}:
                raise SQLCanonicalizationError(
                    "Physical file scan functions are not allowed in run_sql. Use logical table names only."
                )
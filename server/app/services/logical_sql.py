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

    try:
        tree = sqlglot.parse_one(
            logical_sql,
            dialect="duckdb",
            error_level=sqlglot.errors.ErrorLevel.RAISE,
        )
    except Exception as exc:  # noqa: BLE001
        raise SQLCanonicalizationError(f"SQL could not be parsed: {str(exc)[:240]}") from exc

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

        if allowed_file_ids is not None and identity.canonical_id not in allowed_file_ids:
            raise SQLCanonicalizationError(
                f"Logical table '{table_name}' is not authorised for this request."
            )

        referenced[identity.canonical_id] = identity
        table.replace(_physical_table_expression(identity, table))

    if not referenced:
        raise SQLCanonicalizationError(
            "SQL must reference at least one logical table from the current catalog."
        )

    executable_sql = tree.sql(dialect="duckdb")
    identities = list(referenced.values())
    return CanonicalSQL(
        logical_sql=logical_sql,
        executable_sql=executable_sql,
        referenced_file_ids=[identity.canonical_id for identity in identities],
        referenced_tables=[identity.sql_name for identity in identities],
        physical_uris=[identity.execution_uri for identity in identities],
    )


def _physical_table_expression(identity: FileIdentity, original_table):
    if identity.execution_format == "parquet":
        source = exp.ReadParquet(expressions=[exp.Literal.string(identity.execution_uri)])
    else:
        source = exp.Anonymous(
            this="READ_CSV_AUTO",
            expressions=[exp.Literal.string(identity.execution_uri)],
        )
    new_table = exp.Table(this=source)
    alias = original_table.args.get("alias")
    if alias:
        new_table.set("alias", alias.copy())
    else:
        new_table.set("alias", exp.TableAlias(this=exp.to_identifier(original_table.name)))
    return new_table


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
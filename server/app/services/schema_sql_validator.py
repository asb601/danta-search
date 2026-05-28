"""Schema-grounded SQL validation before execution.

The agent writes logical SQL, but the runtime already knows the authorized
catalog schema. This module checks the SQL against that schema before the query
engine sees it, so obvious hallucinations fail fast with actionable feedback.

Scope is intentionally generic:
  - no ERP table names
  - no business-specific status meanings
  - no static source presets

All decisions come from the request-local catalog metadata and inspected schema
results. Unsupported or unparsable SQL falls through to the existing SQL parser
and execution safety layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

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


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    normalised_name: str
    data_type: str = ""
    known_values: tuple[str, ...] = ()
    values_are_exhaustive: bool = False


@dataclass(frozen=True)
class TableSchema:
    file_id: str
    logical_table: str
    columns: dict[str, ColumnSchema] = field(default_factory=dict)

    def column(self, name: str) -> ColumnSchema | None:
        return self.columns.get(_column_key(name))

    def known_column_names(self, limit: int = 40) -> list[str]:
        return [column.name for column in list(self.columns.values())[:limit]]


@dataclass(frozen=True)
class SchemaSQLIssue:
    code: str
    message: str
    table: str | None = None
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.table:
            payload["table"] = self.table
        if self.column:
            payload["column"] = self.column
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class SchemaSQLValidationReport:
    errors: tuple[SchemaSQLIssue, ...] = ()
    warnings: tuple[SchemaSQLIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_error_payload(self) -> dict[str, Any]:
        return {
            "error": "SQL references schema, filter values, labels, or join predicates that are not supported by inspected evidence.",
            "fatal_execution_error": True,
            "schema_validation_error": True,
            "issues": [issue.to_dict() for issue in self.errors[:8]],
            "warnings": [issue.to_dict() for issue in self.warnings[:6]],
            "hint": (
                "Use only columns shown by get_file_schema/inspect_column, and only join tables through approved joins "
                "or strong same-name key columns. Do not project invented string labels as if they came from data. "
                "If the requested business concept is absent, call search_catalog for another logical table "
                "or state that the concept is not available in the inspected schema. Do not call get_file_schema again "
                "for the same tables just to retry the same unsupported SQL."
            ),
        }


@dataclass(frozen=True)
class _Scope:
    alias_to_file_id: dict[str, str]
    derived_aliases: set[str]
    file_ids: list[str]


@dataclass(frozen=True)
class _ColumnResolution:
    status: str
    file_id: str | None = None
    table_schema: TableSchema | None = None
    column_schema: ColumnSchema | None = None
    candidate_tables: tuple[TableSchema, ...] = ()


def build_schema_index(
    catalog: list[dict],
    file_identities: FileIdentityMap | None,
) -> dict[str, TableSchema]:
    """Build a file_id -> TableSchema index from the request catalog."""
    if file_identities is None:
        return {}

    index: dict[str, TableSchema] = {}
    for entry in catalog:
        file_id = str(entry.get("file_id") or "")
        if not file_id:
            continue
        identity = file_identities.by_id.get(file_id)
        if not identity:
            continue
        table_schema = _table_schema_from_columns(
            file_id=file_id,
            logical_table=identity.sql_name,
            raw_columns=_extract_raw_columns(entry),
        )
        if table_schema.columns:
            index[file_id] = table_schema
    return index


def overlay_inspected_schemas(
    base_index: dict[str, TableSchema] | None,
    inspected_columns_by_file_id: dict[str, list[dict]] | None,
) -> dict[str, TableSchema]:
    """Overlay get_file_schema results onto the base catalog schema index."""
    if not inspected_columns_by_file_id:
        return base_index or {}

    merged = dict(base_index or {})
    for file_id, columns in inspected_columns_by_file_id.items():
        if not file_id or not isinstance(columns, list):
            continue
        existing = merged.get(file_id)
        logical_table = existing.logical_table if existing else file_id
        table_schema = _table_schema_from_columns(
            file_id=file_id,
            logical_table=logical_table,
            raw_columns=columns,
        )
        if table_schema.columns:
            merged[file_id] = table_schema
    return merged


def validate_logical_sql_schema(
    sql: str,
    file_identities: FileIdentityMap | None,
    schema_index: dict[str, TableSchema] | None,
    *,
    allowed_file_ids: set[str] | None = None,
    sql_ctx: Any | None = None,
) -> SchemaSQLValidationReport:
    """Validate logical SQL against known table schemas.

    The validator is best-effort by design. It returns errors only when the
    evidence is clear: an in-scope table has known columns and the SQL references
    a column that is absent, an equality filter uses a value outside an
    exhaustive known value set, or the query contains a logically empty
    HAVING COUNT(*) = 0 pattern, unsupported string literal labels, or a
    cross-table predicate that is not supported by approved join evidence or a
    strong same-name key.
    """
    if not sql or not file_identities or not schema_index or not _SQLGLOT_AVAILABLE:
        return SchemaSQLValidationReport()

    try:
        tree = sqlglot.parse_one(
            sql,
            dialect="duckdb",
            error_level=sqlglot.errors.ErrorLevel.RAISE,
        )
    except Exception:
        return SchemaSQLValidationReport()

    cte_names = _cte_names(tree)
    join_contracts = _join_contracts(sql_ctx)
    errors: list[SchemaSQLIssue] = []
    warnings: list[SchemaSQLIssue] = []

    for select_expr in tree.find_all(exp.Select):
        scope = _scope_for_select(select_expr, file_identities, cte_names, allowed_file_ids)
        if not scope.file_ids and not scope.derived_aliases:
            continue

        select_aliases = _select_aliases(select_expr)
        for column_expr in _iter_current_scope(select_expr, exp.Column):
            column_name = getattr(column_expr, "name", "") or ""
            if not column_name or column_name == "*":
                continue
            if not getattr(column_expr, "table", None) and _column_key(column_name) in select_aliases:
                continue
            resolution = _resolve_column(column_expr, scope, schema_index)
            issue = _issue_for_column_resolution(column_expr, resolution)
            if issue:
                errors.append(issue)

        for issue in _validate_exhaustive_filter_values(select_expr, scope, schema_index):
            if issue.code == "partially_unknown_filter_values":
                warnings.append(issue)
            else:
                errors.append(issue)

        errors.extend(_validate_literal_label_projections(select_expr))
        errors.extend(_validate_cross_table_eq_contracts(select_expr, scope, schema_index, join_contracts))
        errors.extend(_validate_in_subquery_contracts(
            select_expr,
            scope,
            schema_index,
            file_identities,
            cte_names,
            allowed_file_ids,
            join_contracts,
        ))

        count_zero_issue = _validate_having_count_star_zero(select_expr)
        if count_zero_issue:
            errors.append(count_zero_issue)

    return SchemaSQLValidationReport(tuple(_dedupe_issues(errors)), tuple(_dedupe_issues(warnings)))


def _extract_raw_columns(entry: dict) -> list[dict]:
    raw_columns = [
        column_entry
        for column_entry in (entry.get("columns_info") or [])
        if isinstance(column_entry, dict) and column_entry.get("name")
    ]
    if raw_columns:
        return raw_columns
    return [
        {"name": column_name, "type": "unknown"}
        for column_name in (entry.get("column_names") or [])
        if isinstance(column_name, str) and column_name
    ]


def _table_schema_from_columns(
    *,
    file_id: str,
    logical_table: str,
    raw_columns: list[dict],
) -> TableSchema:
    columns: dict[str, ColumnSchema] = {}
    for column_entry in raw_columns:
        name = str(column_entry.get("name") or "").strip()
        if not name:
            continue
        values, exhaustive = _known_values(column_entry)
        columns[_column_key(name)] = ColumnSchema(
            name=name,
            normalised_name=_column_key(name),
            data_type=str(column_entry.get("type") or column_entry.get("dtype") or ""),
            known_values=tuple(values),
            values_are_exhaustive=exhaustive,
        )
    return TableSchema(file_id=file_id, logical_table=logical_table, columns=columns)


def _known_values(column_entry: dict) -> tuple[list[str], bool]:
    unique_values = column_entry.get("unique_values")
    if isinstance(unique_values, list):
        values = _dedupe_values(unique_values)
        return values, bool(values)

    sample_values = column_entry.get("sample_values") or column_entry.get("top_values") or []
    values = _dedupe_values(sample_values if isinstance(sample_values, list) else [])
    distinct_count = _int_or_none(column_entry.get("distinct_count"))
    if distinct_count is None:
        distinct_count = _int_or_none(column_entry.get("unique_count"))
    exhaustive = bool(values) and distinct_count is not None and distinct_count > 0 and distinct_count <= len(values)
    return values, exhaustive


def _dedupe_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        key = _value_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _cte_names(tree) -> set[str]:
    names: set[str] = set()
    for cte_expr in tree.find_all(exp.CTE):
        alias = getattr(cte_expr, "alias_or_name", "") or ""
        if alias:
            names.add(normalise_identity_key(alias))
    return names


def _scope_for_select(
    select_expr,
    file_identities: FileIdentityMap,
    cte_names: set[str],
    allowed_file_ids: set[str] | None,
) -> _Scope:
    alias_to_file_id: dict[str, str] = {}
    derived_aliases: set[str] = set()
    file_ids: list[str] = []

    for source_expr in _source_expressions(select_expr):
        if isinstance(source_expr, exp.Subquery):
            alias = getattr(source_expr, "alias_or_name", "") or ""
            if alias:
                derived_aliases.add(normalise_identity_key(alias))
            continue
        if not isinstance(source_expr, exp.Table):
            continue

        table_name = source_expr.name or ""
        table_key = normalise_identity_key(table_name)
        if table_key in cte_names:
            derived_aliases.add(normalise_identity_key(getattr(source_expr, "alias_or_name", "") or table_name))
            continue

        identity = _resolve_identity(file_identities, table_name)
        if identity is None:
            continue
        if allowed_file_ids is not None and identity.canonical_id not in allowed_file_ids:
            continue
        if identity.canonical_id not in file_ids:
            file_ids.append(identity.canonical_id)
        for alias in _source_aliases(source_expr, identity):
            alias_key = normalise_identity_key(alias)
            if alias_key:
                alias_to_file_id[alias_key] = identity.canonical_id

    return _Scope(alias_to_file_id=alias_to_file_id, derived_aliases=derived_aliases, file_ids=file_ids)


def _source_expressions(select_expr) -> Iterable[Any]:
    from_clause = select_expr.args.get("from_")
    if from_clause is not None:
        source = getattr(from_clause, "this", None)
        if source is not None:
            yield source
        for source in getattr(from_clause, "expressions", []) or []:
            yield source
    for join_expr in select_expr.args.get("joins") or []:
        source = getattr(join_expr, "this", None)
        if source is not None:
            yield source


def _resolve_identity(file_identities: FileIdentityMap, table_name: str) -> FileIdentity | None:
    try:
        return file_identities.resolve_table(table_name)
    except (KeyError, ValueError):
        return None


def _source_aliases(table_expr, identity: FileIdentity) -> set[str]:
    aliases = {
        table_expr.name or "",
        getattr(table_expr, "alias_or_name", "") or "",
        identity.sql_name,
        identity.logical_name,
    }
    alias_expr = table_expr.args.get("alias")
    alias_name = getattr(alias_expr, "this", None) if alias_expr is not None else None
    if alias_name is not None:
        aliases.add(str(alias_name))
    return {alias for alias in aliases if alias}


def _select_aliases(select_expr) -> set[str]:
    aliases: set[str] = set()
    for projection in getattr(select_expr, "expressions", []) or []:
        alias = getattr(projection, "alias", "") or ""
        if alias:
            aliases.add(_column_key(alias))
    return aliases


def _iter_current_scope(root, node_type) -> Iterable[Any]:
    stack = [root]
    while stack:
        current = stack.pop()
        if current is not root and isinstance(current, exp.Select):
            continue
        if isinstance(current, node_type):
            yield current
        for child in _children(current):
            stack.append(child)


def _children(node) -> Iterable[Any]:
    for value in getattr(node, "args", {}).values():
        if isinstance(value, exp.Expression):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, exp.Expression):
                    yield item


def _resolve_column(column_expr, scope: _Scope, schema_index: dict[str, TableSchema]) -> _ColumnResolution:
    column_name = getattr(column_expr, "name", "") or ""
    qualifier = getattr(column_expr, "table", "") or ""
    column_key = _column_key(column_name)

    if qualifier:
        qualifier_key = normalise_identity_key(qualifier)
        if qualifier_key in scope.derived_aliases:
            return _ColumnResolution(status="derived")
        file_id = scope.alias_to_file_id.get(qualifier_key)
        if not file_id:
            return _ColumnResolution(status="unknown_scope")
        table_schema = schema_index.get(file_id)
        if not table_schema or not table_schema.columns:
            return _ColumnResolution(status="schema_unavailable", file_id=file_id)
        column_schema = table_schema.column(column_name)
        if not column_schema:
            return _ColumnResolution(status="missing", file_id=file_id, table_schema=table_schema)
        return _ColumnResolution(
            status="resolved",
            file_id=file_id,
            table_schema=table_schema,
            column_schema=column_schema,
        )

    candidates: list[tuple[str, TableSchema, ColumnSchema]] = []
    tables_with_schema: list[TableSchema] = []
    for file_id in scope.file_ids:
        table_schema = schema_index.get(file_id)
        if not table_schema or not table_schema.columns:
            continue
        tables_with_schema.append(table_schema)
        column_schema = table_schema.column(column_key)
        if column_schema:
            candidates.append((file_id, table_schema, column_schema))

    if not tables_with_schema:
        return _ColumnResolution(status="schema_unavailable")
    if not candidates:
        return _ColumnResolution(status="missing_unqualified", candidate_tables=tuple(tables_with_schema))
    if len(candidates) > 1:
        return _ColumnResolution(
            status="ambiguous",
            candidate_tables=tuple(candidate[1] for candidate in candidates),
        )
    file_id, table_schema, column_schema = candidates[0]
    return _ColumnResolution(
        status="resolved",
        file_id=file_id,
        table_schema=table_schema,
        column_schema=column_schema,
    )


def _issue_for_column_resolution(column_expr, resolution: _ColumnResolution) -> SchemaSQLIssue | None:
    column_name = getattr(column_expr, "name", "") or ""
    if resolution.status == "missing" and resolution.table_schema:
        known_columns = resolution.table_schema.known_column_names()
        return SchemaSQLIssue(
            code="unknown_column",
            table=resolution.table_schema.logical_table,
            column=column_name,
            message=f"Column '{column_name}' is not present in logical table '{resolution.table_schema.logical_table}'.",
            details={"known_columns": known_columns},
        )
    if resolution.status == "missing_unqualified":
        tables = list(resolution.candidate_tables)
        table_name = tables[0].logical_table if len(tables) == 1 else None
        return SchemaSQLIssue(
            code="unknown_column",
            table=table_name,
            column=column_name,
            message=f"Column '{column_name}' is not present in the current SQL scope.",
            details={
                "tables_checked": [table.logical_table for table in tables[:8]],
                "known_columns_by_table": {
                    table.logical_table: table.known_column_names(limit=24)
                    for table in tables[:4]
                },
            },
        )
    if resolution.status == "ambiguous":
        return SchemaSQLIssue(
            code="ambiguous_column",
            column=column_name,
            message=f"Column '{column_name}' exists in multiple tables in the same SQL scope; qualify it with a table alias.",
            details={"candidate_tables": [table.logical_table for table in resolution.candidate_tables[:8]]},
        )
    return None


def _validate_exhaustive_filter_values(
    select_expr,
    scope: _Scope,
    schema_index: dict[str, TableSchema],
) -> list[SchemaSQLIssue]:
    issues: list[SchemaSQLIssue] = []
    for comparison_expr in _iter_current_scope(select_expr, exp.EQ):
        column_expr, values = _comparison_column_and_values(comparison_expr)
        if column_expr is None or not values:
            continue
        issue = _issue_for_known_values(column_expr, values, scope, schema_index)
        if issue:
            issues.append(issue)

    for in_expr in _iter_current_scope(select_expr, exp.In):
        column_expr = getattr(in_expr, "this", None)
        if not isinstance(column_expr, exp.Column):
            continue
        values = [
            literal.this
            for literal in (getattr(in_expr, "expressions", []) or [])
            if isinstance(literal, exp.Literal) and literal.is_string
        ]
        if not values:
            continue
        issue = _issue_for_known_values(column_expr, values, scope, schema_index)
        if issue:
            issues.append(issue)
    return issues


def _validate_literal_label_projections(select_expr) -> list[SchemaSQLIssue]:
    issues: list[SchemaSQLIssue] = []
    for projection in getattr(select_expr, "expressions", []) or []:
        alias = getattr(projection, "alias", "") or ""
        inner_expr = getattr(projection, "this", None) if isinstance(projection, exp.Alias) else projection
        literal_values = _projection_literal_label_values(inner_expr)
        if not literal_values:
            continue
        literal_preview = literal_values[:4]
        issues.append(SchemaSQLIssue(
            code="literal_label_projection",
            column=alias or None,
            message=(
                f"String literal result value(s) {literal_preview} are being projected as labels"
                + (f" for '{alias}'" if alias else "")
                + ". This fabricates a category that is not sourced from a column."
            ),
            details={
                "literals": literal_preview,
                "alias": alias,
                "fix": "Select the real status/code column or aggregate only; explain labels outside SQL only when supported by field definitions or inspected data.",
            },
        ))
    return issues


def _projection_literal_label_values(expression) -> list[str]:
    if isinstance(expression, exp.Literal) and expression.is_string:
        return [str(expression.this)]
    if isinstance(expression, exp.Case):
        values: list[str] = []
        for if_expr in expression.args.get("ifs") or []:
            true_expr = if_expr.args.get("true")
            if isinstance(true_expr, exp.Literal) and true_expr.is_string:
                values.append(str(true_expr.this))
        default_expr = expression.args.get("default")
        if isinstance(default_expr, exp.Literal) and default_expr.is_string:
            values.append(str(default_expr.this))
        return _dedupe_values(values)
    return []


def _validate_cross_table_eq_contracts(
    select_expr,
    scope: _Scope,
    schema_index: dict[str, TableSchema],
    join_contracts: set[frozenset[tuple[str, str]]],
) -> list[SchemaSQLIssue]:
    issues: list[SchemaSQLIssue] = []
    for comparison_expr in _iter_current_scope(select_expr, exp.EQ):
        left_expr = getattr(comparison_expr, "this", None)
        right_expr = getattr(comparison_expr, "expression", None)
        if not isinstance(left_expr, exp.Column) or not isinstance(right_expr, exp.Column):
            continue
        issue = _validate_column_relation(
            left_expr,
            right_expr,
            scope,
            scope,
            schema_index,
            join_contracts,
        )
        if issue:
            issues.append(issue)
    return issues


def _validate_in_subquery_contracts(
    select_expr,
    scope: _Scope,
    schema_index: dict[str, TableSchema],
    file_identities: FileIdentityMap,
    cte_names: set[str],
    allowed_file_ids: set[str] | None,
    join_contracts: set[frozenset[tuple[str, str]]],
) -> list[SchemaSQLIssue]:
    issues: list[SchemaSQLIssue] = []
    for in_expr in _iter_current_scope(select_expr, exp.In):
        outer_column = getattr(in_expr, "this", None)
        query_expr = in_expr.args.get("query")
        if not isinstance(outer_column, exp.Column) or not isinstance(query_expr, exp.Subquery):
            continue
        inner_select = getattr(query_expr, "this", None)
        if not isinstance(inner_select, exp.Select):
            continue
        inner_column = _first_projected_column(inner_select)
        if inner_column is None:
            continue
        inner_scope = _scope_for_select(inner_select, file_identities, cte_names, allowed_file_ids)
        issue = _validate_column_relation(
            outer_column,
            inner_column,
            scope,
            inner_scope,
            schema_index,
            join_contracts,
        )
        if issue:
            issues.append(issue)
    return issues


def _first_projected_column(select_expr) -> Any | None:
    projections = list(getattr(select_expr, "expressions", []) or [])
    if len(projections) != 1:
        return None
    projection = projections[0]
    inner_expr = getattr(projection, "this", None) if isinstance(projection, exp.Alias) else projection
    return inner_expr if isinstance(inner_expr, exp.Column) else None


def _validate_column_relation(
    left_expr,
    right_expr,
    left_scope: _Scope,
    right_scope: _Scope,
    schema_index: dict[str, TableSchema],
    join_contracts: set[frozenset[tuple[str, str]]],
) -> SchemaSQLIssue | None:
    left = _resolve_column(left_expr, left_scope, schema_index)
    right = _resolve_column(right_expr, right_scope, schema_index)
    if left.status != "resolved" or right.status != "resolved":
        return None
    if not left.file_id or not right.file_id or not left.column_schema or not right.column_schema:
        return None
    if left.file_id == right.file_id:
        return None

    relation_key = _relation_key(
        left.file_id,
        left.column_schema.name,
        right.file_id,
        right.column_schema.name,
    )
    if relation_key in join_contracts:
        return None
    if _strong_same_name_join(left.column_schema.name, right.column_schema.name):
        return None

    left_ref = _column_ref(left)
    right_ref = _column_ref(right)
    return SchemaSQLIssue(
        code="unverified_cross_table_relation",
        message=(
            f"Cross-table predicate {left_ref} = {right_ref} is not backed by an approved join "
            "and the columns do not share the same strong key name."
        ),
        details={
            "left": left_ref,
            "right": right_ref,
            "fix": "Use extract_relations/search_catalog for an approved relationship, or compare strong same-name keys such as the same *_id column.",
        },
    )


def _join_contracts(sql_ctx: Any | None) -> set[frozenset[tuple[str, str]]]:
    contracts: set[frozenset[tuple[str, str]]] = set()
    for join in list(getattr(sql_ctx, "approved_joins", []) or []):
        left_file_id = str(getattr(join, "left_file_id", "") or "")
        right_file_id = str(getattr(join, "right_file_id", "") or "")
        left_col = str(getattr(join, "left_col", "") or "")
        right_col = str(getattr(join, "right_col", "") or "")
        if not left_file_id or not right_file_id or not left_col or not right_col:
            continue
        contracts.add(_relation_key(left_file_id, left_col, right_file_id, right_col))
    return contracts


def _relation_key(left_file_id: str, left_col: str, right_file_id: str, right_col: str) -> frozenset[tuple[str, str]]:
    return frozenset({
        (str(left_file_id), _column_key(left_col)),
        (str(right_file_id), _column_key(right_col)),
    })


def _strong_same_name_join(left_col: str, right_col: str) -> bool:
    left_key = _column_key(left_col)
    right_key = _column_key(right_col)
    if left_key != right_key:
        return False
    if len(left_key) < 4:
        return False
    return left_key.endswith(("_id", "_key", "_code", "_num", "_number")) or "_id_" in left_key


def _column_ref(resolution: _ColumnResolution) -> str:
    table = resolution.table_schema.logical_table if resolution.table_schema else (resolution.file_id or "?")
    column = resolution.column_schema.name if resolution.column_schema else "?"
    return f"{table}.{column}"


def _dedupe_issues(issues: list[SchemaSQLIssue]) -> list[SchemaSQLIssue]:
    seen: set[tuple[str, str | None, str | None, str]] = set()
    result: list[SchemaSQLIssue] = []
    for issue in issues:
        key = (issue.code, issue.table, issue.column, issue.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _comparison_column_and_values(comparison_expr) -> tuple[Any | None, list[str]]:
    left_expr = getattr(comparison_expr, "this", None)
    right_expr = getattr(comparison_expr, "expression", None)
    if isinstance(left_expr, exp.Column) and isinstance(right_expr, exp.Literal) and right_expr.is_string:
        return left_expr, [str(right_expr.this)]
    if isinstance(right_expr, exp.Column) and isinstance(left_expr, exp.Literal) and left_expr.is_string:
        return right_expr, [str(left_expr.this)]
    return None, []


def _issue_for_known_values(
    column_expr,
    values: list[str],
    scope: _Scope,
    schema_index: dict[str, TableSchema],
) -> SchemaSQLIssue | None:
    resolution = _resolve_column(column_expr, scope, schema_index)
    if resolution.status != "resolved" or not resolution.table_schema or not resolution.column_schema:
        return None
    column_schema = resolution.column_schema
    if not column_schema.values_are_exhaustive or not column_schema.known_values:
        return None

    known_keys = {_value_key(value) for value in column_schema.known_values}
    missing_values = [value for value in values if _value_key(value) not in known_keys]
    if not missing_values:
        return None

    issue_code = "unknown_filter_value" if len(missing_values) == len(values) else "partially_unknown_filter_values"
    verb = "was not observed" if len(missing_values) == 1 else "were not observed"
    return SchemaSQLIssue(
        code=issue_code,
        table=resolution.table_schema.logical_table,
        column=column_schema.name,
        message=(
            f"Filter value(s) {missing_values[:8]} {verb} for "
            f"{resolution.table_schema.logical_table}.{column_schema.name}."
        ),
        details={
            "known_values": list(column_schema.known_values[:20]),
            "values_checked": values[:20],
        },
    )


def _validate_having_count_star_zero(select_expr) -> SchemaSQLIssue | None:
    if select_expr.args.get("group") is None:
        return None
    having_expr = select_expr.args.get("having")
    if having_expr is None:
        return None
    for comparison_expr in _iter_current_scope(having_expr, exp.EQ):
        left_expr = getattr(comparison_expr, "this", None)
        right_expr = getattr(comparison_expr, "expression", None)
        if (_is_count_star(left_expr) and _is_zero_literal(right_expr)) or (
            _is_count_star(right_expr) and _is_zero_literal(left_expr)
        ):
            return SchemaSQLIssue(
                code="unsatisfiable_having_count_star_zero",
                message=(
                    "HAVING COUNT(*) = 0 cannot return grouped rows because every emitted group has at least one row. "
                    "Use a LEFT JOIN/NOT EXISTS pattern when looking for missing matches."
                ),
            )
    return None


def _is_count_star(expression) -> bool:
    if not isinstance(expression, exp.Count):
        return False
    return any(isinstance(child, exp.Star) for child in expression.find_all(exp.Star))


def _is_zero_literal(expression) -> bool:
    return isinstance(expression, exp.Literal) and not expression.is_string and str(expression.this).strip() in {"0", "0.0"}


def _column_key(value: str) -> str:
    return str(value or "").strip().strip('`"[]').lower()


def _value_key(value: Any) -> str:
    return str(value).strip().casefold()

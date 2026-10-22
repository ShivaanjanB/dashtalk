"""The semantic layer: dashboards, metrics and dimensions defined in YAML, compiled to safe SQL.

The model never writes SQL. It names a metric and some filters; this module maps them to columns it
knows, binds every value as a parameter, resolves names to exact matches ("UK" -> United Kingdom,
"apple" -> one company or a question back), adds the user's row-level scope, and refuses anything it
can't resolve.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from dashtalk.paths import SEMANTIC

MAX_LIMIT = 100
TIME_DIMENSIONS = ("year", "quarter", "month")


class QueryError(Exception):
    """A request the layer won't run: not_found, ambiguous, denied or invalid. Returned to the model as data."""

    def __init__(self, status: str, message: str, **extra: object) -> None:
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra

    def as_dict(self) -> dict:
        return {"status": self.status, "message": self.message, **self.extra}


@dataclass(frozen=True)
class Dimension:
    name: str
    column: str
    match: str = "exact"  # exact | contains | name_or_code
    type: str = "str"
    code_column: str | None = None
    description: str = ""
    synonyms: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    expr: str
    unit: str
    description: str = ""
    synonyms: tuple[str, ...] = ()
    caveats: str = ""
    period: dict | None = None  # overrides the dashboard's, e.g. balance-sheet dates for a balance-sheet ratio


@dataclass(frozen=True)
class Dashboard:
    name: str
    title: str
    description: str
    table: str
    grain: str
    source: str
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]
    caveats: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    row_level_security: str | None = None
    period: dict | None = None  # which dates a figure covers, when "the year" isn't the obvious reading


def key(text: str) -> str:
    """Normalise a name for matching: case, underscores, hyphens and spacing don't matter."""
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower())


def load_dashboards(folder: Path = SEMANTIC) -> dict[str, Dashboard]:
    dashboards = {}
    for path in sorted(folder.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        dims = {name: Dimension(name=name, column=d["column"], match=d.get("match", "exact"), type=d.get("type", "str"),
                                code_column=d.get("code_column"), description=d.get("description", ""),
                                synonyms=tuple(d.get("synonyms", [])),
                                aliases={key(k): v for k, v in (d.get("aliases") or {}).items()})
                for name, d in spec["dimensions"].items()}
        metrics = {name: Metric(name=name, label=m["label"], expr=m["expr"], unit=m["unit"],
                                description=m.get("description", ""), synonyms=tuple(m.get("synonyms", [])),
                                caveats=m.get("caveats", ""), period=m.get("period"))
                   for name, m in spec["metrics"].items()}
        dashboards[spec["name"]] = Dashboard(
            name=spec["name"], title=spec["title"], description=spec["description"], table=spec["table"],
            grain=spec["grain"], source=spec["source"], dimensions=dims, metrics=metrics,
            caveats=tuple(spec.get("caveats", [])), examples=tuple(spec.get("examples", [])),
            row_level_security=spec.get("row_level_security"), period=spec.get("period"))
    return dashboards


def find_metric(dash: Dashboard, name: str) -> Metric:
    wanted = key(name)
    for metric in dash.metrics.values():
        if wanted in {key(metric.name), key(metric.label), *map(key, metric.synonyms)}:
            return metric
    raise QueryError("not_found", f"No metric called '{name}' on {dash.name}.", available_metrics=list(dash.metrics))


def find_dimension(dash: Dashboard, name: str) -> Dimension:
    wanted = key(name)
    for dim in dash.dimensions.values():
        if wanted in {key(dim.name), key(dim.column), *map(key, dim.synonyms)}:
            return dim
    raise QueryError("not_found", f"No dimension called '{name}' on {dash.name}.",
                     available_dimensions=list(dash.dimensions))


def _as_list(value: object) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def resolve_values(con: duckdb.DuckDBPyConnection, dash: Dashboard, dim: Dimension, requested: list,
                   scope: list[str] | None) -> dict[str, list]:
    """Map each requested value to the exact values present in the table, within the user's scope.

    Exact names, aliases and codes win; otherwise "contains" dimensions fall back to a partial match.
    """
    if dim.type == "int":
        try:
            return {str(v): [int(v)] for v in requested}
        except (TypeError, ValueError) as err:
            raise QueryError("invalid", f"{dim.name} must be a whole number, got {requested}.") from err
    scope_sql, scope_params = "", []
    if scope is not None:
        scope_sql = f" AND {dim.column} IN ({', '.join('?' * len(scope))})"
        scope_params = list(scope)
    resolved = {}
    for value in requested:
        text = dim.aliases.get(key(value), str(value))
        exact = f"lower({dim.column}) = lower(?)"
        params = [text]
        if dim.code_column:
            exact += f" OR upper({dim.code_column}) = upper(?)"
            params.append(text)
        rows = con.execute(f"SELECT DISTINCT {dim.column} FROM {dash.table} WHERE ({exact}){scope_sql} ORDER BY 1",
                           params + scope_params).fetchall()
        if not rows and dim.match in ("contains", "name_or_code"):
            rows = con.execute(f"SELECT DISTINCT {dim.column} FROM {dash.table} WHERE {dim.column} ILIKE ?{scope_sql} "
                               "ORDER BY 1", [f"%{text}%"] + scope_params).fetchall()
        resolved[str(value)] = [r[0] for r in rows]
    return resolved


def run_query(con: duckdb.DuckDBPyConnection, dash: Dashboard, metric_name: str, filters: dict | None = None,
              group_by: list | None = None, order: str = "desc", limit: int = 25,
              row_scope: dict[str, list[str]] | None = None) -> dict:
    """Compile and run one metric query. Raises QueryError for anything it won't answer."""
    metric = find_metric(dash, metric_name)
    filters, group_by, row_scope = filters or {}, group_by or [], row_scope or {}
    if order not in ("desc", "asc"):
        raise QueryError("invalid", "order must be 'desc' or 'asc'.")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise QueryError("invalid", f"limit must be between 1 and {MAX_LIMIT}.")
    group_dims = [find_dimension(dash, g) for g in group_by]
    where, params, matched = [], [], {}

    for name, value in filters.items():
        dim = find_dimension(dash, name)
        scope = row_scope.get(dim.name)
        if scope is not None:
            # A value outside the user's scope is refused outright, not silently dropped.
            outside = resolve_values(con, dash, dim, _as_list(value), None)
            hidden = sorted({v for vals in outside.values() for v in vals} - set(scope))
            if hidden:
                raise QueryError("denied", f"You can see {dash.title} for {', '.join(scope)} only; "
                                           f"{', '.join(hidden)} is outside your access.")
        resolved = resolve_values(con, dash, dim, _as_list(value), scope)
        for asked, found in resolved.items():
            if not found:
                raise QueryError("not_found", f"No {dim.name} matching '{asked}' on {dash.name}.")
            if len(found) > 1 and dim not in group_dims and dim.type != "int":
                raise QueryError("ambiguous", f"'{asked}' matches {len(found)} values of {dim.name}. Which one?",
                                 candidates=found[:10])
        values = sorted({v for vals in resolved.values() for v in vals}, key=str)
        matched[dim.name] = values
        where.append(f"{dim.column} IN ({', '.join('?' * len(values))})")
        params += values

    scope_note = None
    for dim_name, allowed in row_scope.items():
        if dim_name not in matched:
            dim = dash.dimensions[dim_name]
            where.append(f"{dim.column} IN ({', '.join('?' * len(allowed))})")
            params += list(allowed)
            scope_note = f"Limited to your access: {dim_name} in {', '.join(allowed)}."

    period = metric.period or dash.period
    select = [f'{d.column} AS "{d.name}"' for d in group_dims] + [f"{metric.expr} AS value"]
    if period:
        select += [f"min({period['end']}) AS period_first", f"max({period['end']}) AS period_last"]
    sql = f"SELECT {', '.join(select)} FROM {dash.table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group_dims:
        sql += " GROUP BY " + ", ".join(d.column for d in group_dims)
        if all(d.name in TIME_DIMENSIONS for d in group_dims):
            sql += " ORDER BY " + ", ".join(f'"{d.name}"' for d in group_dims)
        else:
            sql += f" ORDER BY value {order.upper()} NULLS LAST"
    sql += f" LIMIT {limit}"
    frame = con.execute(sql, params).fetchdf()
    rows = []
    for r in frame.to_dict("records"):  # not iterrows: it would turn a missing value into a date beside the dates
        row = {d.name: _plain(r[d.name]) for d in group_dims}
        shown, covers = display(r["value"], metric.unit), _period_label(period, r, row, matched)
        rows.append({**row, "value": _number(r["value"]), "display": f"{shown} ({covers})" if covers else shown})
    note = (dash.period or {}).get("note")
    return {"status": "ok", "dashboard": dash.name, "metric": metric.name, "label": metric.label, "unit": metric.unit,
            "matched": matched, "rows": rows, "scope": scope_note, **({"period": note} if note else {}),
            "caveats": [c for c in (metric.caveats, *dash.caveats) if c], "sql": sql, "params": [_plain(p) for p in params]}


def _period_label(period: dict | None, r: dict, row: dict, matched: dict) -> str:
    """Which dates a figure covers: the one year end every row behind it shares, else the year it's filed under."""
    if not period:
        return ""
    first, last = r["period_first"], r["period_last"]
    if pd.notna(last) and first == last:
        return period["label"].format(end=f"{last.day} {last:%b %Y}")
    dim = period.get("dimension", "year")
    years = [row[dim]] if dim in row else matched.get(dim, [])
    return period["mixed"].format(year=", ".join(map(str, years))) if years and period.get("mixed") else ""


def _plain(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _number(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


def display(value: object, unit: str) -> str:
    """A human-readable figure: $383.3bn, 5.05%, 3.2x."""
    number = _number(value)
    if number is None:
        return "no data"
    if unit == "usd":
        for size, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
            if abs(number) >= size:
                return f"${number / size:,.1f}{suffix}"
        return f"${number:,.0f}"
    if unit.startswith("percent"):
        return f"{number:.2f}%" + (" of GDP" if unit == "percent of GDP" else "")
    if unit == "times":
        return f"{number:.2f}x"
    if unit == "units":
        return f"{number:,.0f} units"
    return f"{number:.3f}"

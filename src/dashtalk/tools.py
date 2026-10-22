"""The three tools, bound to one user: list dashboards, describe one, and query a metric.

The user comes from how the server was started, never from the model. Every call is checked against
config/users.yaml and written to an audit log (logs/audit.jsonl), allowed or not.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import yaml

from dashtalk.paths import CONFIG, LOGS, WAREHOUSE
from dashtalk.semantic import Dashboard, QueryError, load_dashboards, run_query

_audit_lock = threading.Lock()


@dataclass(frozen=True)
class User:
    id: str
    name: str
    dashboards: frozenset[str]
    row_filters: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def load_users(path: Path = CONFIG / "users.yaml") -> dict[str, User]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["users"]
    return {uid: User(id=uid, name=u.get("name", uid), dashboards=frozenset(u.get("dashboards", [])),
                      row_filters=u.get("row_filters") or {}) for uid, u in spec.items()}


class DashTalk:
    def __init__(self, user_id: str, dashboards: dict[str, Dashboard] | None = None, users: dict[str, User] | None = None,
                 warehouse: Path = WAREHOUSE, audit_path: Path | None = LOGS / "audit.jsonl") -> None:
        self.users = users if users is not None else load_users()
        self.user = self.users.get(user_id)
        self.user_id = user_id
        self.dashboards = dashboards if dashboards is not None else load_dashboards()
        self.warehouse = warehouse
        self.audit_path = audit_path

    # ---- the three tools -----------------------------------------------------------------------

    def list_dashboards(self) -> dict:
        """The dashboards this user may use (others aren't mentioned)."""
        result = {"status": "ok", "user": self.user_id,
                  "dashboards": [{"name": d.name, "title": d.title, "description": d.description}
                                 for d in self.dashboards.values() if self._allowed(d.name)]}
        return self._log("list_dashboards", {}, result)

    def describe_dashboard(self, dashboard: str) -> dict:
        """Metrics (with synonyms, units and caveats), dimensions and example questions for one dashboard."""
        try:
            dash = self._dashboard(dashboard)
            result = {
                "status": "ok", "dashboard": dash.name, "title": dash.title, "description": dash.description,
                "grain": dash.grain, "source": dash.source, "caveats": list(dash.caveats),
                **({"period": dash.period["note"]} if dash.period and dash.period.get("note") else {}),
                "metrics": [{"name": m.name, "label": m.label, "unit": m.unit, "description": m.description,
                             "synonyms": list(m.synonyms), **({"caveats": m.caveats} if m.caveats else {})}
                            for m in dash.metrics.values()],
                "dimensions": [{"name": d.name, "description": d.description, "synonyms": list(d.synonyms)}
                               for d in dash.dimensions.values()],
                "examples": list(dash.examples),
                "your_access": self._scope_text(dash.name),
            }
        except QueryError as err:
            result = err.as_dict()
        return self._log("describe_dashboard", {"dashboard": dashboard}, result)

    def query_metric(self, dashboard: str, metric: str, filters: dict | None = None, group_by: list | None = None,
                     order: str = "desc", limit: int = 25) -> dict:
        """One metric, optionally filtered and grouped, within this user's access."""
        args = {"dashboard": dashboard, "metric": metric, "filters": filters or {}, "group_by": group_by or [],
                "order": order, "limit": limit}
        try:
            dash = self._dashboard(dashboard)
            with duckdb.connect(str(self.warehouse), read_only=True) as con:
                result = run_query(con, dash, metric, filters, group_by, order, limit,
                                   row_scope=self.user.row_filters.get(dash.name))
        except QueryError as err:
            result = err.as_dict()
        return self._log("query_metric", args, result)

    # ---- helpers ---------------------------------------------------------------------------------

    def _allowed(self, dashboard: str) -> bool:
        return self.user is not None and dashboard in self.user.dashboards

    def _dashboard(self, name: str) -> Dashboard:
        dash = self.dashboards.get(name) or next((d for d in self.dashboards.values() if d.title.lower() == str(name).lower()), None)
        if dash is None:
            raise QueryError("not_found", f"No dashboard called '{name}'. Use list_dashboards.")
        if not self._allowed(dash.name):
            raise QueryError("denied", f"You don't have access to the {dash.title} dashboard.")
        return dash

    def _scope_text(self, dashboard: str) -> str:
        scope = self.user.row_filters.get(dashboard) if self.user else None
        if not scope:
            return "all rows"
        return "; ".join(f"{dim} in {', '.join(vals)}" for dim, vals in scope.items())

    def _log(self, tool: str, args: dict, result: dict) -> dict:
        if self.audit_path is not None:
            entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "user": self.user_id, "tool": tool,
                     "args": args, "status": result.get("status"), "rows": len(result.get("rows", []))}
            with _audit_lock:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        return result

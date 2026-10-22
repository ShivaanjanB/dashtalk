"""DashTalk as an MCP server: three tools over the semantic layer, bound to one user.

Usage:  dashtalk-server --user analyst_emea      (stdio; for Claude Desktop, Claude Code or dashtalk-ask)
The user is fixed when the server starts. A prompt can ask for anything; the server only returns what
that user may see, and logs every call to logs/audit.jsonl.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from dashtalk.tools import DashTalk

INSTRUCTIONS = (
    "Answer questions from company dashboards. Call list_dashboards, then describe_dashboard to learn the "
    "metric and dimension names, then query_metric. Never state a figure no tool returned. If a tool "
    "returns status 'ambiguous', ask the user which one they mean; if 'denied', tell them they don't have access."
)


def create_server(user_id: str, **talk_options: Any) -> MCPServer:
    talk = DashTalk(user_id, **talk_options)
    server = MCPServer(name="dashtalk", instructions=INSTRUCTIONS)

    @server.tool()
    def list_dashboards() -> dict:
        """List the dashboards you can query, with a one-line description of each."""
        return talk.list_dashboards()

    @server.tool()
    def describe_dashboard(dashboard: str) -> dict:
        """Metrics (with synonyms, units and caveats), dimensions and example questions for one dashboard."""
        return talk.describe_dashboard(dashboard)

    @server.tool()
    def query_metric(dashboard: str, metric: str, filters: dict[str, Any] | None = None,
                     group_by: list[str] | None = None, order: str = "desc", limit: int = 25) -> dict:
        """Get one metric from one dashboard.

        filters maps dimension names to a value or a list of values, e.g. {"country": "UK", "year": 2023}.
        group_by lists dimensions to break the metric down by, e.g. ["region"]. Results are sorted by value
        (order "desc" or "asc"), or by time when grouping only by year, quarter or month.
        Returns status ok with rows, or not_found, ambiguous (with candidates), denied or invalid.
        """
        return talk.query_metric(dashboard, metric, filters, group_by, order, limit)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default=os.environ.get("DASHTALK_USER", ""), help="who is asking (see config/users.yaml)")
    args = parser.parse_args(argv)
    if not args.user:
        parser.error("pass --user or set DASHTALK_USER")
    create_server(args.user).run()


if __name__ == "__main__":
    main()

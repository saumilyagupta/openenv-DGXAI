from __future__ import annotations

import json
import logging
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.audit_report import handle_audit_report
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build
from groundloop.mcp_shell.tools.ground_check import handle_ground_check
from groundloop.mcp_shell.tools.ingest_sources import handle_ingest_sources
from groundloop.mcp_shell.tools.interrogate import handle_interrogate

_log = logging.getLogger(__name__)

TOOL_NAMES = (
    "interrogate",
    "ingest_sources",
    "ground_check",
    "autonomous_build",
    "audit_report",
)

_HANDLERS = {
    "interrogate": handle_interrogate,
    "ingest_sources": handle_ingest_sources,
    "ground_check": handle_ground_check,
    "autonomous_build": handle_autonomous_build,
    "audit_report": handle_audit_report,
}

_TOOL_DESCRIPTIONS = {
    "interrogate": "Return Socratic clarifying questions about a project brief.",
    "ingest_sources": "Build a KB graph from skill sources. Returns a graph_id.",
    "ground_check": "Search a KB graph for evidence grounding a claim; returns verdict + citations.",
    "autonomous_build": "Run the Ralph-orchestrator loop on a spec against a KB graph; returns final score + files.",
    "audit_report": "Return the structured audit report for a previously started run_id.",
}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "interrogate": {
        "type": "object",
        "properties": {
            "brief": {"type": "string", "minLength": 1},
            "graph_id": {"type": ["string", "null"], "default": None},
        },
        "required": ["brief"],
        "additionalProperties": False,
    },
    "ingest_sources": {
        "type": "object",
        "properties": {
            "source_globs": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "default": None,
            }
        },
        "additionalProperties": False,
    },
    "ground_check": {
        "type": "object",
        "properties": {
            "claim": {"type": "string", "minLength": 1},
            "graph_id": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            "required_tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["claim", "graph_id"],
        "additionalProperties": False,
    },
    "autonomous_build": {
        "type": "object",
        "properties": {
            "spec": {"type": "string", "minLength": 1},
            "graph_id": {"type": "string", "minLength": 1},
            "max_iters": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
        },
        "required": ["spec", "graph_id"],
        "additionalProperties": False,
    },
    "audit_report": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "minLength": 1}},
        "required": ["run_id"],
        "additionalProperties": False,
    },
}


def dispatch(name: str, args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    handler = _HANDLERS.get(name)
    if handler is None:
        _log.warning("dispatch: unknown tool %s", name)
        return {"status": "error", "reason": "unknown_tool", "detail": name}
    return handler(args, session)


def build_server() -> tuple[Server, SessionState]:
    server: Server = Server("groundloop")
    session = SessionState()

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=_TOOL_DESCRIPTIONS[name],
                inputSchema=_TOOL_SCHEMAS[name],
            )
            for name in TOOL_NAMES
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        result = dispatch(name, arguments, session)
        return [types.TextContent(type="text", text=json.dumps(result))]

    return server, session


async def _run_server() -> None:
    logging.basicConfig(level=logging.INFO)
    server, _ = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

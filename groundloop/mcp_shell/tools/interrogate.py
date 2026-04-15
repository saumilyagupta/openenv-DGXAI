from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundloop.interrogator.interrogator import Interrogator
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import InterrogateInput


def handle_interrogate(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = InterrogateInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")

    index = session.get_graph(inp.graph_id) if inp.graph_id else None
    result = Interrogator(index).generate(inp.brief)
    return {
        "status": "ok",
        "questions": list(result.questions),
        "cited_node_ids": list(result.cited_node_ids),
    }

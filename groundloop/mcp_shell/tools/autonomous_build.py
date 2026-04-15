from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AutonomousBuildInput


def handle_autonomous_build(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = AutonomousBuildInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")
    record = session.create_run(spec=inp.spec, graph_id=inp.graph_id)
    record.notes.append(
        "ralph-orchestrator (#7) not yet wired in; run is registered but no iteration will occur."
    )
    return {
        "status": "ok",
        "run_id": record.run_id,
        "run_status": record.status,
        "iterations": record.iterations,
        "notes": list(record.notes),
    }

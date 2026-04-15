from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.schemas import AutonomousBuildInput
from groundloop.ralph_orchestrator.loop import run_loop
from groundloop.ralph_orchestrator.models import LoopConfig
from groundloop.ralph_orchestrator.stub_synthesizer import StubSynthesizer

_log = logging.getLogger(__name__)


def handle_autonomous_build(args: dict[str, Any], session: SessionState) -> dict[str, Any]:
    try:
        inp = AutonomousBuildInput(**args)
    except ValidationError as e:
        return {"status": "error", "reason": "invalid_params", "detail": str(e)}
    session.inc("tool_calls")

    index = session.get_graph(inp.graph_id)
    if index is None:
        return {"status": "error", "reason": "unknown_graph_id", "detail": inp.graph_id}

    initial_files = {"main.py": "from __future__ import annotations\n"}
    synth = StubSynthesizer()
    config = LoopConfig(max_iters=inp.max_iters, target_score=0.95)

    try:
        result = run_loop(
            spec=inp.spec,
            initial_files=initial_files,
            index=index,
            synthesizer=synth,
            config=config,
        )
    except Exception as e:  # noqa: BLE001 - MCP handler boundary
        _log.exception("autonomous_build loop failed")
        return {"status": "error", "reason": "loop_failed", "detail": str(e)}

    session.register_run(result)

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_status": result.terminated_by,
        "iterations": len(result.iterations),
        "final_score": result.final_score,
        "final_files": result.final_files,
    }

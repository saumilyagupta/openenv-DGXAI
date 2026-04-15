from __future__ import annotations

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def test_autonomous_build_returns_pending_run():
    s = SessionState()
    out = handle_autonomous_build({"spec": "build an API", "graph_id": "g"}, s)
    assert out["status"] == "ok"
    assert out["run_status"] == "pending_orchestrator"
    assert out["run_id"].startswith("run_")
    assert s.get_run(out["run_id"]) is not None


def test_autonomous_build_invalid_params():
    s = SessionState()
    out = handle_autonomous_build({"spec": "", "graph_id": "g"}, s)
    assert out["status"] == "error"

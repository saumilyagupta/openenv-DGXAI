from __future__ import annotations

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.audit_report import handle_audit_report
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def test_audit_report_known_run():
    s = SessionState()
    run = handle_autonomous_build({"spec": "x", "graph_id": "g"}, s)
    out = handle_audit_report({"run_id": run["run_id"]}, s)
    assert out["status"] == "ok"
    assert out["run_id"] == run["run_id"]
    assert "tool_calls" in out["metrics"]


def test_audit_report_unknown_run():
    s = SessionState()
    out = handle_audit_report({"run_id": "missing"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_run_id"


def test_audit_report_invalid_params():
    s = SessionState()
    out = handle_audit_report({"run_id": ""}, s)
    assert out["status"] == "error"
    assert out["reason"] == "invalid_params"

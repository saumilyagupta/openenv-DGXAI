from __future__ import annotations

from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.audit_report import handle_audit_report
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def test_audit_report_known_run(tiny_corpus_path: Path) -> None:
    s = SessionState()
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    s.register_graph("g", idx)

    run = handle_autonomous_build(
        {"spec": "ship an api", "graph_id": "g", "max_iters": 1}, s
    )
    assert run["status"] == "ok"

    out = handle_audit_report({"run_id": run["run_id"]}, s)
    assert out["status"] == "ok"
    assert out["run_id"] == run["run_id"]
    assert "report" in out
    assert out["report"]["run_id"] == run["run_id"]
    assert "iterations_total" in out["report"]
    assert "tool_calls" in out["metrics"]


def test_audit_report_unknown_run() -> None:
    s = SessionState()
    out = handle_audit_report({"run_id": "missing"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_run_id"


def test_audit_report_invalid_params() -> None:
    s = SessionState()
    out = handle_audit_report({"run_id": ""}, s)
    assert out["status"] == "error"
    assert out["reason"] == "invalid_params"

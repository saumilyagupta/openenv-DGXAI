from __future__ import annotations

from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.autonomous_build import handle_autonomous_build


def _session_with_graph(tiny_corpus_path: Path) -> tuple[SessionState, str]:
    s = SessionState()
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    gid = "g-test"
    s.register_graph(gid, idx)
    return s, gid


def test_autonomous_build_runs_loop(tiny_corpus_path: Path) -> None:
    s, gid = _session_with_graph(tiny_corpus_path)
    out = handle_autonomous_build(
        {"spec": "build greet(name) function", "graph_id": gid, "max_iters": 1}, s
    )
    assert out["status"] == "ok"
    assert out["run_id"].startswith("ralph_")
    assert isinstance(out["iterations"], int)
    assert out["iterations"] >= 0
    assert "final_score" in out
    assert s.get_run_result(out["run_id"]) is not None


def test_autonomous_build_unknown_graph() -> None:
    s = SessionState()
    out = handle_autonomous_build({"spec": "x", "graph_id": "missing"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_graph_id"


def test_autonomous_build_invalid_params() -> None:
    s = SessionState()
    out = handle_autonomous_build({"spec": "", "graph_id": "g"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "invalid_params"

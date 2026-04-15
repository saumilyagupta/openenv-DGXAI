from __future__ import annotations

from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.ground_check import handle_ground_check


def _prepare_session_with_index(tiny_corpus_path: Path) -> tuple[SessionState, str]:
    s = SessionState()
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    graph_id = "graph_test"
    s.register_graph(graph_id, idx)
    return s, graph_id


def test_ground_check_grounded(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check({"claim": "pytest fixtures", "graph_id": gid}, s)
    assert out["status"] == "ok"
    assert out["verdict"] in {"grounded", "uncertain"}
    assert len(out["citations"]) >= 1


def test_ground_check_unknown_graph():
    s = SessionState()
    out = handle_ground_check({"claim": "x", "graph_id": "nope"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "unknown_graph_id"


def test_ground_check_with_tag_filter(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check(
        {"claim": "security", "graph_id": gid, "required_tags": ["domain:security"]}, s
    )
    assert out["status"] == "ok"
    for c in out["citations"]:
        assert "domain:security" in c["tags"]


def test_ground_check_empty_result_is_ungrounded(tiny_corpus_path: Path):
    s, gid = _prepare_session_with_index(tiny_corpus_path)
    out = handle_ground_check({"claim": "!!!", "graph_id": gid}, s)
    assert out["status"] == "ok"
    assert out["verdict"] == "ungrounded"
    assert out["citations"] == []


def test_ground_check_invalid_params():
    s = SessionState()
    out = handle_ground_check({"claim": "", "graph_id": "g"}, s)
    assert out["status"] == "error"
    assert out["reason"] == "invalid_params"

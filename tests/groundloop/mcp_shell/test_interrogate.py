from __future__ import annotations

from pathlib import Path

from groundloop.kb_indexer.index import SkillsIndex
from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.interrogate import handle_interrogate


def test_interrogate_returns_five_questions() -> None:
    s = SessionState()
    out = handle_interrogate({"brief": "build a python REST API"}, s)
    assert out["status"] == "ok"
    assert len(out["questions"]) == 5
    assert out["cited_node_ids"] == []


def test_interrogate_questions_deterministic() -> None:
    s = SessionState()
    a = handle_interrogate({"brief": "build a python REST API"}, s)
    b = handle_interrogate({"brief": "build a python REST API"}, s)
    assert a == b


def test_interrogate_rejects_empty() -> None:
    s = SessionState()
    out = handle_interrogate({"brief": ""}, s)
    assert out.get("status") == "error"
    assert out.get("reason") == "invalid_params"


def test_interrogate_with_graph_id(tiny_corpus_path: Path) -> None:
    s = SessionState()
    idx = SkillsIndex(corpus_path=tiny_corpus_path)
    idx.build()
    s.register_graph("g1", idx)
    out = handle_interrogate(
        {"brief": "pytest fixtures for the api", "graph_id": "g1"}, s
    )
    assert out["status"] == "ok"
    assert len(out["questions"]) == 5
    assert len(out["cited_node_ids"]) >= 1


def test_interrogate_unknown_graph_falls_back() -> None:
    s = SessionState()
    out = handle_interrogate(
        {"brief": "build a thing", "graph_id": "does_not_exist"}, s
    )
    assert out["status"] == "ok"
    assert out["cited_node_ids"] == []

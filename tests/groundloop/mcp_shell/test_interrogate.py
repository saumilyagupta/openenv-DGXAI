from __future__ import annotations

from groundloop.mcp_shell.session import SessionState
from groundloop.mcp_shell.tools.interrogate import handle_interrogate


def test_interrogate_returns_three_questions():
    s = SessionState()
    out = handle_interrogate({"brief": "build a python REST API"}, s)
    assert "questions" in out
    assert len(out["questions"]) == 3


def test_interrogate_questions_deterministic():
    s = SessionState()
    a = handle_interrogate({"brief": "build a python REST API"}, s)
    b = handle_interrogate({"brief": "build a python REST API"}, s)
    assert a == b


def test_interrogate_rejects_empty():
    s = SessionState()
    out = handle_interrogate({"brief": ""}, s)
    assert out.get("status") == "error"

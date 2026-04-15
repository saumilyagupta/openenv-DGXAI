from __future__ import annotations

from groundloop.mcp_shell.server import TOOL_NAMES, dispatch
from groundloop.mcp_shell.session import SessionState


def test_tool_names_are_five_expected():
    assert set(TOOL_NAMES) == {
        "interrogate",
        "ingest_sources",
        "ground_check",
        "autonomous_build",
        "audit_report",
    }


def test_dispatch_unknown_tool():
    out = dispatch("nope", {}, SessionState())
    assert out["status"] == "error"
    assert out["reason"] == "unknown_tool"


def test_dispatch_interrogate_routes_correctly():
    out = dispatch("interrogate", {"brief": "hello"}, SessionState())
    assert out["status"] == "ok"
    assert "questions" in out


def test_build_server_returns_server_and_session():
    from groundloop.mcp_shell.server import build_server
    server, session = build_server()
    assert server is not None
    assert isinstance(session, SessionState)


def test_dispatch_unknown_tool_emits_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="groundloop.mcp_shell.server"):
        out = dispatch("nope", {}, SessionState())
    assert out["status"] == "error"
    assert out["reason"] == "unknown_tool"
    assert any(
        "unknown tool" in r.getMessage() and "nope" in r.getMessage()
        for r in caplog.records
    )

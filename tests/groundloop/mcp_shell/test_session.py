from __future__ import annotations

from groundloop.mcp_shell.session import RunRecord, SessionState


def test_session_state_initial_counts_zero():
    s = SessionState()
    m = s.metrics_snapshot()
    assert m == {"tool_calls": 0, "graphs_built": 0, "ground_checks": 0}


def test_session_state_increments():
    s = SessionState()
    s.inc("tool_calls")
    s.inc("tool_calls")
    s.inc("graphs_built")
    assert s.metrics_snapshot()["tool_calls"] == 2
    assert s.metrics_snapshot()["graphs_built"] == 1


def test_session_state_registers_graph():
    s = SessionState()
    s.register_graph("g1", object())
    assert s.get_graph("g1") is not None


def test_session_state_missing_graph_returns_none():
    s = SessionState()
    assert s.get_graph("missing") is None


def test_session_state_creates_run_record():
    s = SessionState()
    r = s.create_run(spec="build a thing", graph_id="g1")
    assert isinstance(r, RunRecord)
    assert r.spec == "build a thing"
    assert s.get_run(r.run_id) is r


def test_session_state_missing_run_returns_none():
    assert SessionState().get_run("missing") is None

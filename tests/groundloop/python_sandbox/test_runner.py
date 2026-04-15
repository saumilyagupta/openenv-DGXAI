from __future__ import annotations

from pathlib import Path

from groundloop.python_sandbox.runner import run_tool


def test_run_tool_captures_exit_code(tmp_path: Path) -> None:
    r = run_tool("true", ["true"], cwd=tmp_path)
    assert r.exit_code == 0
    assert r.timed_out is False


def test_run_tool_captures_stdout(tmp_path: Path) -> None:
    r = run_tool("echo", ["echo", "hello"], cwd=tmp_path)
    assert "hello" in r.stdout


def test_run_tool_times_out(tmp_path: Path) -> None:
    r = run_tool("sleep", ["sleep", "5"], cwd=tmp_path, timeout=0.2)
    assert r.timed_out is True
    assert r.exit_code != 0


def test_run_tool_binary_missing(tmp_path: Path) -> None:
    r = run_tool("nope", ["definitely_not_installed_xyz123"], cwd=tmp_path)
    assert r.exit_code == -1
    assert "not found" in r.stderr.lower() or "no such file" in r.stderr.lower()

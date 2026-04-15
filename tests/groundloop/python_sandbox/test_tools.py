from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.python_sandbox.models import ToolResult
from groundloop.python_sandbox.tools import DEFAULT_TOOLS, argv_for, parse


def test_default_tools_includes_essentials() -> None:
    assert "ruff" in DEFAULT_TOOLS
    assert "mypy" in DEFAULT_TOOLS
    assert "pytest" in DEFAULT_TOOLS


def test_argv_for_ruff(tmp_path: Path) -> None:
    argv = argv_for("ruff", tmp_path)
    assert argv[0] == "ruff"
    assert "check" in argv


def test_argv_for_mypy(tmp_path: Path) -> None:
    argv = argv_for("mypy", tmp_path)
    assert argv[0] == "mypy"


def test_argv_for_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        argv_for("unknown_tool", tmp_path)


def _tr(name: str, exit_code: int, stdout: str = "", stderr: str = "") -> ToolResult:
    return ToolResult(name=name, argv=(name,), exit_code=exit_code,
                      stdout=stdout, stderr=stderr, duration_ms=0, timed_out=False)


def test_parse_ruff_empty_success() -> None:
    p = parse("ruff", _tr("ruff", 0, "[]"))
    assert p.ok is True
    assert p.count == 0


def test_parse_ruff_violations() -> None:
    p = parse("ruff", _tr("ruff", 1, '[{"code":"E702"},{"code":"F401"}]'))
    assert p.ok is False
    assert p.count == 2


def test_parse_mypy_success() -> None:
    p = parse("mypy", _tr("mypy", 0, "Success: no issues found in 1 source file"))
    assert p.ok is True
    assert p.count == 0


def test_parse_mypy_errors() -> None:
    p = parse("mypy", _tr("mypy", 1, "foo.py:3: error: x\nFound 2 errors in 1 file (checked 1 source file)"))
    assert p.ok is False
    assert p.count == 2


def test_parse_pytest_success() -> None:
    p = parse("pytest", _tr("pytest", 0, "5 passed in 0.5s"))
    assert p.ok is True


def test_parse_pytest_failures() -> None:
    p = parse("pytest", _tr("pytest", 1, "1 failed, 2 passed"))
    assert p.ok is False

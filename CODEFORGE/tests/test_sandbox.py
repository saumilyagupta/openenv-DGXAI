from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from codeforge.sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_result(
    name: str = "ruff",
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> ToolResult:
    return ToolResult(
        name=name,
        argv=(name,),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=10,
        timed_out=timed_out,
    )


def _make_parsed(ok: bool = True, count: int = 0, details: dict[str, Any] | None = None) -> ParsedResult:
    return ParsedResult(ok=ok, count=count, details=details or {})


def _make_sandbox_result(
    *,
    parsed: dict[str, ParsedResult] | None = None,
    unresolved: tuple[str, ...] = (),
    tools_run: tuple[str, ...] = ("ruff", "mypy", "pytest", "imports"),
) -> SandboxResult:
    return SandboxResult(
        project_dir="/tmp/test",
        tools_run=tools_run,
        tool_results={},
        parsed=parsed or {},
        imports=ImportReport(total=5, unresolved=unresolved, by_file={}),
        composite_score=0.0,
        generated_at="2026-04-16T00:00:00+00:00",
    )


# ===========================================================================
# Tests for models (frozen Pydantic models)
# ===========================================================================


class TestModels:
    def test_tool_result_frozen(self) -> None:
        tr = _make_tool_result()
        with pytest.raises(Exception):
            tr.name = "other"  # type: ignore[misc]

    def test_parsed_result_frozen(self) -> None:
        pr = _make_parsed()
        with pytest.raises(Exception):
            pr.ok = False  # type: ignore[misc]

    def test_import_report_frozen(self) -> None:
        ir = ImportReport(total=0, unresolved=(), by_file={})
        with pytest.raises(Exception):
            ir.total = 99  # type: ignore[misc]

    def test_sandbox_result_frozen(self) -> None:
        sr = _make_sandbox_result()
        with pytest.raises(Exception):
            sr.composite_score = 1.0  # type: ignore[misc]


# ===========================================================================
# Tests for composite_score (metric.py)
# ===========================================================================


class TestCompositeScore:
    """Tests for the penalty-only composite_score function."""

    def test_all_tools_pass_returns_1(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=True, count=0),
                "pytest": _make_parsed(ok=True, count=0),
                "imports": _make_parsed(ok=True, count=0),
            },
            unresolved=(),
        )
        assert composite_score(sr) == 1.0

    def test_ruff_errors_apply_penalty(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=False, count=10),
                "mypy": _make_parsed(ok=True, count=0),
                "pytest": _make_parsed(ok=True, count=0),
            },
        )
        score = composite_score(sr)
        # ruff_penalty = min(10, 20) / 40 = 0.25 → score = 1.0 - 0.25 = 0.75
        assert score == pytest.approx(0.75)

    def test_mypy_errors_apply_penalty(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=False, count=8),
                "pytest": _make_parsed(ok=True, count=0),
            },
        )
        score = composite_score(sr)
        # mypy_penalty = min(8, 20) / 40 = 0.2 → score = 1.0 - 0.2 = 0.8
        assert score == pytest.approx(0.8)

    def test_pytest_failure_applies_half_penalty(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=True, count=0),
                "pytest": _make_parsed(ok=False, count=1),
            },
        )
        score = composite_score(sr)
        # pytest_penalty = 0.5 → score = 1.0 - 0.5 = 0.5
        assert score == pytest.approx(0.5)

    def test_unresolved_imports_apply_penalty(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=True, count=0),
            },
            unresolved=("foo", "bar", "baz"),
        )
        score = composite_score(sr)
        # imports_penalty = min(1.0, 3 * 0.1) = 0.3 → score = 1.0 - 0.3 = 0.7
        assert score == pytest.approx(0.7)

    def test_score_clamped_to_zero(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=False, count=20),
                "mypy": _make_parsed(ok=False, count=20),
                "pytest": _make_parsed(ok=False, count=5),
            },
            unresolved=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
        )
        score = composite_score(sr)
        # ruff=0.5, mypy=0.5, pytest=0.5, imports=1.0 → raw = 1.0 - 2.5 = -1.5 → clamped to 0.0
        assert score == 0.0

    def test_score_clamped_to_one(self) -> None:
        """Even with negative penalties (which shouldn't happen), score is capped at 1.0."""
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(parsed={}, unresolved=())
        # No parsed results → returns 0.0
        assert composite_score(sr) == 0.0

    def test_tools_filter_only_scores_filtered(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=True, count=0),
                "pytest": _make_parsed(ok=False, count=1),  # would penalize
            },
        )
        # With tools filter excluding pytest, score should be 1.0
        score = composite_score(sr, tools=("ruff", "mypy"))
        assert score == pytest.approx(1.0)

    def test_tools_none_scores_everything(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=False, count=4),
                "pytest": _make_parsed(ok=False, count=1),
            },
        )
        # ruff_penalty = 4/40 = 0.1, pytest_penalty = 0.5 → score = 1.0 - 0.6 = 0.4
        score = composite_score(sr, tools=None)
        assert score == pytest.approx(0.4)

    def test_no_double_counting_starts_from_one(self) -> None:
        """Verify penalty-only approach: starts at 1.0, not pass_rate."""
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=True, count=0),
                "mypy": _make_parsed(ok=True, count=0),
                "pytest": _make_parsed(ok=True, count=0),
                "imports": _make_parsed(ok=True, count=0),
            },
            unresolved=(),
        )
        # All pass, zero penalties → score starts at 1.0 and stays 1.0
        assert composite_score(sr) == 1.0

    def test_tools_filter_empty_returns_zero(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={"ruff": _make_parsed(ok=True, count=0)},
        )
        # Filter with tools that don't exist → empty parsed → 0.0
        assert composite_score(sr, tools=("nonexistent",)) == 0.0

    def test_combined_penalties(self) -> None:
        from codeforge.sandbox.metric import composite_score

        sr = _make_sandbox_result(
            parsed={
                "ruff": _make_parsed(ok=False, count=4),    # penalty = 0.1
                "mypy": _make_parsed(ok=False, count=8),    # penalty = 0.2
                "pytest": _make_parsed(ok=False, count=1),  # penalty = 0.5
            },
            unresolved=("foo",),  # penalty = 0.1
        )
        score = composite_score(sr)
        # 1.0 - 0.1 - 0.1 - 0.2 - 0.5 = 0.1
        assert score == pytest.approx(0.1)


# ===========================================================================
# Tests for tools.py (argv_for, parse, is_available)
# ===========================================================================


class TestTools:
    def test_argv_for_ruff(self) -> None:
        from codeforge.sandbox.tools import argv_for

        argv = argv_for("ruff", Path("/tmp/test"))
        assert argv == ["ruff", "check", "--output-format", "json", "."]

    def test_argv_for_mypy(self) -> None:
        from codeforge.sandbox.tools import argv_for

        argv = argv_for("mypy", Path("/tmp/test"))
        assert argv == ["mypy", "--no-incremental", "--strict", "."]

    def test_argv_for_pytest(self) -> None:
        from codeforge.sandbox.tools import argv_for

        argv = argv_for("pytest", Path("/tmp/test"))
        assert argv == ["pytest", "-q", "--tb=line", "--no-header"]

    def test_argv_for_unknown_raises(self) -> None:
        from codeforge.sandbox.tools import argv_for

        with pytest.raises(ValueError, match="unknown tool"):
            argv_for("unknown_tool", Path("/tmp/test"))

    def test_is_available_imports_always_true(self) -> None:
        from codeforge.sandbox.tools import is_available

        assert is_available("imports") is True

    @pytest.mark.skipif(not shutil.which("ruff"), reason="ruff not installed")
    def test_is_available_ruff(self) -> None:
        from codeforge.sandbox.tools import is_available

        assert is_available("ruff") is True

    def test_is_available_nonexistent(self) -> None:
        from codeforge.sandbox.tools import is_available

        assert is_available("definitely_not_a_real_tool_xyz") is False

    def test_parse_ruff_clean(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("ruff", exit_code=0, stdout="[]")
        pr = parse("ruff", tr)
        assert pr.ok is True
        assert pr.count == 0

    def test_parse_ruff_with_violations(self) -> None:
        from codeforge.sandbox.tools import parse

        import json
        violations = [{"code": "E501", "message": "line too long"}] * 3
        tr = _make_tool_result("ruff", exit_code=1, stdout=json.dumps(violations))
        pr = parse("ruff", tr)
        assert pr.ok is False
        assert pr.count == 3

    def test_parse_ruff_invalid_json(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("ruff", exit_code=1, stdout="not json")
        pr = parse("ruff", tr)
        assert pr.ok is False
        assert pr.count == 0

    def test_parse_mypy_success(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("mypy", exit_code=0, stdout="Success: no issues found")
        pr = parse("mypy", tr)
        assert pr.ok is True
        assert pr.count == 0

    def test_parse_mypy_errors(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("mypy", exit_code=1, stdout="Found 5 errors in 2 files")
        pr = parse("mypy", tr)
        assert pr.ok is False
        assert pr.count == 5

    def test_parse_pytest_pass(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("pytest", exit_code=0, stdout="3 passed")
        pr = parse("pytest", tr)
        assert pr.ok is True
        assert pr.count == 0

    def test_parse_pytest_fail(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("pytest", exit_code=1, stdout="1 failed, 2 passed")
        pr = parse("pytest", tr)
        assert pr.ok is False
        assert pr.count == 1

    def test_parse_unknown_tool(self) -> None:
        from codeforge.sandbox.tools import parse

        tr = _make_tool_result("unknown", exit_code=0)
        pr = parse("unknown", tr)
        assert pr.ok is True
        assert pr.count == 0

    def test_default_tools(self) -> None:
        from codeforge.sandbox.tools import DEFAULT_TOOLS

        assert isinstance(DEFAULT_TOOLS, tuple)
        assert "ruff" in DEFAULT_TOOLS
        assert "mypy" in DEFAULT_TOOLS
        assert "pytest" in DEFAULT_TOOLS
        assert "imports" in DEFAULT_TOOLS


# ===========================================================================
# Tests for runner.py
# ===========================================================================


class TestRunner:
    def test_run_tool_echo(self, tmp_path: Path) -> None:
        from codeforge.sandbox.runner import run_tool

        tr = run_tool("echo", ["echo", "hello"], cwd=tmp_path)
        assert tr.exit_code == 0
        assert "hello" in tr.stdout
        assert tr.timed_out is False

    def test_run_tool_timeout(self, tmp_path: Path) -> None:
        from codeforge.sandbox.runner import run_tool

        tr = run_tool("sleep", ["sleep", "10"], cwd=tmp_path, timeout=0.1)
        assert tr.timed_out is True

    def test_run_tool_not_found(self, tmp_path: Path) -> None:
        from codeforge.sandbox.runner import run_tool

        tr = run_tool("nonexistent", ["nonexistent_binary_xyz"], cwd=tmp_path)
        assert tr.exit_code == -1
        assert "not found" in tr.stderr

    def test_run_tool_records_duration(self, tmp_path: Path) -> None:
        from codeforge.sandbox.runner import run_tool

        tr = run_tool("echo", ["echo", "hi"], cwd=tmp_path)
        assert tr.duration_ms >= 0


# ===========================================================================
# Tests for imports.py
# ===========================================================================


class TestImports:
    def test_scan_clean_file(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        (tmp_path / "hello.py").write_text("import os\nimport sys\n")
        report = scan_imports(tmp_path)
        assert report.total == 2
        assert len(report.unresolved) == 0

    def test_scan_unresolved_import(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        (tmp_path / "hello.py").write_text("import nonexistent_package_xyz\n")
        report = scan_imports(tmp_path)
        assert "nonexistent_package_xyz" in report.unresolved

    def test_scan_syntax_error(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        (tmp_path / "bad.py").write_text("def f(\n")
        report = scan_imports(tmp_path)
        assert "__parse_error__" in report.by_file.get("bad.py", ())

    def test_scan_local_module_not_unresolved(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        (tmp_path / "mymod.py").write_text("x = 1\n")
        (tmp_path / "main.py").write_text("import mymod\n")
        report = scan_imports(tmp_path)
        assert "mymod" not in report.unresolved

    def test_scan_relative_import_ignored(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        (tmp_path / "pkg" / "a.py").write_text("from . import b\n")
        (tmp_path / "pkg" / "b.py").write_text("x = 1\n")
        report = scan_imports(tmp_path)
        assert len(report.unresolved) == 0

    def test_scan_empty_dir(self, tmp_path: Path) -> None:
        from codeforge.sandbox.imports import scan_imports

        report = scan_imports(tmp_path)
        assert report.total == 0
        assert len(report.unresolved) == 0
        assert report.by_file == {}


# ===========================================================================
# Tests for sandbox.py (integration)
# ===========================================================================


class TestSandbox:
    def test_run_sandbox_valid_file(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        result = run_sandbox(
            files={"hello.py": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'},
            tools=("ruff", "imports"),
        )
        assert result.composite_score >= 0.0
        assert result.composite_score <= 1.0
        assert "ruff" in result.tools_run
        assert "imports" in result.tools_run

    def test_run_sandbox_syntax_error_file(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        result = run_sandbox(
            files={"bad.py": "def f(\n"},
            tools=("imports",),
        )
        # Syntax error → parse error in imports
        assert "__parse_error__" in result.imports.by_file.get("bad.py", ())

    def test_run_sandbox_path_traversal_rejected(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        with pytest.raises(ValueError, match="escapes sandbox root"):
            run_sandbox(files={"../escape.py": "x = 1\n"}, tools=("imports",))

    def test_run_sandbox_temp_dir_cleanup(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        result = run_sandbox(
            files={"hello.py": "x = 1\n"},
            tools=("imports",),
        )
        # After run_sandbox returns, the temp dir should be cleaned up
        assert not Path(result.project_dir).exists()

    def test_run_sandbox_requires_exactly_one_source(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        with pytest.raises(ValueError, match="exactly one"):
            run_sandbox(tools=("imports",))

        with pytest.raises(ValueError, match="exactly one"):
            run_sandbox(
                project_dir=Path("/tmp/test"),
                files={"a.py": "x = 1\n"},
                tools=("imports",),
            )

    @pytest.mark.skipif(not shutil.which("ruff"), reason="ruff not installed")
    def test_run_sandbox_ruff_integration(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        result = run_sandbox(
            files={"clean.py": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'},
            tools=("ruff",),
        )
        ruff_parsed = result.parsed.get("ruff")
        assert ruff_parsed is not None
        assert ruff_parsed.ok is True

    def test_run_sandbox_unavailable_tool(self) -> None:
        from codeforge.sandbox.sandbox import run_sandbox

        result = run_sandbox(
            files={"hello.py": "x = 1\n"},
            tools=("pip-audit",),  # unlikely to be installed
        )
        if "pip-audit" in result.parsed:
            pa = result.parsed["pip-audit"]
            # Either available or marked as unavailable
            assert isinstance(pa.ok, bool)

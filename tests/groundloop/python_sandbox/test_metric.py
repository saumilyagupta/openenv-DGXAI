from __future__ import annotations

from groundloop.python_sandbox.metric import composite_score
from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)


def _sr(parsed: dict[str, ParsedResult], unresolved: tuple[str, ...] = ()) -> SandboxResult:
    return SandboxResult(
        project_dir="/x",
        tools_run=tuple(parsed.keys()),
        tool_results={
            k: ToolResult(
                name=k, argv=(k,), exit_code=0 if v.ok else 1,
                stdout="", stderr="", duration_ms=0, timed_out=False,
            )
            for k, v in parsed.items()
        },
        parsed=parsed,
        imports=ImportReport(total=0, unresolved=unresolved, by_file={}),
        composite_score=0.0,
        generated_at="t",
    )


def test_all_pass_score_1() -> None:
    parsed = {
        "ruff": ParsedResult(ok=True, count=0, details={}),
        "mypy": ParsedResult(ok=True, count=0, details={}),
        "pytest": ParsedResult(ok=True, count=0, details={}),
    }
    assert composite_score(_sr(parsed)) == 1.0


def test_pytest_fail_halves_score() -> None:
    parsed = {
        "ruff": ParsedResult(ok=True, count=0, details={}),
        "mypy": ParsedResult(ok=True, count=0, details={}),
        "pytest": ParsedResult(ok=False, count=1, details={}),
    }
    s = composite_score(_sr(parsed))
    assert 0.1 <= s <= 0.3


def test_unresolved_imports_penalize() -> None:
    parsed = {"ruff": ParsedResult(ok=True, count=0, details={})}
    s = composite_score(_sr(parsed, unresolved=("foo", "bar")))
    assert 0.7 <= s <= 0.9


def test_score_clamped_to_zero() -> None:
    parsed = {
        "ruff": ParsedResult(ok=False, count=100, details={}),
        "mypy": ParsedResult(ok=False, count=100, details={}),
        "pytest": ParsedResult(ok=False, count=1, details={}),
    }
    s = composite_score(
        _sr(parsed, unresolved=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")),
    )
    assert s == 0.0

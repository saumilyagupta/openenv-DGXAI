from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)


def test_tool_result_frozen() -> None:
    r = ToolResult(name="ruff", argv=("ruff", "check"), exit_code=0,
                   stdout="", stderr="", duration_ms=10, timed_out=False)
    with pytest.raises(ValidationError):
        r.exit_code = 1  # type: ignore[misc]


def test_parsed_result_shape() -> None:
    p = ParsedResult(ok=True, count=0, details={})
    assert p.ok is True


def test_import_report_defaults() -> None:
    ir = ImportReport(total=0, unresolved=(), by_file={})
    assert ir.total == 0


def test_sandbox_result_requires_fields() -> None:
    with pytest.raises(ValidationError):
        SandboxResult()  # type: ignore[call-arg]

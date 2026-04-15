from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from groundloop.python_sandbox.imports import scan_imports
from groundloop.python_sandbox.metric import composite_score
from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)
from groundloop.python_sandbox.runner import run_tool
from groundloop.python_sandbox.tools import (
    DEFAULT_TOOLS,
    argv_for,
    is_available,
    parse,
)


def run_sandbox(
    *,
    project_dir: Path | None = None,
    files: dict[str, str] | None = None,
    tools: Iterable[str] = DEFAULT_TOOLS,
    timeout_per_tool: float = 60.0,
) -> SandboxResult:
    if (project_dir is None) == (files is None):
        msg = "exactly one of project_dir / files must be set"
        raise ValueError(msg)

    tmp_root: Path | None = None
    try:
        if files is not None:
            tmp_root = Path(tempfile.mkdtemp(prefix="groundloop_sandbox_"))
            for name, content in files.items():
                target = tmp_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            project_dir = tmp_root

        assert project_dir is not None  # noqa: S101
        tool_list = tuple(tools)
        tool_results: dict[str, ToolResult] = {}
        parsed_results: dict[str, ParsedResult] = {}
        imports_report: ImportReport | None = None

        for name in tool_list:
            if name == "imports":
                imports_report = scan_imports(project_dir)
                parsed_results[name] = ParsedResult(
                    ok=len(imports_report.unresolved) == 0,
                    count=len(imports_report.unresolved),
                    details={"unresolved": list(imports_report.unresolved)},
                )
                continue
            if not is_available(name):
                tool_results[name] = ToolResult(
                    name=name, argv=(name,), exit_code=-1,
                    stdout="", stderr="binary not found",
                    duration_ms=0, timed_out=False,
                )
                parsed_results[name] = ParsedResult(
                    ok=False, count=0, details={"unavailable": True},
                )
                continue
            argv = argv_for(name, project_dir)
            tr = run_tool(name, argv, cwd=project_dir, timeout=timeout_per_tool)
            tool_results[name] = tr
            parsed_results[name] = parse(name, tr)

        if imports_report is None:
            imports_report = ImportReport(total=0, unresolved=(), by_file={})

        result = SandboxResult(
            project_dir=str(project_dir),
            tools_run=tool_list,
            tool_results=tool_results,
            parsed=parsed_results,
            imports=imports_report,
            composite_score=0.0,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        score = composite_score(result)
        return result.model_copy(update={"composite_score": score})
    finally:
        if tmp_root is not None and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

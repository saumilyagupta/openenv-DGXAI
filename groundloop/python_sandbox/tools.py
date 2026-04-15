from __future__ import annotations

import json
import re
import shutil
from typing import TYPE_CHECKING

from groundloop.python_sandbox.models import ParsedResult, ToolResult

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_TOOLS: tuple[str, ...] = ("ruff", "mypy", "pytest", "imports")

_MYPY_ERROR_RE = re.compile(r"Found (\d+) errors?")


def argv_for(name: str, project_dir: Path) -> list[str]:
    # Spec §4.2: argv is cwd-relative. Runner sets cwd=project_dir; tools
    # target "." (ruff/mypy) or rely on cwd (pytest). Prevents path double-
    # expansion (which produced spurious ruff E902 when project_dir was
    # also used as cwd).
    del project_dir
    if name == "ruff":
        return ["ruff", "check", "--output-format", "json", "."]
    if name == "mypy":
        return ["mypy", "--no-incremental", "--strict", "."]
    if name == "pytest":
        return ["pytest", "-q", "--tb=line", "--no-header"]
    if name == "pip-audit":
        return ["pip-audit", "--format", "json"]
    msg = f"unknown tool: {name}"
    raise ValueError(msg)


def parse(name: str, tool_result: ToolResult) -> ParsedResult:
    if name == "ruff":
        return _parse_ruff(tool_result)
    if name == "mypy":
        return _parse_mypy(tool_result)
    if name == "pytest":
        return _parse_pytest(tool_result)
    if name == "pip-audit":
        return _parse_pip_audit(tool_result)
    return ParsedResult(ok=tool_result.exit_code == 0, count=0, details={})


def _parse_ruff(tr: ToolResult) -> ParsedResult:
    try:
        items = json.loads(tr.stdout or "[]")
    except json.JSONDecodeError:
        return ParsedResult(ok=False, count=0, details={"parse_error": tr.stdout[:500]})
    count = len(items) if isinstance(items, list) else 0
    return ParsedResult(
        ok=count == 0 and tr.exit_code == 0,
        count=count,
        details={"violations": items[:20]},
    )


def _parse_mypy(tr: ToolResult) -> ParsedResult:
    if tr.exit_code == 0 and "Success" in tr.stdout:
        return ParsedResult(ok=True, count=0, details={})
    m = _MYPY_ERROR_RE.search(tr.stdout)
    count = int(m.group(1)) if m else 0
    return ParsedResult(ok=False, count=count, details={"tail": tr.stdout[-500:]})


def _parse_pytest(tr: ToolResult) -> ParsedResult:
    ok = tr.exit_code == 0
    return ParsedResult(ok=ok, count=0 if ok else 1, details={"tail": tr.stdout[-500:]})


def _parse_pip_audit(tr: ToolResult) -> ParsedResult:
    if tr.exit_code == 0:
        return ParsedResult(ok=True, count=0, details={})
    try:
        data = json.loads(tr.stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    count = len(data.get("vulnerabilities", []))
    return ParsedResult(ok=count == 0, count=count, details={"tail": tr.stdout[-500:]})


def is_available(name: str) -> bool:
    if name == "imports":
        return True
    return shutil.which(name) is not None

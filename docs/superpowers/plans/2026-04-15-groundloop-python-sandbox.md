# GroundLoop Python-Sandbox Implementation Plan

> Use superpowers:subagent-driven-development.

**Goal:** Run `ruff`/`mypy`/`pytest` + AST import probe on candidate Python code; return structured `SandboxResult` + composite_score for the Ralph loop.

**Spec:** `docs/superpowers/specs/2026-04-15-groundloop-python-sandbox-design.md`.

---

## File Structure

```
groundloop/python_sandbox/
  __init__.py            # re-exports SandboxResult, run_sandbox
  __main__.py
  models.py              # ToolResult, ParsedResult, ImportReport, SandboxResult
  runner.py              # run_tool(name, argv, *, cwd, timeout) -> ToolResult
  tools.py               # argv_for, parse, DEFAULT_TOOLS
  imports.py             # scan_imports(project_dir) -> ImportReport
  metric.py              # composite_score(SandboxResult) -> float
  sandbox.py             # run_sandbox(*, project_dir|files, tools, ...) -> SandboxResult
  cli.py                 # python -m groundloop.python_sandbox <path>
tests/groundloop/python_sandbox/
  fixtures/
    clean_project/
      main.py
      test_main.py
      pyproject.toml
    broken_project/
      main.py
      test_broken.py
  conftest.py
  test_models.py
  test_runner.py
  test_tools.py
  test_imports.py
  test_metric.py
  test_sandbox.py
  test_cli.py
  test_e2e.py
```

---

## Task 1: Scaffold + fixtures

- [ ] Create package + test directories. Empty `__init__.py` files.
- [ ] Create `tests/groundloop/python_sandbox/fixtures/clean_project/main.py`:
```python
from __future__ import annotations


def add(a: int, b: int) -> int:
    return a + b
```
- [ ] `tests/groundloop/python_sandbox/fixtures/clean_project/test_main.py`:
```python
from __future__ import annotations

from main import add


def test_add() -> None:
    assert add(2, 3) == 5
```
- [ ] `tests/groundloop/python_sandbox/fixtures/clean_project/pyproject.toml`:
```toml
[tool.ruff]
line-length = 100

[tool.mypy]
strict = true
```
- [ ] `tests/groundloop/python_sandbox/fixtures/broken_project/main.py`:
```python
# ruff-ignored garbage + type error + bad import
import nonexistent_zzz  # unresolved

def add(a, b):  # missing type hints -> mypy --strict flags this
    x = 1;  # ruff E702
    return a + b + nonexistent_zzz.unknown()
```
- [ ] `tests/groundloop/python_sandbox/fixtures/broken_project/test_broken.py`:
```python
def test_fails() -> None:
    assert False
```
- [ ] `conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def clean_project(tmp_path: Path) -> Path:
    src = Path(__file__).parent / "fixtures" / "clean_project"
    dst = tmp_path / "clean_project"
    import shutil
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def broken_project(tmp_path: Path) -> Path:
    src = Path(__file__).parent / "fixtures" / "broken_project"
    dst = tmp_path / "broken_project"
    import shutil
    shutil.copytree(src, dst)
    return dst
```
- [ ] Commit: `chore: scaffold groundloop/python_sandbox package + fixtures`

---

## Task 2: Models

- [ ] Write failing tests in `test_models.py`:
```python
import pytest
from pydantic import ValidationError

from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)


def test_tool_result_frozen():
    r = ToolResult(name="ruff", argv=("ruff", "check"), exit_code=0,
                   stdout="", stderr="", duration_ms=10, timed_out=False)
    with pytest.raises(ValidationError):
        r.exit_code = 1  # type: ignore[misc]


def test_parsed_result_shape():
    p = ParsedResult(ok=True, count=0, details={})
    assert p.ok is True


def test_import_report_defaults():
    ir = ImportReport(total=0, unresolved=(), by_file={})
    assert ir.total == 0


def test_sandbox_result_requires_fields():
    with pytest.raises(ValidationError):
        SandboxResult()  # type: ignore[call-arg]
```
- [ ] Implement `models.py`:
```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool


class ParsedResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ok: bool
    count: int
    details: dict[str, Any]


class ImportReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    total: int
    unresolved: tuple[str, ...]
    by_file: dict[str, tuple[str, ...]]


class SandboxResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    project_dir: str
    tools_run: tuple[str, ...]
    tool_results: dict[str, ToolResult]
    parsed: dict[str, ParsedResult]
    imports: ImportReport
    composite_score: float
    generated_at: str
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): Pydantic models`

---

## Task 3: Runner (subprocess abstraction)

- [ ] Write failing tests in `test_runner.py`:
```python
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
```
- [ ] Implement `runner.py`:
```python
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from groundloop.python_sandbox.models import ToolResult


def run_tool(
    name: str,
    argv: list[str],
    *,
    cwd: Path,
    timeout: float = 60.0,
    env_overrides: dict[str, str] | None = None,
) -> ToolResult:
    t0 = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = -1

    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=None if env_overrides is None else {**_os_env(), **env_overrides},
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    except (FileNotFoundError, OSError) as e:
        stderr = f"binary not found: {e}"

    duration_ms = int((time.monotonic() - t0) * 1000)
    return ToolResult(
        name=name,
        argv=tuple(argv),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


def _os_env() -> dict[str, str]:
    import os
    return dict(os.environ)
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): subprocess runner with timeout + missing-binary handling`

---

## Task 4: Tool argv builders + parsers

- [ ] Write failing tests in `test_tools.py`:
```python
from pathlib import Path

from groundloop.python_sandbox.models import ToolResult
from groundloop.python_sandbox.tools import DEFAULT_TOOLS, argv_for, parse


def test_default_tools_includes_essentials():
    assert "ruff" in DEFAULT_TOOLS
    assert "mypy" in DEFAULT_TOOLS
    assert "pytest" in DEFAULT_TOOLS


def test_argv_for_ruff(tmp_path: Path):
    argv = argv_for("ruff", tmp_path)
    assert argv[0] == "ruff"
    assert "check" in argv


def test_argv_for_mypy(tmp_path: Path):
    argv = argv_for("mypy", tmp_path)
    assert argv[0] == "mypy"


def test_argv_for_unknown_raises(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError):
        argv_for("unknown_tool", tmp_path)


def _tr(name: str, exit_code: int, stdout: str = "", stderr: str = "") -> ToolResult:
    return ToolResult(name=name, argv=(name,), exit_code=exit_code,
                      stdout=stdout, stderr=stderr, duration_ms=0, timed_out=False)


def test_parse_ruff_empty_success():
    p = parse("ruff", _tr("ruff", 0, "[]"))
    assert p.ok is True
    assert p.count == 0


def test_parse_ruff_violations():
    p = parse("ruff", _tr("ruff", 1, '[{"code":"E702"},{"code":"F401"}]'))
    assert p.ok is False
    assert p.count == 2


def test_parse_mypy_success():
    p = parse("mypy", _tr("mypy", 0, "Success: no issues found in 1 source file"))
    assert p.ok is True
    assert p.count == 0


def test_parse_mypy_errors():
    p = parse("mypy", _tr("mypy", 1, "foo.py:3: error: x\nFound 2 errors in 1 file (checked 1 source file)"))
    assert p.ok is False
    assert p.count == 2


def test_parse_pytest_success():
    p = parse("pytest", _tr("pytest", 0, "5 passed in 0.5s"))
    assert p.ok is True


def test_parse_pytest_failures():
    p = parse("pytest", _tr("pytest", 1, "1 failed, 2 passed"))
    assert p.ok is False
```
- [ ] Implement `tools.py`:
```python
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from groundloop.python_sandbox.models import ParsedResult, ToolResult

DEFAULT_TOOLS: tuple[str, ...] = ("ruff", "mypy", "pytest", "imports")

_MYPY_ERROR_RE = re.compile(r"Found (\d+) errors?")


def argv_for(name: str, project_dir: Path) -> list[str]:
    if name == "ruff":
        return ["ruff", "check", "--output-format", "json", str(project_dir)]
    if name == "mypy":
        return ["mypy", "--no-incremental", "--strict", str(project_dir)]
    if name == "pytest":
        return ["pytest", "-q", "--tb=line", "--no-header", str(project_dir)]
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
    return ParsedResult(ok=count == 0 and tr.exit_code == 0, count=count, details={"violations": items[:20]})


def _parse_mypy(tr: ToolResult) -> ParsedResult:
    if tr.exit_code == 0 and "Success" in tr.stdout:
        return ParsedResult(ok=True, count=0, details={})
    m = _MYPY_ERROR_RE.search(tr.stdout)
    count = int(m.group(1)) if m else 0
    return ParsedResult(ok=False, count=count, details={"tail": tr.stdout[-500:]})


def _parse_pytest(tr: ToolResult) -> ParsedResult:
    ok = tr.exit_code == 0
    # Exit codes: 0=ok, 1=failed, 2=interrupt, 5=no tests collected
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
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): argv builders + parsers for ruff/mypy/pytest/pip-audit`

---

## Task 5: Import scanner

- [ ] Write failing tests in `test_imports.py`:
```python
from __future__ import annotations

from pathlib import Path

from groundloop.python_sandbox.imports import scan_imports


def test_clean_project_has_no_unresolved(clean_project: Path):
    r = scan_imports(clean_project)
    assert r.unresolved == ()
    assert r.total >= 1


def test_broken_project_flags_nonexistent(broken_project: Path):
    r = scan_imports(broken_project)
    assert "nonexistent_zzz" in r.unresolved


def test_scan_empty_dir(tmp_path: Path):
    r = scan_imports(tmp_path)
    assert r.total == 0
    assert r.unresolved == ()


def test_scan_ignores_relative_imports(tmp_path: Path):
    (tmp_path / "a.py").write_text("from . import b\n")
    r = scan_imports(tmp_path)
    assert r.unresolved == ()
```
- [ ] Implement `imports.py`:
```python
from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path

from groundloop.python_sandbox.models import ImportReport

_log = logging.getLogger(__name__)


def _top_level(name: str) -> str:
    return name.split(".")[0]


def _extract_imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(_top_level(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            out.add(_top_level(node.module))
    return out


def scan_imports(project_dir: Path) -> ImportReport:
    by_file: dict[str, tuple[str, ...]] = {}
    all_pkgs: set[str] = set()
    total = 0

    for py in sorted(project_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as e:
            _log.warning("imports: parse error %s: %s", py, e)
            by_file[str(py.relative_to(project_dir))] = ("__parse_error__",)
            continue
        pkgs = _extract_imports(tree)
        total += len(pkgs)
        by_file[str(py.relative_to(project_dir))] = tuple(sorted(pkgs))
        all_pkgs.update(pkgs)

    unresolved = tuple(
        sorted(p for p in all_pkgs if importlib.util.find_spec(p) is None)
    )
    return ImportReport(total=total, unresolved=unresolved, by_file=by_file)
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): AST-based import scanner`

---

## Task 6: Metric

- [ ] Write failing tests in `test_metric.py`:
```python
from groundloop.python_sandbox.metric import composite_score
from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)


def _sr(parsed: dict, unresolved: tuple = ()) -> SandboxResult:
    return SandboxResult(
        project_dir="/x",
        tools_run=tuple(parsed.keys()),
        tool_results={k: ToolResult(name=k, argv=(k,), exit_code=0 if v.ok else 1,
                                     stdout="", stderr="", duration_ms=0, timed_out=False)
                      for k, v in parsed.items()},
        parsed=parsed,
        imports=ImportReport(total=0, unresolved=unresolved, by_file={}),
        composite_score=0.0,
        generated_at="t",
    )


def test_all_pass_score_1():
    parsed = {
        "ruff": ParsedResult(ok=True, count=0, details={}),
        "mypy": ParsedResult(ok=True, count=0, details={}),
        "pytest": ParsedResult(ok=True, count=0, details={}),
    }
    assert composite_score(_sr(parsed)) == 1.0


def test_pytest_fail_halves_score():
    parsed = {
        "ruff": ParsedResult(ok=True, count=0, details={}),
        "mypy": ParsedResult(ok=True, count=0, details={}),
        "pytest": ParsedResult(ok=False, count=1, details={}),
    }
    # 2/3 pass rate - 0.5 pytest penalty = 0.167
    s = composite_score(_sr(parsed))
    assert 0.1 <= s <= 0.3


def test_unresolved_imports_penalize():
    parsed = {"ruff": ParsedResult(ok=True, count=0, details={})}
    s = composite_score(_sr(parsed, unresolved=("foo", "bar")))
    # 1.0 - 0.2 = 0.8
    assert 0.7 <= s <= 0.9


def test_score_clamped_to_zero():
    parsed = {
        "ruff": ParsedResult(ok=False, count=100, details={}),
        "mypy": ParsedResult(ok=False, count=100, details={}),
        "pytest": ParsedResult(ok=False, count=1, details={}),
    }
    s = composite_score(_sr(parsed, unresolved=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")))
    assert s == 0.0
```
- [ ] Implement `metric.py`:
```python
from __future__ import annotations

from groundloop.python_sandbox.models import SandboxResult


def composite_score(result: SandboxResult) -> float:
    parsed = result.parsed
    if not parsed:
        return 0.0
    pass_rate = sum(1 for p in parsed.values() if p.ok) / len(parsed)

    imports_penalty = min(1.0, len(result.imports.unresolved) * 0.1)
    ruff = parsed.get("ruff")
    mypy = parsed.get("mypy")
    pytest = parsed.get("pytest")

    ruff_penalty = min(ruff.count, 20) / 40 if ruff else 0.0
    mypy_penalty = min(mypy.count, 20) / 40 if mypy else 0.0
    pytest_penalty = 0.5 if pytest and not pytest.ok else 0.0

    raw = pass_rate - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
    return max(0.0, min(1.0, raw))
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): composite score metric`

---

## Task 7: Sandbox orchestration

- [ ] Write failing tests in `test_sandbox.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from groundloop.python_sandbox.sandbox import run_sandbox


def test_sandbox_clean_project_scores_high(clean_project: Path):
    r = run_sandbox(project_dir=clean_project, tools=("ruff", "imports"))
    assert r.composite_score >= 0.9
    assert r.imports.unresolved == ()


def test_sandbox_broken_project_scores_low(broken_project: Path):
    r = run_sandbox(project_dir=broken_project)
    assert r.composite_score <= 0.5
    assert "nonexistent_zzz" in r.imports.unresolved


def test_sandbox_files_dict_creates_tmp(tmp_path: Path):
    files = {"main.py": "x = 1\n"}
    r = run_sandbox(files=files, tools=("imports",))
    assert r.imports.total == 0


def test_sandbox_requires_one_of_inputs():
    with pytest.raises(ValueError):
        run_sandbox()
```
- [ ] Implement `sandbox.py`:
```python
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
                parsed_results[name] = ParsedResult(ok=False, count=0, details={"unavailable": True})
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
```
- [ ] Run: PASS.
- [ ] Commit: `feat(python-sandbox): run_sandbox orchestrator`

---

## Task 8: CLI + e2e + public API

- [ ] Write failing tests in `test_cli.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

from groundloop.python_sandbox.cli import main


def test_cli_json_output(clean_project: Path, capsys):
    rc = main([str(clean_project), "--tool", "imports", "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "composite_score" in data


def test_cli_missing_dir(tmp_path: Path):
    rc = main([str(tmp_path / "nonexistent"), "--format", "json"])
    assert rc == 1
```
- [ ] Write failing test `test_e2e.py`:
```python
from __future__ import annotations

from groundloop.python_sandbox import run_sandbox


def test_e2e_files_dict_end_to_end():
    files = {"main.py": "from __future__ import annotations\n\ndef f() -> int:\n    return 1\n"}
    r = run_sandbox(files=files, tools=("imports",))
    assert r.composite_score >= 0.9
```
- [ ] Implement `cli.py`:
```python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from groundloop.python_sandbox.sandbox import run_sandbox
from groundloop.python_sandbox.tools import DEFAULT_TOOLS


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="groundloop.python_sandbox")
    p.add_argument("project_dir", type=Path)
    p.add_argument("--tool", action="append", default=None)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv or sys.argv[1:])
    if not args.project_dir.is_dir():
        print(f"ERROR: project_dir not found: {args.project_dir}", file=sys.stderr)
        return 1
    tools = tuple(args.tool) if args.tool else DEFAULT_TOOLS
    result = run_sandbox(project_dir=args.project_dir, tools=tools, timeout_per_tool=args.timeout)
    if args.format == "json":
        print(result.model_dump_json())
    else:
        print(f"composite_score={result.composite_score:.3f}")
        for name, p in result.parsed.items():
            print(f"  {name}: ok={p.ok} count={p.count}")
        if result.imports.unresolved:
            print(f"  unresolved imports: {list(result.imports.unresolved)}")
    return 0
```
- [ ] `__main__.py`:
```python
from groundloop.python_sandbox.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] `__init__.py`:
```python
from __future__ import annotations

from groundloop.python_sandbox.models import (
    ImportReport,
    ParsedResult,
    SandboxResult,
    ToolResult,
)
from groundloop.python_sandbox.sandbox import run_sandbox

__all__ = ["ImportReport", "ParsedResult", "SandboxResult", "ToolResult", "run_sandbox"]
```
- [ ] Run full suite: `python3 -m pytest tests/groundloop/python_sandbox/ -v --cov=groundloop.python_sandbox --cov-report=term` — expect ≥ 85% coverage, all pass.
- [ ] `ruff check groundloop/python_sandbox/` — expect clean.
- [ ] `mypy --strict groundloop/python_sandbox/` — expect clean.
- [ ] Smoke test the CLI: `python3 -m groundloop.python_sandbox tests/groundloop/python_sandbox/fixtures/clean_project --format json --tool imports` — expect JSON with `"composite_score": 1.0`.
- [ ] Append a `### Python Sandbox` subsection to `README.md`.
- [ ] Commit: `feat(python-sandbox): CLI + public API; coverage + linters clean`

---

## Self-Review

- ✅ Every spec §4 component has an implementing task.
- ✅ Acceptance criteria §7.1–§7.8 covered by sandbox + cli tests + final smoke.
- ✅ No placeholders.
- ✅ Type consistency across modules (ToolResult field order identical in models.py and runner.py; ParsedResult shape consistent across parsers).

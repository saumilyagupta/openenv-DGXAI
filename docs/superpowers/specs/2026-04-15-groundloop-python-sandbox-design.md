# GroundLoop Python-Sandbox — Design Spec

**Date:** 2026-04-15
**Sub-project:** #6 of 8
**Depends on:** nothing (pure utility)
**Consumed by:** ralph-orchestrator (#7)

---

## 1. Purpose

Run verification tools (`ruff`, `mypy`, `pytest`, `pip-audit`, `python -c "import x"` import probe) on a candidate Python project directory and return a structured `SandboxResult`. The Ralph loop uses this as the per-iteration signal — analogous to Karpathy's Autoresearch `val_bpb` metric.

## 2. Scope

**In scope:**

- Accept a project dir path OR a `{filename: content}` dict (write to tmp dir).
- Run a configurable list of tools per invocation.
- Capture stdout/stderr/exit-code per tool.
- Parse structured output where cheap: ruff JSON, mypy `--error-summary`, pytest `--tb=line` + exit code, pip-audit JSON.
- Run with a per-tool timeout (default 60s).
- Resolve imports via AST-level analysis (no execution): parse all `.py` files, collect `import`/`from` statements, confirm each top-level package is either in stdlib or resolvable in the sandbox's Python (`importlib.util.find_spec`).
- Return a deterministic `SandboxResult` with per-tool sub-results + an aggregate `composite_score` (0.0 to 1.0) the orchestrator can use as the iteration metric.

**Out of scope:**

- Network isolation / firejail / Docker — local tools only for the MVP. Trust model: orchestrator-controlled input.
- Actually running the candidate code (only linters + tests via pytest are invoked).
- Auto-fixing — we only report.
- Fuzz / mutation testing.

## 3. Architecture

```
groundloop/python_sandbox/
  __init__.py
  __main__.py
  runner.py        # run_tool(name, cwd, timeout) -> ToolResult
  tools.py         # DEFAULT_TOOLS, tool-specific arg builders + parsers
  imports.py       # scan_imports(project_dir) -> ImportReport
  metric.py        # composite_score(SandboxResult) -> float
  models.py        # ToolResult, SandboxResult, ImportReport (Pydantic v2 frozen)
  sandbox.py       # run_sandbox(project_dir|files, tools, ...) -> SandboxResult
  cli.py           # python -m groundloop.python_sandbox <path>
tests/groundloop/python_sandbox/
  fixtures/
    clean_project/        # a project that passes everything
    broken_project/       # ruff + mypy failures, missing import
  conftest.py
  test_runner.py
  test_tools.py
  test_imports.py
  test_metric.py
  test_sandbox.py
  test_cli.py
  test_e2e.py
```

Each module ≤ 200 lines.

## 4. Component Contracts

### 4.1 `runner.py`

```python
def run_tool(
    name: str,
    argv: list[str],
    *,
    cwd: Path,
    timeout: float = 60.0,
    env_overrides: dict[str, str] | None = None,
) -> ToolResult: ...
```

Uses `subprocess.run` with `capture_output=True`, `text=True`, `timeout=timeout`, `cwd=cwd`. No `shell=True`. Returns `ToolResult(name, argv, exit_code, stdout, stderr, duration_ms, timed_out)`.

### 4.2 `tools.py`

```python
DEFAULT_TOOLS: tuple[str, ...] = ("ruff", "mypy", "pytest", "imports")

def argv_for(name: str, project_dir: Path) -> list[str]: ...
def parse(name: str, tool_result: ToolResult) -> ParsedResult: ...
```

- `ruff`: `["ruff", "check", "--output-format", "json", "."]`
- `mypy`: `["mypy", "--no-incremental", "--strict", "."]`
- `pytest`: `["pytest", "-q", "--tb=line", "--no-header"]`
- `imports`: handled specially (no subprocess — calls `scan_imports`).
- `pip-audit` (optional): `["pip-audit", "--format", "json"]` — only included if in tool list.

`ParsedResult` has `{ok: bool, count: int, details: dict[str, Any]}` — e.g., ruff: `count = number of violations`, `details = {"violations": [...]}`.

### 4.3 `imports.py`

```python
@dataclass(frozen=True)
class ImportReport:
    total: int
    unresolved: tuple[str, ...]     # top-level package names that failed find_spec
    by_file: dict[str, tuple[str, ...]]

def scan_imports(project_dir: Path) -> ImportReport: ...
```

Parse every `*.py` under `project_dir` via `ast`. Extract top-level package (e.g. `from foo.bar import baz` → `foo`). Skip relative imports (level > 0). For each unique package, call `importlib.util.find_spec(package)`. If `None` → unresolved.

### 4.4 `metric.py`

```python
def composite_score(result: SandboxResult) -> float: ...
```

Deterministic formula in [0.0, 1.0]:

```
tool_pass_rate = (# tools with ok=True) / (# tools run)
imports_penalty = len(imports.unresolved) * 0.1  (capped at 1.0)
ruff_penalty = min(ruff.count, 20) / 40  (so 20+ violations = 0.5 penalty)
mypy_penalty = min(mypy.count, 20) / 40
pytest_failure_penalty = 0.5 if pytest exit != 0 else 0.0

raw = tool_pass_rate - imports_penalty - ruff_penalty - mypy_penalty - pytest_failure_penalty
return max(0.0, min(1.0, raw))
```

### 4.5 `sandbox.py`

```python
def run_sandbox(
    *,
    project_dir: Path | None = None,
    files: dict[str, str] | None = None,
    tools: Iterable[str] = DEFAULT_TOOLS,
    timeout_per_tool: float = 60.0,
) -> SandboxResult: ...
```

Exactly one of `project_dir`/`files` must be set. If `files`, create a tmp dir under the system tmp root, write each file, run the pipeline, return result. Caller does not need to delete the tmp dir (sandbox will tear it down before returning, but preserves the `SandboxResult` in memory).

### 4.6 `models.py`

```python
class ToolResult(BaseModel, frozen=True):
    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

class ParsedResult(BaseModel, frozen=True):
    ok: bool
    count: int
    details: dict

class ImportReport(BaseModel, frozen=True):
    total: int
    unresolved: tuple[str, ...]
    by_file: dict[str, tuple[str, ...]]

class SandboxResult(BaseModel, frozen=True):
    project_dir: str
    tools_run: tuple[str, ...]
    tool_results: dict[str, ToolResult]
    parsed: dict[str, ParsedResult]
    imports: ImportReport
    composite_score: float
    generated_at: str
```

### 4.7 `cli.py`

```
python -m groundloop.python_sandbox <project_dir> [--tool ruff] [--tool pytest] [--format json|text]
```

Exit 0 on any successful run (even if composite_score < 1.0 — the score is in the output, not the exit). Exit 1 on missing project_dir or internal error.

## 5. Error Handling

| Condition | Action |
|---|---|
| Missing `project_dir` | Raise `FileNotFoundError` / CLI exit 1 |
| Tool binary not installed | ToolResult with `exit_code=-1`, `stderr="binary not found"`, `timed_out=False`, `ok=False` |
| Tool timeout | `timed_out=True`, `ok=False` |
| Tool non-zero exit | `ok=False`, captured stdout/stderr |
| AST parse error | Log WARN, record in `ImportReport.by_file[filename] = ("__parse_error__",)` |

Each tool failure is isolated — sandbox ALWAYS returns a SandboxResult, never raises from the main entry point (except for missing-input).

## 6. Testing

Coverage target: **85%** (subprocess boundaries are harder to cover; tool-specific parsers mock subprocess output).

Fixtures:
- `fixtures/clean_project/` — one `main.py` with `print("ok")`, one `test_main.py`, `requirements.txt`. Passes ruff + mypy + pytest + has no unresolved imports.
- `fixtures/broken_project/` — one file with a syntax-clean but ruff-failing line (`x = 1;`), a type error for mypy, a test that asserts False, and an `import nonexistent_zzz`. Exercises every penalty.

Tests:
- `test_runner.py` — run_tool captures stdout/exit; timeout path uses a `sleep 5` command with 0.5s timeout.
- `test_tools.py` — argv builders produce expected lists; parsers handle valid + malformed output.
- `test_imports.py` — scan_imports on fixtures; unresolved detection; AST parse error path.
- `test_metric.py` — composite_score across synthetic SandboxResults (all-pass, all-fail, partial).
- `test_sandbox.py` — run_sandbox on clean fixture (score → 1.0), broken fixture (score ≤ 0.5).
- `test_cli.py` — CLI roundtrip with `--format json`.
- `test_e2e.py` — `files=` variant (inline synthesis).

## 7. Acceptance Criteria

1. `run_sandbox(project_dir=fixtures/clean_project)` returns `composite_score >= 0.9` and all 4 default tools report `ok=True`.
2. `run_sandbox(project_dir=fixtures/broken_project)` returns `composite_score <= 0.5`, with `imports.unresolved` containing `"nonexistent_zzz"`, ruff count > 0, pytest exit != 0.
3. `run_sandbox(files={"main.py": "print('hi')\n"})` succeeds and creates + cleans up a tmp dir.
4. CLI smoke: `python -m groundloop.python_sandbox <dir> --format json` produces parseable JSON with a `composite_score` key.
5. Every tool result is deterministic: `generated_at` aside, running the same input twice produces byte-identical `parsed` dicts (may differ in `duration_ms`).
6. `ruff check` + `mypy --strict` clean on `groundloop/python_sandbox/`.
7. Coverage ≥ 85% on the package.
8. Tool timeout (0.5s on `sleep 5`) returns `timed_out=True` without crashing the sandbox.

## 8. Dependencies

Already installed: `ruff`, `mypy`, `pytest` (used as CLI tools). No new Python imports beyond stdlib + `pydantic`.

Optional: `pip-audit` — if not installed, it's silently skipped from default tools (detected via `shutil.which`).

## 9. Deliverables

Package, tests, fixtures, CLI, README subsection.

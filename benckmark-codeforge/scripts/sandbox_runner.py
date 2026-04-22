from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CODE_FENCE_CLOSED = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_FENCE_OPEN = re.compile(r"```(?:python|py)?\s*\n(.*)$", re.DOTALL | re.IGNORECASE)


@dataclass
class ExtractResult:
    code: str
    reason: str  # "ok" | "no_code_block" | "syntax_error"


def _try_parse_prefix(code: str) -> str | None:
    """Greedily shrink code from end until it parses, to salvage truncated output."""
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return None


def extract_code(response: str) -> ExtractResult:
    if not response:
        return ExtractResult("", "no_code_block")

    m = CODE_FENCE_CLOSED.search(response)
    if m:
        code = m.group(1).strip()
        if not code:
            return ExtractResult("", "no_code_block")
        try:
            ast.parse(code)
            return ExtractResult(code, "ok")
        except SyntaxError:
            salvaged = _try_parse_prefix(code)
            if salvaged:
                return ExtractResult(salvaged, "ok")
            return ExtractResult(code, "syntax_error")

    # unclosed fence (truncation): take everything after the opening fence and salvage
    m = CODE_FENCE_OPEN.search(response)
    if m:
        code = m.group(1).strip()
        salvaged = _try_parse_prefix(code)
        if salvaged:
            return ExtractResult(salvaged, "ok")
        return ExtractResult(code, "syntax_error") if code else ExtractResult("", "no_code_block")

    # no fence at all — try raw response
    salvaged = _try_parse_prefix(response.strip())
    if salvaged:
        return ExtractResult(salvaged, "ok")
    return ExtractResult("", "no_code_block")


@dataclass
class SandboxResult:
    passed: bool
    reason: str
    stdout: str
    stderr: str
    returncode: int


def _classify_stderr(stderr: str, returncode: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "pass"
    if "AssertionError" in stderr:
        return "assertion_error"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "NameError" in stderr:
        return "name_error"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    return "runtime_error"


def run_sandbox(
    code: str,
    test_list: list[str],
    test_setup_code: str = "",
    *,
    python_exec: str = sys.executable,
    wall_timeout_seconds: float = 15.0,
) -> SandboxResult:
    script_parts: list[str] = []
    if test_setup_code.strip():
        script_parts.append(test_setup_code)
    script_parts.append(code)
    script_parts.extend(test_list)
    script = "\n\n".join(script_parts) + "\n"

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "candidate.py"
        p.write_text(script, encoding="utf-8")
        timed_out = False
        try:
            cp = subprocess.run(
                [python_exec, str(p)],
                capture_output=True,
                text=True,
                timeout=wall_timeout_seconds,
                cwd=td,
            )
            stdout = cp.stdout
            stderr = cp.stderr
            rc = cp.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rc = -1

    reason = _classify_stderr(stderr, rc, timed_out)
    return SandboxResult(
        passed=(reason == "pass"),
        reason=reason,
        stdout=stdout[-4000:],
        stderr=stderr[-4000:],
        returncode=rc,
    )

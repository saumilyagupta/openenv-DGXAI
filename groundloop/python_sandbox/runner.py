from __future__ import annotations

import os
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
    return dict(os.environ)

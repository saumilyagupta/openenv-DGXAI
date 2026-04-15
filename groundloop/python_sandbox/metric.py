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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeforge.sandbox.models import SandboxResult


def composite_score(
    result: SandboxResult,
    *,
    tools: tuple[str, ...] | None = None,
) -> float:
    """Compute composite score from sandbox results.

    When tools is None, score all tools that were run (full-project scoring).
    When tools is provided, score only those tools (subtask scoring).
    This lets the planner score 'implement core.py' with only ruff+mypy+imports,
    without pytest destroying the score because tests aren't written yet.
    """
    parsed = result.parsed
    if tools is not None:
        parsed = {k: v for k, v in parsed.items() if k in tools}
    if not parsed:
        return 0.0

    # Penalty-only scoring (no double-counting with pass_rate)
    imports_penalty = min(1.0, len(result.imports.unresolved) * 0.1)

    ruff = parsed.get("ruff")
    mypy = parsed.get("mypy")
    pytest_result = parsed.get("pytest")

    ruff_penalty = min(ruff.count, 20) / 40 if ruff else 0.0
    mypy_penalty = min(mypy.count, 20) / 40 if mypy else 0.0
    pytest_penalty = 0.5 if pytest_result and not pytest_result.ok else 0.0

    # Start at 1.0, subtract penalties. No pass_rate to avoid double-counting.
    raw = 1.0 - imports_penalty - ruff_penalty - mypy_penalty - pytest_penalty
    return max(0.0, min(1.0, raw))

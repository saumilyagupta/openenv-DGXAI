from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    task_level: str
    brief: str
    initial_files: dict[str, str]
    target_score: float
    max_budget: int
    tools: tuple[str, ...]


TASKS: tuple[Task, ...] = (
    Task(
        task_id="greet_single_file",
        task_level="easy",
        brief=(
            "Implement `greet(name)` in `main.py` so that `greet(\"Alice\")` returns "
            "`\"Hello, Alice!\"`. Use type hints. Keep the module under 15 lines."
        ),
        initial_files={"main.py": "def greet(name):\n    pass\n"},
        target_score=0.90,
        max_budget=4,
        tools=("ruff", "imports", "mypy"),
    ),
    Task(
        task_id="greet_with_tests",
        task_level="medium",
        brief=(
            "Extend `main.py` so that `greet(None)` raises `ValueError`, "
            "and add a `test_main.py` with pytest assertions. Keep `ruff` and "
            "`mypy --strict` clean."
        ),
        initial_files={
            "main.py": (
                "from __future__ import annotations\n\n\n"
                "def greet(name: str) -> str:\n"
                "    return f\"Hello, {name}!\"\n"
            ),
            "test_main.py": "",
        },
        target_score=0.80,
        max_budget=6,
        tools=("ruff", "imports", "mypy", "pytest"),
    ),
    Task(
        task_id="multi_file_module",
        task_level="hard",
        brief=(
            "Split into three files: `main.py` (entry), `core.py` (the greet "
            "function), `test_core.py` (tests). Every function must be type-hinted. "
            "All tests pass. `mypy --strict` clean."
        ),
        initial_files={
            "main.py": (
                "from __future__ import annotations\n\nfrom core import greet\n\n\n"
                "if __name__ == \"__main__\":\n"
                "    print(greet(\"World\"))\n"
            ),
            "core.py": "",
            "test_core.py": "",
        },
        target_score=0.70,
        max_budget=10,
        tools=("ruff", "imports", "mypy", "pytest"),
    ),
)


def get_task(task_level: str) -> Task:
    for t in TASKS:
        if t.task_level == task_level:
            return t
    msg = f"unknown task_level: {task_level!r} (expected easy|medium|hard)"
    raise ValueError(msg)

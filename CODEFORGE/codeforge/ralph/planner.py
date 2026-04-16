from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Subtask:
    """A single step in a decomposed task plan."""

    description: str
    target_files: tuple[str, ...]
    acceptance: str
    tools: tuple[str, ...]


class Planner:
    """Decomposes a task spec into ordered subtasks.

    The planner analyzes the spec and initial files to determine:
    1. Which files need to be created/modified
    2. In what order (dependency-based)
    3. Which tools are relevant for scoring each subtask

    This enables incremental scoring: implement core.py with ruff+mypy only,
    then add tests with ruff+mypy+pytest.
    """

    def plan(
        self,
        spec: str,
        initial_files: dict[str, str],
    ) -> list[Subtask]:
        """Decompose spec into ordered subtasks.

        Strategy:
        1. Identify empty files (need implementation)
        2. Identify test files (depends on implementation files)
        3. Order: implementation files first, then test files
        4. Implementation files get ruff+mypy+imports tools
        5. Test files get ruff+mypy+imports+pytest tools
        """
        empty_files = [f for f, content in initial_files.items() if not content.strip()]

        # Separate test files from implementation files
        test_files = [f for f in empty_files if f.startswith("test_")]
        impl_files = [f for f in empty_files if not f.startswith("test_")]

        subtasks: list[Subtask] = []

        # Phase 1: Implement empty non-test files
        for f in impl_files:
            subtasks.append(
                Subtask(
                    description=f"Implement {f}",
                    target_files=(f,),
                    acceptance=f"{f} passes ruff, mypy --strict, and imports check",
                    tools=("ruff", "imports", "mypy"),
                )
            )

        # Phase 2: Write tests
        for f in test_files:
            subtasks.append(
                Subtask(
                    description=f"Write tests in {f}",
                    target_files=(f,),
                    acceptance=f"{f} passes all tools including pytest",
                    tools=("ruff", "imports", "mypy", "pytest"),
                )
            )

        # If no subtasks were generated, create one for everything
        if not subtasks:
            all_files = tuple(initial_files.keys())
            has_tests = any(f.startswith("test_") for f in initial_files)
            tools = (
                ("ruff", "imports", "mypy", "pytest")
                if has_tests
                else ("ruff", "imports", "mypy")
            )
            subtasks.append(
                Subtask(
                    description="Implement the complete task",
                    target_files=all_files,
                    acceptance="All tools pass",
                    tools=tools,
                )
            )

        return subtasks

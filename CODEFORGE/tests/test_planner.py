from __future__ import annotations

import pytest

from codeforge.ralph.planner import Planner, Subtask
from codeforge.tasks import get_task


# ---------------------------------------------------------------------------
# Subtask dataclass tests
# ---------------------------------------------------------------------------


class TestSubtask:
    def test_frozen(self) -> None:
        s = Subtask(
            description="Implement core.py",
            target_files=("core.py",),
            acceptance="ruff clean",
            tools=("ruff", "imports", "mypy"),
        )
        with pytest.raises(AttributeError):
            s.description = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        s = Subtask(
            description="Write tests",
            target_files=("test_core.py",),
            acceptance="pytest passes",
            tools=("ruff", "imports", "mypy", "pytest"),
        )
        assert s.description == "Write tests"
        assert s.target_files == ("test_core.py",)
        assert s.acceptance == "pytest passes"
        assert s.tools == ("ruff", "imports", "mypy", "pytest")


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestPlanner:
    def setup_method(self) -> None:
        self.planner = Planner()

    # -- Easy task: single file with content, no empty files ----------------

    def test_plan_easy_task(self) -> None:
        task = get_task("easy")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        # Easy task: main.py has content, no empty files → fallback subtask
        assert len(subtasks) >= 1
        assert all(isinstance(s, Subtask) for s in subtasks)

    def test_easy_task_no_pytest(self) -> None:
        task = get_task("easy")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        # Easy task has no test files → tools should not include pytest
        for s in subtasks:
            assert "pytest" not in s.tools

    # -- Medium task: main.py has content, test_main.py empty ---------------

    def test_plan_medium_task(self) -> None:
        task = get_task("medium")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        # Medium task: test_main.py is empty → at least one subtask for tests
        assert len(subtasks) >= 1
        test_subtasks = [s for s in subtasks if any("test_" in f for f in s.target_files)]
        assert len(test_subtasks) >= 1

    def test_medium_test_subtask_has_pytest(self) -> None:
        task = get_task("medium")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        test_subtasks = [s for s in subtasks if any("test_" in f for f in s.target_files)]
        for s in test_subtasks:
            assert "pytest" in s.tools

    # -- Hard task: core.py empty, test_core.py empty -----------------------

    def test_plan_hard_task(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        # Hard task: core.py empty, test_core.py empty → at least 2 subtasks
        assert len(subtasks) >= 2

    def test_hard_task_ordering_impl_before_tests(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        impl_idx = next(i for i, s in enumerate(subtasks) if "core.py" in s.target_files)
        test_idx = next(i for i, s in enumerate(subtasks) if "test_core.py" in s.target_files)
        assert impl_idx < test_idx

    def test_hard_impl_subtask_no_pytest(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        impl_subtask = next(s for s in subtasks if "core.py" in s.target_files and "test_core.py" not in s.target_files)
        assert "pytest" not in impl_subtask.tools
        assert "ruff" in impl_subtask.tools
        assert "imports" in impl_subtask.tools
        assert "mypy" in impl_subtask.tools

    def test_hard_test_subtask_has_pytest(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        test_subtask = next(s for s in subtasks if "test_core.py" in s.target_files)
        assert "pytest" in test_subtask.tools

    # -- Target files correctness -------------------------------------------

    def test_subtask_target_files_correct(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        all_targets = set()
        for s in subtasks:
            all_targets.update(s.target_files)
        # All empty files should be covered by some subtask
        empty_files = {f for f, c in task.initial_files.items() if not c.strip()}
        assert empty_files.issubset(all_targets)

    # -- Acceptance descriptions are meaningful -----------------------------

    def test_acceptance_descriptions_nonempty(self) -> None:
        task = get_task("hard")
        subtasks = self.planner.plan(task.brief, dict(task.initial_files))
        for s in subtasks:
            assert len(s.acceptance) > 0
            assert len(s.description) > 0

    # -- Fallback when no empty files ---------------------------------------

    def test_fallback_subtask_when_no_empty_files(self) -> None:
        files = {"main.py": "print('hello')\n"}
        subtasks = self.planner.plan("Implement greeting", files)
        assert len(subtasks) == 1
        assert subtasks[0].target_files == ("main.py",)

    def test_fallback_with_test_file_includes_pytest(self) -> None:
        files = {"main.py": "def f(): pass\n", "test_main.py": "def test_f(): pass\n"}
        subtasks = self.planner.plan("Implement and test", files)
        assert len(subtasks) == 1
        assert "pytest" in subtasks[0].tools

    def test_fallback_without_test_file_no_pytest(self) -> None:
        files = {"main.py": "def f(): pass\n"}
        subtasks = self.planner.plan("Implement", files)
        assert len(subtasks) == 1
        assert "pytest" not in subtasks[0].tools

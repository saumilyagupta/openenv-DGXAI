from __future__ import annotations

import pytest

from codeforge.tasks import TASKS, Task, get_task


class TestGetTask:
    """Test get_task for all 3 levels."""

    def test_get_easy_task(self) -> None:
        task = get_task("easy")
        assert task.task_id == "greet_single_file"
        assert task.task_level == "easy"
        assert task.target_score == 0.90
        assert task.max_budget == 4

    def test_get_medium_task(self) -> None:
        task = get_task("medium")
        assert task.task_id == "greet_with_tests"
        assert task.task_level == "medium"
        assert task.target_score == 0.80
        assert task.max_budget == 6

    def test_get_hard_task(self) -> None:
        task = get_task("hard")
        assert task.task_id == "multi_file_module"
        assert task.task_level == "hard"
        assert task.target_score == 0.70
        assert task.max_budget == 10


class TestGetTaskUnknown:
    """Test that unknown levels raise ValueError."""

    def test_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="unknown task_level"):
            get_task("nightmare")

    def test_empty_level(self) -> None:
        with pytest.raises(ValueError, match="unknown task_level"):
            get_task("")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="unknown task_level"):
            get_task("Easy")


class TestTaskFields:
    """Test task fields are correct."""

    def test_easy_task_fields(self) -> None:
        task = get_task("easy")
        assert task.brief
        assert "greet" in task.brief.lower()
        assert "main.py" in task.initial_files
        assert task.tools == ("ruff", "imports", "mypy")

    def test_medium_task_fields(self) -> None:
        task = get_task("medium")
        assert "test_main.py" in task.initial_files
        assert "pytest" in task.tools
        assert task.tools == ("ruff", "imports", "mypy", "pytest")

    def test_hard_task_fields(self) -> None:
        task = get_task("hard")
        assert "core.py" in task.initial_files
        assert "test_core.py" in task.initial_files
        assert "main.py" in task.initial_files
        assert task.tools == ("ruff", "imports", "mypy", "pytest")

    def test_task_is_frozen(self) -> None:
        task = get_task("easy")
        with pytest.raises(AttributeError):
            task.task_id = "hacked"  # type: ignore[misc]


class TestTasksTuple:
    """Test the TASKS constant."""

    def test_tasks_has_three_entries(self) -> None:
        assert len(TASKS) == 3

    def test_tasks_are_task_instances(self) -> None:
        for task in TASKS:
            assert isinstance(task, Task)

    def test_all_task_ids_unique(self) -> None:
        ids = [t.task_id for t in TASKS]
        assert len(ids) == len(set(ids))

    def test_all_task_levels_unique(self) -> None:
        levels = [t.task_level for t in TASKS]
        assert len(levels) == len(set(levels))

    def test_target_scores_are_above_floor(self) -> None:
        """All task target scores must be above the uncertain floor (0.50)."""
        for task in TASKS:
            assert task.target_score > 0.50, (
                f"Task {task.task_id} target {task.target_score} must be above floor 0.50"
            )

from __future__ import annotations

from groundloop_env.tasks import TASKS, get_task


def test_three_tasks_present():
    levels = {t.task_level for t in TASKS}
    assert levels == {"easy", "medium", "hard"}


def test_get_task_by_level():
    t = get_task("easy")
    assert t.task_level == "easy"
    assert t.brief
    assert t.initial_files
    assert 0.0 < t.target_score <= 1.0


def test_get_task_invalid_level_raises():
    import pytest
    with pytest.raises(ValueError):
        get_task("trivial")

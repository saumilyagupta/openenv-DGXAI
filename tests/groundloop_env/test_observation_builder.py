from __future__ import annotations

from groundloop_env.observation_builder import build_observation
from groundloop_env.tasks import get_task


def test_build_minimal_observation():
    t = get_task("easy")
    obs = build_observation(
        episode_id="e1", task=t, current_files=t.initial_files,
        budget_remaining=4, previous_score=0.0,
        last_citations=(), last_grounding=None, is_done=False, last_reward=0.0,
    )
    assert obs.episode_id == "e1"
    assert obs.task_level == "easy"
    assert obs.initial_files == t.initial_files
    assert obs.is_done is False

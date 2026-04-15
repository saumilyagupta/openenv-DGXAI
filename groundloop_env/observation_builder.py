from __future__ import annotations

from models import CodeForgeObservation
from groundloop_env.tasks import Task


def build_observation(
    *,
    episode_id: str,
    task: Task,
    current_files: dict[str, str],
    budget_remaining: int,
    previous_score: float,
    last_citations: tuple[dict[str, object], ...] = (),
    last_grounding: dict[str, object] | None = None,
    is_done: bool = False,
    last_reward: float = 0.0,
) -> CodeForgeObservation:
    return CodeForgeObservation(
        episode_id=episode_id,
        task_id=task.task_id,
        task_level=task.task_level,
        task_brief=task.brief,
        initial_files=dict(task.initial_files),
        current_files=dict(current_files),
        budget_remaining=budget_remaining,
        previous_score=previous_score,
        last_citations=last_citations,
        last_grounding=last_grounding,
        is_done=is_done,
        last_reward=last_reward,
        reward=last_reward,
        done=is_done,
    )

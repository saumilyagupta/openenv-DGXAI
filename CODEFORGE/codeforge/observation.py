from __future__ import annotations

from typing import TYPE_CHECKING

from codeforge.models import CodeForgeObservation

if TYPE_CHECKING:
    from codeforge.tasks import Task


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
    last_cluster_hits: tuple[str, ...] = (),
    last_interrogation_questions: tuple[str, ...] = (),
    last_ralph_run_id: str | None = None,
    last_ralph_iterations: tuple[dict[str, object], ...] = (),
    cumulative_audit_summary: dict[str, object] | None = None,
    error: str | None = None,
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
        last_cluster_hits=last_cluster_hits,
        last_interrogation_questions=last_interrogation_questions,
        last_ralph_run_id=last_ralph_run_id,
        last_ralph_iterations=last_ralph_iterations,
        cumulative_audit_summary=cumulative_audit_summary or {},
        error=error,
        reward=last_reward,
        done=is_done,
    )

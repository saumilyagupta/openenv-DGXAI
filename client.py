from openenv.core.generic_client import GenericEnvClient
from openenv.core.client_types import StepResult
from models import EpistemicAction, EpistemicObservation


class EpistemicEnv(GenericEnvClient):
    def __init__(self, base_url: str):
        super().__init__(base_url=base_url)

    def reset(self, task_level: str = "easy") -> StepResult:
        return super().reset(task_level=task_level)

    def _step_payload(self, action: EpistemicAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(
            observation=EpistemicObservation(**payload["observation"]),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> EpistemicObservation:
        return EpistemicObservation(**payload)


from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class ActionType(str, Enum):
    QUERY = "query"
    COMMIT = "commit"


class EpistemicAction(Action):
    action_type: ActionType
    # For QUERY
    query_text: Optional[str] = None
    # For COMMIT
    verdict: Optional[str] = None          # "true" | "false" | "uncertain"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class EvidenceSnippet(BaseModel):
    id: str
    text: str
    relevance_score: float


class EpistemicObservation(Observation):
    claim: str
    evidence_gathered: list[EvidenceSnippet]
    budget_remaining: int
    task_level: str                         # "easy" | "medium" | "hard"
    episode_id: str
    is_done: bool = False
    last_reward: Optional[float] = None

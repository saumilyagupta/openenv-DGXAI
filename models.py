from __future__ import annotations

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


class CodeForgeActionType(str, Enum):
    QUERY_KB = "query_kb"
    SUBMIT = "submit"


class CodeForgeAction(Action):
    action_type: CodeForgeActionType
    claim: Optional[str] = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    files: Optional[dict[str, str]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CodeForgeObservation(Observation):
    episode_id: str
    task_id: str
    task_level: str
    task_brief: str
    initial_files: dict[str, str]
    current_files: dict[str, str]
    budget_remaining: int
    previous_score: float
    last_citations: tuple[dict[str, object], ...] = ()
    last_grounding: Optional[dict[str, object]] = None
    is_done: bool = False
    last_reward: float = 0.0

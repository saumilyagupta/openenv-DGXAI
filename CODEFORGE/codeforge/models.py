from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CodeForgeActionType(StrEnum):
    QUERY_KB = "query_kb"
    QUERY_CLUSTER = "query_cluster"
    INTERROGATE = "interrogate"
    RUN_RALPH = "run_ralph"
    SUBMIT = "submit"
    GET_AUDIT = "get_audit"


class CodeForgeAction(Action):
    action_type: CodeForgeActionType
    # query_kb fields
    claim: str | None = None
    top_k: int = 5
    required_tags: tuple[str, ...] = ()
    # submit fields
    files: dict[str, str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # query_cluster fields
    cluster_label: str | None = None
    # run_ralph fields
    max_iters: int = Field(default=3, ge=1, le=10)
    # get_audit fields
    target_run_id: str | None = None


class CodeForgeObservation(Observation):
    episode_id: str
    task_id: str
    task_level: str
    task_brief: str
    initial_files: dict[str, str]
    current_files: dict[str, str]
    budget_remaining: int
    previous_score: float
    last_reward: float
    is_done: bool
    # KB results
    last_citations: tuple[dict[str, object], ...] = ()
    last_grounding: dict[str, object] | None = None
    # Cluster results
    last_cluster_hits: tuple[str, ...] = ()
    # Interrogation results
    last_interrogation_questions: tuple[str, ...] = ()
    # Ralph results
    last_ralph_run_id: str | None = None
    last_ralph_iterations: tuple[dict[str, object], ...] = ()
    # Audit summary
    cumulative_audit_summary: dict[str, object] = Field(default_factory=dict)
    # Error field
    error: str | None = None


@dataclass(frozen=True)
class AuditEntry:
    step_index: int
    action_type: str
    cited_skill_ids: tuple[str, ...]
    cited_clusters: tuple[str, ...]
    grounding_report: dict[str, object] | None
    reward: float
    brier_penalty: float | None
    confidence_declared: float | None
    quality: float

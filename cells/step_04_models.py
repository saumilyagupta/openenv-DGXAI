"""DriftCall core dataclasses.

Implements docs/modules/models.md §2. Every declaration is pure shape; no
runtime logic lives here. All dataclasses are frozen. Invariants in §3.5 are
enforced by downstream modules (env.py, drift_injector.py, vendors/*), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class ActionType(StrEnum):
    TOOL_CALL = "tool_call"
    SPEAK = "speak"
    CLARIFY = "clarify"
    PROBE_SCHEMA = "probe_schema"
    SUBMIT = "submit"
    ABORT = "abort"


@dataclass(frozen=True)
class DriftCallAction:
    action_type: ActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    message: str | None = None
    confidence: float | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: Literal["ok", "schema_error", "policy_error", "auth_error", "timeout"]
    response: dict[str, Any]
    schema_version: str
    latency_ms: int


@dataclass(frozen=True)
class DriftEvent:
    turn: int
    drift_type: Literal["schema", "policy", "tnc", "pricing", "auth"]
    domain: str
    description: str
    from_version: str
    to_version: str
    pattern_id: str


@dataclass(frozen=True)
class GoalSpec:
    domain: str
    intent: str
    slots: dict[str, Any]
    constraints: dict[str, Any]
    language: Literal["hi", "ta", "kn", "en", "hinglish"]
    seed_utterance: str


@dataclass(frozen=True)
class DriftCallObservation:
    turn: int
    goal: GoalSpec
    last_transcript: str
    last_lang: str
    last_confidence: float
    tool_results: tuple[ToolResult, ...]
    drift_log: tuple[DriftEvent, ...]
    budget_remaining: int
    available_tools: tuple[str, ...]


@dataclass(frozen=True)
class DriftCallState:
    episode_id: str
    goal: GoalSpec
    vendor_states: dict[str, dict[str, Any]]
    schema_versions: dict[str, str]
    drift_schedule: tuple[DriftEvent, ...]
    drift_fired: tuple[DriftEvent, ...]
    turn: int
    max_turns: int
    actions: tuple[DriftCallAction, ...]
    done: bool


__all__ = [
    "ActionType",
    "DriftCallAction",
    "ToolResult",
    "DriftEvent",
    "GoalSpec",
    "DriftCallObservation",
    "DriftCallState",
]

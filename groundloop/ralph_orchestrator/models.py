from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TerminationReason = Literal["target_hit", "max_iters", "stuck"]
IterationReason = Literal[
    "score_improved",
    "score_regressed",
    "score_plateau",
    "target_hit",
    "sandbox_error",
    "synthesizer_error",
]


class LoopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_iters: int = Field(default=5, gt=0, le=100)
    target_score: float = Field(default=0.95, gt=0.0, le=2.0)
    tools: tuple[str, ...] = ("ruff", "imports")
    timeout_per_tool: float = 60.0
    top_k_citations: int = Field(default=5, gt=0, le=50)


class SynthesisResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    proposed_files: dict[str, str]
    rationale: str
    cited_node_ids: tuple[str, ...]


class Iteration(BaseModel):
    model_config = ConfigDict(frozen=True)
    index: int
    cited_node_ids: tuple[str, ...]
    rationale: str
    proposed_files: dict[str, str]
    sandbox_score_before: float
    sandbox_score_after: float
    kept: bool
    reason: IterationReason


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    spec: str
    started_at: str
    ended_at: str
    final_score: float
    final_files: dict[str, str]
    iterations: tuple[Iteration, ...]
    terminated_by: TerminationReason

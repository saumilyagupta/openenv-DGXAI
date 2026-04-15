from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.ralph_orchestrator.models import Iteration, LoopConfig, RunResult, SynthesisResult


def test_loop_config_defaults() -> None:
    c = LoopConfig()
    assert c.max_iters == 5
    assert 0.0 < c.target_score <= 1.0


def test_iteration_frozen() -> None:
    it = Iteration(
        index=0, cited_node_ids=(), rationale="r",
        proposed_files={"a.py": ""}, sandbox_score_before=0.0,
        sandbox_score_after=1.0, kept=True, reason="score_improved",
    )
    with pytest.raises(ValidationError):
        it.kept = False  # type: ignore[misc]


def test_synthesis_result_requires_fields() -> None:
    with pytest.raises(ValidationError):
        SynthesisResult()  # type: ignore[call-arg]


def test_run_result_iterations_immutable() -> None:
    r = RunResult(
        run_id="r", spec="s", started_at="t", ended_at="t",
        final_score=1.0, final_files={"main.py": ""}, iterations=(),
        terminated_by="target_hit",
    )
    assert r.iterations == ()

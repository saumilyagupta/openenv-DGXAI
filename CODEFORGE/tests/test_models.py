from __future__ import annotations

import pytest
from pydantic import ValidationError

from codeforge.models import (
    AuditEntry,
    CodeForgeAction,
    CodeForgeActionType,
    CodeForgeObservation,
)


class TestCodeForgeActionType:
    """Test all 6 CodeForgeActionType enum values."""

    def test_query_kb_value(self) -> None:
        assert CodeForgeActionType.QUERY_KB == "query_kb"

    def test_query_cluster_value(self) -> None:
        assert CodeForgeActionType.QUERY_CLUSTER == "query_cluster"

    def test_interrogate_value(self) -> None:
        assert CodeForgeActionType.INTERROGATE == "interrogate"

    def test_run_ralph_value(self) -> None:
        assert CodeForgeActionType.RUN_RALPH == "run_ralph"

    def test_submit_value(self) -> None:
        assert CodeForgeActionType.SUBMIT == "submit"

    def test_get_audit_value(self) -> None:
        assert CodeForgeActionType.GET_AUDIT == "get_audit"

    def test_enum_has_exactly_six_members(self) -> None:
        assert len(CodeForgeActionType) == 6

    def test_enum_is_str_subclass(self) -> None:
        assert isinstance(CodeForgeActionType.SUBMIT, str)


class TestCodeForgeAction:
    """Test CodeForgeAction validation and defaults."""

    def test_minimal_query_kb(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.QUERY_KB,
            claim="How to use pathlib?",
        )
        assert action.action_type == CodeForgeActionType.QUERY_KB
        assert action.claim == "How to use pathlib?"
        assert action.top_k == 5
        assert action.required_tags == ()

    def test_minimal_submit(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            files={"main.py": "print('hello')"},
            confidence=0.8,
        )
        assert action.files == {"main.py": "print('hello')"}
        assert action.confidence == 0.8

    def test_confidence_at_lower_bound(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            confidence=0.0,
        )
        assert action.confidence == 0.0

    def test_confidence_at_upper_bound(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.SUBMIT,
            confidence=1.0,
        )
        assert action.confidence == 1.0

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                confidence=-0.01,
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeForgeAction(
                action_type=CodeForgeActionType.SUBMIT,
                confidence=1.01,
            )

    def test_confidence_none_is_default(self) -> None:
        action = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT)
        assert action.confidence is None

    def test_max_iters_default(self) -> None:
        action = CodeForgeAction(action_type=CodeForgeActionType.RUN_RALPH)
        assert action.max_iters == 3

    def test_max_iters_minimum(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.RUN_RALPH,
            max_iters=1,
        )
        assert action.max_iters == 1

    def test_max_iters_maximum(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.RUN_RALPH,
            max_iters=10,
        )
        assert action.max_iters == 10

    def test_max_iters_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeForgeAction(
                action_type=CodeForgeActionType.RUN_RALPH,
                max_iters=0,
            )

    def test_max_iters_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeForgeAction(
                action_type=CodeForgeActionType.RUN_RALPH,
                max_iters=11,
            )

    def test_query_cluster_action(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.QUERY_CLUSTER,
            cluster_label="testing",
        )
        assert action.cluster_label == "testing"

    def test_get_audit_action(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.GET_AUDIT,
            target_run_id="run-123",
        )
        assert action.target_run_id == "run-123"

    def test_interrogate_action(self) -> None:
        action = CodeForgeAction(
            action_type=CodeForgeActionType.INTERROGATE,
            claim="How does asyncio work?",
        )
        assert action.action_type == CodeForgeActionType.INTERROGATE


class TestCodeForgeObservation:
    """Test CodeForgeObservation construction."""

    def test_full_observation(self) -> None:
        obs = CodeForgeObservation(
            episode_id="ep-001",
            task_id="greet_single_file",
            task_level="easy",
            task_brief="Implement greet(name)",
            initial_files={"main.py": "def greet(name): pass"},
            current_files={"main.py": "def greet(name): return f'Hello, {name}!'"},
            budget_remaining=3,
            previous_score=0.0,
            last_reward=0.0,
            is_done=False,
        )
        assert obs.episode_id == "ep-001"
        assert obs.task_id == "greet_single_file"
        assert obs.budget_remaining == 3
        assert obs.is_done is False

    def test_observation_defaults(self) -> None:
        obs = CodeForgeObservation(
            episode_id="ep-002",
            task_id="greet_single_file",
            task_level="easy",
            task_brief="Implement greet",
            initial_files={},
            current_files={},
            budget_remaining=4,
            previous_score=0.0,
            last_reward=0.0,
            is_done=False,
        )
        assert obs.last_citations == ()
        assert obs.last_grounding is None
        assert obs.last_cluster_hits == ()
        assert obs.last_interrogation_questions == ()
        assert obs.last_ralph_run_id is None
        assert obs.last_ralph_iterations == ()
        assert obs.cumulative_audit_summary == {}
        assert obs.error is None

    def test_observation_with_kb_results(self) -> None:
        obs = CodeForgeObservation(
            episode_id="ep-003",
            task_id="greet_single_file",
            task_level="easy",
            task_brief="Test",
            initial_files={},
            current_files={},
            budget_remaining=2,
            previous_score=0.5,
            last_reward=0.5,
            is_done=False,
            last_citations=({"node_id": "n1", "score": 0.9},),
            last_grounding={"groundedness": 0.85, "total": 10},
        )
        assert len(obs.last_citations) == 1
        assert obs.last_grounding is not None

    def test_observation_with_ralph_results(self) -> None:
        obs = CodeForgeObservation(
            episode_id="ep-004",
            task_id="greet_with_tests",
            task_level="medium",
            task_brief="Extend greet",
            initial_files={},
            current_files={},
            budget_remaining=0,
            previous_score=0.8,
            last_reward=0.85,
            is_done=True,
            last_ralph_run_id="ralph-001",
            last_ralph_iterations=({"iter": 1, "score": 0.7}, {"iter": 2, "score": 0.85}),
        )
        assert obs.last_ralph_run_id == "ralph-001"
        assert len(obs.last_ralph_iterations) == 2


class TestAuditEntry:
    """Test AuditEntry frozen dataclass."""

    def test_construction(self) -> None:
        entry = AuditEntry(
            step_index=0,
            action_type="submit",
            cited_skill_ids=("node-1", "node-2"),
            cited_clusters=("cluster-a",),
            grounding_report={"groundedness": 0.9, "total": 5},
            reward=0.85,
            brier_penalty=0.003,
            confidence_declared=0.85,
            quality=0.88,
        )
        assert entry.step_index == 0
        assert entry.action_type == "submit"
        assert len(entry.cited_skill_ids) == 2
        assert entry.reward == 0.85

    def test_frozen(self) -> None:
        entry = AuditEntry(
            step_index=1,
            action_type="query_kb",
            cited_skill_ids=(),
            cited_clusters=(),
            grounding_report=None,
            reward=0.0,
            brier_penalty=None,
            confidence_declared=None,
            quality=0.0,
        )
        with pytest.raises(AttributeError):
            entry.step_index = 2  # type: ignore[misc]

    def test_none_fields(self) -> None:
        entry = AuditEntry(
            step_index=0,
            action_type="query_kb",
            cited_skill_ids=(),
            cited_clusters=(),
            grounding_report=None,
            reward=0.0,
            brier_penalty=None,
            confidence_declared=None,
            quality=0.0,
        )
        assert entry.grounding_report is None
        assert entry.brier_penalty is None
        assert entry.confidence_declared is None

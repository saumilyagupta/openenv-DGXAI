"""Tests for the EpistemicNav environment."""

import pytest
from server.environment import EpistemicNavEnvironment
from models import EpistemicAction, ActionType


@pytest.fixture
def env():
    return EpistemicNavEnvironment(max_budget=8, data_dir="data")


class TestReset:

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_level="easy")
        assert obs.claim != ""
        assert obs.budget_remaining == 8
        assert obs.evidence_gathered == []
        assert obs.episode_id != ""

    def test_reset_easy_claim(self, env):
        obs = env.reset(task_level="easy")
        assert obs.task_level == "easy"

    def test_reset_medium_claim(self, env):
        obs = env.reset(task_level="medium")
        assert obs.task_level == "medium"

    def test_reset_hard_claim(self, env):
        obs = env.reset(task_level="hard")
        assert obs.task_level == "hard"

    def test_reset_clears_state(self, env):
        env.reset(task_level="easy")
        # Take a step
        env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="test"))
        assert env.budget_remaining == 7

        # Reset should clear
        obs = env.reset(task_level="easy")
        assert obs.budget_remaining == 8
        assert obs.evidence_gathered == []


class TestStepQuery:

    def test_query_decrements_budget(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="test"))
        assert obs.budget_remaining == 7

    def test_query_not_done(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="water"))
        assert obs.done is False

    def test_query_gathers_evidence(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="water boiling point temperature"))
        assert len(obs.evidence_gathered) > 0

    def test_query_reward_nonnegative(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="DNA genetics"))
        assert obs.reward >= 0.0
        assert obs.reward <= 0.05  # capped at 0.05

    def test_duplicate_evidence_not_added(self, env):
        env.reset(task_level="easy")
        obs1 = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="water boiling"))
        count1 = len(obs1.evidence_gathered)
        obs2 = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="water boiling"))
        count2 = len(obs2.evidence_gathered)
        # Second identical query should add fewer (or zero) new snippets
        assert count2 - count1 < count1


class TestStepCommit:

    def test_commit_ends_episode(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(
            action_type=ActionType.COMMIT, verdict="true", confidence=0.8
        ))
        assert obs.done is True

    def test_commit_reward_in_range(self, env):
        env.reset(task_level="easy")
        obs = env.step(EpistemicAction(
            action_type=ActionType.COMMIT, verdict="true", confidence=0.8
        ))
        assert 0.0 <= obs.reward <= 1.0

    def test_commit_uncertain_on_hard(self, env):
        env.reset(task_level="hard")
        obs = env.step(EpistemicAction(
            action_type=ActionType.COMMIT, verdict="uncertain", confidence=0.5
        ))
        assert obs.done is True
        assert obs.reward >= 0.70  # minimum for correct uncertain


class TestBudgetExhaustion:

    def test_forced_commit_on_budget_zero(self, env):
        env.reset(task_level="easy")
        # Exhaust budget
        for _ in range(8):
            obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="test"))

        # Next non-commit action should force commit
        obs = env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="test"))
        assert obs.done is True
        assert 0.0 <= obs.reward <= 1.0


class TestState:

    def test_state_reflects_current(self, env):
        env.reset(task_level="easy")
        state = env.state
        assert state.budget_remaining == 8
        assert state.is_done is False

    def test_state_after_step(self, env):
        env.reset(task_level="easy")
        env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="test"))
        state = env.state
        assert state.budget_remaining == 7

    def test_state_after_commit(self, env):
        env.reset(task_level="easy")
        env.step(EpistemicAction(
            action_type=ActionType.COMMIT, verdict="true", confidence=0.9
        ))
        state = env.state
        assert state.is_done is True
        assert state.last_reward is not None


class TestFullEpisode:

    def test_query_then_commit(self, env):
        env.reset(task_level="medium")
        # Query twice
        env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="evidence search"))
        env.step(EpistemicAction(action_type=ActionType.QUERY, query_text="more evidence"))
        # Commit
        obs = env.step(EpistemicAction(
            action_type=ActionType.COMMIT, verdict="true", confidence=0.7
        ))
        assert obs.done is True
        assert 0.0 <= obs.reward <= 1.0

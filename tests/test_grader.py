"""Tests for the Brier score grader."""

import pytest
from server.grader import compute_reward


class TestComputeReward:
    """Verify reward function produces correct, bounded scores."""

    def test_correct_high_confidence(self):
        """Correct verdict with high confidence should score near 1.0."""
        reward = compute_reward("true", 0.95, "true", budget_remaining=4, max_budget=8)
        assert 0.85 <= reward <= 1.0

    def test_correct_low_confidence(self):
        """Correct verdict with low confidence should score lower (underconfidence penalty)."""
        reward = compute_reward("true", 0.3, "true", budget_remaining=4, max_budget=8)
        high = compute_reward("true", 0.95, "true", budget_remaining=4, max_budget=8)
        assert reward < high

    def test_wrong_high_confidence(self):
        """Wrong verdict with high confidence should score very low (overconfidence penalty)."""
        reward = compute_reward("false", 0.95, "true", budget_remaining=4, max_budget=8)
        assert reward < 0.2

    def test_wrong_low_confidence(self):
        """Wrong verdict with low confidence should score better than wrong + high confidence."""
        low_conf = compute_reward("false", 0.2, "true", budget_remaining=4, max_budget=8)
        high_conf = compute_reward("false", 0.95, "true", budget_remaining=4, max_budget=8)
        assert low_conf > high_conf

    def test_uncertain_on_uncertain(self):
        """Uncertain verdict when ground truth is uncertain should give min 0.70."""
        reward = compute_reward("uncertain", 0.5, "uncertain", budget_remaining=0, max_budget=8)
        assert reward >= 0.70

    def test_uncertain_calibrated_confidence(self):
        """Uncertain with confidence in [0.4, 0.7] gets bonus."""
        calibrated = compute_reward("uncertain", 0.5, "uncertain", budget_remaining=4, max_budget=8)
        extreme = compute_reward("uncertain", 0.1, "uncertain", budget_remaining=4, max_budget=8)
        assert calibrated > extreme

    def test_reward_always_in_range(self):
        """All reward values must be in [0.0, 1.0]."""
        cases = [
            ("true", 1.0, "true", 8),
            ("true", 0.0, "true", 0),
            ("false", 1.0, "true", 8),
            ("false", 0.0, "false", 0),
            ("uncertain", 0.5, "uncertain", 4),
            ("true", 0.5, "false", 4),
            ("uncertain", 0.0, "true", 8),
            ("uncertain", 1.0, "false", 0),
        ]
        for verdict, conf, gt, budget in cases:
            reward = compute_reward(verdict, conf, gt, budget_remaining=budget, max_budget=8)
            assert 0.0 <= reward <= 1.0, f"Out of range: {verdict}/{conf}/{gt}/{budget} -> {reward}"

    def test_efficiency_bonus_only_when_correct(self):
        """Efficiency bonus should only apply when verdict is correct."""
        correct_full = compute_reward("true", 0.9, "true", budget_remaining=8, max_budget=8)
        correct_none = compute_reward("true", 0.9, "true", budget_remaining=0, max_budget=8)
        wrong_full = compute_reward("false", 0.9, "true", budget_remaining=8, max_budget=8)
        wrong_none = compute_reward("false", 0.9, "true", budget_remaining=0, max_budget=8)

        assert correct_full > correct_none  # efficiency bonus matters when correct
        assert wrong_full == wrong_none  # no bonus when wrong

    def test_deterministic(self):
        """Same inputs must produce same output."""
        r1 = compute_reward("true", 0.8, "true", budget_remaining=3, max_budget=8)
        r2 = compute_reward("true", 0.8, "true", budget_remaining=3, max_budget=8)
        assert r1 == r2

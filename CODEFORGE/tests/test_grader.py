from __future__ import annotations

import pytest

from codeforge.grader import compute_reward


class TestQualityCalculation:
    """Test quality formula with explicit confidence to isolate Brier effects."""

    def test_perfect_scores_perfect_calibration(self) -> None:
        # quality=1.0, confidence=1.0, brier=(1.0-1.0)²=0.0
        reward = compute_reward(sandbox_score=1.0, groundedness=1.0, confidence=1.0)
        assert reward == 1.0

    def test_zero_scores(self) -> None:
        # quality=0.0, effective_conf=0.5, brier=(0.5-0.0)²=0.25
        # reward = 0.0 * (1-0.25) = 0.0
        reward = compute_reward(sandbox_score=0.0, groundedness=0.0)
        assert reward == 0.0

    def test_weighted_combination(self) -> None:
        # quality = 0.6 * 0.5 + 0.4 * 1.0 = 0.70, confidence=0.70 → brier=0
        reward = compute_reward(sandbox_score=0.5, groundedness=1.0, confidence=0.70)
        assert reward == 0.7

    def test_sandbox_only(self) -> None:
        # quality = 0.6, confidence=0.6 → brier=0
        reward = compute_reward(sandbox_score=1.0, groundedness=0.0, confidence=0.6)
        assert reward == 0.6

    def test_grounding_only(self) -> None:
        # quality = 0.4, confidence=0.4 → brier=0
        reward = compute_reward(sandbox_score=0.0, groundedness=1.0, confidence=0.4)
        assert reward == 0.4


class TestBrierPenalty:
    """Test the Brier calibration penalty."""

    def test_well_calibrated(self) -> None:
        # quality = 0.6*0.9 + 0.4*0.9 = 0.9
        # brier = (0.85 - 0.9)^2 = 0.0025
        # reward = 0.9 * (1 - 0.0025) = 0.8978 → 0.898
        reward = compute_reward(sandbox_score=0.9, groundedness=0.9, confidence=0.85)
        assert reward == 0.898

    def test_overconfident(self) -> None:
        # quality = 0.6*0.7 + 0.4*0.7 = 0.7
        # brier = (0.99 - 0.7)^2 = 0.0841
        # reward = 0.7 * (1 - 0.0841) = 0.641
        reward = compute_reward(sandbox_score=0.7, groundedness=0.7, confidence=0.99)
        assert reward == 0.641

    def test_dishonestly_confident_bad_code(self) -> None:
        # quality = 0.6*0.3 + 0.4*0.3 = 0.3
        # brier = (0.9 - 0.3)^2 = 0.36
        # reward = 0.3 * (1 - 0.36) = 0.192
        reward = compute_reward(sandbox_score=0.3, groundedness=0.3, confidence=0.9)
        assert reward == 0.192

    def test_brier_capped_at_half(self) -> None:
        # quality = 0.6*0.1 + 0.4*0.1 = 0.1
        # brier = (1.0 - 0.1)^2 = 0.81, capped at 0.5
        # reward = 0.1 * (1 - 0.5) = 0.05
        reward = compute_reward(sandbox_score=0.1, groundedness=0.1, confidence=1.0)
        assert reward == 0.05


class TestUncertainFloor:
    """Test the uncertain floor at 0.50 (NOT 0.70)."""

    def test_floor_is_0_50(self) -> None:
        # quality = 0.6*0.2 + 0.4*0.2 = 0.2, confidence=0.1 < 0.3, quality < 0.5
        # brier = (0.1 - 0.2)^2 = 0.01
        # raw reward = 0.2 * (1 - 0.01) = 0.198
        # floor triggered: max(0.198, 0.50) = 0.50
        reward = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.1)
        assert reward == 0.5

    def test_floor_does_not_trigger_when_confidence_ge_0_3(self) -> None:
        # quality = 0.6*0.2 + 0.4*0.2 = 0.2, confidence=0.3 >= 0.3 → no floor
        # brier = (0.3 - 0.2)^2 = 0.01
        # reward = 0.2 * (1 - 0.01) = 0.198
        reward = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.3)
        assert reward == 0.198

    def test_floor_does_not_trigger_when_quality_ge_0_5(self) -> None:
        # quality = 0.6*0.5 + 0.4*0.5 = 0.5, confidence=0.1 < 0.3 but quality=0.5 not < 0.5
        # brier = (0.1 - 0.5)^2 = 0.16
        # reward = 0.5 * (1 - 0.16) = 0.42
        reward = compute_reward(sandbox_score=0.5, groundedness=0.5, confidence=0.1)
        assert reward == 0.42

    def test_exploit_scenario_garbage_low_confidence_gives_0_50(self) -> None:
        """The FIXED exploit: garbage code + low confidence = 0.50, below ALL task targets."""
        # quality = 0.6*0.0 + 0.4*0.0 = 0.0, confidence=0.05
        # brier = (0.05 - 0.0)^2 = 0.0025
        # raw reward = 0.0 * (1 - 0.0025) = 0.0
        # floor triggered: max(0.0, 0.50) = 0.50
        reward = compute_reward(sandbox_score=0.0, groundedness=0.0, confidence=0.05)
        assert reward == 0.5
        # 0.50 < easy(0.90) < medium(0.80) < hard(0.70) — cannot complete any task
        assert reward < 0.70

    def test_floor_not_0_70(self) -> None:
        """Ensure the old exploit (floor=0.70 == hard target) is gone."""
        reward = compute_reward(sandbox_score=0.0, groundedness=0.0, confidence=0.05)
        assert reward != 0.70
        assert reward == 0.50


class TestConfidenceNone:
    """confidence=None is treated as 0.5 (mediocre calibration) — not a free pass."""

    def test_none_treated_as_half(self) -> None:
        # quality = 0.8, effective_conf = 0.5, brier = (0.5 - 0.8)² = 0.09
        # reward = 0.8 * (1 - 0.09) = 0.728
        reward = compute_reward(sandbox_score=0.8, groundedness=0.8)
        assert reward == 0.728

    def test_none_with_quality_half_no_brier(self) -> None:
        # quality = 0.5, effective_conf = 0.5, brier = (0.5 - 0.5)² = 0.0
        # reward = 0.5 * (1 - 0.0) = 0.5
        reward = compute_reward(sandbox_score=0.5, groundedness=0.5, confidence=None)
        assert reward == 0.5

    def test_none_no_floor_trigger(self) -> None:
        # quality = 0.2, effective_conf = 0.5, brier = (0.5 - 0.2)² = 0.09
        # reward = 0.2 * (1 - 0.09) = 0.182
        # floor requires confidence is not None → not triggered
        reward = compute_reward(sandbox_score=0.2, groundedness=0.2)
        assert reward == 0.182

    def test_explicit_confidence_better_than_none(self) -> None:
        # Providing accurate confidence always beats omitting it
        # quality = 0.8, conf=0.8 → brier=(0.8-0.8)²=0 → reward=0.8
        # quality = 0.8, conf=None → brier=(0.5-0.8)²=0.09 → reward=0.728
        with_conf = compute_reward(sandbox_score=0.8, groundedness=0.8, confidence=0.8)
        without_conf = compute_reward(sandbox_score=0.8, groundedness=0.8)
        assert with_conf > without_conf


class TestClamping:
    """Test clamping to [0.0, 1.0] and rounding."""

    def test_result_never_negative(self) -> None:
        reward = compute_reward(sandbox_score=0.0, groundedness=0.0)
        assert reward >= 0.0

    def test_result_never_above_one(self) -> None:
        reward = compute_reward(sandbox_score=1.0, groundedness=1.0, confidence=1.0)
        assert reward <= 1.0

    def test_rounded_to_3_decimals(self) -> None:
        # quality = 0.6*0.33 + 0.4*0.77 = 0.198 + 0.308 = 0.506
        reward = compute_reward(sandbox_score=0.33, groundedness=0.77)
        assert reward == 0.506
        # Check it's actually 3 decimals
        reward_str = str(reward)
        if "." in reward_str:
            decimals = len(reward_str.split(".")[1])
            assert decimals <= 3


class TestEdgeCases:
    """Edge cases for grader."""

    def test_confidence_zero(self) -> None:
        # quality = 0.6*0.3 + 0.4*0.3 = 0.3
        # confidence=0.0 < 0.3, quality=0.3 < 0.5 → floor
        # brier = (0.0 - 0.3)^2 = 0.09
        # raw = 0.3 * (1 - 0.09) = 0.273
        # floor: max(0.273, 0.50) = 0.50
        reward = compute_reward(sandbox_score=0.3, groundedness=0.3, confidence=0.0)
        assert reward == 0.5

    def test_confidence_one(self) -> None:
        # quality = 0.6*1.0 + 0.4*1.0 = 1.0
        # brier = (1.0 - 1.0)^2 = 0.0
        # reward = 1.0 * (1 - 0.0) = 1.0
        reward = compute_reward(sandbox_score=1.0, groundedness=1.0, confidence=1.0)
        assert reward == 1.0

    @pytest.mark.parametrize(
        ("sandbox", "ground", "conf", "expected_min", "expected_max"),
        [
            (0.0, 0.0, None, 0.0, 0.0),
            (1.0, 1.0, None, 0.7, 0.8),  # conf=None→0.5, brier=(0.5-1.0)²=0.25
            (0.5, 0.5, 0.5, 0.0, 1.0),
            (0.0, 0.0, 0.05, 0.5, 0.5),  # floor
        ],
    )
    def test_reward_in_valid_range(
        self,
        sandbox: float,
        ground: float,
        conf: float | None,
        expected_min: float,
        expected_max: float,
    ) -> None:
        reward = compute_reward(
            sandbox_score=sandbox, groundedness=ground, confidence=conf
        )
        assert expected_min <= reward <= expected_max

from __future__ import annotations

from groundloop_env.grader import compute_reward


def test_reward_in_range_monotonic_in_both_inputs():
    r_low = compute_reward(sandbox_score=0.2, groundedness=0.5)
    r_hi = compute_reward(sandbox_score=0.9, groundedness=0.5)
    assert 0.0 <= r_low <= r_hi <= 1.0

    r_ungrounded = compute_reward(sandbox_score=0.5, groundedness=0.0)
    r_grounded = compute_reward(sandbox_score=0.5, groundedness=1.0)
    assert r_ungrounded < r_grounded


def test_reward_clamped_to_zero_one():
    assert compute_reward(sandbox_score=-1.0, groundedness=2.0) <= 1.0
    assert compute_reward(sandbox_score=-5.0, groundedness=-5.0) >= 0.0


def test_reward_deterministic():
    a = compute_reward(sandbox_score=0.7, groundedness=0.8)
    b = compute_reward(sandbox_score=0.7, groundedness=0.8)
    assert a == b


def test_reward_without_confidence_backward_compatible():
    assert compute_reward(sandbox_score=1.0, groundedness=1.0) == 1.0
    assert compute_reward(sandbox_score=0.5, groundedness=0.5) == 0.5


def test_reward_perfect_calibration_no_penalty():
    assert compute_reward(sandbox_score=0.5, groundedness=0.5, confidence=0.5) == 0.5


def test_reward_overconfident_wrong_heavily_penalised():
    r = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.95)
    assert r < 0.2, f"expected < 0.2, got {r}"
    assert r <= 0.11


def test_reward_underconfident_right_moderately_penalised():
    r = compute_reward(sandbox_score=0.9, groundedness=0.9, confidence=0.1)
    assert 0.4 <= r <= 0.5


def test_reward_uncertain_floor_activates():
    r = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.1)
    assert r >= 0.70


def test_reward_uncertain_floor_not_triggered_above_threshold():
    r = compute_reward(sandbox_score=0.6, groundedness=0.6, confidence=0.1)
    assert r < 0.70


def test_reward_clamped_to_range():
    assert 0.0 <= compute_reward(sandbox_score=-5.0, groundedness=-5.0, confidence=0.5) <= 1.0
    assert 0.0 <= compute_reward(sandbox_score=5.0, groundedness=5.0, confidence=0.5) <= 1.0


def test_reward_deterministic_across_calls():
    a = compute_reward(sandbox_score=0.7, groundedness=0.8, confidence=0.6)
    b = compute_reward(sandbox_score=0.7, groundedness=0.8, confidence=0.6)
    assert a == b

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

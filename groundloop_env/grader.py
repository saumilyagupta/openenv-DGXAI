from __future__ import annotations

_SANDBOX_WEIGHT = 0.6
_GROUNDING_WEIGHT = 0.4


def compute_reward(*, sandbox_score: float, groundedness: float) -> float:
    raw = _SANDBOX_WEIGHT * sandbox_score + _GROUNDING_WEIGHT * groundedness
    clamped = max(0.0, min(1.0, raw))
    return round(clamped, 3)

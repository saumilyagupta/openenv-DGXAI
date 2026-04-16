from __future__ import annotations

_SANDBOX_WEIGHT = 0.6
_GROUNDING_WEIGHT = 0.4
_BRIER_CAP = 0.5
_UNCERTAIN_CONFIDENCE_THRESHOLD = 0.3
_UNCERTAIN_QUALITY_THRESHOLD = 0.5
_UNCERTAIN_FLOOR = 0.50


def compute_reward(
    *,
    sandbox_score: float,
    groundedness: float,
    confidence: float | None = None,
) -> float:
    """Compute the final reward for a submit action.

    quality   = weighted combination of sandbox and grounding signals
    brier     = calibration penalty (overconfidence on bad code is punished)
    uncertain = floor reward for honest uncertainty (below all task targets)
    """
    quality = _SANDBOX_WEIGHT * sandbox_score + _GROUNDING_WEIGHT * groundedness

    brier_penalty = 0.0
    if confidence is not None:
        brier_penalty = min((confidence - quality) ** 2, _BRIER_CAP)

    reward = quality * (1.0 - brier_penalty)

    if (
        confidence is not None
        and confidence < _UNCERTAIN_CONFIDENCE_THRESHOLD
        and quality < _UNCERTAIN_QUALITY_THRESHOLD
    ):
        reward = max(reward, _UNCERTAIN_FLOOR)

    return round(max(0.0, min(1.0, reward)), 3)

def compute_reward(
    verdict: str,           # "true" | "false" | "uncertain"
    confidence: float,      # [0.0, 1.0]
    ground_truth: str,      # "true" | "false" | "uncertain"
    budget_remaining: int,
    max_budget: int = 8,
) -> float:

    # Special case: genuine uncertainty rewarded
    if verdict == "uncertain" and ground_truth == "uncertain":
        base = 0.70 + 0.10 * (confidence >= 0.4 and confidence <= 0.7)
        efficiency = 0.1 * (budget_remaining / max_budget)
        return min(1.0, base + efficiency)

    correct = 1.0 if verdict == ground_truth else 0.0

    # Brier score: 1 - (confidence - correctness)^2
    # Penalises overconfidence on wrong answers AND underconfidence on right ones
    calibration = 1.0 - (confidence - correct) ** 2

    # Small efficiency bonus — only when correct
    efficiency = budget_remaining / max_budget
    efficiency_bonus = 0.1 * efficiency if correct else 0.0

    # Final reward — always in [0.0, 1.0]
    reward = calibration * 0.9 + efficiency_bonus
    return round(float(reward), 4)

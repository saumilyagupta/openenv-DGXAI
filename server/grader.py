def compute_reward(
    verdict: str,           # "true" | "false" | "uncertain"
    confidence: float,      # [0.0, 1.0]
    ground_truth: str,      # "true" | "false" | "uncertain"
    budget_remaining: int,
    max_budget: int = 8,
) -> float:
    # Clamp confidence to valid range (defense in depth)
    confidence = max(0.0, min(1.0, confidence))

    # Guard against division by zero
    if max_budget <= 0:
        max_budget = 1

    # Special case: genuine uncertainty rewarded
    if verdict == "uncertain" and ground_truth == "uncertain":
        base = 0.70 + 0.10 * (confidence >= 0.4 and confidence <= 0.7)
        efficiency = 0.1 * (budget_remaining / max_budget)
        return min(1.0, base + efficiency)

    correct = verdict == ground_truth
    efficiency = budget_remaining / max_budget

    if correct:
        # Reward scales with confidence when correct
        # conf=0 → 0.15 base (still better than wrong)
        # conf=1 → 1.0 ceiling
        calibration = 1.0 - (1.0 - confidence) ** 2
        reward = 0.15 + calibration * 0.75 + 0.1 * efficiency
    else:
        # Wrong verdict: penalize proportional to confidence
        # conf=0 → 0.10 max (always less than correct at 0.15)
        # conf=1 → 0.00
        penalty = confidence ** 2
        reward = 0.1 * (1.0 - penalty)

    return round(min(1.0, max(0.0, float(reward))), 4)

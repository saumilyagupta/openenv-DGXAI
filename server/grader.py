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

    if correct:
        # Brier-style: reward scales with confidence when correct
        # High confidence + correct = high reward
        # Low confidence + correct = lower reward (underconfidence penalty)
        calibration = 1.0 - (1.0 - confidence) ** 2
        efficiency = 0.1 * (budget_remaining / max_budget)
        reward = calibration * 0.9 + efficiency
    else:
        # Wrong verdict: penalize proportional to confidence
        # High confidence + wrong = near-zero reward (overconfidence penalty)
        # Low confidence + wrong = small consolation (still wrong, just less cocky)
        calibration = (1.0 - confidence) ** 2
        reward = calibration * 0.3  # cap at 0.3 max — wrong is always bad

    return round(min(1.0, max(0.0, float(reward))), 4)

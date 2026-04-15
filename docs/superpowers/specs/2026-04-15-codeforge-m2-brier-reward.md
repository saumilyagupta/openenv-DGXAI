# CodeForge M2 — Brier-Calibrated Reward

**Date:** 2026-04-15
**Module:** M2 of fully-wired CodeForge (per CLAUDE.md §2).
**Depends on:** `models.py` (CodeForgeAction), `groundloop_env/grader.py`.
**Consumed by:** M5 (`run_ralph` uses calibrated reward internally), M6 (AuditLedger logs brier penalty per submit), M7 (schema update).

---

## 1. Purpose

Wire in **Pillar 2 — EpistemicNav Brier calibration**. Agents can declare a `confidence` when submitting; reward rewards well-calibrated agents, penalises overconfident-wrong and underconfident-right submissions. Preserves round-1's "uncertain is a valid answer" floor.

## 2. Scope

**In scope:**

- Extend `CodeForgeAction` with `confidence: float | None = None` field (range [0, 1]).
- Extend `groundloop_env/grader.py::compute_reward` to accept `confidence` and apply Brier penalty + uncertain-floor.
- Update `CodeForgeEnvironment._handle_submit` to pass `action.confidence` through.
- Tests.

**Out of scope:**

- Confidence on `query_kb` / `query_cluster` / `interrogate` / `run_ralph` / `get_audit` — only `submit` uses it.
- Multi-sample confidence calibration (ECE curves) — Phase 3.
- Punishing `confidence=None` (backward-compat: no confidence = no Brier penalty).

## 3. Reward Formula (target)

```
quality  = 0.6 * sandbox_composite_score + 0.4 * grounding_score
brier    = (confidence - quality)² if confidence is not None else 0.0
penalty  = min(brier, 0.5)
reward   = quality * (1 - penalty)

# Uncertain-floor (round-1 rule): agent admits "I don't know" AND output is weak → still rewarded
if confidence is not None and confidence < 0.3 and quality < 0.5:
    reward = max(reward, 0.70)

reward = round(max(0.0, min(1.0, reward)), 3)
```

Deterministic in `(sandbox_score, grounding_score, confidence)`.

## 4. API changes

### 4.1 `models.py`

Add `confidence: float | None = Field(default=None, ge=0.0, le=1.0)` to `CodeForgeAction`. Pydantic validates range.

### 4.2 `grader.py`

```python
def compute_reward(
    *,
    sandbox_score: float,
    groundedness: float,
    confidence: float | None = None,
) -> float:
    quality = 0.6 * sandbox_score + 0.4 * groundedness
    brier_penalty = 0.0
    if confidence is not None:
        brier_penalty = min((confidence - quality) ** 2, 0.5)
    reward = quality * (1.0 - brier_penalty)
    if confidence is not None and confidence < 0.3 and quality < 0.5:
        reward = max(reward, 0.70)
    return round(max(0.0, min(1.0, reward)), 3)
```

### 4.3 `environment.py::_handle_submit`

After computing `sandbox_score` and `grounding_report`, pass `action.confidence` to `compute_reward`:

```python
reward = compute_reward(
    sandbox_score=sandbox_score,
    groundedness=grounding_report.groundedness,
    confidence=action.confidence,
)
```

## 5. Backward Compatibility

- Existing submits without `confidence` → unchanged reward (Brier penalty = 0, no uncertain floor).
- All existing tests must pass without modification.
- Existing `test_submit_easy_task_with_correct_code_meets_target` continues to assert reward > 0.8 (confidence omitted → no Brier penalty → same reward).

## 6. Testing

Coverage target: **95%** on `grader.py` (small module).

Tests:
- `test_reward_without_confidence_backward_compatible` — no confidence → same as current formula.
- `test_reward_perfect_calibration_no_penalty` — confidence = quality → brier = 0 → no penalty.
- `test_reward_overconfident_wrong_heavily_penalised` — confidence=0.95, quality=0.2 → reward much lower than quality.
- `test_reward_underconfident_right_moderately_penalised` — confidence=0.1, quality=0.9 → some penalty but not catastrophic.
- `test_reward_uncertain_floor_activates` — confidence=0.1, quality=0.3 → reward ≥ 0.70.
- `test_reward_uncertain_floor_not_triggered_above_quality_threshold` — confidence=0.1, quality=0.6 → no floor.
- `test_reward_clamped_to_range` — extreme inputs → reward in [0, 1].
- `test_reward_deterministic_across_calls` — same inputs → same output.

Env-level tests:
- `test_submit_with_confidence_passes_to_grader` — env `_handle_submit` forwards `action.confidence`.
- `test_submit_with_high_confidence_good_code_high_reward` — confidence=0.95 + correct code → reward ≥ 0.85.
- `test_submit_with_high_confidence_bad_code_low_reward` — confidence=0.95 + bogus code → reward drops sharply below quality.

## 7. Acceptance

1. `compute_reward(sandbox_score=1.0, groundedness=1.0)` (no confidence) → 1.0 (back-compat).
2. `compute_reward(sandbox_score=1.0, groundedness=1.0, confidence=1.0)` → 1.0 (perfect calibration).
3. `compute_reward(sandbox_score=0.0, groundedness=0.0, confidence=0.95)` → 0.0 (clamped; brier penalty but quality = 0).
4. `compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.1)` → ≥ 0.70 (uncertain floor, quality=0.2<0.5, confidence<0.3).
5. `compute_reward(sandbox_score=0.6, groundedness=0.6, confidence=0.1)` → no floor (quality=0.6, not <0.5).
6. Env: submitting correct easy-task code with `confidence=0.9` → reward > 0.8.
7. `ruff check` + `mypy --strict` clean on `groundloop_env/` and `models.py`.
8. All 56 existing env tests still pass.
9. Coverage ≥ 95% on `grader.py`.

## 8. Deliverables

- Updated `models.py` (add `confidence` field).
- Updated `groundloop_env/grader.py`.
- Updated `groundloop_env/environment.py::_handle_submit`.
- New tests (added to existing `test_grader.py` + `test_environment.py`).
- No new dependencies.

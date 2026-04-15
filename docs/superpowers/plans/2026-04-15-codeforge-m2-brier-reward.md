# CodeForge M2 — Brier-Calibrated Reward Plan

> Use superpowers:subagent-driven-development.

**Goal:** Agents declare `confidence` on submit; reward applies Brier penalty + uncertain-floor per round-1 EpistemicNav rules.

**Spec:** `docs/superpowers/specs/2026-04-15-codeforge-m2-brier-reward.md`.

---

## Task 1: Extend `CodeForgeAction` with `confidence` field

**Files:** `models.py`, `tests/test_models_codeforge.py`.

- [ ] Write failing tests:
```python
def test_action_accepts_confidence():
    a = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files={"x.py": "y"}, confidence=0.8)
    assert a.confidence == 0.8


def test_action_confidence_defaults_none():
    a = CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files={"x.py": "y"})
    assert a.confidence is None


def test_action_confidence_out_of_range_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files={"x.py": "y"}, confidence=1.5)
    with pytest.raises(ValidationError):
        CodeForgeAction(action_type=CodeForgeActionType.SUBMIT, files={"x.py": "y"}, confidence=-0.1)
```

- [ ] Implement: in `models.py`, add to `CodeForgeAction`:
```python
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
```
(Add the `Field` import if not already there.)

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m2): add confidence field to CodeForgeAction`

---

## Task 2: Brier-calibrated reward in grader

**Files:** `groundloop_env/grader.py`, `tests/groundloop_env/test_grader.py`.

- [ ] Add failing tests (keep existing ones, add new ones):
```python
def test_reward_without_confidence_backward_compatible():
    # No confidence → current formula (0.6*sandbox + 0.4*grounding, clamped, rounded)
    assert compute_reward(sandbox_score=1.0, groundedness=1.0) == 1.0
    assert compute_reward(sandbox_score=0.5, groundedness=0.5) == 0.5


def test_reward_perfect_calibration_no_penalty():
    # confidence == quality → brier = 0 → no penalty
    # quality = 0.6*0.5 + 0.4*0.5 = 0.5, confidence=0.5
    assert compute_reward(sandbox_score=0.5, groundedness=0.5, confidence=0.5) == 0.5


def test_reward_overconfident_wrong_heavily_penalised():
    # quality=0.2, confidence=0.95 → brier=(0.75)²=0.5625 clamped to 0.5 → reward = 0.2 * 0.5 = 0.1
    r = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.95)
    assert r < 0.2, f"expected < 0.2, got {r}"
    assert r <= 0.11


def test_reward_underconfident_right_moderately_penalised():
    # quality=0.9, confidence=0.1 → brier=(0.8)²=0.64 clamped to 0.5 → reward = 0.9 * 0.5 = 0.45
    r = compute_reward(sandbox_score=0.9, groundedness=0.9, confidence=0.1)
    assert 0.4 <= r <= 0.5


def test_reward_uncertain_floor_activates():
    # quality=0.2, confidence=0.1 → would yield ~0.18, floor bumps to 0.70
    r = compute_reward(sandbox_score=0.2, groundedness=0.2, confidence=0.1)
    assert r >= 0.70


def test_reward_uncertain_floor_not_triggered_above_threshold():
    # quality=0.6, confidence=0.1 → no floor (quality ≥ 0.5)
    r = compute_reward(sandbox_score=0.6, groundedness=0.6, confidence=0.1)
    assert r < 0.70


def test_reward_clamped_to_range():
    assert 0.0 <= compute_reward(sandbox_score=-5.0, groundedness=-5.0, confidence=0.5) <= 1.0
    assert 0.0 <= compute_reward(sandbox_score=5.0, groundedness=5.0, confidence=0.5) <= 1.0


def test_reward_deterministic_across_calls():
    a = compute_reward(sandbox_score=0.7, groundedness=0.8, confidence=0.6)
    b = compute_reward(sandbox_score=0.7, groundedness=0.8, confidence=0.6)
    assert a == b
```

- [ ] Implement new `grader.py`:
```python
from __future__ import annotations

_SANDBOX_WEIGHT = 0.6
_GROUNDING_WEIGHT = 0.4
_BRIER_CAP = 0.5
_UNCERTAIN_CONFIDENCE_THRESHOLD = 0.3
_UNCERTAIN_QUALITY_THRESHOLD = 0.5
_UNCERTAIN_FLOOR = 0.70


def compute_reward(
    *,
    sandbox_score: float,
    groundedness: float,
    confidence: float | None = None,
) -> float:
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
```

- [ ] Run tests: PASS.
- [ ] Commit: `feat(codeforge-m2): Brier-calibrated reward with uncertain-floor`

---

## Task 3: Env wires `action.confidence` through

**Files:** `groundloop_env/environment.py`, `tests/groundloop_env/test_environment.py`.

- [ ] Add failing test:
```python
def test_submit_with_high_confidence_correct_code_still_meets_target(tiny_corpus_path):
    env = CodeForgeEnvironment(corpus_path=tiny_corpus_path)
    env.reset(task_level="easy")
    good = {
        "main.py": (
            "from __future__ import annotations\n\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n"
        ),
    }
    action = CodeForgeAction(
        action_type=CodeForgeActionType.SUBMIT, files=good, confidence=0.9,
    )
    obs = env.step(action)
    # quality should be ~1.0, confidence 0.9 → brier = 0.01 → reward ~0.99
    assert obs.last_reward >= 0.85, f"got {obs.last_reward}"


def test_submit_overconfident_wrong_code_low_reward(tiny_corpus_path):
    env = CodeForgeEnvironment(corpus_path=tiny_corpus_path)
    env.reset(task_level="easy")
    bad = {"main.py": "import nonexistent_zzz_pkg\nfake_function()\n"}
    action = CodeForgeAction(
        action_type=CodeForgeActionType.SUBMIT, files=bad, confidence=0.95,
    )
    obs = env.step(action)
    # quality low, confidence high → heavy Brier penalty
    assert obs.last_reward < 0.3
```

- [ ] Implement: in `environment.py::_handle_submit`, update the `compute_reward` call:
```python
reward = compute_reward(
    sandbox_score=sandbox_score,
    groundedness=grounding_report.groundedness,
    confidence=action.confidence,
)
```

- [ ] Run tests: both PASS, plus all existing env tests still pass.
- [ ] Commit: `feat(codeforge-m2): env forwards action.confidence into grader`

---

## Task 4: Full verification

- [ ] Run full test suite: `python3 -m pytest tests/ -v --cov=groundloop_env --cov-report=term`. Expect all pass, ≥85% coverage on `groundloop_env/`.
- [ ] `ruff check groundloop_env/` + `mypy --strict groundloop_env/` — expect clean.
- [ ] Verify §7 AC §11.5 (easy task with spec-literal greet code, no confidence) still passes with `reward > 0.8`:
  ```python
  import httpx  # if available; otherwise use env directly
  # ... (test without confidence, reward must still be > 0.8)
  ```
  This is covered by the existing `test_submit_easy_task_with_correct_code_meets_target` test which omits `confidence`.
- [ ] Commit: `test(codeforge-m2): final verification of M2 integration`

---

## Self-Review

- ✅ §3 formula implemented exactly (constants named, ordering explicit).
- ✅ §5 backward compat preserved (no-confidence case = prior behavior).
- ✅ §7 acceptance criteria 1–9 all covered by tests.
- ✅ No placeholders.
- ✅ Type consistency: `confidence: float | None` identical across action/grader/env.

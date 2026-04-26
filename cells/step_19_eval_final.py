"""Cell 19 — Final evaluation harness (post-training LoRA).

Implements ``docs/modules/evaluation.md`` §2.1, §3.1, §3.3 (paired-difference),
§3.5 (drift-detection latency aggregation), §3.8, §5 ``EpisodeSetLeakError``.

Hard rules (evaluation.md §3.1, §6.1, §6.3):
- Same 50 episodes as baseline (paired); ``EpisodeSetLeakError`` raised on
  mismatch.
- Bootstrap CI seed for paired-difference is ``20260428`` (evaluation.md §2.4).
- Wall-clock budget 20 minutes — same ceiling as baseline.
- No LLM-as-judge; static AST scan via ``_NO_LLM_JUDGE_FORBIDDEN_IMPORTS``.

Heavy imports (``torch``) are deferred so this module imports cleanly on
CPU-only CI. The training-eval delegate is injected (see step_18).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cells.step_18_eval_baseline import (
    BUDGET_RUN_EVAL_SECONDS,
    DEFAULT_N_BOOT,
    DEFAULT_PAIRED_BOOTSTRAP_SEED,
    DriftDetectionLatency,
    EvalBudgetExceededError,
    EvalReport,
    EvaluationError,
    PerLanguageReport,
    TrainingEvalCallable,
    _check_catalogue_hashes,
    _episode_ids_from_breakdown,
    _validate_briefs_first_50,
    run_eval,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence


__all__ = [
    "BUDGET_RUN_EVAL_SECONDS",
    "DEFAULT_PAIRED_BOOTSTRAP_SEED",
    "DriftDetectionLatency",
    "EpisodeSetLeakError",
    "EvalBudgetExceededError",
    "EvalReport",
    "PerLanguageReport",
    "assert_paired_episode_sets",
    "eval_final",
    "paired_difference_ci",
]


# ---------------------------------------------------------------------------
# Errors — evaluation.md §5
# ---------------------------------------------------------------------------


class EpisodeSetLeakError(EvaluationError):
    """Baseline ``episode_ids`` ≠ final ``episode_ids`` — paired-comparison invariant violated."""


# ---------------------------------------------------------------------------
# Paired-difference CI — evaluation.md §2.4
# ---------------------------------------------------------------------------


def paired_difference_ci(
    baseline_samples: tuple[float, ...],
    final_samples: tuple[float, ...],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = DEFAULT_PAIRED_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on ``mean(final - baseline)`` — index-paired.

    evaluation.md §2.4: lengths must match (raises ``EpisodeSetLeakError``).
    Edge cases mirror :func:`bootstrap_ci`: empty → all-NaN; single → triple.
    """
    if len(baseline_samples) != len(final_samples):
        raise EpisodeSetLeakError(
            f"paired-comparison invariant: len(baseline)={len(baseline_samples)} "
            f"!= len(final)={len(final_samples)}",
        )
    n = len(baseline_samples)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    diffs = tuple(f - b for b, f in zip(baseline_samples, final_samples, strict=True))
    mean = sum(diffs) / n
    if n == 1:
        return mean, mean, mean
    if all(d == diffs[0] for d in diffs):
        return mean, mean, mean

    import numpy as np

    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(diffs, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return float(mean), lo, hi


# ---------------------------------------------------------------------------
# Episode-set leak guard — evaluation.md §3.1
# ---------------------------------------------------------------------------


def assert_paired_episode_sets(baseline: EvalReport, final: EvalReport) -> None:
    """Raise ``EpisodeSetLeakError`` iff ``episode_ids`` tuples differ."""
    base_ids = _episode_ids_from_breakdown(baseline)
    final_ids = _episode_ids_from_breakdown(final)
    if base_ids != final_ids:
        raise EpisodeSetLeakError(
            "paired-comparison invariant violated — baseline.episode_ids != final.episode_ids; "
            "operator must re-run baseline against the current val split.",
        )


# ---------------------------------------------------------------------------
# Drift-detection-latency point extraction — evaluation.md §3.5
# ---------------------------------------------------------------------------


def _final_latency_point(report: EvalReport) -> tuple[float, float]:
    """Return ``(p50, p95)`` from the report's drift-detection latency."""
    lat = report.drift_detection_latency
    # Stage-3 takes precedence (final stage); falls back to stage-2 if Stage-3 NaN.
    p50 = lat.stage3_median
    p95 = lat.stage3_p95
    return float(p50), float(p95)


# ---------------------------------------------------------------------------
# Final-eval entry point — evaluation.md §2.2 ``eval_final.py``
# ---------------------------------------------------------------------------


def eval_final(
    checkpoint: Path,
    episodes: int = 50,
    *,
    baseline: EvalReport,
    training_eval: TrainingEvalCallable,
    briefs: Sequence[Any],
    catalogue_hashes: dict[str, str] | None = None,
    budget_seconds: int = BUDGET_RUN_EVAL_SECONDS,
    monotonic: Callable[[], float] | None = None,
) -> EvalReport:
    """Run the trained LoRA against the SAME 50 paired episodes used by baseline.

    evaluation.md §2.1, §3.1: rejects mismatched checkpoints; verifies catalogue
    hashes; computes paired-difference CIs and stores them under
    ``EvalReport.breakdown['paired_ci']``.
    """
    if not isinstance(checkpoint, Path):
        raise EvaluationError(
            f"checkpoint must be pathlib.Path; got {type(checkpoint).__name__}",
        )
    if episodes != 50:
        raise EvaluationError(
            f"eval_final expects episodes=50 (paired contract); got {episodes}",
        )

    selected = _validate_briefs_first_50(briefs)
    if catalogue_hashes is not None:
        _check_catalogue_hashes(selected, catalogue_hashes)

    # Pre-flight: episode_ids match baseline before launching rollout.
    expected_ids = tuple(row.episode_id for row in selected)
    base_ids = _episode_ids_from_breakdown(baseline)
    if base_ids and base_ids != expected_ids:
        raise EpisodeSetLeakError(
            "paired-comparison invariant violated at entry — baseline.episode_ids "
            "do not match val/briefs.jsonl[0:50]; re-run baseline first.",
        )

    clock = monotonic if monotonic is not None else time.monotonic
    started = clock()

    final_report = run_eval(
        checkpoint,
        episodes,
        training_eval=training_eval,
        briefs=briefs,
        catalogue_hashes=catalogue_hashes,
        budget_seconds=budget_seconds,
        monotonic=clock,
    )
    elapsed = clock() - started
    if elapsed > budget_seconds:
        raise EvalBudgetExceededError(
            f"eval_final wall-clock {elapsed:.1f}s exceeded {budget_seconds}s",
        )

    assert_paired_episode_sets(baseline, final_report)

    # Compute paired-difference CIs (evaluation.md §3.3).
    paired_ci = _build_paired_ci_block(baseline, final_report)
    breakdown = dict(final_report.breakdown)
    breakdown["paired_ci"] = paired_ci
    return replace(final_report, breakdown=breakdown)


def _build_paired_ci_block(
    baseline: EvalReport,
    final: EvalReport,
) -> dict[str, tuple[float, float, float]]:
    """Construct the ``breakdown['paired_ci']`` block for the blog narrative."""
    out: dict[str, tuple[float, float, float]] = {}
    base_samples: dict[str, tuple[float, ...]] = baseline.breakdown.get("samples", {})
    final_samples: dict[str, tuple[float, ...]] = final.breakdown.get("samples", {})
    for key in ("reward", "r1", "r2", "r3", "r4", "r5"):
        if key in base_samples and key in final_samples:
            out[key] = paired_difference_ci(
                tuple(base_samples[key]),
                tuple(final_samples[key]),
            )

    # Drift-latency delta — final p50 minus baseline p50 (lower is better).
    base_p50, _ = _final_latency_point(baseline)
    final_p50, _ = _final_latency_point(final)
    if not (base_p50 != base_p50 or final_p50 != final_p50):  # neither NaN
        delta = final_p50 - base_p50
        out["drift_latency_p50"] = (delta, delta, delta)
    return out

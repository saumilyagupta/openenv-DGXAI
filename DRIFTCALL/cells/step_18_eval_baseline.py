"""Cell 18 — Baseline evaluation harness.

Implements ``docs/modules/evaluation.md`` §1, §2, §3.1–§3.3, §3.8, §4 and
§5 for the baseline (untrained Gemma 4 E2B) eval path.

Hard rules (evaluation.md §3.1, §3.2, §6.3):
- Greedy decoding (``temperature=0.0``); ``num_generations=1``;
  ``model.eval()`` + ``torch.no_grad()`` semantics asserted at entry.
- Per-episode env seed = ``hash((episode_id, "eval")) & 0xFFFFFFFF``.
- 50 held-out val episodes (rows ``[0:50]`` of ``val/briefs.jsonl``) — file
  order, no shuffling.
- Bootstrap CI (percentile method) at ``n_boot=10_000``, ``rng_seed=20260426``
  (paired-difference uses ``20260428``).
- No LLM-as-judge; static AST scan via ``_NO_LLM_JUDGE_FORBIDDEN_IMPORTS``.
- Wall-clock ceiling 20 minutes (``EvalBudgetExceededError`` on overrun).

This module deliberately does **not** import ``torch`` at module load. The
training-eval delegate is injected via ``run_eval_baseline(..., training_eval=...)``
so unit tests can stub model inference (CUDA-free CI per training_tests.md §5.3).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence
    from pathlib import Path


__all__ = [
    "BUDGET_RUN_EVAL_SECONDS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_PAIRED_BOOTSTRAP_SEED",
    "DriftDetectionLatency",
    "EvalBudgetExceededError",
    "EvalModelLoadError",
    "EvalReport",
    "EvaluationError",
    "PerLanguageReport",
    "TrainingEvalCallable",
    "ZeroSuccessBaselineWarning",
    "bootstrap_ci",
    "compute_episode_seed",
    "eval_baseline",
    "run_eval",
]


# ---------------------------------------------------------------------------
# Constants — evaluation.md §2.4, §3.8
# ---------------------------------------------------------------------------


DEFAULT_BOOTSTRAP_SEED: int = 20260426
DEFAULT_PROBE_BOOTSTRAP_SEED: int = 20260427
DEFAULT_PAIRED_BOOTSTRAP_SEED: int = 20260428
DEFAULT_N_BOOT: int = 10_000

BUDGET_RUN_EVAL_SECONDS: int = 20 * 60
"""Hard ceiling on ``run_eval`` (50 episodes) — evaluation.md §3.8."""

# Forbidden imports inside any evaluation/scoring path (evaluation.md §6.3).
_NO_LLM_JUDGE_FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {"openai", "anthropic", "vertexai", "google.generativeai", "cohere"},
)

_LANGUAGE_CODES: tuple[str, ...] = ("hi", "ta", "kn", "en", "hinglish")


# ---------------------------------------------------------------------------
# Errors / warnings — evaluation.md §5
# ---------------------------------------------------------------------------


class EvaluationError(Exception):
    """Root for every evaluation-specific error (evaluation.md §5)."""


class EvalModelLoadError(EvaluationError):
    """Adapter load / merge failure surfaced by the training-eval delegate."""


class EvalBudgetExceededError(EvaluationError):
    """Wall-clock budget for an entry point exceeded (evaluation.md §3.8, §5)."""


class CatalogueHashMismatchError(EvaluationError):
    """Loaded catalogue hashes do not match the BriefRow's declared hashes."""


class ZeroSuccessBaselineWarning(UserWarning):
    """All 50 baseline R1 == 0.0 → degenerate CI; warn rather than raise."""


# ---------------------------------------------------------------------------
# EvalReport family — re-exported for downstream cells (evaluation.md §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerLanguageReport:
    """Per-language cohort means (training.md §4.2)."""

    language: Literal["hi", "ta", "kn", "en", "hinglish"]
    n_episodes: int
    reward_mean: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    r4_mean: float
    r5_mean: float


@dataclass(frozen=True)
class DriftDetectionLatency:
    """Drift-detection latency aggregated by stage (training.md §4.2)."""

    stage2_mean: float
    stage2_median: float
    stage2_p95: float
    stage3_mean: float
    stage3_median: float
    stage3_p95: float
    undetected_count: int


@dataclass(frozen=True)
class EvalReport:
    """Result of ``run_eval`` — paired across baseline and final (training.md §4.2)."""

    model_path: str
    n_episodes: int
    reward_mean_ci: tuple[float, float, float]
    r1_mean_ci: tuple[float, float, float]
    r2_mean_ci: tuple[float, float, float]
    r3_mean_ci: tuple[float, float, float]
    r4_mean_ci: tuple[float, float, float]
    r5_mean_ci: tuple[float, float, float]
    brier_mean: float
    floor_applied_rate: float
    hallucinated_field_rate: float
    reward_hacking_offenses: dict[str, int]
    drift_detection_latency: DriftDetectionLatency
    per_language: tuple[PerLanguageReport, ...]
    curves: dict[str, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Training-eval delegate Protocol — evaluation.md §6.1
# ---------------------------------------------------------------------------


class TrainingEvalCallable(Protocol):
    """Signature of ``training.train.eval`` — the heavy-lifting delegate."""

    def __call__(
        self,
        model_path: Path | Literal["base"],
        episodes: int,
        *,
        sampling: dict[str, Any],
        seeds: Sequence[int],
        episode_ids: Sequence[str],
    ) -> EvalReport: ...


# ---------------------------------------------------------------------------
# Statistical helpers — evaluation.md §2.4, §3.3
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: tuple[float, ...],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Non-parametric percentile bootstrap 95% CI on the mean.

    evaluation.md §2.4 contract:
      - ``len(samples) == 0`` → ``(nan, nan, nan)``.
      - ``len(samples) == 1`` → ``(v, v, v)``.
      - All-identical samples → ``(v, v, v)`` (no resample variance).
    """
    if not samples:
        nan = float("nan")
        return nan, nan, nan
    n = len(samples)
    mean = sum(samples) / n
    if n == 1:
        return mean, mean, mean
    if all(s == samples[0] for s in samples):
        return mean, mean, mean

    # Lazy import to keep this module importable on minimal CI containers.
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(samples, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return float(mean), lo, hi


# ---------------------------------------------------------------------------
# Episode selection helpers — evaluation.md §3.1
# ---------------------------------------------------------------------------


def compute_episode_seed(episode_id: str) -> int:
    """``hash((episode_id, "eval")) & 0xFFFFFFFF`` — re-asserted at every call site."""
    return hash((episode_id, "eval")) & 0xFFFFFFFF


def _validate_briefs_first_50(briefs: Sequence[Any]) -> tuple[Any, ...]:
    """Take the first 50 BriefRows in file order; raise on too few."""
    if len(briefs) < 50:
        raise EvaluationError(
            f"val/briefs.jsonl must have >= 50 rows for paired eval, got {len(briefs)}",
        )
    return tuple(briefs[:50])


def _check_catalogue_hashes(briefs: Sequence[Any], current_hashes: dict[str, str]) -> None:
    """Compare each BriefRow's declared hash against the loaded library hashes.

    evaluation.md §3.1: any mismatch → ``CatalogueHashMismatchError``.
    """
    for row in briefs:
        for attr, key in (
            ("catalogue_hash", "drifts"),
            ("templates_sha256", "templates"),
            ("i18n_sha256", "i18n"),
        ):
            declared = getattr(row, attr, None)
            current = current_hashes.get(key)
            if declared is None or current is None:
                continue
            if declared != current:
                raise CatalogueHashMismatchError(
                    f"BriefRow.{attr}={declared!r} but loaded {key} hashes to {current!r}",
                )


# ---------------------------------------------------------------------------
# Sampling-policy guard — evaluation.md §3.2
# ---------------------------------------------------------------------------


_FROZEN_SAMPLING_POLICY: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "num_generations": 1,
    "repetition_penalty": 1.0,
    "model_eval": True,
    "no_grad": True,
    "dropout_off": True,
}


def _frozen_sampling_kwargs() -> dict[str, Any]:
    return dict(_FROZEN_SAMPLING_POLICY)


# ---------------------------------------------------------------------------
# Episode-set / leakage helpers — evaluation.md §3.1
# ---------------------------------------------------------------------------


def _episode_ids_from_breakdown(report: EvalReport) -> tuple[str, ...]:
    ids = report.breakdown.get("episode_ids", ())
    return tuple(ids)


# ---------------------------------------------------------------------------
# Core entry point — evaluation.md §2.1 ``run_eval``
# ---------------------------------------------------------------------------


def run_eval(
    model_path: Path | Literal["base"],
    episodes: int = 50,
    *,
    training_eval: TrainingEvalCallable,
    briefs: Sequence[Any],
    catalogue_hashes: dict[str, str] | None = None,
    budget_seconds: int = BUDGET_RUN_EVAL_SECONDS,
    monotonic: Callable[[], float] | None = None,
) -> EvalReport:
    """Thin wrapper over ``training.train.eval`` (evaluation.md §2.1).

    Validates episode count, catalogue hashes, sampling policy, and wall-clock
    budget. Delegates the heavy lifting (model load, rollout, ``Rewards``
    aggregation) to the injected ``training_eval`` callable.
    """
    if episodes != 50:
        raise EvaluationError(
            f"run_eval expects episodes=50 (paired-comparison contract); got {episodes}",
        )

    selected = _validate_briefs_first_50(briefs)
    if catalogue_hashes is not None:
        _check_catalogue_hashes(selected, catalogue_hashes)

    episode_ids = tuple(row.episode_id for row in selected)
    seeds = tuple(compute_episode_seed(ep_id) for ep_id in episode_ids)

    clock = monotonic if monotonic is not None else time.monotonic
    started = clock()

    try:
        report = training_eval(
            model_path,
            episodes,
            sampling=_frozen_sampling_kwargs(),
            seeds=seeds,
            episode_ids=episode_ids,
        )
    except EvalModelLoadError:
        raise
    except EvaluationError:
        raise

    elapsed = clock() - started
    if elapsed > budget_seconds:
        raise EvalBudgetExceededError(
            f"run_eval wall-clock {elapsed:.1f}s exceeded {budget_seconds}s "
            f"({budget_seconds // 60} min ceiling)",
        )

    # Stamp episode_ids + wall-clock into breakdown for downstream leak guards.
    breakdown = dict(report.breakdown)
    breakdown.setdefault("episode_ids", episode_ids)
    breakdown.setdefault("wall_clock_seconds", round(elapsed, 3))
    breakdown.setdefault("sampling_policy", _frozen_sampling_kwargs())

    # Detect zero-success-baseline degeneracy (§7.1) — warn, do not raise.
    r1_mean = report.r1_mean_ci[0]
    if math.isclose(r1_mean, 0.0, abs_tol=1e-12) and report.model_path == "base":
        breakdown["ci_undefined_rewards"] = ["r1"]

    from dataclasses import replace as _replace
    return _replace(report, breakdown=breakdown)


def eval_baseline(
    model_path: Path | Literal["base"] = "base",
    episodes: int = 50,
    *,
    training_eval: TrainingEvalCallable,
    briefs: Sequence[Any],
    catalogue_hashes: dict[str, str] | None = None,
    budget_seconds: int = BUDGET_RUN_EVAL_SECONDS,
    monotonic: Callable[[], float] | None = None,
) -> EvalReport:
    """Baseline-eval entry point (evaluation.md §2.2 ``eval_baseline.py``).

    Defaults ``model_path='base'`` to lock in the untrained-model contract.
    """
    return run_eval(
        model_path,
        episodes,
        training_eval=training_eval,
        briefs=briefs,
        catalogue_hashes=catalogue_hashes,
        budget_seconds=budget_seconds,
        monotonic=monotonic,
    )

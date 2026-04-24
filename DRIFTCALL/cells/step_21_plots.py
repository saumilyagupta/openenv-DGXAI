"""Cell 21 — Eval-curve renderer (4 plot panels for DESIGN.md §15 pitch).

Implements ``docs/modules/evaluation.md`` §2.1 ``render_plots``, §3.4
(per-language bars), §3.5 (drift-detection latency curve), §3.8 (2-min
budget), §5 ``PlotRenderError`` / ``WandBHistoryUnavailableWarning``,
§7 edge cases 2 (empty cohort), 3 (Stage-1 NaN), 6 (WandB purged).

Hard rules (evaluation.md §3.8, §6.3):
- ``matplotlib`` only; no seaborn.
- Canonical figsize ``(16, 9)`` inches at ``dpi=100`` → ``1600x900`` px PNGs.
- ``wandb_run_id is None`` → skip the two history-driven plots, render the
  other two; warn via ``WandBHistoryUnavailableWarning``.
- Wall-clock budget 2 minutes (``EvalBudgetExceededError``).
- No LLM-as-judge; static AST scan via ``_NO_LLM_JUDGE_FORBIDDEN_IMPORTS``.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cells.step_18_eval_baseline import (
    EvalBudgetExceededError,
    EvalReport,
    EvaluationError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable


__all__ = [
    "BUDGET_RENDER_PLOTS_SECONDS",
    "CANONICAL_FIGSIZE",
    "CANONICAL_DPI",
    "PlotRenderError",
    "WandBHistoryUnavailableWarning",
    "render_plots",
]


# ---------------------------------------------------------------------------
# Constants — evaluation.md §3.8
# ---------------------------------------------------------------------------


CANONICAL_FIGSIZE: tuple[float, float] = (16.0, 9.0)
"""evaluation.md integration §3.4 — every PNG is 1600x900 px at dpi=100."""

CANONICAL_DPI: int = 100

BUDGET_RENDER_PLOTS_SECONDS: int = 120
"""evaluation.md §3.8 — 2-minute hard ceiling on ``render_plots``."""

_NO_LLM_JUDGE_FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {"openai", "anthropic", "vertexai", "google.generativeai", "cohere"},
)


# ---------------------------------------------------------------------------
# Errors / warnings — evaluation.md §5
# ---------------------------------------------------------------------------


class PlotRenderError(EvaluationError):
    """``matplotlib`` save failure (disk full / unwriteable / missing font)."""


class WandBHistoryUnavailableWarning(UserWarning):
    """WandB history fetch failed — degrade gracefully (skip 2 plots)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_figure(title: str) -> Any:
    """Return a new (fig, ax) pair pinned to the canonical figsize."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=CANONICAL_FIGSIZE, dpi=CANONICAL_DPI)
    ax.set_title(title)
    return fig, ax


def _save_figure(fig: Any, out_path: Path) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=CANONICAL_DPI, bbox_inches="tight")
    except OSError as exc:  # disk full, unwriteable
        raise PlotRenderError(
            f"failed to save plot to {out_path}: {exc}",
        ) from exc
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def _wandb_curves(wandb_run_id: str | None) -> dict[str, list[tuple[int, float]]]:
    """Try to fetch WandB history; return ``{}`` and warn on any failure."""
    if wandb_run_id is None:
        warnings.warn(
            "WandB run id is None — per_reward_stack and drift_latency_vs_step skipped.",
            WandBHistoryUnavailableWarning,
            stacklevel=2,
        )
        return {}
    wandb = _try_import_wandb()
    if wandb is None:
        warnings.warn(
            f"wandb import failed — history for {wandb_run_id!r} unavailable.",
            WandBHistoryUnavailableWarning,
            stacklevel=2,
        )
        return {}
    history = _try_fetch_wandb_history(wandb, wandb_run_id)
    if history is None:
        warnings.warn(
            f"WandB fetch failed for run {wandb_run_id!r}.",
            WandBHistoryUnavailableWarning,
            stacklevel=2,
        )
        return {}
    return _coerce_history(history)


def _try_import_wandb() -> Any:
    """Best-effort wandb import; returns ``None`` on failure."""
    import importlib
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def _try_fetch_wandb_history(wandb_mod: Any, run_id: str) -> Any:
    """Best-effort history fetch; returns ``None`` on any failure."""
    try:
        api = wandb_mod.Api()
        run = api.run(run_id)
        return run.history()
    except (RuntimeError, ValueError, ImportError, AttributeError, KeyError, TypeError):
        return None


def _coerce_history(history: Any) -> dict[str, list[tuple[int, float]]]:
    """Coerce a WandB history (DataFrame-like) into per-key (step, value) pairs."""
    if isinstance(history, dict):
        out: dict[str, list[tuple[int, float]]] = {}
        for key, rows in history.items():
            if isinstance(rows, list):
                out[key] = [(int(r[0]), float(r[1])) for r in rows]
        return out
    return {}


# ---------------------------------------------------------------------------
# Plot 1 — per-reward stack — evaluation.md §3.5 (over training steps)
# ---------------------------------------------------------------------------


def _plot_per_reward_stack(curves: dict[str, list[tuple[int, float]]], out_path: Path) -> Path:
    fig, ax = _new_figure("Per-reward means vs training step")
    keys = ("R1_mean", "R2_mean", "R3_mean", "R4_mean", "R5_mean")
    found_any = False
    for key in keys:
        rows = curves.get(f"train/{key}") or curves.get(key)
        if not rows:
            continue
        found_any = True
        steps = [r[0] for r in rows]
        values = [r[1] for r in rows]
        ax.plot(steps, values, label=key)
    if not found_any:
        ax.text(0.5, 0.5, "No WandB history available", ha="center", va="center")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward mean")
    ax.legend(loc="best")
    _save_figure(fig, out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Plot 2 — drift-detection latency vs step — evaluation.md §3.5
# ---------------------------------------------------------------------------


def _plot_drift_latency_vs_step(
    curves: dict[str, list[tuple[int, float]]],
    final: EvalReport,
    out_path: Path,
) -> Path:
    fig, ax = _new_figure("Drift-detection latency vs training step")
    p50_rows = curves.get("eval/drift_latency_p50") or []
    p95_rows = curves.get("eval/drift_latency_p95") or []
    if p50_rows:
        ax.plot([r[0] for r in p50_rows], [r[1] for r in p50_rows], label="p50")
    if p95_rows:
        ax.plot([r[0] for r in p95_rows], [r[1] for r in p95_rows], label="p95")

    # Final point (rightmost) from the held-out 50 (evaluation.md §3.5 fusion).
    p50_final = final.drift_detection_latency.stage3_median
    if not math.isnan(p50_final) and p50_rows:
        last_step = p50_rows[-1][0] + 50
        ax.scatter([last_step], [p50_final], label="final p50", marker="*", s=120)

    if not p50_rows and not p95_rows:
        ax.text(0.5, 0.5, "Stage 1 eval — no drift events", ha="center", va="center")
    ax.set_xlabel("training step")
    ax.set_ylabel("turns to adapt")
    ax.legend(loc="best")
    _save_figure(fig, out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Plot 3 — per-language bars — evaluation.md §3.4
# ---------------------------------------------------------------------------


def _plot_per_language_bars(final: EvalReport, out_path: Path) -> Path:
    fig, ax = _new_figure("Per-language reward breakdown (final)")
    cohorts = [c for c in final.per_language if c.n_episodes > 0]
    if not cohorts:
        ax.text(0.5, 0.5, "No non-empty per-language cohorts", ha="center", va="center")
        _save_figure(fig, out_path)
        return out_path.resolve()

    languages = [c.language for c in cohorts]
    rewards = ("r1_mean", "r2_mean", "r3_mean", "r4_mean", "r5_mean")
    n_groups = len(languages)
    bar_width = 0.15
    import numpy as np

    x = np.arange(n_groups)
    for i, key in enumerate(rewards):
        values = [getattr(c, key) for c in cohorts]
        ax.bar(x + i * bar_width, values, bar_width, label=key.upper())
    ax.set_xticks(x + 2 * bar_width)
    ax.set_xticklabels(languages)
    ax.set_xlabel("language")
    ax.set_ylabel("mean")
    ax.legend(loc="best")

    # Annotate low-n cohorts (1-4) with '(low-n)' suffix per evaluation.md §3.4.
    for c, xi in zip(cohorts, x, strict=True):
        if 1 <= c.n_episodes <= 4:
            ax.annotate(
                f"(low-n n={c.n_episodes})",
                xy=(xi + 2 * bar_width, 0),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    _save_figure(fig, out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Plot 4 — before/after bars — evaluation.md §2.1
# ---------------------------------------------------------------------------


def _plot_before_after_bars(
    baseline: EvalReport,
    final: EvalReport,
    out_path: Path,
) -> Path:
    fig, ax = _new_figure("Baseline vs Final — per-reward means with 95% CI")
    keys = ("reward", "r1", "r2", "r3", "r4", "r5")
    n_groups = len(keys)
    import numpy as np

    x = np.arange(n_groups)
    bar_w = 0.35
    base_means: list[float] = []
    base_errs: list[tuple[float, float]] = []
    final_means: list[float] = []
    final_errs: list[tuple[float, float]] = []
    for key in keys:
        b_mean, b_lo, b_hi = getattr(baseline, f"{key}_mean_ci")
        f_mean, f_lo, f_hi = getattr(final, f"{key}_mean_ci")
        base_means.append(b_mean)
        base_errs.append((b_mean - b_lo, b_hi - b_mean))
        final_means.append(f_mean)
        final_errs.append((f_mean - f_lo, f_hi - f_mean))

    base_err_arr = np.asarray(base_errs).T
    final_err_arr = np.asarray(final_errs).T
    ax.bar(x - bar_w / 2, base_means, bar_w, yerr=base_err_arr, label="baseline", capsize=4)
    ax.bar(x + bar_w / 2, final_means, bar_w, yerr=final_err_arr, label="final", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in keys])
    ax.set_xlabel("reward channel")
    ax.set_ylabel("mean (95% CI)")
    ax.legend(loc="best")

    # Zero-success-baseline annotation per evaluation.md §7.1.
    if math.isclose(baseline.r1_mean_ci[0], 0.0, abs_tol=1e-12):
        ax.annotate(
            "0 of 50 successes",
            xy=(1 - bar_w / 2, 0),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    _save_figure(fig, out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Public entry point — evaluation.md §2.1
# ---------------------------------------------------------------------------


def render_plots(
    baseline: EvalReport,
    final: EvalReport,
    wandb_run_id: str | None,
    out_dir: Path,
    *,
    budget_seconds: int = BUDGET_RENDER_PLOTS_SECONDS,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, Path]:
    """Render the 4 plot panels (evaluation.md §2.1, §3.5).

    ``wandb_run_id=None`` → skip the two history-driven plots, render the
    other two; warn via ``WandBHistoryUnavailableWarning``.
    """
    if not isinstance(out_dir, Path):
        raise EvaluationError(
            f"out_dir must be pathlib.Path; got {type(out_dir).__name__}",
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    clock = monotonic if monotonic is not None else time.monotonic
    started = clock()

    paths: dict[str, Path] = {}
    curves = _wandb_curves(wandb_run_id)

    if wandb_run_id is not None and curves:
        paths["per_reward_stack"] = _plot_per_reward_stack(
            curves, out_dir / "per_reward_stack.png",
        )
        paths["drift_latency_vs_step"] = _plot_drift_latency_vs_step(
            curves, final, out_dir / "drift_latency_vs_step.png",
        )

    paths["per_language_bars"] = _plot_per_language_bars(
        final, out_dir / "per_language_bars.png",
    )
    paths["before_after_bars"] = _plot_before_after_bars(
        baseline, final, out_dir / "before_after_bars.png",
    )

    elapsed = clock() - started
    if elapsed > budget_seconds:
        raise EvalBudgetExceededError(
            f"render_plots wall-clock {elapsed:.1f}s exceeded {budget_seconds}s "
            f"({budget_seconds // 60} min ceiling)",
        )
    return paths

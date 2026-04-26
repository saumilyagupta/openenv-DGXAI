"""Tests for cells/step_21_plots.py.

Covers evaluation.md §2.1 (render_plots), §3.4, §3.5, §3.8, §5
PlotRenderError / WandBHistoryUnavailableWarning. matplotlib uses Agg backend.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from cells.step_18_eval_baseline import (
    DriftDetectionLatency,
    EvalBudgetExceededError,
    EvalReport,
    PerLanguageReport,
)
from cells.step_21_plots import (
    BUDGET_RENDER_PLOTS_SECONDS,
    CANONICAL_DPI,
    CANONICAL_FIGSIZE,
    PlotRenderError,
    WandBHistoryUnavailableWarning,
    render_plots,
)


def _empty_latency() -> DriftDetectionLatency:
    nan = float("nan")
    return DriftDetectionLatency(nan, nan, nan, nan, nan, nan, 0)


def _populated_latency() -> DriftDetectionLatency:
    return DriftDetectionLatency(
        stage2_mean=1.2, stage2_median=1.0, stage2_p95=2.0,
        stage3_mean=1.6, stage3_median=1.0, stage3_p95=2.0,
        undetected_count=9,
    )


def _baseline_report() -> EvalReport:
    return EvalReport(
        model_path="base",
        n_episodes=50,
        reward_mean_ci=(0.118, 0.086, 0.152),
        r1_mean_ci=(0.0, 0.0, 0.0),  # zero-success baseline
        r2_mean_ci=(0.254, 0.198, 0.310),
        r3_mean_ci=(0.320, 0.262, 0.378),
        r4_mean_ci=(0.640, 0.588, 0.692),
        r5_mean_ci=(-0.186, -0.240, -0.132),
        brier_mean=0.412,
        floor_applied_rate=0.08,
        hallucinated_field_rate=0.14,
        reward_hacking_offenses={"hallucinated_field": 7},
        drift_detection_latency=_empty_latency(),
        per_language=(
            PerLanguageReport("hi", 11, 0.103, 0.09, 0.20, 0.31, 0.64, -0.18),
            PerLanguageReport("ta", 10, 0.098, 0.10, 0.25, 0.28, 0.60, -0.22),
            PerLanguageReport("kn",  9, 0.081, 0.00, 0.22, 0.30, 0.58, -0.24),
            PerLanguageReport("en", 10, 0.184, 0.20, 0.30, 0.38, 0.71, -0.12),
            PerLanguageReport("hinglish", 10, 0.124, 0.10, 0.28, 0.33, 0.67, -0.17),
        ),
        breakdown={},
    )


def _final_report() -> EvalReport:
    return EvalReport(
        model_path="/ckpt/stage3_final",
        n_episodes=50,
        reward_mean_ci=(0.542, 0.480, 0.604),
        r1_mean_ci=(0.580, 0.460, 0.700),
        r2_mean_ci=(0.740, 0.680, 0.800),
        r3_mean_ci=(0.610, 0.548, 0.672),
        r4_mean_ci=(0.880, 0.842, 0.918),
        r5_mean_ci=(-0.040, -0.080, 0.000),
        brier_mean=0.081,
        floor_applied_rate=0.04,
        hallucinated_field_rate=0.02,
        reward_hacking_offenses={"hallucinated_field": 1},
        drift_detection_latency=_populated_latency(),
        per_language=(
            PerLanguageReport("hi", 11, 0.55, 0.6, 0.75, 0.6, 0.9, -0.05),
            PerLanguageReport("ta", 10, 0.52, 0.55, 0.7, 0.6, 0.85, -0.06),
            PerLanguageReport("kn",  9, 0.48, 0.5, 0.7, 0.55, 0.85, -0.07),
            PerLanguageReport("en", 10, 0.60, 0.65, 0.78, 0.65, 0.91, -0.03),
            PerLanguageReport("hinglish", 10, 0.55, 0.6, 0.75, 0.6, 0.9, -0.05),
        ),
        breakdown={},
    )


# ---------------------------------------------------------------------------
# render_plots — happy path (WandB None)
# ---------------------------------------------------------------------------


def test_render_plots_without_wandb_emits_2_plots(tmp_path: Path) -> None:
    with pytest.warns(WandBHistoryUnavailableWarning):
        paths = render_plots(_baseline_report(), _final_report(), None, tmp_path)
    assert set(paths) == {"per_language_bars", "before_after_bars"}
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 0


def test_render_plots_creates_canonical_figsize_pngs(tmp_path: Path) -> None:
    """PNGs are saved at the canonical figsize (16x9 in @ dpi=100)."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    with pytest.warns(WandBHistoryUnavailableWarning):
        paths = render_plots(_baseline_report(), _final_report(), None, tmp_path)
    img = Image.open(paths["before_after_bars"])
    # bbox_inches='tight' may shrink slightly; assert >= floor.
    assert img.width >= 1200
    assert img.height >= 600


def test_render_plots_rejects_non_path_out_dir() -> None:
    from cells.step_18_eval_baseline import EvaluationError
    with pytest.raises(EvaluationError, match="pathlib.Path"):
        render_plots(_baseline_report(), _final_report(), None, "/tmp")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Empty per-language cohorts
# ---------------------------------------------------------------------------


def test_render_per_language_handles_empty_cohort(tmp_path: Path) -> None:
    """A cohort with n_episodes == 0 must be filtered out (evaluation.md §7.2)."""
    from dataclasses import replace as _replace
    base = _baseline_report()
    final = _final_report()
    nan = float("nan")
    swapped = tuple(
        PerLanguageReport("kn", 0, nan, nan, nan, nan, nan, nan)
        if pl.language == "kn" else pl
        for pl in final.per_language
    )
    final_no_kn = _replace(final, per_language=swapped)
    with pytest.warns(WandBHistoryUnavailableWarning):
        paths = render_plots(base, final_no_kn, None, tmp_path)
    assert paths["per_language_bars"].exists()


def test_render_per_language_all_empty(tmp_path: Path) -> None:
    """All cohorts empty → renderer still emits the PNG (with placeholder text)."""
    from dataclasses import replace as _replace
    base = _baseline_report()
    final = _final_report()
    nan = float("nan")
    final_empty = _replace(
        final,
        per_language=tuple(
            PerLanguageReport(pl.language, 0, nan, nan, nan, nan, nan, nan)
            for pl in final.per_language
        ),
    )
    with pytest.warns(WandBHistoryUnavailableWarning):
        paths = render_plots(base, final_empty, None, tmp_path)
    assert paths["per_language_bars"].exists()


# ---------------------------------------------------------------------------
# Wall-clock budget — evaluation.md §3.8
# ---------------------------------------------------------------------------


def test_render_plots_budget_2min_exceeded_raises(tmp_path: Path) -> None:
    fake_clock = iter([0.0, BUDGET_RENDER_PLOTS_SECONDS + 1.0])
    with pytest.warns(WandBHistoryUnavailableWarning), pytest.raises(EvalBudgetExceededError, match="2 min"):
        render_plots(
            _baseline_report(), _final_report(), None, tmp_path,
            monotonic=lambda: next(fake_clock),
        )


# ---------------------------------------------------------------------------
# WandB unavailable degrade path
# ---------------------------------------------------------------------------


def test_render_plots_warns_when_wandb_run_id_provided_but_lib_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When wandb is unavailable / fails, render_plots warns + renders 2 plots."""
    import sys
    # Force wandb import to fail.
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.warns(WandBHistoryUnavailableWarning):
        paths = render_plots(_baseline_report(), _final_report(), "stub-run", tmp_path)
    # History-driven plots are skipped (graceful degrade).
    assert "per_language_bars" in paths
    assert "before_after_bars" in paths


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------


def test_canonical_figsize_is_16x9() -> None:
    assert CANONICAL_FIGSIZE == (16.0, 9.0)
    assert CANONICAL_DPI == 100


# ---------------------------------------------------------------------------
# No LLM-as-judge static AST scan
# ---------------------------------------------------------------------------


def test_no_llm_judge_imports_in_plots_module() -> None:
    src = Path("cells/step_21_plots.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in ("openai", "anthropic", "vertexai"):
                    assert forbidden not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for forbidden in ("openai", "anthropic", "vertexai"):
                assert forbidden not in node.module


# ---------------------------------------------------------------------------
# PlotRenderError on unwriteable path
# ---------------------------------------------------------------------------


def test_plot_render_error_raised_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.figure  # noqa: PLC0415

    original_savefig = matplotlib.figure.Figure.savefig

    def boom(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise OSError("disk full simulation")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", boom)
    with pytest.warns(WandBHistoryUnavailableWarning), pytest.raises(PlotRenderError, match="disk full"):
        render_plots(_baseline_report(), _final_report(), None, tmp_path)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", original_savefig)


# ---------------------------------------------------------------------------
# Integration test — full plot rendering with WandB stub
# ---------------------------------------------------------------------------


def test_integration_render_plots_with_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: history-driven + bars-driven plots all render to disk."""
    import sys
    import types

    fake_history = {
        "train/R1_mean": [(0, 0.1), (50, 0.18), (100, 0.26), (200, 0.41)],
        "train/R2_mean": [(0, 0.25), (50, 0.32), (100, 0.44), (200, 0.6)],
        "eval/drift_latency_p50": [(50, 2.0), (100, 1.5), (200, 1.0)],
        "eval/drift_latency_p95": [(50, 2.5), (100, 2.0), (200, 1.5)],
    }

    fake_run = types.SimpleNamespace(history=lambda: fake_history)
    fake_api = types.SimpleNamespace(run=lambda _id: fake_run)
    fake_wandb = types.SimpleNamespace(Api=lambda: fake_api)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    paths = render_plots(_baseline_report(), _final_report(), "team/run-id", tmp_path)
    assert {"per_reward_stack", "drift_latency_vs_step", "per_language_bars",
            "before_after_bars"} <= set(paths)
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 0

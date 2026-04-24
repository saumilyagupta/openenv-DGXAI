"""Tests for cells/step_22_summary.py.

Covers evaluation.md §3.3, §3.4, §3.5; the markdown summary table that ships
in the HF blog and DESIGN.md §15 pitch.
"""

from __future__ import annotations

import ast
from pathlib import Path

from cells.step_18_eval_baseline import (
    DriftDetectionLatency,
    EvalReport,
    PerLanguageReport,
)
from cells.step_22_summary import (
    format_drift_latency_table,
    format_per_language_table,
    format_per_reward_table,
    print_summary_table,
)


def _baseline() -> EvalReport:
    return EvalReport(
        model_path="base",
        n_episodes=50,
        reward_mean_ci=(0.118, 0.086, 0.152),
        r1_mean_ci=(0.100, 0.040, 0.180),
        r2_mean_ci=(0.254, 0.198, 0.310),
        r3_mean_ci=(0.320, 0.262, 0.378),
        r4_mean_ci=(0.640, 0.588, 0.692),
        r5_mean_ci=(-0.186, -0.240, -0.132),
        brier_mean=0.412,
        floor_applied_rate=0.08,
        hallucinated_field_rate=0.14,
        reward_hacking_offenses={"hallucinated_field": 7, "bare_drift_claim": 5},
        drift_detection_latency=DriftDetectionLatency(
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), 27,
        ),
        per_language=(
            PerLanguageReport("hi", 11, 0.103, 0.09, 0.20, 0.31, 0.64, -0.18),
            PerLanguageReport("en", 10, 0.184, 0.20, 0.30, 0.38, 0.71, -0.12),
        ),
        breakdown={},
    )


def _final() -> EvalReport:
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
        reward_hacking_offenses={"hallucinated_field": 1, "bare_drift_claim": 1},
        drift_detection_latency=DriftDetectionLatency(
            stage2_mean=1.2, stage2_median=1.0, stage2_p95=2.0,
            stage3_mean=1.6, stage3_median=1.0, stage3_p95=2.0,
            undetected_count=9,
        ),
        per_language=(
            PerLanguageReport("hi", 11, 0.55, 0.6, 0.75, 0.6, 0.9, -0.05),
            PerLanguageReport("en", 10, 0.60, 0.65, 0.78, 0.65, 0.91, -0.03),
        ),
        breakdown={
            "paired_ci": {
                "reward": (0.424, 0.362, 0.486),
                "r1": (0.480, 0.372, 0.588),
                "r2": (0.486, 0.410, 0.562),
                "r3": (0.290, 0.240, 0.340),
                "r4": (0.240, 0.190, 0.290),
                "r5": (0.146, 0.096, 0.196),
            },
        },
    )


# ---------------------------------------------------------------------------
# format_per_reward_table
# ---------------------------------------------------------------------------


def test_per_reward_table_lists_all_six_channels() -> None:
    md = format_per_reward_table(_baseline(), _final())
    for key in ("REWARD", "R1", "R2", "R3", "R4", "R5"):
        assert key in md


def test_per_reward_table_renders_paired_ci_with_sign() -> None:
    md = format_per_reward_table(_baseline(), _final())
    # Positive deltas show '+'.
    assert "+0.424" in md
    assert "+0.480" in md


def test_per_reward_table_handles_missing_paired_ci() -> None:
    final = _final()
    no_paired = EvalReport(
        **{k: v for k, v in final.__dict__.items() if k != "breakdown"},
        breakdown={},
    )
    md = format_per_reward_table(_baseline(), no_paired)
    # Em-dash placeholder when paired_ci absent.
    assert "—" in md


# ---------------------------------------------------------------------------
# format_per_language_table
# ---------------------------------------------------------------------------


def test_per_language_table_includes_all_languages() -> None:
    md = format_per_language_table(_baseline(), _final())
    assert "hi" in md
    assert "en" in md


def test_per_language_table_renders_delta_with_sign() -> None:
    md = format_per_language_table(_baseline(), _final())
    # hi: 0.55 - 0.103 = +0.447
    assert "+0.447" in md or "+0.45" in md


def test_per_language_table_renders_nan_cohorts_as_NaN() -> None:
    nan = float("nan")
    base = _baseline()
    final = _final()
    from dataclasses import replace as _replace
    nan_final = _replace(
        final,
        per_language=(PerLanguageReport("kn", 0, nan, nan, nan, nan, nan, nan),),
    )
    md = format_per_language_table(base, nan_final)
    assert "NaN" in md or "—" in md


# ---------------------------------------------------------------------------
# format_drift_latency_table
# ---------------------------------------------------------------------------


def test_drift_latency_table_shows_stage2_and_stage3_rows() -> None:
    md = format_drift_latency_table(_baseline(), _final())
    assert "Stage 2" in md
    assert "Stage 3" in md
    assert "1.00" in md  # final p50 = 1.0
    assert "2.00" in md  # final p95


def test_drift_latency_table_renders_nan_for_baseline() -> None:
    md = format_drift_latency_table(_baseline(), _final())
    assert "NaN" in md


# ---------------------------------------------------------------------------
# print_summary_table
# ---------------------------------------------------------------------------


def test_print_summary_table_includes_all_sections() -> None:
    md = print_summary_table(_baseline(), _final())
    assert "## Per-reward" in md
    assert "## Per-language breakdown" in md
    assert "## Drift-detection latency" in md
    assert "## Reward-hacking offenses" in md
    assert "Baseline model" in md
    assert "Final model" in md


def test_print_summary_table_includes_offense_table() -> None:
    md = print_summary_table(_baseline(), _final())
    assert "hallucinated_field" in md
    assert "bare_drift_claim" in md


def test_print_summary_table_returns_str_only() -> None:
    md = print_summary_table(_baseline(), _final())
    assert isinstance(md, str)
    assert len(md) > 0


def test_print_summary_table_idempotent() -> None:
    a = print_summary_table(_baseline(), _final())
    b = print_summary_table(_baseline(), _final())
    assert a == b


# ---------------------------------------------------------------------------
# No LLM-as-judge static AST scan
# ---------------------------------------------------------------------------


def test_no_llm_judge_imports_in_summary_module() -> None:
    src = Path("cells/step_22_summary.py").read_text(encoding="utf-8")
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
# Integration test
# ---------------------------------------------------------------------------


def test_integration_summary_table_blog_ready() -> None:
    """End-to-end: full summary contains every claim the blog narrative needs."""
    md = print_summary_table(_baseline(), _final())
    # All 6 reward channels.
    for key in ("REWARD", "R1", "R2", "R3", "R4", "R5"):
        assert key in md
    # Both stages.
    assert "Stage 2" in md
    assert "Stage 3" in md
    # Paired Δ R1 was +0.480 [+0.372, +0.588] per evaluation.md §8.2.
    assert "+0.480" in md

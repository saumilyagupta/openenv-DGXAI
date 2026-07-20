"""Tests for cells/step_18_eval_baseline.py.

Covers evaluation.md §1, §2.1, §2.4, §3.1–§3.3, §3.8, §4, §5 for the baseline
eval entry point. Model inference is mocked end-to-end — zero CUDA calls.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cells.step_18_eval_baseline import (
    BUDGET_RUN_EVAL_SECONDS,
    DEFAULT_BOOTSTRAP_SEED,
    DriftDetectionLatency,
    EvalBudgetExceededError,
    EvalReport,
    EvaluationError,
    bootstrap_ci,
    compute_episode_seed,
    eval_baseline,
    run_eval,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BriefRow:
    episode_id: str
    seed: int = 0
    catalogue_hash: str = "drifts-v1"
    templates_sha256: str = "templates-v1"
    i18n_sha256: str = "i18n-v1"


def _fake_briefs(n: int = 60, prefix: str = "ep") -> list[_BriefRow]:
    return [_BriefRow(episode_id=f"{prefix}_{i:04d}") for i in range(n)]


def _empty_latency() -> DriftDetectionLatency:
    nan = float("nan")
    return DriftDetectionLatency(nan, nan, nan, nan, nan, nan, 0)


def _eval_report(model_path: str = "base", r1_mean: float = 0.1) -> EvalReport:
    return EvalReport(
        model_path=model_path,
        n_episodes=50,
        reward_mean_ci=(0.118, 0.086, 0.152),
        r1_mean_ci=(r1_mean, max(0.0, r1_mean - 0.06), r1_mean + 0.08),
        r2_mean_ci=(0.254, 0.198, 0.310),
        r3_mean_ci=(0.320, 0.262, 0.378),
        r4_mean_ci=(0.640, 0.588, 0.692),
        r5_mean_ci=(-0.186, -0.240, -0.132),
        brier_mean=0.412,
        floor_applied_rate=0.08,
        hallucinated_field_rate=0.14,
        reward_hacking_offenses={"hallucinated_field": 7},
        drift_detection_latency=_empty_latency(),
        per_language=(),
        curves={},
        breakdown={},
    )


def _stub_eval(report: EvalReport | None = None) -> tuple[MagicMock, EvalReport]:
    rep = report if report is not None else _eval_report()
    stub = MagicMock(return_value=rep)
    return stub, rep


# ---------------------------------------------------------------------------
# Unit tests — bootstrap_ci edge cases (evaluation.md §2.4)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_empty_returns_all_nan() -> None:
    m, lo, hi = bootstrap_ci(())
    assert math.isnan(m)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_ci_single_sample_returns_triple_v() -> None:
    assert bootstrap_ci((0.42,)) == (0.42, 0.42, 0.42)


def test_bootstrap_ci_all_identical_returns_no_variance() -> None:
    m, lo, hi = bootstrap_ci((0.5, 0.5, 0.5, 0.5, 0.5))
    assert m == 0.5
    assert lo == 0.5
    assert hi == 0.5


def test_bootstrap_ci_default_n_boot_brackets_mean() -> None:
    samples = tuple(float(i) / 50.0 for i in range(50))
    m, lo, hi = bootstrap_ci(samples, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    assert lo <= m <= hi
    assert math.isclose(m, sum(samples) / 50, abs_tol=1e-9)


def test_bootstrap_ci_deterministic_under_same_seed() -> None:
    samples = tuple(0.1 * i for i in range(50))
    a = bootstrap_ci(samples, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    b = bootstrap_ci(samples, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    assert a == b


# ---------------------------------------------------------------------------
# Episode selection — evaluation.md §3.1
# ---------------------------------------------------------------------------


def test_compute_episode_seed_is_pure_function() -> None:
    s1 = compute_episode_seed("ep_0001")
    s2 = compute_episode_seed("ep_0001")
    assert s1 == s2
    assert 0 <= s1 <= 0xFFFFFFFF


def test_run_eval_uses_first_50_briefs_in_file_order() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    run_eval("base", 50, training_eval=stub, briefs=briefs)
    call_kwargs = stub.call_args.kwargs
    assert call_kwargs["episode_ids"] == tuple(b.episode_id for b in briefs[:50])
    assert len(call_kwargs["episode_ids"]) == 50


def test_run_eval_passes_seeded_seeds_to_delegate() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    run_eval("base", 50, training_eval=stub, briefs=briefs)
    seeds = stub.call_args.kwargs["seeds"]
    expected = tuple(compute_episode_seed(b.episode_id) for b in briefs[:50])
    assert seeds == expected


def test_run_eval_rejects_non_50_episodes() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    with pytest.raises(EvaluationError, match="paired-comparison contract"):
        run_eval("base", 25, training_eval=stub, briefs=briefs)


def test_run_eval_rejects_too_few_briefs() -> None:
    briefs = _fake_briefs(10)
    stub, _ = _stub_eval()
    with pytest.raises(EvaluationError, match=">= 50 rows"):
        run_eval("base", 50, training_eval=stub, briefs=briefs)


# ---------------------------------------------------------------------------
# Sampling-policy guard — evaluation.md §3.2
# ---------------------------------------------------------------------------


def test_run_eval_pins_frozen_greedy_sampling() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    run_eval("base", 50, training_eval=stub, briefs=briefs)
    sampling = stub.call_args.kwargs["sampling"]
    assert sampling["temperature"] == 0.0
    assert sampling["top_k"] == 1
    assert sampling["num_generations"] == 1
    assert sampling["model_eval"] is True
    assert sampling["no_grad"] is True
    assert sampling["dropout_off"] is True


# ---------------------------------------------------------------------------
# Catalogue hash mismatch — evaluation.md §3.1
# ---------------------------------------------------------------------------


def test_run_eval_raises_on_catalogue_hash_mismatch() -> None:
    from cells.step_18_eval_baseline import CatalogueHashMismatchError
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    with pytest.raises(CatalogueHashMismatchError, match="drifts"):
        run_eval(
            "base",
            50,
            training_eval=stub,
            briefs=briefs,
            catalogue_hashes={"drifts": "stale", "templates": "templates-v1", "i18n": "i18n-v1"},
        )
    stub.assert_not_called()


# ---------------------------------------------------------------------------
# Wall-clock budget — evaluation.md §3.8
# ---------------------------------------------------------------------------


def test_run_eval_raises_when_budget_exceeded() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    fake_clock_values = iter([0.0, BUDGET_RUN_EVAL_SECONDS + 1.0])

    def fake_monotonic() -> float:
        return next(fake_clock_values)

    with pytest.raises(EvalBudgetExceededError, match="run_eval"):
        run_eval(
            "base", 50,
            training_eval=stub, briefs=briefs, monotonic=fake_monotonic,
        )


def test_run_eval_passes_under_budget() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    fake_clock_values = iter([0.0, 0.1, 0.1])

    def fake_monotonic() -> float:
        return next(fake_clock_values)

    out = run_eval("base", 50, training_eval=stub, briefs=briefs, monotonic=fake_monotonic)
    assert out.breakdown["wall_clock_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Episode-id stamping + zero-success warning
# ---------------------------------------------------------------------------


def test_run_eval_stamps_episode_ids_into_breakdown() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    out = run_eval("base", 50, training_eval=stub, briefs=briefs)
    assert out.breakdown["episode_ids"] == tuple(b.episode_id for b in briefs[:50])


def test_run_eval_marks_ci_undefined_for_zero_success_baseline() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval(_eval_report(model_path="base", r1_mean=0.0))
    out = run_eval("base", 50, training_eval=stub, briefs=briefs)
    assert out.breakdown.get("ci_undefined_rewards") == ["r1"]


# ---------------------------------------------------------------------------
# eval_baseline wrapper
# ---------------------------------------------------------------------------


def test_eval_baseline_defaults_model_to_base() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    eval_baseline(training_eval=stub, briefs=briefs)
    args, _ = stub.call_args
    assert args[0] == "base"
    assert args[1] == 50


def test_eval_baseline_accepts_path_object() -> None:
    briefs = _fake_briefs(60)
    stub, _ = _stub_eval()
    out = eval_baseline(Path("/tmp/ckpt"), training_eval=stub, briefs=briefs)
    args, _ = stub.call_args
    assert args[0] == Path("/tmp/ckpt")
    assert isinstance(out, EvalReport)


# ---------------------------------------------------------------------------
# No LLM-as-judge static-AST scan — evaluation.md §6.3
# ---------------------------------------------------------------------------


def test_no_llm_judge_imports_in_eval_baseline_module() -> None:
    src = Path("cells/step_18_eval_baseline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "openai" not in alias.name
                assert "anthropic" not in alias.name
                assert "vertexai" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert "openai" not in node.module
            assert "anthropic" not in node.module
            assert "vertexai" not in node.module


# ---------------------------------------------------------------------------
# Integration test — full baseline path with stubbed delegate
# ---------------------------------------------------------------------------


def test_integration_baseline_end_to_end_with_stub() -> None:
    """Stubbed `training.eval` returns a canonical baseline report; full
    paired-comparison contract validated end-to-end (evaluation.md §3.1).
    """
    briefs = _fake_briefs(60, prefix="s2_ep")
    canonical = _eval_report(model_path="base", r1_mean=0.1)
    stub = MagicMock(return_value=canonical)

    out = eval_baseline(
        training_eval=stub,
        briefs=briefs,
        catalogue_hashes={"drifts": "drifts-v1", "templates": "templates-v1", "i18n": "i18n-v1"},
    )

    assert isinstance(out, EvalReport)
    assert out.model_path == "base"
    assert out.n_episodes == 50
    assert out.breakdown["episode_ids"] == tuple(b.episode_id for b in briefs[:50])
    assert out.breakdown["sampling_policy"]["temperature"] == 0.0
    stub.assert_called_once()

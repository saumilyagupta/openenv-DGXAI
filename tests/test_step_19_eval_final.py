"""Tests for cells/step_19_eval_final.py.

Covers evaluation.md §2.1, §2.4 (paired_difference_ci), §3.1, §3.3, §3.8,
§5 EpisodeSetLeakError. Stubbed training.eval — zero CUDA.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cells.step_18_eval_baseline import (
    BUDGET_RUN_EVAL_SECONDS,
    DriftDetectionLatency,
    EvalBudgetExceededError,
    EvalReport,
    EvaluationError,
)
from cells.step_19_eval_final import (
    DEFAULT_PAIRED_BOOTSTRAP_SEED,
    EpisodeSetLeakError,
    assert_paired_episode_sets,
    eval_final,
    paired_difference_ci,
)


@dataclass(frozen=True)
class _BriefRow:
    episode_id: str
    seed: int = 0
    catalogue_hash: str = "drifts-v1"
    templates_sha256: str = "templates-v1"
    i18n_sha256: str = "i18n-v1"


def _fake_briefs(n: int = 60) -> list[_BriefRow]:
    return [_BriefRow(episode_id=f"ep_{i:04d}") for i in range(n)]


def _empty_latency() -> DriftDetectionLatency:
    nan = float("nan")
    return DriftDetectionLatency(nan, nan, nan, nan, nan, nan, 0)


def _make_report(
    model_path: str,
    episode_ids: tuple[str, ...],
    *,
    samples: dict[str, tuple[float, ...]] | None = None,
    r1_mean: float = 0.4,
) -> EvalReport:
    breakdown: dict[str, Any] = {"episode_ids": episode_ids}
    if samples is not None:
        breakdown["samples"] = samples
    return EvalReport(
        model_path=model_path,
        n_episodes=50,
        reward_mean_ci=(0.5, 0.4, 0.6),
        r1_mean_ci=(r1_mean, max(0.0, r1_mean - 0.1), r1_mean + 0.1),
        r2_mean_ci=(0.5, 0.4, 0.6),
        r3_mean_ci=(0.5, 0.4, 0.6),
        r4_mean_ci=(0.6, 0.5, 0.7),
        r5_mean_ci=(-0.1, -0.2, 0.0),
        brier_mean=0.1,
        floor_applied_rate=0.05,
        hallucinated_field_rate=0.02,
        reward_hacking_offenses={},
        drift_detection_latency=_empty_latency(),
        per_language=(),
        curves={},
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# paired_difference_ci — evaluation.md §2.4
# ---------------------------------------------------------------------------


def test_paired_difference_ci_rejects_unequal_lengths() -> None:
    with pytest.raises(EpisodeSetLeakError, match="paired-comparison invariant"):
        paired_difference_ci((0.1, 0.2), (0.1,))


def test_paired_difference_ci_empty_returns_nan_triple() -> None:
    m, lo, hi = paired_difference_ci((), ())
    assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)


def test_paired_difference_ci_is_index_paired() -> None:
    base = tuple(0.1 * i for i in range(50))
    final = tuple(0.1 * i + 0.42 for i in range(50))
    m, lo, hi = paired_difference_ci(base, final, rng_seed=DEFAULT_PAIRED_BOOTSTRAP_SEED)
    assert math.isclose(m, 0.42, abs_tol=1e-9)
    assert lo <= m <= hi


def test_paired_difference_ci_single_pair_returns_triple() -> None:
    m, lo, hi = paired_difference_ci((0.3,), (0.5,))
    assert m == 0.2
    assert lo == 0.2
    assert hi == 0.2


def test_paired_difference_ci_all_identical_diff() -> None:
    base = (0.1, 0.1, 0.1)
    final = (0.5, 0.5, 0.5)
    m, lo, hi = paired_difference_ci(base, final)
    assert math.isclose(m, 0.4, abs_tol=1e-9)
    assert lo == m == hi


# ---------------------------------------------------------------------------
# assert_paired_episode_sets — evaluation.md §3.1
# ---------------------------------------------------------------------------


def test_assert_paired_episode_sets_passes_on_match() -> None:
    ids = tuple(f"ep_{i}" for i in range(50))
    a = _make_report("base", ids)
    b = _make_report("/ckpt", ids)
    assert_paired_episode_sets(a, b)


def test_assert_paired_episode_sets_raises_on_mismatch() -> None:
    a = _make_report("base", tuple(f"ep_{i}" for i in range(50)))
    b = _make_report("/ckpt", tuple(f"ep_{i + 1}" for i in range(50)))
    with pytest.raises(EpisodeSetLeakError):
        assert_paired_episode_sets(a, b)


# ---------------------------------------------------------------------------
# eval_final entry point
# ---------------------------------------------------------------------------


def test_eval_final_rejects_str_checkpoint() -> None:
    briefs = _fake_briefs()
    baseline = _make_report("base", tuple(b.episode_id for b in briefs[:50]))
    with pytest.raises(EvaluationError, match="pathlib.Path"):
        eval_final(
            "/not/a/path",  # type: ignore[arg-type]
            baseline=baseline,
            training_eval=MagicMock(),
            briefs=briefs,
        )


def test_eval_final_rejects_non_50_episodes() -> None:
    briefs = _fake_briefs()
    baseline = _make_report("base", tuple(b.episode_id for b in briefs[:50]))
    with pytest.raises(EvaluationError, match="paired contract"):
        eval_final(
            Path("/ckpt"),
            episodes=25,
            baseline=baseline,
            training_eval=MagicMock(),
            briefs=briefs,
        )


def test_eval_final_raises_episode_set_leak_at_entry() -> None:
    briefs = _fake_briefs()
    # Baseline episode_ids do NOT match val/briefs.jsonl[0:50].
    baseline = _make_report("base", tuple(f"DIFF_{i}" for i in range(50)))
    with pytest.raises(EpisodeSetLeakError, match="paired-comparison invariant violated at entry"):
        eval_final(
            Path("/ckpt"),
            baseline=baseline,
            training_eval=MagicMock(),
            briefs=briefs,
        )


def test_eval_final_runs_paired_comparison_and_stores_paired_ci() -> None:
    briefs = _fake_briefs()
    paired_ids = tuple(b.episode_id for b in briefs[:50])
    base_samples = {
        "reward": tuple(0.1 for _ in range(50)),
        "r1": tuple(0.0 for _ in range(50)),
        "r2": tuple(0.5 for _ in range(50)),
        "r3": tuple(0.3 for _ in range(50)),
        "r4": tuple(0.6 for _ in range(50)),
        "r5": tuple(-0.2 for _ in range(50)),
    }
    final_samples = {
        "reward": tuple(0.6 for _ in range(50)),
        "r1": tuple(0.7 for _ in range(50)),
        "r2": tuple(0.8 for _ in range(50)),
        "r3": tuple(0.6 for _ in range(50)),
        "r4": tuple(0.9 for _ in range(50)),
        "r5": tuple(-0.05 for _ in range(50)),
    }
    baseline = _make_report("base", paired_ids, samples=base_samples, r1_mean=0.0)
    final_report = _make_report("/ckpt/stage3", paired_ids, samples=final_samples, r1_mean=0.7)

    stub = MagicMock(return_value=final_report)
    out = eval_final(
        Path("/ckpt/stage3"),
        baseline=baseline,
        training_eval=stub,
        briefs=briefs,
    )
    assert isinstance(out, EvalReport)
    paired = out.breakdown["paired_ci"]
    assert "reward" in paired
    assert "r1" in paired
    assert math.isclose(paired["r1"][0], 0.7, abs_tol=1e-9)


def test_eval_final_budget_exceeded_raises() -> None:
    briefs = _fake_briefs()
    paired_ids = tuple(b.episode_id for b in briefs[:50])
    baseline = _make_report("base", paired_ids)
    final_report = _make_report("/ckpt", paired_ids)
    stub = MagicMock(return_value=final_report)

    # Budget guard inside eval_final (after run_eval delegate returns).
    # Clock is invoked: outer started; run_eval inner started; run_eval inner end;
    # outer end → exceed budget on the outer end.
    times = [0.0, 0.1, 0.2, BUDGET_RUN_EVAL_SECONDS + 5.0]
    idx = {"i": 0}

    def fake_monotonic() -> float:
        i = idx["i"]
        idx["i"] = min(i + 1, len(times) - 1)
        return times[i]

    with pytest.raises(EvalBudgetExceededError):
        eval_final(
            Path("/ckpt"),
            baseline=baseline,
            training_eval=stub,
            briefs=briefs,
            monotonic=fake_monotonic,
        )


def test_eval_final_post_run_leak_guard_raises() -> None:
    briefs = _fake_briefs()
    paired_ids = tuple(b.episode_id for b in briefs[:50])
    baseline = _make_report("base", paired_ids)

    # The training-eval delegate returns a report whose episode_ids differ —
    # this simulates a corrupted val split between baseline and final.
    rogue_ids = tuple(f"ROGUE_{i}" for i in range(50))
    rogue = _make_report("/ckpt", rogue_ids)
    stub = MagicMock(return_value=rogue)

    with pytest.raises(EpisodeSetLeakError):
        eval_final(
            Path("/ckpt"),
            baseline=baseline,
            training_eval=stub,
            briefs=briefs,
        )


# ---------------------------------------------------------------------------
# No LLM-as-judge static AST scan
# ---------------------------------------------------------------------------


def test_no_llm_judge_imports_in_eval_final_module() -> None:
    src = Path("cells/step_19_eval_final.py").read_text(encoding="utf-8")
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


def test_integration_full_paired_eval_with_paired_ci_block() -> None:
    """End-to-end: baseline + final reports merge into a paired-CI summary
    consumable by the blog narrative (evaluation.md §3.3).
    """
    briefs = _fake_briefs(120)
    paired_ids = tuple(b.episode_id for b in briefs[:50])
    base_samples = {f"r{i}": tuple(0.1 for _ in range(50)) for i in range(1, 6)}
    base_samples["reward"] = tuple(0.1 for _ in range(50))
    final_samples = {f"r{i}": tuple(0.5 for _ in range(50)) for i in range(1, 6)}
    final_samples["reward"] = tuple(0.5 for _ in range(50))

    baseline = _make_report("base", paired_ids, samples=base_samples, r1_mean=0.1)
    final_rep = _make_report("/ckpt/stage3_final", paired_ids, samples=final_samples)
    stub = MagicMock(return_value=final_rep)

    out = eval_final(
        Path("/ckpt/stage3_final"),
        baseline=baseline,
        training_eval=stub,
        briefs=briefs,
    )
    paired = out.breakdown["paired_ci"]
    assert set(paired) >= {"reward", "r1", "r2", "r3", "r4", "r5"}
    for triple in paired.values():
        assert len(triple) == 3

"""Tests for cells/step_13_grpo_config.py.

Mocks ``trl.GRPOConfig`` with a permissive stub so CPU-only CI can run. The
stub captures kwargs and exposes them as attributes — the invariant asserter
is what we actually test.

Covers training_tests.md U1–U13 (config invariants), U19–U23 (reward_fn
TRL-0.23 signature), plus P7-style determinism smoke.
"""

from __future__ import annotations

import inspect
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cells.step_13_grpo_config import (
    ALLOWED_NUM_GENERATIONS,
    BETA_KL,
    DEFAULT_NUM_GENERATIONS,
    EFFECTIVE_ROLLOUTS_PER_UPDATE,
    MAX_COMPLETION_LENGTH,
    MAX_PROMPT_LENGTH,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    REPORT_TO,
    WARMUP_RATIO_STAGE1,
    WARMUP_RATIO_STAGE2_3,
    assert_config_invariants,
    build_grpo_config,
    reward_fn,
)


class _FakeGRPOConfig:
    """Stub with the exact fields training.md §2.4 lists."""

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            "bf16": False,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def _install_fake_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    trl_mod = types.ModuleType("trl")
    trl_mod.GRPOConfig = _FakeGRPOConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl_mod)


@dataclass(frozen=True)
class _FakeRewards:
    reward: float


@dataclass(frozen=True)
class _FakeEpisode:
    ident: str


def _install_fake_rewards(
    monkeypatch: pytest.MonkeyPatch, reward_value: float = 0.5
) -> MagicMock:
    mod = types.ModuleType("cells.step_08_rewards")
    fn = MagicMock(side_effect=lambda ep: _FakeRewards(reward=reward_value))
    mod.compute_rewards = fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cells.step_08_rewards", mod)
    return fn


class TestBuildGrpoConfigStage1:
    def test_warmup_ratio_0_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert math.isclose(cfg.warmup_ratio, 0.1, abs_tol=1e-9)

    def test_lr_scheduler_cosine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.lr_scheduler_type == "cosine"

    def test_num_generations_default_2_grad_accum_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v1 default aligned with Sudoku GRPO notebook: G=2, ga=2.
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.num_generations == 2
        assert cfg.gradient_accumulation_steps == 2
        # Effective rollout product is informational; Sudoku notebook = 4.
        assert cfg.num_generations * cfg.gradient_accumulation_steps == 4

    def test_run_name_stage_scoped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.run_name == "driftcall-stage1"

    def test_default_output_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.output_dir == "checkpoints/stage1"

    def test_resume_output_dir_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1, resume_output_dir=Path("/tmp/resume"))
        assert cfg.output_dir == "/tmp/resume"


class TestBuildGrpoConfigStage2And3:
    def test_stage2_warmup_is_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=2)
        assert cfg.warmup_ratio == 0.0

    def test_stage3_warmup_is_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=3)
        assert cfg.warmup_ratio == 0.0

    def test_stage2_run_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=2)
        assert cfg.run_name == "driftcall-stage2"

    def test_stage3_run_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=3)
        assert cfg.run_name == "driftcall-stage3"


class TestConfigInvariantFields:
    def test_bias_correction_kl_when_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        # TRL 0.24 (Unsloth-pinned) does not expose use_bias_correction_kl;
        # newer TRL versions do. Accept either absent OR True.
        ubc = getattr(cfg, "use_bias_correction_kl", None)
        assert ubc is None or ubc is True

    def test_fp16_true_bf16_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.fp16 is True
        assert getattr(cfg, "bf16", False) is False

    def test_gradient_checkpointing_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.gradient_checkpointing is True

    def test_per_device_batch_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.per_device_train_batch_size == PER_DEVICE_TRAIN_BATCH_SIZE
        assert cfg.per_device_train_batch_size == 1

    def test_beta_0_04(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=2)
        assert math.isclose(cfg.beta, 0.04, abs_tol=1e-9)
        assert BETA_KL == 0.04

    def test_max_lengths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=3)
        assert cfg.max_prompt_length == MAX_PROMPT_LENGTH == 1024
        assert cfg.max_completion_length == MAX_COMPLETION_LENGTH == 2048

    def test_report_to_wandb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1)
        assert cfg.report_to == "wandb"
        assert REPORT_TO == "wandb"


class TestNumGenerationsFallback:
    def test_g4_flips_grad_accum_to_8(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        cfg = build_grpo_config(stage=1, num_generations=4)
        assert cfg.num_generations == 4
        assert cfg.gradient_accumulation_steps == 8
        assert (
            cfg.num_generations * cfg.gradient_accumulation_steps
            == EFFECTIVE_ROLLOUTS_PER_UPDATE
        )

    def test_rejects_g_not_in_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        with pytest.raises(AssertionError) as exc:
            build_grpo_config(stage=1, num_generations=16)
        assert "num_generations in {4, 8}" in str(exc.value)

    def test_rejects_g_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        with pytest.raises(AssertionError):
            build_grpo_config(stage=1, num_generations=1)

    def test_allowed_set_constant(self) -> None:
        assert set(ALLOWED_NUM_GENERATIONS) == {2, 4, 8}

    def test_default_num_generations_2(self) -> None:
        # v1 default aligned with Sudoku GRPO notebook.
        assert DEFAULT_NUM_GENERATIONS == 2


class TestStageValidation:
    def test_rejects_stage_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        with pytest.raises(AssertionError):
            build_grpo_config(stage=0)  # type: ignore[arg-type]

    def test_rejects_stage_4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        with pytest.raises(AssertionError):
            build_grpo_config(stage=4)  # type: ignore[arg-type]


class TestAssertConfigInvariantsRejects:
    def _base_cfg(self, **overrides: Any) -> _FakeGRPOConfig:
        cfg = _FakeGRPOConfig(
            learning_rate=5e-6,
            warmup_ratio=WARMUP_RATIO_STAGE1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=8,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            beta=BETA_KL,
            use_bias_correction_kl=True,
            fp16=True,
            bf16=False,
            gradient_checkpointing=True,
            report_to=REPORT_TO,
            run_name="driftcall-stage1",
            output_dir="checkpoints/stage1",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_rejects_bias_correction_false(self) -> None:
        cfg = self._base_cfg(use_bias_correction_kl=False)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_fp16_false(self) -> None:
        cfg = self._base_cfg(fp16=False)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_bf16_true(self) -> None:
        cfg = self._base_cfg(bf16=True)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_gradient_checkpointing_false(self) -> None:
        cfg = self._base_cfg(gradient_checkpointing=False)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_batch_not_positive_int(self) -> None:
        # TRL >=0.24 may auto-bump pdb to satisfy pdb*ga*ws == num_generations;
        # we only require a positive int.
        cfg = self._base_cfg(per_device_train_batch_size=0)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_num_generations_mismatch(self) -> None:
        cfg = self._base_cfg(num_generations=4)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_grad_accum_not_positive_int(self) -> None:
        # The strict ga==expected check was relaxed (TRL may auto-correct);
        # we only require a positive int and the rollout-product check below.
        cfg = self._base_cfg(num_generations=4, gradient_accumulation_steps=0)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=4)

    def test_rejects_product_zero(self) -> None:
        # Effective rollout product invariant relaxed (Sudoku notebook uses
        # 2*2=4, original spec 8*4=32 — both valid). Only require >= 1.
        cfg = self._base_cfg(num_generations=2, gradient_accumulation_steps=0)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=2)

    def test_rejects_warmup_mismatch(self) -> None:
        cfg = self._base_cfg(warmup_ratio=0.0)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_beta_mismatch(self) -> None:
        cfg = self._base_cfg(beta=0.1)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_max_prompt_length_mismatch(self) -> None:
        cfg = self._base_cfg(max_prompt_length=512)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_max_completion_length_mismatch(self) -> None:
        cfg = self._base_cfg(max_completion_length=1024)
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_report_to_mismatch(self) -> None:
        cfg = self._base_cfg(report_to="tensorboard")
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_rejects_run_name_mismatch(self) -> None:
        cfg = self._base_cfg(run_name="wrong")
        with pytest.raises(AssertionError):
            assert_config_invariants(cfg, stage=1, num_generations=8)

    def test_happy_path_returns_snapshot(self) -> None:
        cfg = self._base_cfg()
        snap = assert_config_invariants(cfg, stage=1, num_generations=8)
        assert snap.stage == 1
        assert snap.num_generations == 8
        assert snap.gradient_accumulation_steps == 4
        assert snap.warmup_ratio == WARMUP_RATIO_STAGE1
        assert snap.beta == BETA_KL


class TestBuildGrpoDeterminism:
    def test_same_args_same_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        a = build_grpo_config(stage=2, num_generations=4)
        b = build_grpo_config(stage=2, num_generations=4)
        for k in (
            "learning_rate",
            "warmup_ratio",
            "num_generations",
            "gradient_accumulation_steps",
            "beta",
            "fp16",
            "run_name",
        ):
            assert getattr(a, k) == getattr(b, k)


class TestRewardFnSignature:
    def test_signature_is_trl_023_compatible(self) -> None:
        sig = inspect.signature(reward_fn)
        params = list(sig.parameters.values())
        # prompts, completions positional
        assert params[0].name == "prompts"
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
        assert params[1].name == "completions"
        # _meta and episodes must be keyword-only
        names = {p.name: p for p in params}
        assert names["_meta"].kind == inspect.Parameter.KEYWORD_ONLY
        assert names["episodes"].kind == inspect.Parameter.KEYWORD_ONLY
        # kwargs tail
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


class TestRewardFnBehavior:
    def test_returns_list_of_g_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_rewards(monkeypatch, reward_value=0.5)
        eps = [_FakeEpisode(ident=str(i)) for i in range(8)]
        res = reward_fn(
            prompts=["p"] * 8,
            completions=["c"] * 8,
            _meta=[{} for _ in range(8)],
            episodes=eps,
        )
        assert isinstance(res, list)
        assert len(res) == 8
        assert all(isinstance(x, float) for x in res)

    def test_values_in_unit_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # compute_rewards returns out-of-range — reward_fn must clamp.
        _install_fake_rewards(monkeypatch, reward_value=1.7)
        eps = [_FakeEpisode(ident="a")]
        res = reward_fn(
            prompts=["p"],
            completions=["c"],
            _meta=[{}],
            episodes=eps,
        )
        assert res == [1.0]

        _install_fake_rewards(monkeypatch, reward_value=-0.2)
        res = reward_fn(
            prompts=["p"],
            completions=["c"],
            _meta=[{}],
            episodes=eps,
        )
        assert res == [0.0]

    def test_3_decimal_precision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_rewards(monkeypatch, reward_value=0.77712345)
        eps = [_FakeEpisode(ident="a")]
        res = reward_fn(
            prompts=["p"],
            completions=["c"],
            _meta=[{}],
            episodes=eps,
        )
        assert res == [0.777]

    def test_delegates_to_compute_rewards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fn = _install_fake_rewards(monkeypatch, reward_value=0.321)
        eps = [_FakeEpisode(ident="x")]
        res = reward_fn(
            prompts=["p"],
            completions=["c"],
            _meta=[{}],
            episodes=eps,
        )
        assert res == [0.321]
        assert fn.call_count == 1
        assert fn.call_args.args == (eps[0],)

    def test_length_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_rewards(monkeypatch, reward_value=0.5)
        with pytest.raises(ValueError):
            reward_fn(
                prompts=["p", "q"],
                completions=["c"],
                _meta=[{}, {}],
                episodes=[_FakeEpisode(ident="1"), _FakeEpisode(ident="2")],
            )

    def test_meta_length_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_rewards(monkeypatch, reward_value=0.5)
        with pytest.raises(ValueError):
            reward_fn(
                prompts=["p"],
                completions=["c"],
                _meta=[{}, {}],
                episodes=[_FakeEpisode(ident="1")],
            )

    def test_no_cross_rollout_state_leak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Permuting episode order permutes the output in lockstep."""
        # Map each episode.ident to a distinct reward.
        reward_map = {"a": 0.1, "b": 0.5, "c": 0.9}
        mod = types.ModuleType("cells.step_08_rewards")

        def _cr(ep: _FakeEpisode) -> _FakeRewards:
            return _FakeRewards(reward=reward_map[ep.ident])

        mod.compute_rewards = _cr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cells.step_08_rewards", mod)

        eps = [_FakeEpisode(ident=i) for i in ("a", "b", "c")]
        original = reward_fn(
            prompts=["p"] * 3,
            completions=["c"] * 3,
            _meta=[{}] * 3,
            episodes=eps,
        )
        permuted = reward_fn(
            prompts=["p"] * 3,
            completions=["c"] * 3,
            _meta=[{}] * 3,
            episodes=[eps[2], eps[0], eps[1]],
        )
        assert original == [0.1, 0.5, 0.9]
        assert permuted == [0.9, 0.1, 0.5]


class TestStageConstants:
    def test_warmup_ratio_stage1(self) -> None:
        assert WARMUP_RATIO_STAGE1 == 0.1

    def test_warmup_ratio_stage2_3(self) -> None:
        assert WARMUP_RATIO_STAGE2_3 == 0.0

    def test_effective_rollouts_32(self) -> None:
        assert EFFECTIVE_ROLLOUTS_PER_UPDATE == 32

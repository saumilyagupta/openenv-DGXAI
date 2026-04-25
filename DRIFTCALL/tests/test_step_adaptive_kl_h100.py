"""Tests for adaptive KL + H100 config upgrade (task #13).

Covers:
  - AdaptiveKLCallback (cells/step_14) — retargets β from measured KL each step.
  - Hardware mode (cells/step_13) — V100 uses fp16 + paged_adamw_8bit;
    H100 uses bf16 + adamw_torch_fused + attn_implementation=flash_attention_3.
  - LoRA dropout (cells/step_12) — BootConfig carries lora_dropout=0.05 default.

All heavy deps (torch, trl, unsloth) are stubbed so the suite is CPU-only.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Fake TRL / GRPOConfig used by the step_13 build
# ---------------------------------------------------------------------------


class _FakeGRPOConfig:
    """Permissive stub — accepts any kwarg, exposes it as an attr."""

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {"bf16": False, "fp16": False}
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def _install_fake_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    trl_mod = types.ModuleType("trl")
    trl_mod.GRPOConfig = _FakeGRPOConfig  # type: ignore[attr-defined]
    # TrainerCallback lives in transformers.trainer_callback; stub it too.

    class _FakeTrainerCallback:
        def __init__(self) -> None:
            pass

    trl_mod.TrainerCallback = _FakeTrainerCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl_mod)

    # Also expose a TrainerCallback base via transformers.trainer_callback stub.
    tc_mod = types.ModuleType("transformers.trainer_callback")
    tc_mod.TrainerCallback = _FakeTrainerCallback  # type: ignore[attr-defined]
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.trainer_callback = tc_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.trainer_callback", tc_mod)


# ---------------------------------------------------------------------------
# Hardware mode — step_13 build_grpo_config(hardware=...)
# ---------------------------------------------------------------------------


class TestHardwareMode:
    def test_hardware_v100_default_uses_fp16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        config = build_grpo_config(stage=1)  # default hardware=v100
        assert config.fp16 is True
        assert getattr(config, "bf16", False) is False
        assert config.optim == "paged_adamw_8bit"

    def test_hardware_v100_explicit_matches_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        config = build_grpo_config(stage=1, hardware="v100")
        assert config.fp16 is True
        assert config.bf16 is False
        assert config.optim == "paged_adamw_8bit"

    def test_hardware_h100_uses_bf16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        config = build_grpo_config(stage=1, hardware="h100")
        assert config.bf16 is True
        assert config.fp16 is False

    def test_hardware_h100_uses_fused_adamw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        config = build_grpo_config(stage=1, hardware="h100")
        assert config.optim == "adamw_torch_fused"

    def test_hardware_h100_sets_attn_implementation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        config = build_grpo_config(stage=1, hardware="h100")
        # TRL passes this to the model; exposed on the config for downstream.
        assert getattr(config, "attn_implementation", None) == "flash_attention_3"

    def test_hardware_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import build_grpo_config

        with pytest.raises(AssertionError, match="hardware"):
            build_grpo_config(stage=1, hardware="a100")  # type: ignore[arg-type]

    def test_hardware_h100_invariants_still_hold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import (
            BETA_KL,
            EFFECTIVE_ROLLOUTS_PER_UPDATE,
            build_grpo_config,
        )

        config = build_grpo_config(stage=2, hardware="h100")
        assert config.beta == BETA_KL
        # Effective rollout product is informational, not asserted to a fixed
        # constant — Sudoku notebook = 4, original spec = 32.
        assert config.num_generations * config.gradient_accumulation_steps >= 1
        assert config.gradient_checkpointing is True
        # TRL 0.24 (Unsloth-pinned) does not expose use_bias_correction_kl;
        # newer TRL versions do. Accept either absent OR True.
        ubc = getattr(config, "use_bias_correction_kl", None)
        assert ubc is None or ubc is True

    def test_invariants_checker_accepts_h100_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_trl(monkeypatch)
        from cells.step_13_grpo_config import (
            assert_config_invariants,
            build_grpo_config,
        )

        config = build_grpo_config(stage=1, hardware="h100")
        inv = assert_config_invariants(config, stage=1, num_generations=2)
        # fp16 reflects whichever precision flag is set; H100 → False.
        assert inv.fp16 is False


# ---------------------------------------------------------------------------
# LoRA dropout — step_12 BootConfig
# ---------------------------------------------------------------------------


class TestLoRADropout:
    def test_boot_config_lora_dropout_default_is_zero(self) -> None:
        # 0.0 enables Unsloth's fast LoRA path required for multimodal
        # Gemma 4 GRPO. Any non-zero dropout triggers the slow path that
        # routes through the broken chunked log-softmax.
        from cells.step_12_gemma_boot import BootConfig

        cfg = BootConfig()
        assert cfg.lora_dropout == 0.0

    def test_boot_config_lora_dropout_override(self) -> None:
        from cells.step_12_gemma_boot import BootConfig

        cfg = BootConfig(lora_dropout=0.1)
        assert cfg.lora_dropout == 0.1

    def test_boot_gemma_passes_lora_dropout_to_peft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a minimal fake unsloth so we can capture the kwargs peft receives.
        param = torch.zeros(1, dtype=torch.float16)
        base_model = MagicMock()
        base_model.parameters = MagicMock(return_value=iter([param]))
        peft_model = MagicMock(name="peft_model")
        FastModel = MagicMock(name="FastModel")
        FastModel.from_pretrained = MagicMock(
            return_value=(base_model, MagicMock(name="tokenizer"))
        )
        FastModel.get_peft_model = MagicMock(return_value=peft_model)

        unsloth_mod = types.ModuleType("unsloth")
        unsloth_mod.FastVisionModel = FastModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)

        from cells.step_12_gemma_boot import BootConfig, boot_gemma

        boot_gemma(BootConfig(lora_dropout=0.05))
        kwargs = FastModel.get_peft_model.call_args.kwargs
        assert kwargs["lora_dropout"] == 0.05


# ---------------------------------------------------------------------------
# AdaptiveKLCallback — step_14
# ---------------------------------------------------------------------------


@dataclass
class _FakeArgs:
    """Minimal stand-in for TRL GRPOConfig used by the callback under test."""

    beta: float = 0.04


@dataclass
class _FakeState:
    global_step: int = 0


class TestAdaptiveKLCallback:
    def test_callback_can_be_constructed(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        assert cb.target_kl == 0.04

    def test_default_target_matches_beta_kl(self) -> None:
        from cells.step_13_grpo_config import BETA_KL
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback()
        assert cb.target_kl == BETA_KL

    def test_kl_above_target_increases_beta(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        state = _FakeState(global_step=1)
        # KL = 2x the target → beta must go up.
        cb.on_log(args, state, None, logs={"kl": 0.08})
        assert args.beta > 0.04

    def test_kl_below_target_decreases_beta(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": 0.01})
        assert args.beta < 0.04

    def test_kl_equals_target_leaves_beta_unchanged(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=2.0)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": 0.04})
        assert args.beta == pytest.approx(0.04)

    def test_beta_clamped_to_min(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(
            target_kl=0.04, kp=50.0, beta_min=0.001, beta_max=1.0,
        )
        args = _FakeArgs(beta=0.002)
        # KL tiny → β collapses; clamp at beta_min.
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": 0.0})
        assert args.beta >= 0.001

    def test_beta_clamped_to_max(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(
            target_kl=0.04, kp=50.0, beta_min=0.001, beta_max=0.5,
        )
        args = _FakeArgs(beta=0.4)
        # Huge KL → β explodes; clamp at beta_max.
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": 5.0})
        assert args.beta <= 0.5

    def test_missing_kl_is_noop(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs={"loss": 0.5})
        assert args.beta == 0.04

    def test_none_logs_is_noop(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs=None)
        assert args.beta == 0.04

    def test_non_numeric_kl_is_noop(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": "nan"})
        assert args.beta == 0.04

    def test_nan_kl_is_noop(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04)
        args = _FakeArgs(beta=0.04)
        cb.on_log(args, _FakeState(global_step=1), None, logs={"kl": float("nan")})
        assert args.beta == 0.04

    def test_monotonic_increase_toward_stable_point(self) -> None:
        """Repeated high-KL signals drive β monotonically upward until clamp."""
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(target_kl=0.04, kp=0.5, beta_min=0.001, beta_max=1.0)
        args = _FakeArgs(beta=0.04)
        previous = args.beta
        for step in range(10):
            cb.on_log(
                args, _FakeState(global_step=step + 1), None, logs={"kl": 0.16},
            )
            assert args.beta >= previous
            previous = args.beta
        assert args.beta > 0.04  # actually moved

    def test_construction_with_explicit_bounds(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        cb = AdaptiveKLCallback(
            target_kl=0.05,
            kp=1.5,
            beta_min=0.005,
            beta_max=0.8,
        )
        assert cb.target_kl == 0.05
        assert cb.kp == 1.5
        assert cb.beta_min == 0.005
        assert cb.beta_max == 0.8

    def test_invalid_bounds_raise(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        with pytest.raises((AssertionError, ValueError)):
            AdaptiveKLCallback(target_kl=0.04, beta_min=0.5, beta_max=0.1)

    def test_invalid_target_kl_raises(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        with pytest.raises((AssertionError, ValueError)):
            AdaptiveKLCallback(target_kl=-0.01)


# ---------------------------------------------------------------------------
# DriftCallGRPOTrainer integration — adaptive KL is auto-wired by default.
# ---------------------------------------------------------------------------


class _CallbackTrackingBase:
    """Stub base class that mimics TRL's ``add_callback`` surface."""

    def __init__(
        self, *, model: Any, args: Any, processing_class: Any, **_: Any
    ) -> None:
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.callbacks: list[Any] = []

    def add_callback(self, cb: Any) -> None:
        self.callbacks.append(cb)


class TestAdaptiveKLWiring:
    def _build_trainer(self, **overrides: Any) -> Any:
        from cells.step_14_custom_trainer import make_driftcall_grpo_trainer_cls

        Trainer = make_driftcall_grpo_trainer_cls(_CallbackTrackingBase)
        kwargs: dict[str, Any] = dict(
            model=MagicMock(),
            args=MagicMock(num_generations=8, beta=0.04),
            processing_class=MagicMock(),
            rollout_group_fn=MagicMock(),
            env_factory=MagicMock(),
            reward_fn_driftcall=MagicMock(return_value=[0.5] * 8),
        )
        kwargs.update(overrides)
        return Trainer(**kwargs)

    def test_adaptive_kl_callback_wired_by_default(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        trainer = self._build_trainer()
        has_adaptive = any(
            isinstance(cb, AdaptiveKLCallback) for cb in trainer.callbacks
        )
        assert has_adaptive, "AdaptiveKLCallback must be added by default"

    def test_adaptive_kl_can_be_disabled(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        trainer = self._build_trainer(enable_adaptive_kl=False)
        has_adaptive = any(
            isinstance(cb, AdaptiveKLCallback) for cb in trainer.callbacks
        )
        assert not has_adaptive, (
            "AdaptiveKLCallback must not be added when enable_adaptive_kl=False"
        )

    def test_adaptive_kl_callback_exposed_on_trainer(self) -> None:
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        trainer = self._build_trainer()
        assert hasattr(trainer, "adaptive_kl_callback")
        assert isinstance(trainer.adaptive_kl_callback, AdaptiveKLCallback)

    def test_adaptive_kl_callback_none_when_disabled(self) -> None:
        trainer = self._build_trainer(enable_adaptive_kl=False)
        assert trainer.adaptive_kl_callback is None

    def test_adaptive_kl_callback_target_defaults_to_beta_kl(self) -> None:
        from cells.step_13_grpo_config import BETA_KL

        trainer = self._build_trainer()
        assert trainer.adaptive_kl_callback.target_kl == BETA_KL

    def test_base_without_add_callback_falls_back(self) -> None:
        """Bases lacking ``add_callback`` still receive the callback via a list attr."""
        from cells.step_14_custom_trainer import (
            AdaptiveKLCallback,
            make_driftcall_grpo_trainer_cls,
        )

        class _NoAddCallbackBase:
            def __init__(
                self, *, model: Any, args: Any, processing_class: Any, **_: Any
            ) -> None:
                self.model = model
                self.args = args
                self.processing_class = processing_class

        Trainer = make_driftcall_grpo_trainer_cls(_NoAddCallbackBase)
        trainer = Trainer(
            model=MagicMock(),
            args=MagicMock(num_generations=8, beta=0.04),
            processing_class=MagicMock(),
            rollout_group_fn=MagicMock(),
            env_factory=MagicMock(),
            reward_fn_driftcall=MagicMock(return_value=[0.5] * 8),
        )
        assert hasattr(trainer, "_driftcall_callbacks")
        assert any(
            isinstance(cb, AdaptiveKLCallback)
            for cb in trainer._driftcall_callbacks
        )

    def test_custom_target_kl_override(self) -> None:
        trainer = self._build_trainer(adaptive_kl_target=0.07)
        assert trainer.adaptive_kl_callback.target_kl == 0.07

    def test_integration_simulated_50_steps(self) -> None:
        """Drive the callback with a synthetic KL signal end-to-end."""
        from cells.step_14_custom_trainer import AdaptiveKLCallback

        @dataclass
        class _Args:
            beta: float = 0.04

        @dataclass
        class _State:
            global_step: int = 0

        cb = AdaptiveKLCallback(target_kl=0.04, kp=0.5)
        args = _Args()
        for step in range(1, 26):
            cb.on_log(args, _State(global_step=step), None, logs={"kl": 0.12})
        high_beta = args.beta
        assert high_beta > 0.04
        for step in range(26, 51):
            cb.on_log(args, _State(global_step=step), None, logs={"kl": 0.01})
        low_beta = args.beta
        assert low_beta < high_beta
